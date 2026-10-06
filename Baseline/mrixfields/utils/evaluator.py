"""Validation evaluator for CUT pretraining.

Aligns with `Evaluation/evaluate.py` (official challenge evaluation):
- nRMSE, SSIM, LPIPS, PSNR — per-subject averaged, all_reduced.
- FID / KID — distribution-level via InceptionV3 features, all_reduced
  sufficient statistics (no all_gather, safe for DDP).

Supports multi-field: ``source_fields`` + ``field_stats`` for per-field
N(0,1) normalisation.  Metrics named as ``{metric}_{field}_{modality}``.
"""

from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.utils import load_nifti, list_nifti_files, extract_subject_id
from ..data.utils import ABBR_TO_SPLIT
from ..inference.base_inference import predict_volume_slice_by_slice, predict_volume_3d
from .metrics import compute_psnr, compute_ssim

_METRIC_TYPES = ("ssim", "psnr", "nrmse", "lpips")
_EXTRA_METRIC_TYPES = ("fid", "kid")
_D = torch.distributed
_FEAT_DIM = 2048


class _InceptionFeatureExtractor(nn.Module):
    """Lazy-loaded InceptionV3 feature extractor (2048-d pool3 features)."""

    def __init__(self, device: torch.device):
        super().__init__()
        self._device = device
        self._model = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            raw = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
            raw.aux_logits = False
            raw.AuxLogits = nn.Identity()
            raw.fc = nn.Identity()
            self._model = raw.to(self._device)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False
        except ImportError:
            self._model = None

    @torch.no_grad()
    def extract(self, volume: np.ndarray) -> list[torch.Tensor]:
        if self._model is None:
            return [torch.tensor([]) for _ in range(volume.shape[0])]
        C, D, H, W = volume.shape
        results = []
        for c in range(C):
            feats = []
            for i in range(0, D, self._batch_size):
                batch = torch.from_numpy(volume[c, i:i + self._batch_size]).float().to(self._device)
                batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)
                batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
                batch = (batch - 0.5) / 0.5
                feat = self._model(batch).flatten(1)
                feats.append(feat.cpu())
            results.append(torch.cat(feats, dim=0) if feats else torch.empty(0, _FEAT_DIM))
        return results


# ---- FID / KID helpers ---------------------------------------------------

def _compute_fid(mu_fake, sigma_fake, mu_real, sigma_real):
    diff = mu_fake - mu_real
    cov = sigma_fake @ sigma_real
    e, v = torch.linalg.eigh(cov)
    sqrt_cov = v @ torch.diag(e.clamp(min=0).sqrt()) @ v.T
    return float(diff.dot(diff) + torch.trace(sigma_fake + sigma_real - 2 * sqrt_cov))


def _compute_kid(fake_feats, real_feats):
    d = fake_feats.shape[1]
    g = 1.0 / d
    def kern(A, B): return (g * (A @ B.T) + 1.0) ** 3

    n, m = fake_feats.shape[0], real_feats.shape[0]
    kff = kern(fake_feats, fake_feats)
    krr = kern(real_feats, real_feats)
    kfr = kern(fake_feats, real_feats)
    mmd = (kff.sum() - kff.trace()) / (n * (n - 1)) if n > 1 else 0.0
    mmd += (krr.sum() - krr.trace()) / (m * (m - 1)) if m > 1 else 0.0
    mmd -= 2 * kfr.sum() / (n * m) if n > 0 and m > 0 else 0.0
    return float(mmd)


def _all_gather_variable(tensor: torch.Tensor) -> torch.Tensor:
    if not _D.is_initialized():
        return tensor
    world = _D.get_world_size()
    local_n = torch.tensor([tensor.shape[0]], device=tensor.device)
    all_n = [torch.zeros_like(local_n) for _ in range(world)]
    _D.all_gather(all_n, local_n)
    max_n = max(int(n.item()) for n in all_n)
    pad = torch.zeros(max_n - tensor.shape[0], tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
    padded = torch.cat([tensor, pad], dim=0)
    bufs = [torch.zeros_like(padded) for _ in range(world)]
    _D.all_gather(bufs, padded)
    parts = [bufs[r][:int(all_n[r].item())] for r in range(world)]
    return torch.cat(parts, dim=0)


class Evaluator:
    """Per-epoch validator supporting single or multiple source fields.

    Args:
        data_root:   Root dataset directory.
        split:       Validation split.
        source_field: Single source field (backward compat).
        source_fields: List of source fields (supersedes ``source_field``).
        target_field: Target field strength.
        modalities:  List of modality names.
        field_stats: Dict of per-field N(0,1) stats, e.g.
            ``{"0.1T": {"mu": [m], "sigma": [s], "bg_norm": [b]}, ...}``.
        device:      Torch device.
        **infer_kwargs: ``{slice_axis, crop_size, batch_size, mu_tgt, sigma_tgt, slice_range}``.
    """

    def __init__(self, data_root: str | Path, split: str,
                 target_field: str,
                 modalities: list[str], device: torch.device,
                 source_field: str = "",
                 source_fields: list[str] | None = None,
                 field_stats: dict | None = None,
                 metric_names=None, **infer_kwargs):
        self._data_root = Path(data_root)
        self._split = ABBR_TO_SPLIT.get(split, split)
        self._source_fields = source_fields or ([source_field] if source_field else [])
        self._target_field = target_field
        self._modalities = modalities
        self._device = device
        self._field_stats = field_stats or {}
        self._slice_axis = infer_kwargs.pop("slice_axis", 2)
        self._is_3d = infer_kwargs.pop("is_3d", False)
        self._crop_size = infer_kwargs.pop("crop_size", None)
        self._batch_size = infer_kwargs.pop("batch_size", 32)
        self._mu_tgt = infer_kwargs.pop("mu_tgt", None)
        self._sigma_tgt = infer_kwargs.pop("sigma_tgt", None)
        self._mu_src = infer_kwargs.pop("mu_src", None)
        self._sigma_src = infer_kwargs.pop("sigma_src", None)
        self._slice_range = infer_kwargs.pop("slice_range", None)
        self._metric_names = tuple(metric_names or _METRIC_TYPES)
        self._all_metric_names = tuple(
            m for m in self._metric_names if m not in _EXTRA_METRIC_TYPES
        )

        if "lpips" in self._metric_names:
            from lpips import LPIPS as _LPIPS
            self._lpips_net = _LPIPS(net="alex").to(self._device)
            self._lpips_net.eval()

        self._need_inception = any(m in self._metric_names for m in _EXTRA_METRIC_TYPES)
        self._inception = None
        if self._need_inception:
            self._inception = _InceptionFeatureExtractor(self._device)
            self._inception._batch_size = self._batch_size

    def run(self, model: torch.nn.Module | None = None, force_norm: bool = False) -> dict:
        """Evaluate model on all source fields.

        Iterates over ``self._source_fields`` and produces metrics keyed as
        ``{metric}_{field}_{modality}`` (e.g. ``ssim_0.1T_T1W``).

        When ``model`` is ``None``, baseline (source-vs-target) metrics.
        """
        all_results = {}

        for sf in self._source_fields:
            stats = self._field_stats.get(sf) if self._field_stats else None
            if stats is not None:
                mu_src = stats.get("mu")
                sigma_src = stats.get("sigma")
            else:
                mu_src = sigma_src = None
            mu_src = mu_src if mu_src is not None else self._mu_src  # fallback for single-field
            sigma_src = sigma_src if sigma_src is not None else self._sigma_src

            pairs = self._build_pairs(source_field=sf)
            N = len(pairs)
            if N == 0:
                continue

            if _D.is_initialized():
                rank, world = _D.get_rank(), _D.get_world_size()
                my_pairs = pairs[rank:N:world]
            else:
                rank, world = 0, 1
                my_pairs = pairs

            metric_keys = [f"{m}_{sf}_{mod}" for mod in self._modalities for m in self._all_metric_names]
            sums = torch.zeros(len(metric_keys), device=self._device)
            cnt = torch.zeros(1, device=self._device, dtype=torch.int64)

            if self._need_inception:
                pred_feats = [[] for _ in self._modalities]
                tgt_feats = [[] for _ in self._modalities]

            for src_paths, tgt_paths in my_pairs:
                src_vols = [load_nifti(p)[0] for p in src_paths]
                tgt_vols = [load_nifti(p)[0] for p in tgt_paths]
                src = np.stack(src_vols, axis=0)

                if self._is_3d:
                    pred = predict_volume_3d(
                        model, src, self._device,
                        mu_src=mu_src, sigma_src=sigma_src,
                        mu_tgt=self._mu_tgt, sigma_tgt=self._sigma_tgt,
                        force_norm=force_norm,
                    )
                else:
                    pred = predict_volume_slice_by_slice(
                    model, src, self._device,
                    slice_axis=self._slice_axis,
                    crop_size=self._crop_size,
                    batch_size=self._batch_size,
                    mu_src=mu_src, sigma_src=sigma_src,
                    mu_tgt=self._mu_tgt, sigma_tgt=self._sigma_tgt,
                    force_norm=force_norm,
                )

                for c, mod in enumerate(self._modalities):
                    pv = pred[c]
                    tv = tgt_vols[c]

                    if self._slice_range is not None:
                        z0, z1 = self._slice_range
                        pv = pv[z0:z1]
                        tv = tv[z0:z1]

                    msh = tuple(min(s, t) for s, t in zip(pv.shape, tv.shape))
                    pv, tv = pv[:msh[0], :msh[1], :msh[2]], tv[:msh[0], :msh[1], :msh[2]]

                    pt = torch.from_numpy(pv.copy()).float().to(self._device)
                    tt = torch.from_numpy(tv.copy()).float().to(self._device)

                    out = self._evaluate_channel(pt, tt)
                    offset = c * len(self._all_metric_names)
                    for i, name in enumerate(self._all_metric_names):
                        sums[offset + i] += out[name]

                    if self._need_inception:
                        valid = [(s, tv[s]) for s in range(msh[0])
                                 if tv[s].max() >= 1e-10 and (tv[s].max() - tv[s].min()) >= 1e-10]
                        if valid:
                            idx = [s for s, _ in valid]
                            fp = self._inception.extract(pv[idx][np.newaxis])[0]
                            ft = self._inception.extract(tv[idx][np.newaxis])[0]
                            if fp.numel() and ft.numel():
                                pred_feats[c].append(fp)
                                tgt_feats[c].append(ft)

                cnt += 1

            if _D.is_initialized():
                _D.all_reduce(sums)
                _D.all_reduce(cnt)

            n = cnt.item()
            if n > 0:
                result = {k: (sums[i].item() / n) for i, k in enumerate(metric_keys)}
                if self._need_inception:
                    fid_kid = self._compute_fid_kid(pred_feats, tgt_feats, sf)
                    result.update(fid_kid)
                all_results.update(result)

        return all_results

    # ---- per-channel metrics ---------------------------------------------------

    def _evaluate_channel(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        D = pred.shape[0]
        has_nrmse = "nrmse" in self._metric_names
        has_ssim = "ssim" in self._metric_names
        has_psnr = "psnr" in self._metric_names
        has_lpips = "lpips" in self._metric_names

        # Mask to brain region: ensure background mismatch doesn't affect metrics
        brain_mask = (target > 0).float()
        pred = pred * brain_mask
        target = target * brain_mask

        nrmse_val = 0.0
        if has_nrmse:
            denom = target.norm()
            if denom > 1e-10:
                nrmse_val = float((pred - target).norm() / denom)

        ssim_sum = psnr_sum = 0.0
        lpips_pairs = []
        n_valid = 0
        for s in range(D):
            p, t = pred[s], target[s]
            if t.abs().max().item() < 1e-10 or (t.max() - t.min()).item() < 1e-10:
                continue
            if has_ssim:
                ssim_sum += compute_ssim(p, t, data_range=1.0)
            if has_psnr:
                v = compute_psnr(p, t, data_range=1.0)
                if math.isfinite(v):
                    psnr_sum += min(v, 100.0)
            n_valid += 1
            if has_lpips:
                lpips_pairs.append((p, t))

        lpips_val = 0.0
        if has_lpips and lpips_pairs:
            pb = torch.stack([x[0] for x in lpips_pairs], dim=0)
            tb = torch.stack([x[1] for x in lpips_pairs], dim=0)
            p3 = (pb * 2 - 1).unsqueeze(1).expand(-1, 3, -1, -1)
            t3 = (tb * 2 - 1).unsqueeze(1).expand(-1, 3, -1, -1)
            with torch.no_grad():
                lpips_val = float(self._lpips_net(p3, t3).mean().item())

        n = max(n_valid, 1)
        return {
            "ssim": ssim_sum / n if has_ssim else 0.0,
            "psnr": psnr_sum / n if has_psnr else 0.0,
            "nrmse": nrmse_val,
            "lpips": lpips_val,
        }

    # ---- FID / KID ------------------------------------------------------------

    def _compute_fid_kid(self, pred_feats, tgt_feats, field: str):
        has_fid = "fid" in self._metric_names
        has_kid = "kid" in self._metric_names
        if not has_fid and not has_kid:
            return {}
        out = {}

        for c, mod in enumerate(self._modalities):
            pred_local = torch.cat(pred_feats[c], dim=0).to(self._device) if pred_feats[c] else torch.empty(0, _FEAT_DIM, device=self._device)
            tgt_local = torch.cat(tgt_feats[c], dim=0).to(self._device) if tgt_feats[c] else torch.empty(0, _FEAT_DIM, device=self._device)

            if has_fid:
                n_p = torch.tensor([pred_local.shape[0]], device=self._device)
                n_t = torch.tensor([tgt_local.shape[0]], device=self._device)
                sum_p = pred_local.sum(dim=0) if pred_local.shape[0] else torch.zeros(_FEAT_DIM, device=self._device)
                sum_t = tgt_local.sum(dim=0) if tgt_local.shape[0] else torch.zeros(_FEAT_DIM, device=self._device)
                sum_sq_p = pred_local.T @ pred_local if pred_local.shape[0] else torch.zeros(_FEAT_DIM, _FEAT_DIM, device=self._device)
                sum_sq_t = tgt_local.T @ tgt_local if tgt_local.shape[0] else torch.zeros(_FEAT_DIM, _FEAT_DIM, device=self._device)

                if _D.is_initialized():
                    _D.all_reduce(n_p); _D.all_reduce(n_t); _D.all_reduce(sum_p)
                    _D.all_reduce(sum_t); _D.all_reduce(sum_sq_p); _D.all_reduce(sum_sq_t)

                if n_p.item() >= 2 and n_t.item() >= 2:
                    mu_p = sum_p / n_p; mu_t = sum_t / n_t
                    sigma_p = sum_sq_p / n_p - mu_p.outer(mu_p)
                    sigma_t = sum_sq_t / n_t - mu_t.outer(mu_t)
                    out[f"fid_{field}_{mod}"] = _compute_fid(mu_p, sigma_p, mu_t, sigma_t)

            if has_kid:
                pred_all = _all_gather_variable(pred_local)
                tgt_all = _all_gather_variable(tgt_local)
                if pred_all.shape[0] >= 2 and tgt_all.shape[0] >= 2:
                    out[f"kid_{field}_{mod}"] = _compute_kid(pred_all, tgt_all)

        return out

    # ---- internals -------------------------------------------------------------

    def run_on_loader(self, model: nn.Module | None, loader) -> dict:
        """Batch-level evaluation — Lightning ``validation_step`` style.

        Each batch is ``{"source": tensor, "target": tensor, "field": str,
        "modality": str}``.  The evaluator calls ``model(source)``
        (or identity if ``model is None``), then accumulates per-sample
        metrics.  At the end, ``all_reduce`` aggregates across DDP ranks.

        Returns ``{metric}_{field}_{modality: value, ...}``.
        """
        from collections import defaultdict
        per_field_mod = defaultdict(lambda: {"sums": None, "cnt": 0,
                                              "metric_keys": []})
        if self._need_inception:
            for v in per_field_mod.values():
                v["pred_feats"] = []
                v["tgt_feats"] = []

        for batch in loader:
            source = batch["source"].to(self._device)
            target = batch["target"].to(self._device)
            sf = batch.get("field", "unknown")
            mod = batch.get("modality", "T1W")

            if model is not None:
                pred = model(source)
            else:
                pred = source

            key = (sf, mod)
            rec = per_field_mod[key]
            if rec["sums"] is None:
                rec["metric_keys"] = [f"{n}_{sf}_{mod}" for n in self._all_metric_names]
                rec["sums"] = torch.zeros(len(rec["metric_keys"]), device=self._device)

            out = self._evaluate_channel(pred, target)
            for i, name in enumerate(self._all_metric_names):
                rec["sums"][i] += out[name]
            rec["cnt"] += 1

            if self._need_inception:
                if isinstance(pred, torch.Tensor):
                    pred_slices = [pred[z] for z in range(pred.shape[0])
                                   if target[z].abs().max() > 1e-10]
                    tgt_slices = [target[z] for z in range(target.shape[0])
                                  if target[z].abs().max() > 1e-10]
                    if pred_slices:
                        fp = self._inception.extract(
                            torch.stack(pred_slices).unsqueeze(0).cpu().numpy())[0]
                        ft = self._inception.extract(
                            torch.stack(tgt_slices).unsqueeze(0).cpu().numpy())[0]
                        if fp.numel() and ft.numel():
                            rec["pred_feats"].append(fp)
                            rec["tgt_feats"].append(ft)

        all_results = {}
        for (sf, mod), rec in per_field_mod.items():
            if _D.is_initialized():
                cnt_t = torch.tensor([rec["cnt"]], device=self._device)
                _D.all_reduce(rec["sums"])
                _D.all_reduce(cnt_t)
                n = cnt_t.item()
            else:
                n = rec["cnt"]

            if n > 0:
                for i, k in enumerate(rec["metric_keys"]):
                    all_results[k] = rec["sums"][i].item() / n
                if self._need_inception and rec.get("pred_feats"):
                    fid_kid = self._compute_fid_kid(
                        [rec["pred_feats"]], [rec["tgt_feats"]], sf)
                    all_results.update(fid_kid)

        return all_results

    # ---- internals -------------------------------------------------------------

    def _build_pairs(self, source_field: str) -> list[tuple[list[Path], list[Path]]]:
        pairs = []
        for subj_id in self._subject_ids(source_field):
            src_paths, tgt_paths = [], []
            for mod in self._modalities:
                sp = self._file_path(mod, source_field, subj_id)
                tp = self._file_path(mod, self._target_field, subj_id)
                if not sp.exists() or not tp.exists():
                    break
                src_paths.append(sp)
                tgt_paths.append(tp)
            if len(src_paths) == len(self._modalities):
                pairs.append((src_paths, tgt_paths))
        return pairs

    def _subject_ids(self, source_field: str) -> list[str]:
        files = list_nifti_files(self._data_root, self._split, self._modalities[0], source_field)
        return sorted({extract_subject_id(f.name) for f in files})

    def _file_path(self, modality: str, field: str, subj_id: str) -> Path:
        base = self._data_root / self._split / modality / field
        prefix, numeric = subj_id.split("_", 1)
        return base / f"{prefix}_{modality}_{field}_{numeric}.nii.gz"
