"""2D MONAI slice datasets for MRIxFields2026.

Uses MONAI CacheDataset for volumetric loading and caching, then
extracts 2D axial slices on-the-fly. Replaces CachedUnpairedDataset /
CachedPairedDataset for the MONAI code path.

NIfTI paths from the same subject across modalities are passed as a list
to ``LoadImaged``, which automatically stacks them along a new channel
axis → (C, D, H, W). In ``__getitem__`` a single axial slice at z is
extracted → (C, H, W).

Preprocessing pipeline (deterministic, cached):
    LoadImaged (list of paths → stacked)
    -> EnsureChannelFirstd
    -> ScaleIntensityRanged ([0,1] -> [-1,1])
    -> EnsureTyped

When ``normalize=True``, per-channel μ/σ are computed internally from
the cached volumes and N(0,1) normalisation is applied in ``__getitem__``.
The computed stats are exposed via ``.mu``, ``.sigma``, ``.bg_norm``
properties (or ``.src_mu`` / ``.tgt_mu`` for the paired version).
"""

import bisect
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset

from .transforms import CenterCropOrPad
from .utils import (
    FIELD_TO_DOMAIN,
    extract_subject_id,
    get_joint_domain,
    list_nifti_files,
)

from monai.data import PersistentDataset
from monai.transforms import (
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAffine,
    RandAffineD,
    RandFlip,
    RandFlipD,
    RandRotate,
    RandRotateD,
    RandZoom,
    RandZoomD,
    ScaleIntensityRanged,
    SpatialCropd,
    SpatialPadd,
    Spacingd,
)


_AUGMENT_MAP = {
    "rotate": lambda kw, ks: (RandRotateD(keys=ks, range_x=kw, prob=0.5, padding_mode="constant") if ks
                              else RandRotate(range_x=kw, prob=0.5, padding_mode="constant")),
    "scale":  lambda kw, ks: (RandZoomD(keys=ks, min_zoom=kw[0], max_zoom=kw[1], prob=0.5, padding_mode="constant") if ks
                              else RandZoom(min_zoom=kw[0], max_zoom=kw[1], prob=0.5, padding_mode="constant")),
    "translate": lambda kw, ks: (RandAffineD(keys=ks, translate_range=kw, prob=0.5, padding_mode="zeros") if ks
                                 else RandAffine(translate_range=kw, prob=0.5, padding_mode="zeros")),
    "flip_lr": lambda kw, ks: (RandFlipD(keys=ks, spatial_axis=0, prob=0.5) if ks
                               else RandFlip(spatial_axis=0, prob=0.5)) if kw else None,
    "flip_ap": lambda kw, ks: (RandFlipD(keys=ks, spatial_axis=1, prob=0.5) if ks
                               else RandFlip(spatial_axis=1, prob=0.5)) if kw else None,
}


def _build_augment(augment_cfg: dict, keys=None):
    """Build augmentation Compose from config dict.

    ``keys`` controls whether transforms operate on a single tensor
    (``None``) or on multiple keys in a dict (e.g. ``["source", "target"]``).
    """
    if not augment_cfg:
        return None
    steps = []
    for name, fn in _AUGMENT_MAP.items():
        if name not in augment_cfg:
            continue
        t = fn(augment_cfg[name], keys)
        if t is not None:
            steps.append(t)
    return Compose(steps) if steps else None


def _load_norm_stats_from_yaml(yaml_path, modalities, field_strength):
    """Load per-channel μ, σ, bg_norm from a norm_stats.yaml file.

    Keys follow ``{modality}/{field_strength}``, e.g. ``T1W/0.1T``.
    Returns ``(mu, sigma, bg_norm)`` each as Python list.
    """
    import yaml as _yaml
    with open(str(yaml_path)) as f:
        data = _yaml.safe_load(f) or {}
    mu, sigma, bg = [], [], []
    for mod in modalities:
        key = f"{mod}/{field_strength}"
        mu.append(data[key]["mu"])
        sigma.append(data[key]["sigma"])
        bg.append(data[key]["bg_norm"])
    return mu, sigma, bg


def _compute_norm_stats(cache_ds, n_vols, n_channels, key="image", vol_start=0, nonzero=False):
    import torch.distributed as dist

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    my_vols = (n_vols + world_size - 1) // world_size
    start = vol_start + rank * my_vols
    end = min(start + my_vols, vol_start + n_vols)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sum_x = torch.zeros(n_channels, device=dev)
    sum_x2 = torch.zeros(n_channels, device=dev)
    n_pixels = torch.zeros(n_channels, dtype=torch.float32, device=dev)

    for vol_idx in range(start, end):
        vol = cache_ds[vol_idx][key]                     # (C, H, W, D) in [0,1]
        vol = vol * 2.0 - 1.0                             # → [-1,1]
        if dev.type == "cuda":
            vol = vol.to(dev, non_blocking=True)
        if nonzero:
            mask = vol > -1.0                              # brain voxels (bg = -1.0)
            vol = vol * mask
            sum_x += vol.sum(dim=(1, 2, 3))
            sum_x2 += (vol * vol).sum(dim=(1, 2, 3))
            n_pixels += mask.sum(dim=(1, 2, 3))            # (C,) per-channel
        else:
            sum_x += vol.sum(dim=(1, 2, 3))
            sum_x2 += (vol * vol).sum(dim=(1, 2, 3))
            n_pixels += vol.shape[1] * vol.shape[2] * vol.shape[3]

    if dist.is_initialized():
        dist.all_reduce(sum_x)
        dist.all_reduce(sum_x2)
        dist.all_reduce(n_pixels)

    mu = sum_x / n_pixels
    var = (sum_x2 / n_pixels - mu * mu).clamp(min=1e-12)
    sigma = var.sqrt()
    bg_norm = (-1.0 - mu) / sigma
    return mu.cpu().tolist(), sigma.cpu().tolist(), bg_norm.cpu().tolist()


class AutoCropSlicesd(MapTransform):
    """Crop spatial axis to range with foreground signal.

    Designed for the cache-phase transform — runs once per volume.
    Slices with fewer than ``min_frac`` non-zero voxels are trimmed.

    Args:
        keys: Keys to transform (usually ``["image"]``).
        axis: Spatial axis to crop (default -1 = last dim).
        min_frac: Minimum fraction of non-zero voxels to keep a slice.
    """

    def __init__(self, keys, axis=-1, min_frac=0.10):
        super().__init__(keys)
        self.axis = axis
        self.min_frac = min_frac

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # voxels per slice (product of all dims except axis)
            per_slice = img.shape[self.axis]
            non_zero = (img != 0).float().sum(
                dim=tuple(i for i in range(img.ndim) if i != self.axis))
            keep = non_zero > img.numel() // per_slice * self.min_frac
            idx = torch.where(keep)[0]
            if len(idx) > 0:
                slc = [slice(None)] * img.ndim
                slc[self.axis] = slice(int(idx[0]), int(idx[-1]) + 1)
                d[key] = img[tuple(slc)]
        return d


class UnpairedMONAI2DDataset(Dataset):
    """Unpaired 2D axial slice dataset with MONAI volume caching.

    Caches full 3D volumes via MONAI ``CacheDataset``, then in
    ``__getitem__`` extracts a single axial slice at a pre-determined
    z-index from the configured ``slice_range``.

    Supports multiple field strengths via ``field_strengths`` parameter.
    When multiple fields are given, volumes from all fields are merged
    into a single dataset.  Each field's N(0,1) stats are computed
    independently via ``_field_stats[field]``.

    Output dict:
        ``{"image": (C, H, W), "field_strength": ..., "modalities": ..., "filename": ...}``

    Args:
        data_root: Root data directory.
        split: Data split name.
        modality: Single modality string (backward compat).
        field_strength: Single field strength (backward compat).
        field_strengths: List of field strengths (supersedes ``field_strength``).
        modalities: Modality names list (supersedes ``modality``).
        slice_range: (z_start, z_end) axial slice range.
        slice_axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
        crop_size: Optional (H, W) to center-crop-or-pad.
        normalize: If True, compute and apply N(0,1) normalisation
            (all voxels).  Mutually exclusive with ``norm_nonzero``.
        norm_nonzero: If True, compute N(0,1) stats from brain-only voxels.
        augment: Dict of augmentation config.  ``None`` or ``{}`` disables.
        cache_dir: Directory for ``PersistentDataset`` disk cache.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: Optional[str] = None,
        field_strength: str = "",
        field_strengths: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        slice_range: Tuple[int, int] = (120, 210),
        slice_axis: int = 2,
        crop_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        norm_nonzero: bool = False,
        augment: Optional[dict] = None,
        cache_dir: Optional[str | Path] = None,
        spacing: Optional[List[float]] = None,
        clip_percentile: Optional[List[float]] = None,
        auto_slice_range: bool = False,
        auto_slice_min_frac: float = 0.10,
        num_workers: int = 0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        if modalities is not None:
            self.modalities = modalities
        elif modality is not None:
            self.modalities = [modality]
        else:
            raise ValueError(
                "UnpairedMONAI2DDataset requires 'modalities' or 'modality'."
            )
        if field_strengths is None:
            field_strengths = [field_strength] if field_strength else []
        self.field_strengths = field_strengths
        self.z0, self.z1 = slice_range
        self._n_slices = self.z1 - self.z0
        self._spatial_dim = 1 + slice_axis

        # ---- Discover NIfTI files across all fields -------------------------
        self.data_dicts = []
        self.file_names = []
        self._vol_fields: List[str] = []  # field for each volume index

        for fs in field_strengths:
            lookups = {}
            for mod in self.modalities:
                files = list_nifti_files(data_root, split, mod, fs)
                lookups[mod] = {extract_subject_id(f.name): f for f in files}
            for subj_id in list(lookups[self.modalities[0]].keys()):
                paths = []
                for mod in self.modalities:
                    p = lookups[mod].get(subj_id)
                    if p is None:
                        break
                    paths.append(str(p))
                if len(paths) == len(self.modalities):
                    self.data_dicts.append({"image": paths})
                    self.file_names.append(lookups[self.modalities[0]][subj_id].name)
                    self._vol_fields.append(fs)

        if not self.data_dicts:
            raise FileNotFoundError(
                f"No NIfTI files found for {self.modalities}/{field_strengths} "
                f"in {data_root}"
            )

        # ---- Cache transform --------------------------------------------------
        cache_steps = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
        ]
        if spacing:
            cache_steps.append(Spacingd(keys=["image"], pixdim=spacing))
        if crop_size:
            cache_steps.append(CenterSpatialCropd(
                keys=["image"], roi_size=(crop_size[0], crop_size[1], -1)))
            cache_steps.append(SpatialPadd(
                keys=["image"], spatial_size=(crop_size[0], crop_size[1], -1)))
        if auto_slice_range:
            cache_steps.append(AutoCropSlicesd(
                keys=["image"], axis=self._spatial_dim,
                min_frac=auto_slice_min_frac))
        else:
            spatial_slice = [slice(None), slice(None), slice(None)]
            spatial_slice[slice_axis] = slice(self.z0, self.z1)
            cache_steps.append(SpatialCropd(keys=["image"], roi_slices=spatial_slice))
        if clip_percentile:
            cache_steps.append(ScaleIntensityRangePercentilesd(
                keys=["image"], lower=clip_percentile[0], upper=clip_percentile[1],
                b_min=0, b_max=1, clip=True))
        cache_steps.append(EnsureTyped(keys=["image"], dtype=torch.float32))
        cache_transform = Compose(cache_steps)

        if cache_dir:
            self._cache_ds = PersistentDataset(
                data=self.data_dicts,
                transform=cache_transform,
                cache_dir=Path(cache_dir),
            )
        else:
            from monai.data import CacheDataset
            self._cache_ds = CacheDataset(
                data=self.data_dicts,
                transform=cache_transform,
                num_workers=num_workers,
            )

        # ---- Dataset length ---------------------------------------------------
        self._n_vols = len(self._cache_ds)

        # ---- Per-volume slice offset table (always built for uniform indexing) -
        self._cum_slices = [0]
        for vi in range(self._n_vols):
            D = self._cache_ds[vi]["image"].shape[-1]
            self._cum_slices.append(self._cum_slices[-1] + D)
        self._n_slices = self._cum_slices[-1]

        # ---- Per-slice transforms ---------------------------------------------
        self._augment = _build_augment(augment) if augment else None

        # ---- Per-field stats --------------------------------------------------
        self._field_stats: Dict[str, dict] = {}
        self.norm_nonzero = norm_nonzero
        do_norm = normalize or norm_nonzero
        if do_norm:
            # Compute stats for each field independently (vols are grouped)
            vol_start = 0
            for fs in field_strengths:
                n_field = self._vol_fields.count(fs)
                if n_field > 0:
                    mu, sigma, bg = _compute_norm_stats(
                        self._cache_ds, n_field, len(self.modalities),
                        vol_start=vol_start, nonzero=norm_nonzero,
                    )
                    self._field_stats[fs] = {"mu": mu, "sigma": sigma, "bg_norm": bg}
                vol_start += n_field

            # Single-field backward compat: expose as flat lists
            if len(field_strengths) == 1 and field_strengths[0] in self._field_stats:
                s = self._field_stats[field_strengths[0]]
                self.mu, self.sigma, self.bg_norm = s["mu"], s["sigma"], s["bg_norm"]
            else:
                self.mu = self.sigma = self.bg_norm = None
        else:
            self.mu = self.sigma = self.bg_norm = None

    def __len__(self) -> int:
        return self._n_slices

    def __getitem__(self, index: int) -> Dict:
        vol_idx = bisect.bisect_right(self._cum_slices, index) - 1
        z_local = index - self._cum_slices[vol_idx]
        z_global = z_local + self.z0

        vol = self._cache_ds[vol_idx]
        fs = self._vol_fields[vol_idx]

        idx = [slice(None)] * vol["image"].ndim
        idx[self._spatial_dim] = z_local
        image = vol["image"][tuple(idx)]

        if self._augment:
            image = self._augment(image)

        image = image * 2.0 - 1.0

        if self._field_stats:
            s = self._field_stats.get(fs)
            if s is not None:
                mu_t = torch.tensor(s["mu"], device=image.device).view(-1, 1, 1)
                sigma_t = torch.tensor(s["sigma"], device=image.device).view(-1, 1, 1)
                image = (image - mu_t) / sigma_t

        return {
            "image": image,
            "field_strength": fs,
            "modalities": self.modalities,
            "filename": self.file_names[vol_idx],
            "source_paths": self.data_dicts[vol_idx]["image"],
            "slice_index": f"{z_global:03d}",
        }


class PairedMONAI2DDataset(Dataset):
    """Paired source-target 2D axial slice dataset with MONAI caching.

    Source and target are loaded as a pair, cached as two 3D volumes.
    ``__getitem__`` extracts the same z from both for spatial correspondence.

    Output dict matches ``CachedPairedDataset``:
        ``{"source": (C, H, W), "target": (C, H, W), "source_field": ...,
           "target_field": ..., "modalities": ..., "filename": ...}``

    When ``normalize=True``, stats for source and target are computed
    separately and exposed via ``.src_mu``, ``.src_sigma``, ``.src_bg_norm``
    and ``.tgt_mu``, ``.tgt_sigma``, ``.tgt_bg_norm``.

    Args:
        data_root: Root data directory.
        split: Data split name.
        modality: Single modality (backward compat).
        source_field: Source field strength.
        target_field: Target field strength.
        modalities: Modality names list.
        slice_range: (z_start, z_end) axial slice range.
        slice_axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
        crop_size: Optional (H, W) to center-crop-or-pad.
        normalize: If True, apply N(0,1) normalisation (all voxels).
        norm_nonzero: If True, compute N(0,1) stats from brain-only voxels.
            Mutually exclusive with ``normalize``.
        norm_stats_path: Path to a norm_stats.yaml from pretrain.
        augment: Dict of augmentation config. Passed to ``_build_augment``
            with ``keys=["source", "target"]`` for synchronized transform.
        cache_dir: Directory for ``PersistentDataset`` disk cache.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: Optional[str] = None,
        source_field: str = "",
        target_field: str = "",
        modalities: Optional[List[str]] = None,
        slice_range: Tuple[int, int] = (120, 210),
        slice_axis: int = 2,
        crop_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        norm_nonzero: bool = False,
        norm_stats_path: Optional[str | Path] = None,
        augment: Optional[dict] = None,
        cache_dir: Optional[str | Path] = None,
        num_workers: int = 0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        if modalities is not None:
            self.modalities = modalities
        elif modality is not None:
            self.modalities = [modality]
        else:
            raise ValueError(
                "PairedMONAI2DDataset requires 'modalities' or 'modality'."
            )
        self.source_field = source_field
        self.target_field = target_field
        self.z0, self.z1 = slice_range
        self._n_slices = self.z1 - self.z0
        self._spatial_dim = 1 + slice_axis

        # ---- Discover paired NIfTI files -------------------------------------
        src_lookups = {}
        tgt_lookups = {}
        for mod in self.modalities:
            src_files = list_nifti_files(data_root, split, mod, source_field)
            tgt_files = list_nifti_files(data_root, split, mod, target_field)
            src_lookups[mod] = {extract_subject_id(f.name): f for f in src_files}
            tgt_lookups[mod] = {extract_subject_id(f.name): f for f in tgt_files}

        self.data_dicts = []
        self.file_names = []
        for subj_id in src_lookups[self.modalities[0]]:
            src_paths = [str(src_lookups[mod][subj_id]) for mod in self.modalities]
            try:
                tgt_paths = [str(tgt_lookups[mod][subj_id]) for mod in self.modalities]
            except KeyError:
                continue
            self.data_dicts.append({"source": src_paths, "target": tgt_paths})
            self.file_names.append(Path(src_paths[0]).name)

        if not self.data_dicts:
            raise FileNotFoundError(
                f"No paired volumes for {source_field} <-> {target_field} in {split}"
            )

        # ---- Cache transform (scale moved to __getitem__) ---------------------
        # roi_slices: one slice per spatial dim (H, W, D) — NOT including channel
        spatial_slice = [slice(None), slice(None), slice(None)]
        spatial_slice[slice_axis] = slice(self.z0, self.z1)
        det_transform = Compose([
            LoadImaged(keys=["source", "target"]),
            EnsureChannelFirstd(keys=["source", "target"]),
            Orientationd(keys=["source", "target"], axcodes="RAS", labels=None),
            SpatialCropd(keys=["source", "target"], roi_slices=spatial_slice),
            EnsureTyped(keys=["source", "target"], dtype=torch.float32),
        ])

        if cache_dir:
            self._cache_ds = PersistentDataset(
                data=self.data_dicts,
                transform=det_transform,
                cache_dir=Path(cache_dir),
            )
        else:
            from monai.data import CacheDataset
            self._cache_ds = CacheDataset(
                data=self.data_dicts,
                transform=det_transform,
                num_workers=num_workers,
            )

        # ---- Dataset length ---------------------------------------------------
        self._n_pairs = len(self._cache_ds)

        # ---- Per-slice transforms ---------------------------------------------
        self._crop = CenterCropOrPad(crop_size) if crop_size else None
        self._augment = _build_augment(augment, keys=["source", "target"]) if augment else None

        # ---- Stats ------------------------------------------------------------
        self.norm_nonzero = norm_nonzero
        self.src_mu = self.src_sigma = self.src_bg_norm = None
        self.tgt_mu = self.tgt_sigma = self.tgt_bg_norm = None

        do_norm = normalize or norm_nonzero
        if do_norm:
            if norm_stats_path and Path(norm_stats_path).exists():
                self.src_mu, self.src_sigma, self.src_bg_norm = _load_norm_stats_from_yaml(
                    norm_stats_path, self.modalities, source_field,
                )
                self.tgt_mu, self.tgt_sigma, self.tgt_bg_norm = _load_norm_stats_from_yaml(
                    norm_stats_path, self.modalities, target_field,
                )
            else:
                self.src_mu, self.src_sigma, self.src_bg_norm = _compute_norm_stats(
                    self._cache_ds, self._n_pairs, len(self.modalities),
                    key="source", nonzero=norm_nonzero,
                )
                self.tgt_mu, self.tgt_sigma, self.tgt_bg_norm = _compute_norm_stats(
                    self._cache_ds, self._n_pairs, len(self.modalities),
                    key="target", nonzero=norm_nonzero,
                )

    def __len__(self) -> int:
        return self._n_pairs * self._n_slices

    def __getitem__(self, index: int) -> Dict:
        pair_idx = index // self._n_slices
        z_local = index % self._n_slices
        z_global = z_local + self.z0

        item = self._cache_ds[pair_idx]

        idx = [slice(None)] * item["source"].ndim
        idx[self._spatial_dim] = z_local
        src = item["source"][tuple(idx)]
        tgt = item["target"][tuple(idx)]

        if self._crop:
            src = self._crop(src)
            tgt = self._crop(tgt)

        if self._augment:
            item = {"source": src, "target": tgt}
            item = self._augment(item)
            src, tgt = item["source"], item["target"]

        src = src * 2.0 - 1.0   # ScaleIntensityRanged [0,1] → [-1,1]
        tgt = tgt * 2.0 - 1.0

        if self.src_mu is not None:
            mu_t = torch.tensor(self.src_mu, device=src.device).view(-1, 1, 1)
            sigma_t = torch.tensor(self.src_sigma, device=src.device).view(-1, 1, 1)
            src = (src - mu_t) / sigma_t

        if self.tgt_mu is not None:
            mu_t = torch.tensor(self.tgt_mu, device=tgt.device).view(-1, 1, 1)
            sigma_t = torch.tensor(self.tgt_sigma, device=tgt.device).view(-1, 1, 1)
            tgt = (tgt - mu_t) / sigma_t

        return {
            "source": src,
            "target": tgt,
            "source_field": self.source_field,
            "target_field": self.target_field,
            "modalities": self.modalities,
            "filename": self.file_names[pair_idx],
            "source_paths": self.data_dicts[pair_idx]["source"],
            "target_paths": self.data_dicts[pair_idx]["target"],
            "slice_index": f"{z_global:03d}",
        }


class MultiDomainMONAI2DDataset(Dataset):
    """MONAI-backed single-channel dataset for StarGAN v2.

    A StarGAN domain is a field strength for single-modality training and a
    ``(modality, field strength)`` pair for multi-modality training.  Each
    sample is kept single-channel so one shared generator can translate among
    every configured domain.  Reference samples are drawn uniformly from a
    different domain, rather than in proportion to domain dataset size.

    The class composes :class:`UnpairedMONAI2DDataset` instances instead of
    duplicating their cache, preprocessing, augmentation, and normalisation
    implementation.  ``domain_stats`` is keyed by integer domain label while
    ``field_stats`` follows the evaluator's per-field/per-modality layout.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modalities: List[str],
        field_strengths: List[str],
        **dataset_kwargs,
    ):
        if not modalities:
            raise ValueError("MultiDomainMONAI2DDataset requires modalities.")
        if not field_strengths:
            raise ValueError("MultiDomainMONAI2DDataset requires field_strengths.")

        self.modalities = list(modalities)
        self.field_strengths = list(field_strengths)
        self._use_joint_domains = len(self.modalities) > 1
        self.datasets = []
        self.domain_labels = []
        self.domain_pairs = {}
        self.domain_stats = {}

        for modality in self.modalities:
            for field_strength in self.field_strengths:
                try:
                    dataset = UnpairedMONAI2DDataset(
                        data_root=data_root,
                        split=split,
                        modalities=[modality],
                        field_strength=field_strength,
                        **dataset_kwargs,
                    )
                except FileNotFoundError:
                    continue
                domain = self.domain_index(modality, field_strength)
                self.datasets.append(dataset)
                self.domain_labels.append(domain)
                self.domain_pairs[domain] = (modality, field_strength)
                if dataset.mu is not None:
                    self.domain_stats[domain] = {
                        "mu": dataset.mu[0],
                        "sigma": dataset.sigma[0],
                        "bg_norm": dataset.bg_norm[0],
                    }

        if len(self.datasets) < 2:
            raise FileNotFoundError(
                "StarGAN v2 needs at least two non-empty MONAI domains; "
                f"found {len(self.datasets)} for modalities={self.modalities}, "
                f"fields={self.field_strengths}."
            )

        self._concat = ConcatDataset(self.datasets)
        self._field_stats = self._build_field_stats()

    def domain_index(self, modality: str, field_strength: str) -> int:
        if self._use_joint_domains:
            return get_joint_domain(modality, field_strength)
        return FIELD_TO_DOMAIN[field_strength]

    def _build_field_stats(self) -> Dict[str, dict]:
        field_stats = {}
        for field_strength in self.field_strengths:
            stats = []
            for modality in self.modalities:
                domain = self.domain_index(modality, field_strength)
                if domain not in self.domain_stats:
                    stats = []
                    break
                stats.append(self.domain_stats[domain])
            if stats:
                field_stats[field_strength] = {
                    key: [item[key] for item in stats]
                    for key in ("mu", "sigma", "bg_norm")
                }
        return field_stats

    def __len__(self) -> int:
        return len(self._concat)

    def _dataset_index(self, index: int) -> Tuple[int, int]:
        dataset_idx = bisect.bisect_right(self._concat.cumulative_sizes, index)
        previous = 0 if dataset_idx == 0 else self._concat.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, index - previous

    def __getitem__(self, index: int) -> Dict:
        dataset_idx, sample_idx = self._dataset_index(index)
        source = self.datasets[dataset_idx][sample_idx]
        source_domain = self.domain_labels[dataset_idx]

        choices = [i for i in range(len(self.datasets)) if i != dataset_idx]
        ref_dataset_idx = random.choice(choices)
        ref_dataset = self.datasets[ref_dataset_idx]
        reference = ref_dataset[random.randrange(len(ref_dataset))]
        reference2 = ref_dataset[random.randrange(len(ref_dataset))]
        ref_domain = self.domain_labels[ref_dataset_idx]
        ref_modality, ref_field = self.domain_pairs[ref_domain]

        return {
            **source,
            "domain": source_domain,
            "modality": self.domain_pairs[source_domain][0],
            "ref_image": reference["image"],
            "ref_image2": reference2["image"],
            "ref_domain": ref_domain,
            "ref_modality": ref_modality,
            "ref_field_strength": ref_field,
            "ref_filename": reference["filename"],
            "ref_slice_index": reference["slice_index"],
        }
