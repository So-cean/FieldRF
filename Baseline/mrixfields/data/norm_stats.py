"""Per-(modality, field) normalisation statistics.

``NormStats`` uses a Dataset + DataLoader to compute per-channel μ, σ, bg_norm
on GPU.  Results are written to (and re-read from) a YAML file for persistence.
"""

import os
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import yaml


class NormStats:
    """Compute or load normalisation statistics for a dataset.

    Args:
        yaml_path: Path to the YAML cache (e.g. ``{preprocessed_dir}/norm_stats.yaml``).

    Usage::

        stats = NormStats(yaml_path)
        norm_infos = stats.ensure(dataset, device)
        # norm_infos ≡ [{mu, sigma, bg_norm}, …]  one per channel
    """

    def __init__(self, yaml_path: str | Path):
        self._path = Path(yaml_path)
        self._data: dict = {}

    def ensure(self, dataset, device) -> list:
        """Ensure stats exist for *all* ``(modality, dataset.field_strength)`` pairs
        that this dataset covers.

        When a key is missing, *all ranks* participate in computing it:
        each rank processes its own data shard (via ``DistributedSampler``),
        accumulates partial sums, then ``all_reduce`` merges across ranks.

        The result is written atomically to the YAML cache by each rank
        (identical content, the last one wins).

        Returns:
            ``[{mu, sigma, bg_norm}, …]``  one per modality in *dataset.modalities*.
        """
        self._lazy_load()
        keys = [f"{m}/{dataset.field_strength}" for m in dataset.modalities]
        missing = [k for k in keys if k not in self._data]

        if missing:
            self._compute_missing(dataset, device, missing)

        return [self._data[k] for k in keys]

    # ------------------------------------------------------------------
    #  internal
    # ------------------------------------------------------------------

    def _compute_missing(self, dataset, device, keys: list[str]):
        D = torch.distributed

        if D.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False,
            )
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=256,
            sampler=sampler,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        C = len(dataset.modalities)
        sum_x = torch.zeros(C, device=device)
        sum_x2 = torch.zeros(C, device=device)
        n_pixels = torch.zeros(C, dtype=torch.float32, device=device)

        nonzero = getattr(dataset, "norm_nonzero", False)

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)   # (B, C, H, W)
            if nonzero:
                mask = img > -1.0
                img = img * mask
                sum_x += img.sum(dim=(0, 2, 3))
                sum_x2 += (img * img).sum(dim=(0, 2, 3))
                n_pixels += mask.sum(dim=(0, 2, 3))
            else:
                sum_x += img.sum(dim=(0, 2, 3))
                sum_x2 += (img * img).sum(dim=(0, 2, 3))
                n_pixels += img.shape[0] * img.shape[2] * img.shape[3]

        # Merge across GPUs
        if D.is_initialized():
            D.all_reduce(sum_x)
            D.all_reduce(sum_x2)
            D.all_reduce(n_pixels)

        mu = sum_x / n_pixels
        var = (sum_x2 / n_pixels - mu * mu).clamp(min=1e-12)
        sigma = var.sqrt()
        bg = (-1.0 - mu) / sigma

        for c, k in enumerate(keys):
            self._data[k] = {
                "mu": mu[c].item(),
                "sigma": sigma[c].item(),
                "bg_norm": bg[c].item(),
            }

        self._flush()

    def _lazy_load(self):
        if self._path.exists():
            with open(self._path) as f:
                self._data = yaml.safe_load(f) or {}

    def _flush(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Unique tmp file per process to avoid DDP race
        tmp = self._path.parent / f".norm_stats_{os.getpid()}.yaml"
        with open(tmp, "w") as f:
            yaml.dump(self._data, f, default_flow_style=None)
        try:
            os.replace(tmp, self._path)
        except OSError:
            # Another rank already wrote the final file — discard ours
            if tmp.exists():
                tmp.unlink()
