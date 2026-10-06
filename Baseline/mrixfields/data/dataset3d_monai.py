"""3D MONAI datasets for MRIxFields2026.

Uses MONAI CacheDataset for efficient volumetric loading and MONAI transforms
for preprocessing that matches scripts/preprocess.py parameters.

Supports arbitrary number of modalities via the ``modalities`` parameter.
NIfTI paths from the same subject across modalities are passed as a list
to ``LoadImaged``, which automatically loads and stacks them along a new
channel axis → (C, D, H, W).

Preprocessing pipeline (deterministic, cached):
    LoadImaged (list of paths → stacked)
    -> EnsureChannelFirstd
    -> CropForegroundd (remove all-zero background margins)
    -> ScaleIntensityRanged ([0,1] -> [-1,1])
    -> EnsureTyped

Random transform (per-epoch):
    SpatialPadd (if volume < patch_size) -> RandSpatialCropd (random patch)

Usage:
    # Single modality (backward compatible)
    from mrixfields.data.dataset3d_monai import UnpairedMONAI3DDataset
    ds = UnpairedMONAI3DDataset(
        data_root="/path/to/data",
        split="Training_retrospective",
        modality="T1W",
        field_strength="0.1T",
        patch_size=(128, 128, 128),
        cache_rate=1.0,
    )

    # Multi-modality
    ds = UnpairedMONAI3DDataset(
        data_root="/path/to/data",
        split="Training_retrospective",
        modalities=["T1W", "T2W"],
        field_strength="0.1T",
        patch_size=(128, 128, 128),
    )
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import list_nifti_files, extract_subject_id

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    EnsureTyped,
)


class UnpairedMONAI3DDataset(Dataset):
    """Unpaired 3D volume dataset with MONAI CacheDataset + random patch cropping.

    **Modality modality**: modalities within the same field strength are
    aligned (same subject). Field strengths are unpaired (different subjects).

    Each cached volume can contribute multiple random patches per epoch,
    increasing the effective dataset size without storing duplicated data.

    ``LoadImaged`` receives a list of NIfTI paths (one per modality) and
    automatically stacks them along channel dim 0 → ``(C, D, H, W)``.
    Random patch cropping is applied once on the stacked channels, so all
    modalities within a subject share the same spatial patch (modalites
    aligned within the same field strength).

    Args:
        data_root: Root data directory.
        split: Data split name, e.g. "Training_retrospective".
        modality: Single modality string (kept for backward compatibility).
        field_strength: Field strength, e.g. "0.1T".
        modalities: Arbitrary number of modality names, e.g. ["T1W", "T2W"].
            When provided, ``modality`` is ignored.
        patch_size: Spatial size of random patches (D, H, W).
        transform: Optional custom deterministic transform for caching.
            If None, defaults to percentile clip + [-1, 1] scaling.
        cache_rate: MONAI cache_rate (1.0 = cache all, 0.0 = no cache).
        cache_num_workers: Workers for building the cache.
        samples_per_volume: Number of random patches to extract from each
            volume per epoch. Effectively multiplies dataset size.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: Optional[str] = None,
        field_strength: str = "",
        modalities: Optional[List[str]] = None,
        patch_size: Sequence[int] = (128, 128, 128),
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        cache_num_workers: int = 0,
        cache_dir: Optional[str | Path] = None,
        samples_per_volume: int = 16,
        seg_root: Optional[str | Path] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        if modalities is not None:
            self.modalities = modalities
        elif modality is not None:
            self.modalities = [modality]
        else:
            raise ValueError(
                "UnpairedMONAI3DDataset requires either 'modalities' (list) or 'modality' (str). "
                "Neither was provided."
            )
        self.field_strength = field_strength
        self.patch_size = tuple(patch_size)
        self.samples_per_volume = samples_per_volume
        self._seg_root = Path(seg_root) if seg_root else None
        self._last_vol_idx = -1
        self._last_patches = None

        lookups = {}
        for mod in self.modalities:
            files = list_nifti_files(data_root, split, mod, field_strength)
            lookups[mod] = {extract_subject_id(f.name): f for f in files}

        self.data_dicts = []
        self.file_names = []
        for subj_id, base_path in lookups[self.modalities[0]].items():
            paths = []
            for mod in self.modalities:
                path = lookups[mod].get(subj_id)
                if path is None:
                    break
                paths.append(str(path))
            if len(paths) == len(self.modalities):
                item = {"image": paths}
                # Attach seg path if available
                if self._seg_root is not None:
                    seg_path = self._resolve_seg(paths[0])
                    if seg_path is not None:
                        item["seg"] = seg_path
                self.data_dicts.append(item)
                self.file_names.append(base_path.name)

        if not self.data_dicts:
            raise FileNotFoundError(
                f"No NIfTI files found for {self.modalities}/{field_strength} "
                f"in {data_root}"
            )

        self._has_seg = "seg" in self.data_dicts[0]

        # Cache transform
        cache_keys = ["image"]
        cache_loaders = [LoadImaged(keys=["image"])]
        if self._has_seg:
            cache_keys.append("seg")
            cache_loaders = [LoadImaged(keys=["image", "seg"])]

        if transform is not None:
            self._cache_transform = transform
        else:
            steps = cache_loaders + [
                EnsureChannelFirstd(keys=cache_keys),
                CropForegroundd(keys=["image"], source_key="image"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0.0,
                    a_max=1.0,
                    b_min=-1.0,
                    b_max=1.0,
                ),
                EnsureTyped(keys=cache_keys, dtype=torch.float32),
            ]
            if self._has_seg:
                steps.append(EnsureTyped(keys=["seg"], dtype=torch.uint8))
            self._cache_transform = Compose(steps)

        if cache_dir:
            from monai.data import PersistentDataset
            self._cache_ds = PersistentDataset(
                data=self.data_dicts,
                transform=self._cache_transform,
                cache_dir=Path(cache_dir),
            )
        else:
            self._cache_ds = CacheDataset(
                data=self.data_dicts,
                transform=self._cache_transform,
                cache_rate=cache_rate,
                num_workers=cache_num_workers,
            )
        self._cache_dir = cache_dir

        # Per-epoch random transform
        pad = SpatialPadd(keys=["image"], spatial_size=self.patch_size, method="symmetric")
        if self._has_seg:
            pad = SpatialPadd(keys=["image", "seg"], spatial_size=self.patch_size, method="symmetric")
            self._cropper = RandCropByPosNegLabeld(
                keys=["image"], label_key="seg",
                spatial_size=self.patch_size,
                pos=5.0, neg=1.0,
                num_samples=samples_per_volume,
            )
            self._rand_transform = Compose([pad, self._cropper])
        else:
            self._rand_transform = Compose([
                pad,
                RandSpatialCropd(keys=["image"], roi_size=self.patch_size, random_size=False),
            ])

    def _resolve_seg(self, image_path: str) -> Optional[str]:
        """Derive segmentation path from an image path."""
        try:
            rel = Path(image_path).relative_to(self.data_root)
        except ValueError:
            return None
        seg_path = self._seg_root / rel
        seg_path = seg_path.with_name(seg_path.stem + "_seg.nii.gz")
        return str(seg_path) if seg_path.exists() else None

    def __len__(self) -> int:
        return len(self._cache_ds) * self.samples_per_volume

    def __getitem__(self, index: int) -> Dict:
        vol_idx = index // self.samples_per_volume
        crop_idx = index % self.samples_per_volume

        if self._has_seg:
            # Cache cropper output per volume (generates samples_per_volume patches)
            if vol_idx != self._last_vol_idx:
                item = self._cache_ds[vol_idx]
                self._last_patches = self._rand_transform(item)
                if isinstance(self._last_patches, dict):
                    self._last_patches = [self._last_patches]
                self._last_vol_idx = vol_idx
            item = dict(self._last_patches[crop_idx])
        else:
            item = self._cache_ds[vol_idx]
            item = self._rand_transform(item)

        return {
            "image": item["image"],
            "field_strength": self.field_strength,
            "modalities": self.modalities,
            "filename": self.file_names[vol_idx],
        }


class PairedMONAI3DDataset(Dataset):
    """Paired 3D volume dataset with synchronized random patch cropping.

    **Source/Target modality**: modalities AND field strengths are both aligned
    (same subject across source and target field strengths).

    Synchronized cropping ensures that source and target are cropped at the
    **same spatial location** (cat → single crop → split). Within each group
    (source or target), all modalities share the same crop location as well.

    Args:
        data_root: Root data directory.
        split: Data split name.
        modality: Single modality string (backward compatibility).
        source_field: Source field strength.
        target_field: Target field strength.
        modalities: Arbitrary number of modality names.
            When provided, ``modality`` is ignored.
        patch_size: Spatial patch size (D, H, W).
        cache_rate: MONAI cache_rate.
        cache_num_workers: Workers for building the cache.
        samples_per_volume: Number of random patches to extract from each
            paired volume per epoch.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: Optional[str] = None,
        source_field: str = "",
        target_field: str = "",
        modalities: Optional[List[str]] = None,
        patch_size: Sequence[int] = (128, 128, 128),
        cache_rate: float = 1.0,
        cache_num_workers: int = 0,
        cache_dir: Optional[str | Path] = None,
        samples_per_volume: int = 16,
    ):
        self.data_root = Path(data_root)
        self.split = split
        if modalities is not None:
            self.modalities = modalities
        elif modality is not None:
            self.modalities = [modality]
        else:
            raise ValueError(
                "PairedMONAI3DDataset requires either 'modalities' (list) or 'modality' (str). "
                "Neither was provided."
            )
        self.source_field = source_field
        self.target_field = target_field
        self.patch_size = tuple(patch_size)
        self.samples_per_volume = samples_per_volume

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
                f"No paired volumes found for {source_field} <-> {target_field} "
                f"in {split}"
            )

        deterministic = Compose([
            LoadImaged(keys=["source", "target"]),
            EnsureChannelFirstd(keys=["source", "target"]),
            CropForegroundd(keys=["source", "target"], source_key="source"),
            ScaleIntensityRanged(
                keys=["source", "target"],
                a_min=0.0,
                a_max=1.0,
                b_min=-1.0,
                b_max=1.0,
            ),
            EnsureTyped(keys=["source", "target"], dtype=torch.float32),
        ])

        if cache_dir:
            from monai.data import PersistentDataset
            self._cache_ds = PersistentDataset(
                data=self.data_dicts,
                transform=deterministic,
                cache_dir=Path(cache_dir),
            )
        else:
            self._cache_ds = CacheDataset(
                data=self.data_dicts,
                transform=deterministic,
                cache_rate=cache_rate,
                num_workers=cache_num_workers,
            )
        self._cache_dir = cache_dir

        self._pad = SpatialPadd(
            keys=["source", "target"], spatial_size=self.patch_size, method="symmetric"
        )
        self._crop = RandSpatialCropd(
            keys=["image"], roi_size=self.patch_size, random_size=False
        )

    def __len__(self) -> int:
        return len(self._cache_ds) * self.samples_per_volume

    def __getitem__(self, index: int) -> Dict:
        vol_idx = index // self.samples_per_volume
        item = self._cache_ds[vol_idx]
        item = self._pad(item)

        C = len(self.modalities)
        # Synchronized crop: cat source and target → single _crop → split
        src = item["source"]                     # (C, D, H, W)
        tgt = item["target"]                     # (C, D, H, W)
        combined = torch.cat([src, tgt], dim=0)  # (2*C, D, H, W)
        cropped = self._crop({"image": combined})["image"]  # (2*C, pd, ph, pw)

        return {
            "source": cropped[:C],
            "target": cropped[C:],
            "source_field": self.source_field,
            "target_field": self.target_field,
            "modalities": self.modalities,
            "filename": self.file_names[vol_idx],
        }
