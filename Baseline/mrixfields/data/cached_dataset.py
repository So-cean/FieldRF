"""Cached PyTorch Dataset classes for preprocessed npz slices.

These load pre-extracted 2D slices from npz files (created by preprocess.py
extract-slices), avoiding the overhead of re-reading 3D NIfTI volumes.

Data structure:
    {preprocessed_dir}/{split}/{modality}/{field_strength}/*.npz
    Each npz contains: image (H, W) float32 in [0,1], slice_idx int32

Output tensors are scaled to [-1, 1] (standard for GAN training).

Three dataset types mirror the on-the-fly versions in dataset.py:
    - CachedUnpairedDataset: Single-domain slices (CUT/CycleGAN)
    - CachedPairedDataset: Matched source-target slices (Hybrid fine-tuning)
    - CachedMultiDomainDataset: All domains (StarGAN v2)
"""

import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import CenterCropOrPad, ToTensor, ScaleToMinusOneOne, Compose
from .utils import FIELD_STRENGTHS, FIELD_TO_DOMAIN, MODALITIES, get_joint_domain


def _cached_transform(crop_size: Optional[Tuple[int, int]] = None) -> Compose:
    """Transform for cached data (npz stored as [0,1], scaled to [-1,1]).

    Args:
        crop_size: Target (H, W) for CenterCropOrPad. None to skip cropping.
    """
    steps = []
    if crop_size is not None:
        steps.append(CenterCropOrPad(crop_size))
    steps += [ToTensor(), ScaleToMinusOneOne()]
    return Compose(steps)


def _list_npz_files(base_dir: Path, split: str, modality: str, field: str) -> List[Path]:
    """List npz files for a given split/modality/field."""
    d = base_dir / split / modality / field
    if not d.exists():
        return []
    return sorted(d.glob("*.npz"))


class CachedUnpairedDataset(Dataset):
    """Single-domain 2D slice dataset from preprocessed npz files.

    Used by CUT and CycleGAN. Two instances (source + target) are
    paired via UnpairedDataLoader.

    Supports arbitrary number of modalities via the ``modalities`` parameter.
    When multiple modalities are provided, slices are concatenated along the
    channel dimension (C, H, W).

    When ``norm_infos`` is provided (a list of ``{mu, sigma, bg_norm}`` per
    channel), the returned tensor is normalised to N(0,1) per channel.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: Optional[str] = None,
        field_strength: str = "",
        modalities: Optional[List[str]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        norm_infos: Optional[List[dict]] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        if modalities is not None:
            self.modalities = modalities
        elif modality is not None:
            self.modalities = [modality]
        else:
            raise ValueError(
                "CachedUnpairedDataset requires either 'modalities' (list) or 'modality' (str). "
                "Neither was provided."
            )
        self.field_strength = field_strength
        self.transform = transform or _cached_transform(crop_size)
        self._norm_infos = norm_infos or []

        def _modality_key(path: Path) -> str:
            parts = path.stem.split("_")
            return "_".join(parts[:2] + parts[3:])  # skip modality at index 2

        lookups = {}
        for mod in self.modalities:
            files = _list_npz_files(self.preprocessed_dir, split, mod, field_strength)
            lookups[mod] = {_modality_key(f): f for f in files}

        base_lookup = lookups[self.modalities[0]]
        self.files = []
        for key, base_path in base_lookup.items():
            group = []
            for mod in self.modalities:
                path = lookups[mod].get(key)
                if path is None:
                    break
                group.append(path)
            if len(group) == len(self.modalities):
                self.files.append(tuple(group))
        if not self.files:
            raise FileNotFoundError(
                f"No cached files found for {self.modalities} in {self.preprocessed_dir / split}."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        slices = [np.load(path)["image"] for path in self.files[index]]
        if self.transform:
            slices = [self.transform(img) for img in slices]
        image = torch.cat(slices, dim=0)

        if self._norm_infos:
            channels = []
            for c in range(len(self.modalities)):
                info = self._norm_infos[c]
                channels.append((image[c] - info["mu"]) / info["sigma"])
            image = torch.stack(channels, dim=0)

        # filename & slice from first modality npz
        fp0 = Path(self.files[index][0])
        filename = fp0.name
        m = re.search(r"_s(\d+)$", fp0.stem)
        slice_index = m.group(1) if m else ""

        return {
            "image": image,
            "filename": filename,
            "slice_index": slice_index,
            "source_paths": [str(p) for p in self.files[index]],
            "field_strength": self.field_strength,
            "modalities": self.modalities,
        }


class CachedPairedDataset(Dataset):
    """Paired source-target 2D slice dataset from preprocessed npz files.

    Used by Hybrid fine-tuning. Matches slices by filename — both source
    and target directories must have been extracted from the same subjects
    (Prospective data).

    Pairing: slices are matched by volume name prefix (e.g., "001_GX").
    For each source slice, the corresponding target slice at the same
    slice index is used.

    Supports arbitrary number of modalities via the ``modalities`` parameter.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: Optional[str] = None,
        source_field: str = "",
        target_field: str = "",
        modalities: Optional[List[str]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        src_norm_infos: Optional[List[dict]] = None,
        tgt_norm_infos: Optional[List[dict]] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        if modalities is not None:
            self.modalities = modalities
        elif modality is not None:
            self.modalities = [modality]
        else:
            raise ValueError(
                "CachedPairedDataset requires either 'modalities' (list) or 'modality' (str). "
                "Neither was provided."
            )
        self.transform = transform or _cached_transform(crop_size)
        self._src_norm_infos = src_norm_infos or []
        self._tgt_norm_infos = tgt_norm_infos or []

        def _pair_key(path: Path) -> str:
            parts = path.stem.split("_")
            return "_".join(parts[-3:])  # subject_prefix + ID + slice, e.g. "R_0001_s072"

        src_lookups = {}
        tgt_lookups = {}
        for mod in self.modalities:
            src_files = _list_npz_files(self.preprocessed_dir, split, mod, source_field)
            tgt_files = _list_npz_files(self.preprocessed_dir, split, mod, target_field)
            src_lookups[mod] = {_pair_key(f): f for f in src_files}
            tgt_lookups[mod] = {_pair_key(f): f for f in tgt_files}

        self.pairs = []
        for key in src_lookups[self.modalities[0]]:
            src_group = [src_lookups[mod][key] for mod in self.modalities]
            # Check if all target modalities exist for this key
            try:
                tgt_group = [tgt_lookups[mod][key] for mod in self.modalities]
            except KeyError:
                continue
            self.pairs.append((tuple(src_group), tuple(tgt_group)))

        if not self.pairs:
            raise ValueError(
                f"No matching pairs found between {source_field} and {target_field}. "
                f"Ensure both were extracted from the same subjects (e.g., Prospective)."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        src_paths, tgt_paths = self.pairs[index]

        src_slices = [np.load(p)["image"] for p in src_paths]
        tgt_slices = [np.load(p)["image"] for p in tgt_paths]
        if self.transform:
            src_slices = [self.transform(s) for s in src_slices]
            tgt_slices = [self.transform(s) for s in tgt_slices]
        src_img = torch.cat(src_slices, dim=0)
        tgt_img = torch.cat(tgt_slices, dim=0)

        if self._src_norm_infos:
            channels = []
            for c in range(len(self.modalities)):
                info = self._src_norm_infos[c]
                channels.append((src_img[c] - info["mu"]) / info["sigma"])
            src_img = torch.stack(channels, dim=0)
        if self._tgt_norm_infos:
            channels = []
            for c in range(len(self.modalities)):
                info = self._tgt_norm_infos[c]
                channels.append((tgt_img[c] - info["mu"]) / info["sigma"])
            tgt_img = torch.stack(channels, dim=0)

        fp0 = Path(src_paths[0])
        filename = fp0.name
        m = re.search(r"_s(\d+)$", fp0.stem)
        slice_index = m.group(1) if m else ""

        return {
            "source": src_img,
            "target": tgt_img,
            "filename": filename,
            "slice_index": slice_index,
            "source_paths": [str(p) for p in src_paths],
            "target_paths": [str(p) for p in tgt_paths],
            "modalities": self.modalities,
        }


class CachedMultiDomainDataset(Dataset):
    """Multi-domain 2D slice dataset from preprocessed npz for StarGAN v2.

    Returns: image + domain label + reference image from a different domain.

    Supports multi-modality training. When multiple modalities are provided,
    domain labels use ``get_joint_domain`` (0..14 for 3 modalities × 5 fields).
    For a single modality, labels fall back to ``FIELD_TO_DOMAIN`` (0..4).
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        field_strengths: List[str] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        if modalities is not None:
            self.modalities = list(modalities)
        elif modality is not None:
            self.modalities = [modality]
        else:
            self.modalities = list(MODALITIES)
        self.field_strengths = field_strengths or FIELD_STRENGTHS
        self.transform = transform or _cached_transform(crop_size)
        self._use_joint = len(self.modalities) > 1

        # Index files per domain
        self.domain_files: Dict[int, List[Path]] = {}
        self.samples: List[Tuple[Path, int, str, str]] = []  # (npz_path, domain_idx, modality, field)
        self._domain_to_pair: Dict[int, Tuple[str, str]] = {}

        for modality_name in self.modalities:
            for fs in self.field_strengths:
                domain_idx = (
                    get_joint_domain(modality_name, fs) if self._use_joint
                    else FIELD_TO_DOMAIN[fs]
                )
                files = _list_npz_files(self.preprocessed_dir, split, modality_name, fs)
                if files:
                    self.domain_files.setdefault(domain_idx, []).extend(files)
                    self._domain_to_pair[domain_idx] = (modality_name, fs)
                    for f in files:
                        self.samples.append((f, domain_idx, modality_name, fs))

        if not self.samples:
            where = self.preprocessed_dir / split
            raise FileNotFoundError(
                f"No npz files found under {where} for modalities={self.modalities}, "
                f"fields={self.field_strengths}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        npz_path, domain_idx, modality_name, field_strength = self.samples[index]
        image = np.load(npz_path)["image"]

        # Sample reference from a different domain
        other_domains = [d for d in self.domain_files if d != domain_idx]
        if other_domains:
            ref_domain = random.choice(other_domains)
        else:
            ref_domain = domain_idx
        ref_path = random.choice(self.domain_files[ref_domain])
        ref_image = np.load(ref_path)["image"]

        if self.transform:
            image = self.transform(image)
            ref_image = self.transform(ref_image)

        fp = Path(npz_path)
        filename = fp.name
        m = re.search(r"_s(\d+)$", fp.stem)
        slice_index = m.group(1) if m else ""

        return {
            "image": image,
            "filename": filename,
            "slice_index": slice_index,
            "domain": domain_idx,
            "field_strength": field_strength,
            "modality": modality_name,
            "ref_image": ref_image,
            "ref_domain": ref_domain,
        }
