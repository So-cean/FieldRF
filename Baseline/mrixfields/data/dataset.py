"""PyTorch Dataset classes for MRIxFields2026 challenge.

Data structure:
    {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

Output tensors are scaled to [-1, 1] (standard for GAN training).

Three dataset types:
    - UnpairedMRIDataset: Single-domain slices (for CUT/CycleGAN unpaired training)
    - PairedMRIDataset: Matched source-target slices (for validation/finetune)
    - MultiDomainMRIDataset: All domains for StarGAN v2 (Task 3)
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    FIELD_STRENGTHS,
    FIELD_TO_DOMAIN,
    MODALITIES,
    extract_subject_id,
    get_joint_domain,
    list_nifti_files,
    load_nifti,
)
from .transforms import CenterCropOrPad, NormalizeMinMax, ToTensor, ScaleToMinusOneOne, Compose


def _default_transform(crop_size: Optional[Tuple[int, int]] = None) -> Compose:
    """Default preprocessing: optional center crop + normalize to [0,1] + to tensor + scale to [-1,1].

    Args:
        crop_size: Target (H, W) for CenterCropOrPad. None to skip cropping.
    """
    steps = []
    if crop_size is not None:
        steps.append(CenterCropOrPad(crop_size))
    steps += [NormalizeMinMax(), ToTensor(), ScaleToMinusOneOne()]
    return Compose(steps)


class UnpairedMRIDataset(Dataset):
    """Single-domain 2D slice dataset for unpaired training.

    Used by CUT and CycleGAN to load slices from one field strength.
    Two instances (source + target) are paired via UnpairedDataLoader.

    Data path: {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

    Supports arbitrary number of modalities via the ``modalities`` parameter.
    When multiple modalities are provided, slices are concatenated along the
    channel dimension (C, H, W).
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: Optional[str] = None,
        field_strength: str = "",
        modalities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        slice_axis: int = 2,
        min_slice_std: float = 0.01,
        norm_infos: Optional[List[dict]] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        if modalities is not None:
            self.modalities = modalities
        elif modality is not None:
            self.modalities = [modality]
        else:
            raise ValueError(
                "UnpairedMRIDataset requires either 'modalities' (list) or 'modality' (str). "
                "Neither was provided."
            )
        self.field_strength = field_strength
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform(crop_size)
        self._norm_infos = norm_infos or []

        # Index: list of (modality_paths_tuple, slice_index)
        self.samples: List[Tuple[Tuple[Path, ...], int]] = []
        self._index_data()

    def _index_data(self):
        lookups = {}
        for mod in self.modalities:
            files = list_nifti_files(self.data_root, self.split, mod, self.field_strength)
            lookups[mod] = {extract_subject_id(f.name): f for f in files}

        for subj_id, base_path in lookups[self.modalities[0]].items():
            modality_paths = []
            for mod in self.modalities:
                path = lookups[mod].get(subj_id)
                if path is None:
                    break
                modality_paths.append(path)
            if len(modality_paths) < len(self.modalities):
                continue
            data, _ = load_nifti(base_path)
            n_slices = data.shape[self.slice_axis]
            for i in range(n_slices):
                if self._get_slice(data, i).std() > self.min_slice_std:
                    self.samples.append((tuple(modality_paths), i))

    def _get_slice(self, volume: np.ndarray, idx: int) -> np.ndarray:
        slicing = [slice(None)] * volume.ndim
        slicing[self.slice_axis] = idx
        return volume[tuple(slicing)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        modality_paths, slice_idx = self.samples[index]
 
        slices = [self.transform(self._get_slice(load_nifti(p)[0], slice_idx)) for p in modality_paths]
        img = torch.cat(slices, dim=0)

        if self._norm_infos:
            channels = []
            for c in range(len(self.modalities)):
                info = self._norm_infos[c]
                channels.append((img[c] - info["mu"]) / info["sigma"])
            img = torch.stack(channels, dim=0)

        return {
            "image": img,
            "field_strength": self.field_strength,
            "modalities": self.modalities,
            "filename": modality_paths[0].name,
            "slice_index": slice_idx,
        }


class PairedMRIDataset(Dataset):
    """Paired source-target 2D slice dataset.

    Used for:
    - Validation/testing (Validating_prospective, Testing_prospective)
    - Paired fine-tuning phase (Training_prospective)

    Pairs slices from source and target field strengths of the SAME subject.
    Subjects are identified by matching filenames across field strength dirs.

    Data path: {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

    Supports arbitrary number of modalities via the ``modalities`` parameter.
    When multiple modalities are provided, slices are concatenated along the
    channel dimension (C, H, W).
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: Optional[str] = None,
        source_field: str = "",
        target_field: str = "",
        modalities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        slice_axis: int = 2,
        min_slice_std: float = 0.01,
    ):
        self.data_root = Path(data_root)
        self.split = split
        if modalities is not None:
            self.modalities = modalities
        elif modality is not None:
            self.modalities = [modality]
        else:
            raise ValueError(
                "PairedMRIDataset requires either 'modalities' (list) or 'modality' (str). "
                "Neither was provided."
            )
        self.source_field = source_field
        self.target_field = target_field
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform(crop_size)

        # Index: list of (src_paths_tuple, tgt_paths_tuple, slice_index)
        self.samples: List[Tuple[Tuple[Path, ...], Tuple[Path, ...], int]] = []
        self._index_data()

    def _index_data(self):
        src_lookups = {}
        tgt_lookups = {}
        for mod in self.modalities:
            src_files = list_nifti_files(self.data_root, self.split, mod, self.source_field)
            tgt_files = list_nifti_files(self.data_root, self.split, mod, self.target_field)
            src_lookups[mod] = {extract_subject_id(f.name): f for f in src_files}
            tgt_lookups[mod] = {extract_subject_id(f.name): f for f in tgt_files}

        for subj_id in src_lookups[self.modalities[0]]:
            src_paths = []
            for mod in self.modalities:
                path = src_lookups[mod].get(subj_id)
                assert path is not None, (
                    f"Missing source {mod} for subject {subj_id} in "
                    f"{self.split}/{self.source_field}"
                )
                src_paths.append(path)

            tgt_paths = []
            for mod in self.modalities:
                path = tgt_lookups[mod].get(subj_id)
                if path is None:
                    break
                tgt_paths.append(path)
            if len(tgt_paths) < len(self.modalities):
                continue  # skip subjects missing any target modality

            src_data, _ = load_nifti(src_paths[0])
            tgt_data, _ = load_nifti(tgt_paths[0])

            n_slices = min(
                src_data.shape[self.slice_axis],
                tgt_data.shape[self.slice_axis],
            )
            for i in range(n_slices):
                src_slc = self._get_slice(src_data, i)
                tgt_slc = self._get_slice(tgt_data, i)
                if (src_slc.std() > self.min_slice_std and
                        tgt_slc.std() > self.min_slice_std):
                    self.samples.append((tuple(src_paths), tuple(tgt_paths), i))

    def _get_slice(self, volume: np.ndarray, idx: int) -> np.ndarray:
        slicing = [slice(None)] * volume.ndim
        slicing[self.slice_axis] = idx
        return volume[tuple(slicing)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        src_paths, tgt_paths, slice_idx = self.samples[index]

        src_slices = [self.transform(self._get_slice(load_nifti(p)[0], slice_idx)) for p in src_paths]
        tgt_slices = [self.transform(self._get_slice(load_nifti(p)[0], slice_idx)) for p in tgt_paths]
        src_slc = torch.cat(src_slices, dim=0)
        tgt_slc = torch.cat(tgt_slices, dim=0)

        return {
            "source": src_slc,
            "target": tgt_slc,
            "source_field": self.source_field,
            "target_field": self.target_field,
            "modalities": self.modalities,
            "filename": src_paths[0].name,
            "slice_index": slice_idx,
        }


class MultiDomainMRIDataset(Dataset):
    """Multi-domain dataset for StarGAN v2 (Task 3: Any -> Any).

    Combines slices from all modalities and field strengths.
    When multiple modalities are used, domain labels use ``get_joint_domain``
    (0..14 for 3 modalities × 5 fields). For a single modality, labels
    fall back to ``FIELD_TO_DOMAIN`` (0..4) for backward compatibility.

    Data path: {data_root}/{split}/{modality}/{field_strength}/*.nii.gz
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modality: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        field_strengths: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        slice_axis: int = 2,
        min_slice_std: float = 0.01,
    ):
        self.data_root = Path(data_root)
        self.split = split
        if modalities is not None:
            self.modalities = list(modalities)
        elif modality is not None:
            self.modalities = [modality]
        else:
            self.modalities = list(MODALITIES)
        self.field_strengths = field_strengths or FIELD_STRENGTHS
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform(crop_size)
        self._use_joint = len(self.modalities) > 1

        # Per-domain sample indices: maps domain_idx → list of sample indices
        self.domain_samples: Dict[int, List[int]] = {}
        # Reverse mapping: domain_idx → (modality, field_strength)
        self._domain_to_pair: Dict[int, Tuple[str, str]] = {}
        # Samples: (nifti_path, slice_index, domain_idx, modality, field_strength)
        self.samples: List[Tuple[Path, int, int, str, str]] = []
        self._index_data()

    def _domain_label(self, modality: str, field_strength: str) -> int:
        if self._use_joint:
            return get_joint_domain(modality, field_strength)
        return FIELD_TO_DOMAIN[field_strength]

    def _index_data(self):
        for mod in self.modalities:
            for fs in self.field_strengths:
                domain_idx = self._domain_label(mod, fs)
                self.domain_samples.setdefault(domain_idx, [])
                self._domain_to_pair[domain_idx] = (mod, fs)
                nifti_files = list_nifti_files(
                    self.data_root, self.split, mod, fs
                )
                for nifti_path in nifti_files:
                    data, _ = load_nifti(nifti_path)
                    n_slices = data.shape[self.slice_axis]
                    for i in range(n_slices):
                        slicing = [slice(None)] * data.ndim
                        slicing[self.slice_axis] = i
                        slc = data[tuple(slicing)]
                        if slc.std() > self.min_slice_std:
                            self.samples.append((nifti_path, i, domain_idx, mod, fs))
                            self.domain_samples[domain_idx].append(len(self.samples) - 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        nifti_path, slice_idx, domain_idx, modality_name, field_strength = self.samples[index]
        data, _ = load_nifti(nifti_path)
        slicing = [slice(None)] * data.ndim
        slicing[self.slice_axis] = slice_idx
        slc = data[tuple(slicing)]
        slc = self.transform(slc)

        # Sample a random reference image from a different domain
        other_domains = [d for d in self.domain_samples if d != domain_idx
                         and len(self.domain_samples[d]) > 0]
        if other_domains:
            ref_domain = np.random.choice(other_domains)
            ref_idx = np.random.choice(self.domain_samples[ref_domain])
            ref_path, ref_slice, _, _, _ = self.samples[ref_idx]
            ref_data, _ = load_nifti(ref_path)
            ref_slicing = [slice(None)] * ref_data.ndim
            ref_slicing[self.slice_axis] = ref_slice
            ref_slc = self.transform(ref_data[tuple(ref_slicing)])
        else:
            ref_slc = slc
            ref_domain = domain_idx

        return {
            "image": slc,
            "domain": domain_idx,
            "ref_image": ref_slc,
            "ref_domain": ref_domain,
            "field_strength": field_strength,
            "modality": modality_name,
            "filename": nifti_path.name,
            "slice_index": slice_idx,
        }

    def get_random_from_domain(self, domain_idx: int) -> Dict:
        """Get a random sample from a specific domain."""
        indices = self.domain_samples[domain_idx]
        idx = np.random.choice(indices)
        return self[idx]
