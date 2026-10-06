"""FieldRF Dataset — 384×384×384 @ 0.5mm, center pad/crop via MONAI.

Output range: [-1,1] (for MAISI VAE). Brain-ID: min-max → [0,1] in cache_brainid_feat.py.
Plain Dataset (no PersistentDataset — dataset is small).
"""

from pathlib import Path

import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged, EnsureTyped,
)

TARGET_SIZE = (384, 384, 384)


def list_nifti_files(data_root, split, modality, field_strength):
    """List one MRI domain without importing the comparison baseline package."""
    return sorted((Path(data_root) / split / modality / str(field_strength)).glob("*.nii.gz"))


def dataset_transforms():
    """Shared transforms: 384³, RAS, [-1,1] (for MAISI VAE). Brain-ID converts to [0,1] elsewhere."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        SpatialPadd(keys=["image"], spatial_size=TARGET_SIZE, method="symmetric"),
        CenterSpatialCropd(keys=["image"], roi_size=TARGET_SIZE),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0),
        EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
    ])


class VolumeDataset(torch.utils.data.Dataset):
    """384×384×384 @ 0.5mm. Plain Dataset — no PersistentDataset overhead."""

    def __init__(self, data_root, split, modality, field_strength,
                 cache_dir=None, max_volumes=0):
        self.data_root = Path(data_root)
        self._modality = modality

        if isinstance(field_strength, str):
            field_strengths = [field_strength]
        else:
            field_strengths = list(field_strength)

        self.data_dicts = []
        self.file_names = []
        self._vol_fields = []

        for fs in field_strengths:
            files = list_nifti_files(data_root, split, modality, fs)
            for f in files:
                self.data_dicts.append({"image": str(f)})
                self.file_names.append(f.name)
                self._vol_fields.append(fs)

        if not self.data_dicts:
            raise FileNotFoundError(f"No files for {modality}/{field_strengths}")

        if max_volumes > 0:
            self.data_dicts = self.data_dicts[:max_volumes]
            self.file_names = self.file_names[:max_volumes]
            self._vol_fields = self._vol_fields[:max_volumes]

        self.transform = dataset_transforms()

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        item = self.transform(self.data_dicts[idx])
        return {
            "image": item["image"],                 # (1, 384, 384, 384)
            "filename": self.file_names[idx],
            "field_strength": self._vol_fields[idx],
            "modality": self.modality,
            "image_path": self.data_dicts[idx]["image"],
        }

    @property
    def modality(self):
        return getattr(self, '_modality', 'T1W')

    @modality.setter
    def modality(self, v):
        self._modality = v
