"""FieldRF Dataset — 384×384×384 @ 0.5mm, center pad/crop via MONAI.

Output range: [-1,1] (for MAISI VAE). Brain-ID: min-max → [0,1] in cache_brainid.py.
Plain Dataset (no PersistentDataset — dataset is small).
"""

from pathlib import Path
from typing import Optional

import sys
_BASELINE = Path(__file__).resolve().parent.parent.parent / "Baseline"
if str(_BASELINE) not in sys.path:
    sys.path.append(str(_BASELINE))

import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged, EnsureTyped,
)

TARGET_SIZE = (384, 384, 384)


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


class FieldRFVolumeDataset(torch.utils.data.Dataset):
    """384×384×384 @ 0.5mm. Plain Dataset — no PersistentDataset overhead."""

    def __init__(self, data_root, split, modality, field_strength,
                 cache_dir=None, max_volumes=0):
        self.data_root = Path(data_root)
        self._modality = modality
        from mrixfields.data.utils import list_nifti_files

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


class PairedCacheDataset(torch.utils.data.Dataset):
    """Loads paired latent + Brain-ID feature from cache, matched by filename.

    Each volume has:
      latent/{field}/{stem}.pt     → {"latent": (4, 96, 96, 96), ...}
      brainid/{field}/{stem}.pt   → {"feature": (128, 96, 96, 96) fp16, ...}

    Returns both together, ensuring same-volume pairing.
    """

    def __init__(self, latent_root, brainid_root, field_names, max_volumes=0):
        self.files = []
        per_field = max_volumes // len(field_names) if max_volumes > 0 and field_names else 0
        for field in field_names:
            latent_dir = Path(latent_root) / str(field)
            brainid_dir = Path(brainid_root) / str(field)
            for pt in sorted(latent_dir.glob("*.pt"))[:per_field if per_field > 0 else None]:
                bp = brainid_dir / pt.name
                if bp.exists():
                    self.files.append((str(pt), str(bp)))
        if not self.files:
            raise FileNotFoundError(f"No paired files in {latent_root}/{{{field_names}}}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        latent_path, brainid_path = self.files[idx]
        d_lat = torch.load(latent_path, map_location="cpu", weights_only=False)
        d_bd = torch.load(brainid_path, map_location="cpu", weights_only=False)
        feat = d_bd["feature"].float()
        if feat.ndim == 5:        # old cache: (1, 128, 96, 96, 96)
            feat = feat[0]        # squeeze batch
        return {
            "z": d_lat["latent"],                        # (4, 96, 96, 96)
            "feat": feat,                                # (128, 96, 96, 96)
            "field_strength": d_lat.get("field_strength", ""),
            "filename": Path(latent_path).stem,
        }


class PairedVisDataset(torch.utils.data.Dataset):
    """Paired prospective data for validation visualization.

    Matches source→target volumes by subject ID across field dirs.
    e.g. P_T1W_0.1T_0016.pt ↔ P_T1W_7T_0016.pt
    Returns (z_src, feat_src, z_tgt, feat_tgt) for each pair.
    """

    @staticmethod
    def _subject_id(filename):
        """Extract subject ID from filename like P_T1W_0.1T_0016.pt → '0016'."""
        return filename.rsplit("_", 1)[-1].replace(".pt", "")

    def __init__(self, latent_root, brainid_root, source_fields, target_field,
                 samples_per_field=1):
        target_latent_dir = Path(latent_root) / str(target_field)
        target_bd_dir = Path(brainid_root) / str(target_field)

        # Build target lookup by subject ID
        target_latent = {}
        target_brainid = {}
        for pt in sorted(target_latent_dir.glob("*.pt")):
            target_latent[self._subject_id(pt.name)] = str(pt)
        for pt in sorted(target_bd_dir.glob("*.pt")):
            target_brainid[self._subject_id(pt.name)] = str(pt)

        self.pairs = []
        common_ids = set(target_latent.keys()) & set(target_brainid.keys())
        for sf in source_fields:
            src_latent_dir = Path(latent_root) / str(sf)
            src_bd_dir = Path(brainid_root) / str(sf)
            count = 0
            for pt in sorted(src_latent_dir.glob("*.pt")):
                if samples_per_field > 0 and count >= samples_per_field:
                    break
                sid = self._subject_id(pt.name)
                bd_pt = src_bd_dir / pt.name
                if sid in common_ids and bd_pt.exists():
                    self.pairs.append({
                        "z_src": str(pt), "feat_src": str(bd_pt),
                        "z_tgt": target_latent[sid], "feat_tgt": target_brainid[sid],
                        "source_field": str(sf),
                    })
                    count += 1
        if not self.pairs:
            print(f"[PairedVisDataset] WARNING: no paired data in {latent_root}/{{{source_fields}}}↔{target_field}")
            self.pairs = []

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        d_src_lat = torch.load(p["z_src"], map_location="cpu", weights_only=False)
        d_src_bd = torch.load(p["feat_src"], map_location="cpu", weights_only=False)
        d_tgt_lat = torch.load(p["z_tgt"], map_location="cpu", weights_only=False)
        d_tgt_bd = torch.load(p["feat_tgt"], map_location="cpu", weights_only=False)

        feat_src = d_src_bd["feature"].float()
        if feat_src.ndim == 5:
            feat_src = feat_src[0]
        feat_tgt = d_tgt_bd["feature"].float()
        if feat_tgt.ndim == 5:
            feat_tgt = feat_tgt[0]

        return {
            "z_src": d_src_lat["latent"],
            "feat_src": feat_src,
            "z_tgt": d_tgt_lat["latent"],
            "feat_tgt": feat_tgt,
            "source_field": p["source_field"],
            "filename": Path(p["z_tgt"]).stem,           # e.g. P_T1W_7T_0006
        }
