"""Matched cached-latent/raw-MRI patches for decoder adaptation."""

from pathlib import Path
import random
import numpy as np
import torch
from fieldrf.utils.visualization import load_orig_ras_384


class LatentImageDataset(torch.utils.data.Dataset):
    """Loads cached VAE latent + original NIfTI, returns matching patches.

    Cached latent: (4, 96, 96, 96) from .pt file.
    Original NIfTI: 384^3, [0,1] after RAS + pad/crop.
    Patch: latent (patch_size//4)^3 -> image patch_size^3, same spatial location.
    """

    def __init__(self, cache_root, data_root, split, modalities, fields,
                 patch_size=128, max_volumes=0):
        self.patch_size = patch_size
        self.latent_patch = patch_size // 4
        self.files = []
        # max_volumes is PER (modality, field) combo (0 = no cap)
        per_mod_field = max_volumes if max_volumes > 0 else 0
        for mod in modalities:
            for field in fields:
                latent_dir = Path(cache_root) / split / "latent" / mod / field
                if not latent_dir.is_dir():
                    continue
                pts = sorted(latent_dir.glob("*.pt"))
                if per_mod_field > 0:
                    pts = pts[:per_mod_field]
                for pt in pts:
                    stem = pt.stem
                    nii = Path(data_root) / split / mod / field / f"{stem}.nii.gz"
                    if nii.exists():
                        self.files.append({"latent": str(pt), "nii": str(nii),
                                           "modality": mod, "field": field})
        mod_counts = {}
        for f in self.files:
            mod_counts[f["modality"]] = mod_counts.get(f["modality"], 0) + 1
        for mod, count in sorted(mod_counts.items()):
            print(f"  {mod}: {count} volumes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        d = torch.load(item["latent"], map_location="cpu", weights_only=False)
        z = d["latent"]  # (4, 96, 96, 96)

        orig_384, _, _, _ = load_orig_ras_384(item["nii"])  # (384,384,384) [0,1]

        # Random crop (latent coords -> image coords via *4)
        D, H, W = z.shape[1:]
        pd = self.latent_patch
        dz = random.randint(0, D - pd)
        dy = random.randint(0, H - pd)
        dx = random.randint(0, W - pd)

        z_patch = z[:, dz:dz+pd, dy:dy+pd, dx:dx+pd]
        ip = self.patch_size
        img_patch = orig_384[dz*4:dz*4+ip, dy*4:dy*4+ip, dx*4:dx*4+ip]

        # Augmentation: random flip (both together)
        for axis in range(3):
            if random.random() < 0.5:
                z_patch = torch.flip(z_patch, [axis + 1])
                img_patch = np.flip(img_patch, axis=axis).copy()

        img_patch = img_patch * 2 - 1  # [0,1] -> [-1,1]
        return {
            "latent": z_patch,
            "image": torch.from_numpy(img_patch.copy()).float().unsqueeze(0),
            "modality": item["modality"],
            "field": item["field"],
        }

