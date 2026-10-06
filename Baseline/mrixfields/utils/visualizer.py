"""Training visualization utilities.

Minimal, standalone module. Inspired by CUT's util/visualizer.py but
stripped down to single-image saving without HTML/Visdom overhead.

Records visualization metadata to visuals_metadata.csv:
    png_filename,filename,slice_index
"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_image(tensor, slice_axis=2):
    """Convert a torch batch tensor to a numpy RGB image.

    Takes ONLY the first sample in the batch (matches CUT behavior).
    Supports both 2D (B, C, H, W) and 3D (B, C, D, H, W) data.
    For 3D data, visualizes the middle slice along the specified axis.
    Assumes tensor range is [-1, 1].
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()
    else:
        img = tensor

    slice_idx = 0
    if img.ndim == 4:
        mid = img.shape[3] // 2  # SI (axial) middle slice
        img = img[..., mid]
        slice_idx = mid

    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8), slice_idx


def save_training_visuals(visuals_dict, save_dir, epoch, prefix="epoch",
                           slice_axis=2, modalities=None,
                           batch_info=None, batch_info_b=None,
                           field_strength="", visual_domains=None):
    """Save a dictionary of visualization tensors to disk.

    Records visuals_metadata.csv with filename, slice index, and field strength.

    Args:
        visuals_dict: Dict[str, Tensor] with keys like
                      {'real_A': ..., 'real_B': ..., 'fake_B': ..., 'idt_B': ...}
        save_dir:     Directory to save images (Path or str).
        epoch:        Current epoch number (used in filename).
        prefix:       Filename prefix, e.g. "epoch" -> epoch010_real_A.png
        slice_axis:   int, axis to slice for 3D visualization (default 2).
        modalities:   list of modality names (e.g. ["T1W", "T2W"]).
        batch_info:   Collated batch dict from source dataset (or paired).
        batch_info_b: Collated batch dict from target dataset (unpaired only).
        field_strength: Source field strength string for multi-field training.
        visual_domains: Optional per-image metadata mapping, e.g.
            ``{"real_A": {"modality": "T1W", "field_strength": "0.1T"}}``.
            This is useful for single-channel multi-domain methods such as
            StarGAN v2, where each visual can belong to a different modality.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    src_fn = src_si = src_fs = tgt_fn = tgt_si = tgt_fs = ""
    if batch_info:
        src_fn = batch_info.get("filename", [""])[0]
        src_si = batch_info.get("slice_index", [""])[0]
        src_fs = batch_info.get("field_strength", [""])[0] or field_strength
    if batch_info_b:
        tgt_fn = batch_info_b.get("filename", [""])[0]
        tgt_si = batch_info_b.get("slice_index", [""])[0]
        tgt_fs = batch_info_b.get("field_strength", [""])[0] or ""
    elif batch_info:
        tgt_fn = batch_info.get("target_paths", [""])[0]
        tgt_si = batch_info.get("slice_index", [""])[0]

    def _fs_key(fs):
        return fs.replace(".", "_") if fs else ""

    metadata_rows = []
    for name, tensor in visuals_dict.items():
        if tensor is None:
            continue
        visual_domain = (visual_domains or {}).get(name, {})
        explicit_modality = visual_domain.get("modality", "")
        explicit_field = visual_domain.get("field_strength", "")
        domain_a = name in {"real_A", "fake_A", "rec_A", "idt_A"}
        domain_b = name in {"real_B", "fake_B", "rec_B", "idt_B"}
        if name == "fake_B":
            im_fs = _fs_key(src_fs) + "_to_" + _fs_key(tgt_fs) if src_fs and tgt_fs else _fs_key(tgt_fs)
        elif name == "fake_A":
            im_fs = _fs_key(tgt_fs) + "_to_" + _fs_key(src_fs) if src_fs and tgt_fs else _fs_key(src_fs)
        elif domain_b:
            im_fs = _fs_key(tgt_fs)
        elif domain_a:
            im_fs = _fs_key(src_fs)
        elif "fake" in name or "pred" in name:
            im_fs = _fs_key(src_fs) + "_to_" + _fs_key(tgt_fs) if src_fs and tgt_fs else _fs_key(tgt_fs)
        else:
            im_fs = _fs_key(src_fs)
        if explicit_field:
            im_fs = _fs_key(explicit_field)
        C = tensor.shape[1]
        for c in range(C):
            ch_tensor = tensor[:, c:c+1, ...]
            if explicit_modality:
                mod_tag = f"_{explicit_modality}"
            else:
                mod_tag = f"_{modalities[c]}" if modalities and len(modalities) > 1 else ""
            fs_tag = f"_{im_fs}" if im_fs else ""
            fname = f"{prefix}{epoch:03d}_{name}{fs_tag}{mod_tag}.png"

            img, _ = tensor_to_image(ch_tensor, slice_axis=slice_axis)
            Image.fromarray(img).save(str(save_dir / fname))

            use_src = name in {"real_A", "fake_B", "rec_A", "idt_A"} or "source" in name
            fn = src_fn if use_src else tgt_fn
            si = src_si if use_src else tgt_si
            if "fake" in name or "pred" in name:
                fn = src_fn; si = src_si
            metadata_rows.append({
                "png_filename": fname,
                "filename": fn,
                "slice_index": si,
                "field_strength": explicit_field or (src_fs if use_src else tgt_fs),
            })

    csv_path = save_dir / "visuals_metadata.csv"
    fieldnames = ["png_filename", "filename", "slice_index", "field_strength"]
    old_rows = []
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            old_rows = list(csv.DictReader(f))
    new_pngs = {r["png_filename"] for r in metadata_rows}
    merged = [r for r in old_rows if r["png_filename"] not in new_pngs] + metadata_rows
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged)
