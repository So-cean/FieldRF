"""3D inference script with MONAI SlidingWindowInferer.

Runs a trained 3D CUT generator on full NIfTI volumes using sliding-window
patch aggregation for memory-efficient inference.

Usage:
    python scripts/inference3d.py \
        --config configs/task1/cut3d/0.1T_to_7T_T1W.yaml \
        --checkpoint $OUTPUT_DIR/task1_0.1T_to_7T_T1W/cut3d/retro_scratch/weights/generator_final.pth \
        --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
        --output_dir $INFERENCE_DIR/
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.models.networks_cut3d import ResnetGenerator3D
from mrixfields.data.utils import load_nifti, save_nifti
from mrixfields.env import load_env, get_inference_dir, get_device

from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    EnsureTyped,
)

import importlib
import shutil
if shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by DDP wrapping."""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


def _extract_generator_state(state_dict: dict) -> dict:
    """Extract generator weights from checkpoint.

    Supports:
        - Full model checkpoint (key "model" with "netG." prefix)
        - DDP-wrapped checkpoints (keys contain "module.", e.g. "netG.module.xxx")
        - Hybrid fine-tune checkpoint (key "generator")
        - Raw generator state_dict
    """
    state_dict = _strip_module_prefix(state_dict)

    if "generator" in state_dict:
        return _strip_module_prefix(state_dict["generator"])
    if "model" in state_dict:
        model_state = _strip_module_prefix(state_dict["model"])
        prefix = "netG."
        gen_state = {k[len(prefix):]: v for k, v in model_state.items()
                     if k.startswith(prefix)}
        if gen_state:
            return _strip_module_prefix(gen_state)
        raise ValueError(
            f"Checkpoint has 'model' key but no '{prefix}' prefix found."
        )
    if "generator" in state_dict:
        return _strip_module_prefix(state_dict["generator"])
    if all(not k.startswith("net") for k in state_dict):
        return state_dict
    raise ValueError(
        f"Unrecognized checkpoint format. Top-level keys: {list(state_dict.keys())}"
    )


def load_generator3d(cfg: dict, checkpoint_path: str, device: torch.device):
    """Load 3D ResNet generator from checkpoint (handles DDP module prefix)."""
    model_cfg = cfg["model"]
    model = ResnetGenerator3D(
        input_nc=model_cfg.get("input_nc", 1),
        output_nc=model_cfg.get("output_nc", 1),
        ngf=model_cfg.get("ngf", 32),
        n_blocks=model_cfg.get("n_blocks", 9),
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    gen_state = _extract_generator_state(state_dict)
    model.load_state_dict(gen_state)
    model.to(device).eval()
    return model


def predict_volume_3d(
    model: torch.nn.Module,
    nifti_path: Path,
    device: torch.device,
    roi_size: tuple = (128, 128, 128),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
) -> tuple:
    """Run sliding-window inference on a single NIfTI volume.

    The ``sw_batch_size`` controls how many patches from the SAME volume
    are processed in parallel by the sliding-window inferer. It does NOT
    batch across multiple subjects / NIfTI files.

    Returns:
        pred_data: numpy array in [0, 1] with the original volume shape.
        original_affine: affine matrix for saving.
    """
    # Load with nibabel to preserve orientation info
    original_img = nib.load(str(nifti_path))
    original_affine = original_img.affine
    original_ornt = nib.io_orientation(original_affine)

    # Load canonical (RAS+) for consistent processing
    data, canonical_affine = load_nifti(nifti_path)
    canonical_ornt = nib.io_orientation(canonical_affine)

    # Brain mask: skull-stripped input has exact 0 in background
    brain_mask = (data > 0).astype(np.float32)

    # Preprocess: percentile clip + scale to [-1, 1]
    # We operate on the canonical volume
    data = data.astype(np.float32)
    upper = np.percentile(data, 99.5)
    data = np.clip(data, 0, upper)
    if upper > 1e-8:
        data = data / upper
    data = data * 2.0 - 1.0  # [0,1] -> [-1,1]

    # Build 5D tensor: (1, 1, D, H, W)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(device)

    # Sliding window inference
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
    )
    with torch.no_grad():
        pred = inferer(inputs=tensor, network=model)

    pred_np = pred.squeeze().cpu().numpy()
    pred_np = np.clip(pred_np, -1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]

    # Apply brain mask: set background back to 0
    pred_np = pred_np * brain_mask

    # Ensure float32
    pred_np = pred_np.astype(np.float32)

    # Convert back from canonical to original orientation
    transform = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
    pred_np = nib.orientations.apply_orientation(pred_np, transform)

    return pred_np, original_affine


def main():
    parser = argparse.ArgumentParser(description="MRIxFields2026 3D Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    load_env()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: INFERENCE_DIR from .env)")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[128, 128, 128],
                        help="Sliding window patch size (D H W)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap ratio")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = get_inference_dir()

    cfg = load_config(args.config)
    if args.device:
        device = torch.device(args.device)
    else:
        device_str = get_device()
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = load_generator3d(cfg, args.checkpoint, device)
    print(f"Loaded 3D generator from: {args.checkpoint}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if str(output_dir) == "/" or str(output_dir) == "":
        raise ValueError(
            f"Invalid output_dir: '{output_dir}'. "
            f"Please specify a valid output directory via --output_dir or set INFERENCE_DIR in .env. "
            f"If using shell variables, make sure they are exported (e.g., export INFERENCE_DIR=...)."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.rglob("*.nii.gz"))
    if not nifti_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    roi_size = tuple(args.roi_size)
    sw_batch_size = cfg.get("inference", {}).get("batch_size", 1)

    for nifti_path in tqdm(nifti_files, desc="3D Inference"):
        pred_np, affine = predict_volume_3d(
            model, nifti_path, device,
            roi_size=roi_size,
            overlap=args.overlap,
            sw_batch_size=sw_batch_size,
        )

        rel_path = nifti_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        save_nifti(pred_np, affine, out_path)

    print(f"3D Inference complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
