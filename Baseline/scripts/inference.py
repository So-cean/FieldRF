"""Unified inference entry point for MRIxFields2026 baselines.

Supports CUT, CycleGAN, StarGAN v2, ITTR, and SwinUNet.

Usage:
    # CUT / CycleGAN / ITTR / SwinUNet
    python scripts/inference.py --config configs/task1/cut/0.1T_to_7T_T1W.yaml \
        --checkpoint $OUTPUT_DIR/task1_0.1T_to_7T_T1W/cut/retro_scratch/weights/generator_final.pth \
        --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
        --output_dir $INFERENCE_DIR/

    # StarGAN v2 (config is embedded in checkpoint)
    python scripts/inference.py \
        --checkpoint $OUTPUT_DIR/task3_any_to_any_multimodal/starganv2/retro_scratch/weights/generator_final.pth \
        --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
        --output_dir $INFERENCE_DIR/ \
        --source_field 0.1T --target_field 7T \
        --source_modality T1W --target_modality T1W

After inference, run SynthSeg segmentation:
    python Evaluation/segment.py \
        --input_dir $INFERENCE_DIR/ \
        --output_dir ${INFERENCE_DIR}_seg/
"""

import argparse
import importlib
import shutil
from pathlib import Path

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NPU initialization (if available)
if shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False

from mrixfields.env import load_env, get_inference_dir, get_device
import mrixfields.inference as I

_METHOD_PREFIX = {
    "cut": "CUT",
    "cut_seg": "CUT",
    "cut3d": "CUT3D",
    "cyclegan": "CycleGAN",
    "nafnet": "NAFNet",
    "restormer": "Restormer",
    "starganv2": "StarGANv2",
    "ittr": "ITTR",
    "unsb": "UNSB",
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_source_field(args, data_cfg: dict) -> str:
    """Determine source_field for norm_stats: explicit CLI arg wins, else from config.

    Single-field (source_field): auto-read from config.
    Multi-field (source_fields): require --source_field.
    """
    if args.source_field is not None:
        return args.source_field
    if "source_field" in data_cfg:
        return data_cfg["source_field"]
    if "source_fields" in data_cfg:
        raise ValueError(
            f"--source_field is required for multi-field models. "
            f"Available: {data_cfg['source_fields']}"
        )
    return None


def _load_config_from_checkpoint(checkpoint_path: str) -> dict:
    """Load full runtime config from checkpoint metadata."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "_metadata" in state and "config_actual" in state["_metadata"]:
        print(f"Loaded config from checkpoint metadata")
        return state["_metadata"]["config_actual"]
    raise ValueError(
        f"Checkpoint {checkpoint_path} is not self-contained "
        "(missing _metadata.config_actual). "
        "Only checkpoints from the unified format are supported."
    )


def main():
    parser = argparse.ArgumentParser(description="MRIxFields2026 Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained generator checkpoint")
    parser.add_argument("--input_dir", type=str, nargs="+", required=True,
                        help="Input directory(s). Single: one path. Multi: one per modality.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: INFERENCE_DIR from .env)")
    parser.add_argument("--source_field", type=str, default=None,
                        help="Source field for norm_stats (required for multi-field models)")
    parser.add_argument("--target_domain", type=int, default=None,
                        help="Target domain index (StarGAN v2 only)")
    parser.add_argument("--target_field", type=str, default=None,
                        help="Target field strength name, e.g. '7T' (StarGAN v2 only)")
    parser.add_argument("--source_modality", type=str, default=None,
                        help="Source modality for joint-domain StarGAN v2")
    parser.add_argument("--target_modality", type=str, default=None,
                        help="Target modality for joint-domain StarGAN v2")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda:0' or 'npu:0'")
    args = parser.parse_args()

    load_env()

    if args.output_dir is None:
        args.output_dir = get_inference_dir()

    cfg = _load_config_from_checkpoint(args.checkpoint)

    if args.device:
        device_str = args.device
    else:
        if torch.cuda.is_available():
            device_str = "cuda:0"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    if device.type == "cpu":
        print("\n" + "=" * 60)
        print("WARNING: Inference is running on CPU!")
        print("This will be very slow. Consider using GPU/NPU for faster inference.")
        print("=" * 60 + "\n")

    method = cfg["method"]
    prefix = _METHOD_PREFIX.get(method)
    inference_cls = getattr(I, f"{prefix}Inference") if prefix else None
    if inference_cls is None:
        raise ValueError(f"Unknown method: {method}. Supported: {list(_METHOD_PREFIX.keys())}")

    # Resolve source_field: CLI arg > config (no fallback)
    data_cfg = cfg.get("data", {})
    source_field = _resolve_source_field(args, data_cfg)

    # StarGAN v2 needs extra target info
    kwargs = {}
    if method == "starganv2":
        kwargs["target_field"] = args.target_field
        kwargs["target_domain"] = args.target_domain
        kwargs["source_modality"] = args.source_modality
        kwargs["target_modality"] = args.target_modality

    inferencer = inference_cls(cfg, args.checkpoint, device,
                               source_field=source_field, **kwargs)

    input_dir = [Path(d) for d in args.input_dir]
    output_dir = Path(args.output_dir)
    if str(output_dir) == "/" or str(output_dir) == "":
        raise ValueError(
            f"Invalid output_dir: '{output_dir}'. "
            f"Please specify a valid output directory via --output_dir or set INFERENCE_DIR in .env."
        )

    crop_size_raw = cfg["data"].get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    slice_axis = cfg["data"].get("slice_axis", 2)
    batch_size = cfg.get("inference", {}).get("batch_size", 1)

    inferencer.run_on_directory(
        input_dir, output_dir,
        slice_axis=slice_axis, crop_size=crop_size, batch_size=batch_size,
    )

    print(f"\nNext step: run SynthSeg segmentation")
    print(f"  python Evaluation/segment.py --input_dir {output_dir} --output_dir {output_dir}_seg/")


if __name__ == "__main__":
    main()
