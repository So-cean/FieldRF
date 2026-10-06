"""SynthSeg brain segmentation for MRIxFields2026 challenge.

Runs SynthSeg 2.0 on NIfTI files to produce segmentation maps.
Required for Task 1 / Task 2 if you want Dice / Volume metrics
(participants submit segmentations alongside generated images).
Task 3 is voxel-level only and does NOT require this step.

Usage:
    # Segment all predictions
    python Evaluation/segment.py \
        --input_dir $INFERENCE_DIR/ \
        --output_dir ${INFERENCE_DIR}_seg/

    # Segment ground truth (needed for evaluation)
    python Evaluation/segment.py \
        --input_dir $DATA_DIR/ground_truth/ \
        --output_dir $DATA_DIR/ground_truth_seg/

Configuration:
    Set SYNTHSEG_DIR in .env (repo root) or as an environment variable.
    See .env.example for details.

Requires:
    - TensorFlow 2.15 with GPU support
    - Standalone SynthSeg (https://github.com/BBillot/SynthSeg)
"""

import os
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
#  Environment: load .env from repo root
# ---------------------------------------------------------------------------

def _load_dotenv():
    """Load .env file from repo root if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value


def _get_synthseg_dir() -> Path:
    """Get SYNTHSEG_DIR from environment and verify required model files exist."""
    _load_dotenv()
    synthseg_dir = os.environ.get("SYNTHSEG_DIR")
    if not synthseg_dir:
        print("ERROR: SYNTHSEG_DIR is not set.")
        print("Set it in .env (repo root) or as an environment variable.")
        print("See .env.example for details.")
        sys.exit(1)
    path = Path(synthseg_dir)
    if not path.exists():
        print(f"ERROR: SYNTHSEG_DIR does not exist: {path}")
        print("Install SynthSeg: git clone https://github.com/BBillot/SynthSeg.git")
        sys.exit(1)

    # Verify SynthSeg 2.0 files that run_synthseg() will load (fail fast).
    required = [
        path / "models" / "synthseg_2.0.h5",
        path / "data" / "labels_classes_priors" / "synthseg_segmentation_labels_2.0.npy",
        path / "data" / "labels_classes_priors" / "synthseg_denoiser_labels_2.0.npy",
        path / "data" / "labels_classes_priors" / "synthseg_topological_classes_2.0.npy",
    ]
    missing = [str(f.relative_to(path)) for f in required if not f.exists()]
    if missing:
        print(f"ERROR: Missing SynthSeg 2.0 files in {path}:")
        for m in missing:
            print(f"  - {m}")
        print("Download from https://github.com/BBillot/SynthSeg")
        sys.exit(1)
    return path


# ---------------------------------------------------------------------------
#  SynthSeg runner
# ---------------------------------------------------------------------------

_synthseg_initialized = False


def _init_synthseg(synthseg_dir: Path):
    """Add SynthSeg to sys.path (once)."""
    global _synthseg_initialized
    if _synthseg_initialized:
        return
    synthseg_str = str(synthseg_dir)
    if synthseg_str not in sys.path:
        sys.path.insert(0, synthseg_str)
    _synthseg_initialized = True


def run_synthseg(nifti_path, output_path, synthseg_dir: Path):
    """Run SynthSeg 2.0 segmentation on a single NIfTI file."""
    _init_synthseg(synthseg_dir)
    from SynthSeg.predict_synthseg import predict

    model_dir = synthseg_dir / "models"
    labels_dir = synthseg_dir / "data" / "labels_classes_priors"

    predict(
        path_images=str(nifti_path),
        path_segmentations=str(output_path),
        path_model_segmentation=str(model_dir / "synthseg_2.0.h5"),
        labels_segmentation=str(labels_dir / "synthseg_segmentation_labels_2.0.npy"),
        robust=False,
        fast=False,
        v1=False,
        n_neutral_labels=19,
        labels_denoiser=str(labels_dir / "synthseg_denoiser_labels_2.0.npy"),
        path_posteriors=None,
        path_resampled=None,
        path_volumes=None,
        do_parcellation=False,
        path_model_parcellation=None,
        labels_parcellation=None,
        path_qc_scores=None,
        path_model_qc=None,
        labels_qc=None,
        cropping=None,
        topology_classes=str(labels_dir / "synthseg_topological_classes_2.0.npy"),
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MRIxFields2026 SynthSeg Segmentation",
        epilog="Run this after inference to produce segmentation maps for submission.",
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing NIfTI files to segment")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save segmentation results")
    parser.add_argument("--suffix", type=str, default="_seg",
                        help="Suffix added to output filenames (default: _seg)")
    args = parser.parse_args()

    synthseg_dir = _get_synthseg_dir()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.rglob("*.nii.gz"))
    if not nifti_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    print(f"SynthSeg dir: {synthseg_dir}")
    print(f"Found {len(nifti_files)} NIfTI files to segment")
    print(f"Output directory: {output_dir}\n")

    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace(".nii.gz", "")
        out_path = output_dir / f"{stem}{args.suffix}.nii.gz"

        if out_path.exists():
            print(f"[{i}/{len(nifti_files)}] Skipping (exists): {out_path.name}")
            continue

        print(f"[{i}/{len(nifti_files)}] Segmenting: {nifti_path.name}")
        run_synthseg(nifti_path, out_path, synthseg_dir)

    print(f"\nSegmentation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
