"""SynthSeg brain segmentation for MRIxFields2026 challenge (FreeSurfer built-in).

Runs FreeSurfer's mri_synthseg on NIfTI files to produce segmentation maps.
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

Requires:
    - FreeSurfer 7.4+ (includes SynthSeg 2.0 via mri_synthseg)
    - $FREESURFER_HOME set and sourced (setup scripts run)
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading


# ---------------------------------------------------------------------------
#  Environment verification
# ---------------------------------------------------------------------------

def _verify_freesurfer() -> str:
    """Check that FreeSurfer's mri_synthseg is available."""
    cmd = shutil.which("mri_synthseg")
    if not cmd:
        print("ERROR: mri_synthseg not found in PATH.")
        print("Ensure FreeSurfer is installed and sourced, e.g.:")
        print("  export FREESURFER_HOME=/usr/local/freesurfer/7.4.1")
        print("  source $FREESURFER_HOME/SetUpFreeSurfer.sh")
        sys.exit(1)
    return cmd


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _safe_print(msg):
    with _print_lock:
        print(msg, flush=True)


def _process_single(nifti_path, output_dir, suffix, mri_synthseg,
                    skip_existing=False, keepgeom=False):
    """Run SynthSeg on a single file. Returns (success, message)."""
    stem = nifti_path.name.replace(".nii.gz", "")
    out_path = output_dir / f"{stem}{suffix}.nii.gz"

    if skip_existing and out_path.exists():
        return True, f"Skipping (exists): {out_path.name}"

    cmd = [
        mri_synthseg,
        "--i", str(nifti_path),
        "--o", str(out_path),
        "--threads", "8",
    ]
    if keepgeom:
        cmd += ["--keepgeom"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        return False, f"FAILED: {nifti_path.name}\n{result.stderr}"
    return True, f"Done: {nifti_path.name}"


def main():
    parser = argparse.ArgumentParser(
        description="MRIxFields2026 SynthSeg Segmentation (FreeSurfer)",
        epilog="Run this after inference to produce segmentation maps for submission.",
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing NIfTI files to segment")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save segmentation results")
    parser.add_argument("--suffix", type=str, default="_seg",
                        help="Suffix added to output filenames (default: _seg)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel processes (default: 8)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip files that already exist in output_dir (default: False)")
    parser.add_argument("--keepgeom", action="store_true",
                        help="Preserve original resolution instead of resampling to 1mm (default: False)")
    args = parser.parse_args()

    mri_synthseg = _verify_freesurfer()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.rglob("*.nii.gz"))
    if not nifti_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    print(f"mri_synthseg: {mri_synthseg}")
    print(f"Found {len(nifti_files)} NIfTI files to segment")
    print(f"Output directory: {output_dir}")
    if args.keepgeom:
        print("--keepgeom: preserving original resolution")
    print(f"Workers: {args.num_workers}\n")

    completed = 0
    failed = 0
    total = len(nifti_files)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                _process_single, f, output_dir, args.suffix, mri_synthseg,
                args.skip_existing, args.keepgeom,
            ): f
            for f in nifti_files
        }
        for future in as_completed(futures):
            nifti_path = futures[future]
            try:
                success, msg = future.result()
                completed += 1
                _safe_print(f"[{completed}/{total}] {msg}")
                if not success:
                    failed += 1
            except Exception as e:
                completed += 1
                failed += 1
                _safe_print(f"[{completed}/{total}] EXCEPTION: {nifti_path.name}: {e}")

    print(f"\nSegmentation complete. {completed - failed}/{total} succeeded, {failed} failed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
