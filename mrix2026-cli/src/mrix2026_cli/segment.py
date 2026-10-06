"""mrix2026 segment: Run SynthSeg on sweep predictions (tree-mirroring)."""
from __future__ import annotations
import os
import subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def _synthseg_executable() -> str:
    """Resolve SynthSeg from an explicit override, FreeSurfer, or PATH."""
    override = os.environ.get("SYNTHSEG_BIN")
    if override:
        return override
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    if freesurfer_home:
        return str(Path(freesurfer_home) / "bin" / "mri_synthseg")
    return "mri_synthseg"


def _seg_one(p: Path, in_root: Path, out_root: Path, suffix: str, dry_run: bool) -> tuple[bool, str]:
    rel = p.relative_to(in_root)
    stem = p.name.replace(".nii.gz", "")
    out = out_root / rel.parent / f"{stem}{suffix}.nii.gz"
    if dry_run:
        return True, f"[dry-run] {rel} -> {out}"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [_synthseg_executable(), "--i", str(p), "--o", str(out), "--robust",
         "--threads", "4", "--keepgeom"],
        capture_output=True, text=True)
    return (r.returncode == 0, f"OK: {rel}" if r.returncode == 0 else f"FAIL: {rel}\n{r.stderr[:200]}")


def run_segment(input_dir: str, output_dir: str, suffix: str = "_seg",
                num_workers: int = 4, dry_run: bool = False) -> int:
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    if not in_root.is_dir():
        print(f"ERROR: input dir not found: {in_root}", file=sys.stderr)
        return 2

    files = sorted(in_root.rglob("*.nii.gz"))
    files = [f for f in files if not f.name.endswith("_seg.nii.gz")]
    if not files:
        print(f"No .nii.gz files under {in_root}")
        return 0
    print(f"Found {len(files)} prediction files, workers={num_workers}")

    done = failed = 0
    n = len(files)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for fut in as_completed({pool.submit(_seg_one, f, in_root, out_root, suffix, dry_run): f for f in files}):
            done += 1
            ok, msg = fut.result()
            if not ok:
                failed += 1
            print(f"[{done}/{n}] {msg}")

    print(f"\nDone: {n - failed}/{n} OK, {failed} failed")
    return 0 if failed == 0 else 1
