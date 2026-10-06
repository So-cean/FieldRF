"""mrix2026 zip: Zip task dirs with task{N}/ prefix (CWL-compatible)."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

def run_zip(submission_dir: str, output_dir: str, tasks_str: str) -> int:
    sub_root = Path(submission_dir)
    out_root = Path(output_dir)
    if not sub_root.is_dir():
        print(f"ERROR: {sub_root} not found", file=sys.stderr)
        return 2
    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
    out_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task_dir = sub_root / task
        if not task_dir.is_dir():
            print(f"WARNING: {task} dir not found")
            continue
        zip_path = out_root / f"{task}.zip"
        print(f"Zipping {task_dir} -> {zip_path} ...")
        r = subprocess.run(["zip", "-r", str(zip_path), task],
                           cwd=str(sub_root), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"ERROR: zip failed: {r.stderr}", file=sys.stderr)
            return r.returncode
        lines = r.stdout.strip().split("\n")
        if lines:
            print(f"  {lines[-1]}")
    print(f"\nZips in: {out_root}")
    return 0
