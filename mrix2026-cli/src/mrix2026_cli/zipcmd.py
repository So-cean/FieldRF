"""mrix2026 zip: Package modality directories at the archive root."""
from __future__ import annotations
import sys
import zipfile
from pathlib import Path
from .constants import MODS

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
        files = sorted(path for mod in MODS for path in (task_dir / mod).rglob("*") if path.is_file())
        if not files:
            print(f"ERROR: no modality files found in {task_dir}", file=sys.stderr)
            return 2
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, arcname=path.relative_to(task_dir).as_posix())
        print(f"  packaged {len(files)} files")
    print(f"\nZips in: {out_root}")
    return 0
