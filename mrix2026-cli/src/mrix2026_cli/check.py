"""mrix2026 check: Self-check submission against GT."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

def _find_score() -> Path | None:
    from .config import find_repo_root
    repo = find_repo_root()
    c = repo / "Submission" / "evaluation-2026" / "score.py"
    return c if c.exists() else None

def run_check(input_dir: str, task: str, gt_dir: str, output_dir: str = "./selfcheck", device: str = "cuda:0") -> int:
    in_path = Path(input_dir)
    gt_path = Path(gt_dir)
    out_path = Path(output_dir)
    if not in_path.is_dir():
        print(f"ERROR: input dir not found: {in_path}", file=sys.stderr)
        return 2
    if not gt_path.is_dir():
        print(f"ERROR: GT dir not found: {gt_path}", file=sys.stderr)
        return 2
    score = _find_score()
    if score is None:
        print("ERROR: score.py not found", file=sys.stderr)
        return 2
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(score), "-i", str(in_path), "-t", task, "-g", str(gt_path), "-o", str(out_path), "--device", device]
    print(f"Running: {' '.join(cmd)}")
    r = subprocess.run(cmd)
    return r.returncode
