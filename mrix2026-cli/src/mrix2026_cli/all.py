"""mrix2026 all: Full pipeline."""
from __future__ import annotations
import sys

def run_all(tasks_str: str = "task1,task2,task3", skip_segment: bool = False,
            skip_check: bool = False, gt_dir: str | None = None,
            zip_output_dir: str = ".", dry_run: bool = False) -> int:
    from .config import get_env
    if dry_run:
        print("=== DRY RUN ===")
        for v in ["INFERENCE_DIR","PREDICTIONS_SEG_DIR","SUBMISSION_DIR"]:
            print(f"  {v} = {get_env(v) or '(not set)'}")
        return 0
    if not skip_segment:
        print("STEP 1/4: Segment"); from .segment import run_segment
        i, o = get_env("INFERENCE_DIR"), get_env("PREDICTIONS_SEG_DIR")
        if not i or not o: print("ERROR: INFERENCE_DIR/PREDICTIONS_SEG_DIR not set", file=sys.stderr); return 2
        rc = run_segment(i, o); 
        if rc: return rc
    print("STEP 2/4: Build"); from .build import run_build
    i, s, o = get_env("INFERENCE_DIR"), get_env("PREDICTIONS_SEG_DIR"), get_env("SUBMISSION_DIR")
    if not all([i, s, o]): print("ERROR: dirs not set", file=sys.stderr); return 2
    rc = run_build(i, s, o, tasks_str=tasks_str)
    if rc: return rc
    if not skip_check and gt_dir:
        print("STEP 3/4: Check"); from .check import run_check
        for t in tasks_str.split(","):
            t = t.strip()
            run_check(f"{o}/{t}", t, f"{gt_dir}/{t}", output_dir=f"./selfcheck_{t}")
    print("STEP 4/4: Zip"); from .zipcmd import run_zip
    rc = run_zip(o, zip_output_dir, tasks_str)
    print("\nDONE. Upload to https://www.synapse.org/Synapse:syn72060672")
    return rc
