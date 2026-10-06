#!/usr/bin/env python3
"""mrix2026: CLI tools for MRIxFields2026."""
from __future__ import annotations
import argparse, sys


def _add_build_parser(sub):
    sub.add_argument("--predictions-dir", help="Inference dir (default: $INFERENCE_DIR)")
    sub.add_argument("--predictions-seg-dir", help="Seg dir (default: $PREDICTIONS_SEG_DIR)")
    sub.add_argument("--output-dir", help="Submission output (default: $SUBMISSION_DIR)")
    sub.add_argument("--tasks", default="task1,task2,task3")
    for t in ("task1","task2","task3"):
        sub.add_argument(f"--{t}-method"); sub.add_argument(f"--{t}-mode"); sub.add_argument(f"--{t}-epoch")
    sub.add_argument("--dry-run", action="store_true"); sub.add_argument("--clean", action="store_true")


def main() -> int:
    p = argparse.ArgumentParser(prog="mrix2026", description="CLI tools for MRIxFields2026")
    s = p.add_subparsers(dest="command", required=True)

    sp = s.add_parser("segment", help="Run SynthSeg on sweep predictions")
    sp.add_argument("--input-dir", help="Pred dir (default: $INFERENCE_DIR)")
    sp.add_argument("--output-dir", help="Seg output (default: $PREDICTIONS_SEG_DIR)")
    sp.add_argument("--suffix", default="_seg"); sp.add_argument("--num-workers", type=int, default=4)
    sp.add_argument("--dry-run", action="store_true")

    bp = s.add_parser("build", help="Build submission directory tree")
    _add_build_parser(bp)

    zp = s.add_parser("zip", help="Zip task directories")
    zp.add_argument("--submission-dir", help="Submission dir (default: $SUBMISSION_DIR)")
    zp.add_argument("--output-dir", default="."); zp.add_argument("--tasks", default="task1,task2,task3")

    cp = s.add_parser("check", help="Self-check against GT")
    cp.add_argument("--input-dir", required=True); cp.add_argument("--task", required=True)
    cp.add_argument("--gt-dir", required=True); cp.add_argument("--output-dir", default="./selfcheck")
    cp.add_argument("--device", default="cuda:0")

    ap = s.add_parser("all", help="Full pipeline")
    ap.add_argument("--tasks", default="task1,task2,task3"); ap.add_argument("--skip-segment", action="store_true")
    ap.add_argument("--skip-check", action="store_true"); ap.add_argument("--gt-dir")
    ap.add_argument("--zip-output-dir", default="."); ap.add_argument("--dry-run", action="store_true")

    args = p.parse_args()
    from . import config as _cfg; _cfg.load_env()

    dispatch = {
        "segment": _run_segment, "build": _run_build, "zip": _run_zip,
        "check": _run_check, "all": _run_all,
    }
    return dispatch[args.command](args)


def _resolve(args, key: str, flag: str) -> str:
    v = getattr(args, flag, None) or ""
    if v: return v
    from .config import get_env
    ev = get_env(key)
    if ev: return ev
    print(f"ERROR: neither --{flag} nor {key} is set", file=sys.stderr)
    sys.exit(1)


def _run_segment(args) -> int:
    from .segment import run_segment
    return run_segment(_resolve(args, "INFERENCE_DIR", "input_dir"),
                       _resolve(args, "PREDICTIONS_SEG_DIR", "output_dir"),
                       args.suffix, args.num_workers, args.dry_run)


def _run_build(args) -> int:
    from .build import run_build
    return run_build(
        _resolve(args, "INFERENCE_DIR", "predictions_dir"),
        _resolve(args, "PREDICTIONS_SEG_DIR", "predictions_seg_dir"),
        _resolve(args, "SUBMISSION_DIR", "output_dir"),
        args.tasks, args.dry_run, args.clean,
        {t: getattr(args, f"{t}_method", None) for t in ("task1","task2","task3")},
        {t: getattr(args, f"{t}_mode", None) for t in ("task1","task2","task3")},
        {t: getattr(args, f"{t}_epoch", None) for t in ("task1","task2","task3")},
    )


def _run_zip(args) -> int:
    from .zipcmd import run_zip
    return run_zip(_resolve(args, "SUBMISSION_DIR", "submission_dir"), args.output_dir, args.tasks)


def _run_check(args) -> int:
    from .check import run_check
    return run_check(args.input_dir, args.task, args.gt_dir, args.output_dir, args.device)


def _run_all(args) -> int:
    from .all import run_all
    return run_all(args.tasks, args.skip_segment, args.skip_check, args.gt_dir, args.zip_output_dir, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
