"""Check FieldRF data, cache geometry, weights and stage configuration contracts."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from fieldrf.utils.artifacts import resolve_artifact
from fieldrf.utils.preflight import CheckReport, check_pipeline, check_stage


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cache", "pretrain", "endpoint", "vae", "paired", "all"], default="all")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--config-dir", default="configs", help="Directory with all four stage YAMLs")
    inputs.add_argument("--config", help="Check only one stage YAML; skips cross-file consistency checks")
    parser.add_argument("--report", help="Optional JSON report path")
    args = parser.parse_args()
    report = CheckReport()
    if args.config:
        if args.stage == "all":
            parser.error("--config requires a specific --stage")
        def check_one():
            with resolve_artifact(args.config, root).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            return check_stage(config, args.stage, root=root)
        result = report.attempt(args.stage, check_one)
        if result is not None:
            report = result
    else:
        report = check_pipeline(resolve_artifact(args.config_dir, root), args.stage, root=root)
    payload = dict(asdict(report), passed=report.passed)
    for item in report.domains:
        print(f"{item['domain']}: {item['volumes']} input volumes")
    for message in report.warnings:
        print(f"WARNING: {message}")
    for message in report.errors[:20]:
        print(f"ERROR: {message}")
    if len(report.errors) > 20:
        print(f"... {len(report.errors) - 20} more errors; use --report to save the full list")
    print(f"{'PASS' if report.passed else 'FAIL'}: {len(report.errors)} errors; cache headers and stage inputs checked")
    if args.report:
        Path(args.report).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
