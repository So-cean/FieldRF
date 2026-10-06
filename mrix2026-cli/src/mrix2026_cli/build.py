"""mrix2026 build: Build Synapse submission tree from sweep outputs."""
from __future__ import annotations
import re, shutil, sys
from pathlib import Path

import nibabel as nib
from .constants import MODS, NEEDS_SEG, PAIRS, EXPECTED_FILES_PER_SUBTASK, DEFAULT_SWEEP, Z_CLIP_RANGE, parse_pair


def _copy_with_axial_clip(src: Path, dst: Path) -> None:
    z0, z1 = Z_CLIP_RANGE
    img = nib.load(str(src))
    nib.save(img.slicer[:, :, z0:z1], str(dst))


def _extract_id(filename: str) -> str | None:
    """Extract trailing subject ID, same logic as score.py (ignores model/field)."""
    base = filename.replace(".nii.gz", "")
    if base.endswith("_seg"):
        base = base[: -len("_seg")]
    parts = base.split("_")
    last = parts[-1] if parts else ""
    return last if re.fullmatch(r"\d+", last) else None


def _sweep_dir(root: Path, task: str, pair: str, mod: str, method: str, mode: str, epoch_tag: str) -> Path:
    return root / f"{task}_{pair}_{mod}" / method / mode / epoch_tag


def _build_one_task(task: str, method: str, mode: str, epoch_tag: str,
                     pred_root: Path, seg_root: Path, out_root: Path, dry_run: bool) -> tuple[int, int, list[str]]:
    needs_seg = NEEDS_SEG[task]
    pairs = PAIRS[task]
    expected_pred = len(pairs) * len(MODS) * EXPECTED_FILES_PER_SUBTASK
    expected_seg = expected_pred if needs_seg else 0
    copied_pred = copied_seg = 0
    warnings: list[str] = []
    tgt_dir = out_root / task

    print(f"\n=== {task}: {method}/{mode}/{epoch_tag} ===")
    print(f"  expected pred: {expected_pred}" + (f", seg: {expected_seg}" if needs_seg else ""))

    for mod in MODS:
        for pair in pairs:
            src_field, tgt_field = parse_pair(pair)
            sp = _sweep_dir(pred_root, task, pair, mod, method, mode, epoch_tag)
            dp = tgt_dir / mod / pair / "pred"
            if not sp.is_dir():
                warnings.append(f"missing pred dir: {sp}")
                continue
            sfiles = sorted(sp.glob("P_*.nii.gz"))
            sfiles = [f for f in sfiles if not f.name.endswith("_seg.nii.gz")]
            if len(sfiles) < EXPECTED_FILES_PER_SUBTASK:
                warnings.append(f"only {len(sfiles)}/{EXPECTED_FILES_PER_SUBTASK} pred in {sp}")
            if not dry_run:
                dp.mkdir(parents=True, exist_ok=True)
            for f in sfiles:
                fid = _extract_id(f.name)
                if fid is None:
                    warnings.append(f"unparseable: {f}")
                    continue
                df = dp / f"P_{mod}_{tgt_field}_{fid}.nii.gz"
                if dry_run:
                    print(f"  [dry-run] {f} -> {df}")
                else:
                    _copy_with_axial_clip(f, df)
                copied_pred += 1

            if needs_seg:
                ss = _sweep_dir(seg_root, task, pair, mod, method, mode, epoch_tag)
                ds = tgt_dir / mod / pair / "seg"
                if not ss.is_dir():
                    warnings.append(f"missing seg dir: {ss}")
                    continue
                sfiles = sorted(ss.glob("P_*_seg.nii.gz"))
                if len(sfiles) < EXPECTED_FILES_PER_SUBTASK:
                    warnings.append(f"only {len(sfiles)}/{EXPECTED_FILES_PER_SUBTASK} seg in {ss}")
                if not dry_run:
                    ds.mkdir(parents=True, exist_ok=True)
                for f in sfiles:
                    fid = _extract_id(f.name)
                    if fid is None:
                        warnings.append(f"unparseable seg: {f}")
                        continue
                    df = ds / f"P_{mod}_{tgt_field}_{fid}_seg.nii.gz"
                    if dry_run:
                        print(f"  [dry-run] {f} -> {df}")
                    else:
                        _copy_with_axial_clip(f, df)
                    copied_seg += 1

    print(f"  copied pred: {copied_pred}/{expected_pred}" + (f", seg: {copied_seg}/{expected_seg}" if needs_seg else ""))
    return copied_pred, copied_seg, warnings


def run_build(predictions_dir: str, predictions_seg_dir: str, output_dir: str,
              tasks_str: str = "task1,task2,task3", dry_run: bool = False, clean: bool = False,
              method_overrides: dict | None = None, mode_overrides: dict | None = None,
              epoch_overrides: dict | None = None) -> int:
    pred_root = Path(predictions_dir)
    seg_root = Path(predictions_seg_dir)
    out_root = Path(output_dir)
    if not pred_root.is_dir():
        print(f"ERROR: predictions dir not found: {pred_root}", file=sys.stderr)
        return 2
    if not seg_root.is_dir():
        print(f"WARNING: seg dir not found: {seg_root}", file=sys.stderr)

    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
    for t in tasks:
        if t not in PAIRS:
            print(f"ERROR: unknown task {t!r}", file=sys.stderr)
            return 2

    overrides = {}
    for t in tasks:
        m = DEFAULT_SWEEP[t]
        overrides[t] = ((method_overrides or {}).get(t) or m[0],
                        (mode_overrides or {}).get(t) or m[1],
                        (epoch_overrides or {}).get(t) or m[2])

    if clean and not dry_run:
        for t in tasks:
            target = out_root / t
            if target.exists():
                print(f"  [clean] removing {target}")
                shutil.rmtree(target)

    all_warnings: list[str] = []
    totals = {"pred": 0, "seg": 0}
    for t in tasks:
        meth, mode, ep = overrides[t]
        cp, cs, ws = _build_one_task(t, meth, mode, ep, pred_root, seg_root, out_root, dry_run)
        totals["pred"] += cp
        totals["seg"] += cs
        all_warnings.extend(f"[{t}] {w}" for w in ws)

    print("\n=== Summary ===")
    print(f"  total pred: {totals['pred']}, seg: {totals['seg']}")
    if all_warnings:
        print(f"\n=== Warnings ({len(all_warnings)}) ===")
        for w in all_warnings:
            print(f"  {w}")
    if not dry_run:
        print(f"\nNext step:\n  cd {out_root} && zip -r ~/task1.zip task1")
    return 0
