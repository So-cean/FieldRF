"""FastSurfer segmentation — single call + streaming post-process."""
import argparse, os, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HOME = Path.home() / "FastSurfer"
_POLL_SECS = 3


def _symlink(src, dst):
    """symlink or hardlink, preferring symlink."""
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        try:
            dst.hardlink_to(src.resolve())
        except OSError:
            shutil.copy2(str(src), str(dst))


def _post(mgz_dir, t1_path, out_path, keepgeom, csf_fill):
    """mri_convert + CSF fill + mgz cleanup. Runs in ProcessPool."""
    sid = mgz_dir.name
    mgz = mgz_dir / "mri" / "aseg.auto_noCCseg.mgz"
    tmp = Path(f"/tmp/fs_pp_{os.getpid()}_{sid}.nii.gz")

    if not mgz.exists():
        return False, f"{sid}: no mgz"

    r = subprocess.run(
        ["mri_convert", "-rl", str(t1_path), "-rt", "nearest", "-odt", "short",
         str(mgz), str(tmp)], capture_output=True, text=True,
    )
    if r.returncode != 0 or not tmp.exists():
        try: shutil.rmtree(mgz_dir, ignore_errors=True)
        except: pass
        return False, f"{sid}: mri_convert failed: {r.stderr[:120] if r.stderr else ''}"

    if csf_fill:
        import nibabel as nib
        import numpy as np
        try:
            seg = nib.load(str(tmp))
            sd = np.round(np.asanyarray(seg.dataobj)).astype(np.int32)
            if keepgeom:
                bm = nib.load(str(t1_path)).get_fdata() > 0
            else:
                tmp_t1 = Path(f"/tmp/fs_t1_{os.getpid()}_{sid}.nii.gz")
                r2 = subprocess.run(
                    ["mri_convert", "-rl", str(tmp), "-rt", "linear", "-odt", "float",
                     str(t1_path), str(tmp_t1)], capture_output=True, text=True,
                )
                if r2.returncode == 0 and tmp_t1.exists():
                    bm = nib.load(str(tmp_t1)).get_fdata() > 0
                    tmp_t1.unlink(missing_ok=True)
                else:
                    tmp_t1.unlink(missing_ok=True)
                    bm = None
            if bm is not None:
                sd[(bm) & (sd == 0)] = 24
            nib.save(nib.Nifti1Image(sd.astype(np.int16), seg.affine), str(out_path))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            try: shutil.rmtree(mgz_dir, ignore_errors=True)
            except: pass
            return False, f"{sid}: CSF fill error: {e}"
    else:
        shutil.move(str(tmp), str(out_path))

    tmp.unlink(missing_ok=True)
    try: shutil.rmtree(mgz_dir, ignore_errors=True)
    except: pass
    return True, f"{sid}: OK"


def _find_ready(todo, outdir, seen):
    """Find subject dirs with complete mgz, not yet submitted."""
    ready = []
    for stem, t1_path in todo:
        if stem in seen:
            continue
        if (outdir / stem / "mri" / "aseg.auto_noCCseg.mgz").exists():
            ready.append((stem, t1_path))
    return ready


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--suffix", default="_seg")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--keepgeom", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_csf_fill", action="store_true")
    args = p.parse_args()

    if not (_HOME / "FastSurferCNN/run_prediction.py").exists():
        print("ERROR: FastSurfer not found at", _HOME); sys.exit(1)
    if not shutil.which("mri_convert"):
        print("ERROR: mri_convert not found"); sys.exit(1)

    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.rglob("*.nii.gz"))
    if not files:
        print("No nii.gz found"); return

    todo = []
    for f in files:
        stem = f.name.replace(".nii.gz", "")
        o = outdir / f"{stem}{args.suffix}.nii.gz"
        if args.skip_existing and o.exists():
            print(f"  Skip: {stem}")
        else:
            todo.append((stem, f))

    if not todo:
        print("All done."); return

    n_total = len(files)
    n_todo = len(todo)
    print(f"Processing {n_todo}/{n_total}, workers={args.num_workers}")
    print(f"Device={args.device}, batch={args.batch_size}, keepgeom={args.keepgeom}")

    # --- prepare work dir with symlinks (zero-copy) ---
    tmpdir = Path(tempfile.mkdtemp(dir=str(outdir)))
    try:
        work_dir = tmpdir / "input"
        work_dir.mkdir()
        for stem, f in todo:
            _symlink(f, work_dir / f.name)

        # --- launch FastSurfer (Popen, model stays loaded) ---
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_HOME)
        env["FASTSURFER_HOME"] = str(_HOME)
        cmd = [
            sys.executable, str(_HOME / "FastSurferCNN/run_prediction.py"),
            "--in_dir", str(work_dir), "--sd", str(outdir),
            "--batch_size", str(args.batch_size), "--device", args.device,
            "--orientation", "native", "--threads", str(args.num_workers),
            "--remove_suffix", ".nii.gz",
        ]
        fs_proc = subprocess.Popen(cmd, env=env)

        # --- streaming: poll for completed subjects while FastSurfer runs ---
        pool = ProcessPoolExecutor(max_workers=args.num_workers)
        futures = {}
        submitted = set()
        ok = fail = 0
        last_report = time.time()

        while fs_proc.poll() is None:
            time.sleep(_POLL_SECS)
            ready = _find_ready(todo, outdir, submitted)

            for stem, t1_path in ready:
                submitted.add(stem)
                out_path = outdir / f"{stem}{args.suffix}.nii.gz"
                mgz_dir = outdir / stem
                if not (mgz_dir / "mri" / "aseg.auto_noCCseg.mgz").exists():
                    continue
                job = pool.submit(_post, mgz_dir, t1_path, out_path,
                                  args.keepgeom, not args.no_csf_fill)
                futures[job] = stem

            if ready and time.time() - last_report > 30:
                print(f"  [stream] {len(submitted)}/{n_todo} submitted, "
                      f"{ok} OK {fail} fail")
                last_report = time.time()

        # --- FastSurfer done, drain remaining ---
        fs_rc = fs_proc.returncode
        if fs_rc != 0:
            print(f"FastSurfer exit code {fs_rc}, draining remaining mgz ...")
        time.sleep(2)  # let async writes finish
        ready = _find_ready(todo, outdir, submitted)
        for stem, t1_path in ready:
            submitted.add(stem)
            out_path = outdir / f"{stem}{args.suffix}.nii.gz"
            mgz_dir = outdir / stem
            if not (mgz_dir / "mri" / "aseg.auto_noCCseg.mgz").exists():
                continue
            job = pool.submit(_post, mgz_dir, t1_path, out_path,
                              args.keepgeom, not args.no_csf_fill)
            futures[job] = stem

        # --- await all post jobs ---
        for job in as_completed(futures):
            stem = futures[job]
            try:
                s, msg = job.result()
                if s: ok += 1
                else: fail += 1
                print(f"  {msg}")
            except Exception as e:
                fail += 1
                print(f"  {stem}: exception: {e}")

        pool.shutdown()

        # --- report missing ---
        for stem, _ in todo:
            if stem not in submitted:
                print(f"  NO MGZ: {stem}")
                fail += 1

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    final = len(list(outdir.glob(f"*{args.suffix}.nii.gz")))
    print(f"\nDone. Post: {ok} OK, {fail} failed. Files: {final}")


if __name__ == "__main__":
    main()
