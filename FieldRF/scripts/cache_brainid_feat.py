"""Cache Brain-ID features — unified script (all levels or single level).

Merges cache_brainid_feat_all.py + cache_brainid_feat.py into one script with
a unified .b2nd storage format (one file per subject, native blosc2 container).

Storage format:
  One .b2nd file per subject, containing all requested feats packed into a
  single 1D byte buffer.  Shape metadata is stored in the blosc2 `meta` dict,
  so the loader (data.dataset._load_feats_b2nd) can slice + reshape each feat.

  blosc2.asarray(1D_buffer, meta={"shapes": {"feat0": [...], ...}, "dtype": "float16"})

Modes (--feats):
  all      : cache feat0-4 (default, for RF multi-scale injection)
  0,1,2,3,4: cache specific levels (comma-separated, e.g. "3,4")

Feature geometry:
  Native feat4 is 192 cubed, while MAISI latents are 96 cubed.
  A two-level ControlNet conditioning pyramid needs unpooled feat4 (pool=1).
  A single-level pyramid needs pooled feat4 (pool=2).
  The default is derived from the bridge configuration; incompatible overrides
  fail before cache generation.

Usage:
  python scripts/cache_brainid_feat.py \\
      --config configs/pretrain.yaml \\
      --split Training_retrospective \\
      --fields 0.1T 1.5T 3T \\
      --feats 4 \\
      --num_workers 8 --save_workers 2 --blosc2_threads 2
"""
import sys
import os
import time
import argparse
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BRAINID_ROOT = Path(os.environ.get("BRAINID_ROOT", str(Path(__file__).resolve().parents[2] / "third_party" / "Brain-ID")))
_FIELDRF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FIELDRF))
from data.volume_dataset import VolumeDataset


def _resolve_feat4_pool(cfg, requested=None):
    """Match native 192-cubed Brain-ID features to 96-cubed MAISI latents."""
    bridge = cfg.get("bridge", {})
    channels = bridge.get("conditioning_embedding_num_channels")
    if channels is None:
        channels = bridge.get("controlnet", {}).get("conditioning_embedding_num_channels", [64, 64])
    factor = 2 ** (len(channels) - 1)
    pool = requested if requested is not None else 2 // factor
    if pool not in {1, 2} or pool * factor != 2:
        raise ValueError(
            f"feat4_pool={pool} is incompatible with conditioning pyramid {channels}: "
            "192 / feat4_pool / conditioning_downsample must equal latent size 96"
        )
    return pool


# ── GPU forward ── #

def _forward_one(batch, model, device, feat_indices, feat4_pool):
    """GPU forward for one volume -> dict of CPU fp16 tensors.

    Uses model.backbone.get_feature() (one forward, all 5 levels).
    feat4_pool=2 applies avg_pool3d to feat4 (192^3 -> 96^3) before saving.
    """
    im = batch["image"].to(device, non_blocking=True)
    im = F.interpolate(im, size=(192, 192, 192), mode="trilinear", align_corners=False)
    im = (im + 1) / 2  # [-1,1] -> [0,1] for Brain-ID

    with torch.no_grad(), torch.amp.autocast("cuda"):
        feats = model.backbone.get_feature(im)  # list of 5 tensors (1, C, D, H, W)

    del im

    feats_dict = {}
    for i in feat_indices:
        f = feats[i][0]  # squeeze batch dim
        if i == 4 and feat4_pool == 2:
            f = F.avg_pool3d(f.unsqueeze(0), 2)[0]  # 192^3 -> 96^3
        feats_dict[f"feat{i}"] = f.detach().cpu().half()

    del feats
    return feats_dict


def _forward_one_with_retry(batch, model, device, feat_indices, feat4_pool):
    """Wrapper around _forward_one with OOM retry."""
    try:
        return None, _forward_one(batch, model, device, feat_indices, feat4_pool)
    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM, retrying...")
            gc.collect()
            torch.cuda.empty_cache()
            try:
                return None, _forward_one(batch, model, device, feat_indices, feat4_pool)
            except Exception as e2:
                return str(e2), None
        return str(e), None


# ── Unified .b2nd save (native blosc2 container, one file per subject) ── #

def _save_feats_b2nd(feats_dict, pt, nthreads):
    """Save feats as one native .b2nd file (fp16 buffer + shape meta).

    All feats are flattened to 1D fp16, concatenated into one 1D fp16 buffer,
    and stored via blosc2.asarray with shape metadata in the `meta` dict.
    The loader reads meta, slices the fp16 buffer by element count, reshapes.

    Atomic write: {pt}.tmp -> rename to {pt} (prevents 0KB on crash).
    """
    import blosc2

    keys = sorted(feats_dict.keys())  # feat0, feat1, ...
    arrays = [feats_dict[k].numpy() for k in keys]  # fp16 numpy arrays
    shapes = {k: list(arrays[i].shape) for i, k in enumerate(keys)}
    dtype_name = str(arrays[0].dtype)  # "float16"

    # Flatten each feat to 1D fp16 and concatenate (keep native dtype,
    # blosc2 SHUFFLE works correctly on 2-byte fp16 groups)
    flat = np.concatenate([a.reshape(-1) for a in arrays])

    cparams = {
        "codec": blosc2.Codec.ZSTD,
        "clevel": 5,
        "filters": [blosc2.Filter.SHUFFLE],
        "filters_meta": [0],
        "nthreads": nthreads,
        "typesize": arrays[0].itemsize,  # 2 for fp16
    }

    # Chunk size: ~4 MB worth of fp16 elements
    chunk_size = min(flat.size, 4 * 1024 * 1024)

    tmp = str(pt) + ".tmp"
    try:
        blosc2.asarray(
            flat,
            urlpath=tmp,
            mode="w",
            chunks=(chunk_size,),
            meta={"shapes": shapes, "dtype": dtype_name, "keys": keys},
            cparams=cparams,
        )
        os.rename(tmp, str(pt))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ── Async save pipeline ── #

def _run_field(field_val, field_dir, ds, model, device, loader_kwargs,
               feat_indices, feat4_pool, blosc2_threads, save_workers, max_volumes=0,
               overwrite=False):
    """Process all volumes for one field with async save pipeline.

    Returns (saved, skipped, errors, elapsed, rate).
    """
    loader_kwargs = dict(loader_kwargs)
    loader = DataLoader(ds, **loader_kwargs)
    total = len(ds)

    save_executor = ThreadPoolExecutor(max_workers=save_workers)
    saved = 0
    skipped = 0
    errors = 0
    t0 = time.time()
    pending = []

    t_load = 0.0
    t_gpu = 0.0
    t_enq = 0.0
    n_timed = 0
    t_last_iter_end = time.time()

    for batch in loader:
        t_iter_start = time.time()
        t_load += t_iter_start - t_last_iter_end

        fn = batch["filename"][0]
        pt = field_dir / f"{fn.replace('.nii.gz', '')}.b2nd"

        if pt.exists() and pt.stat().st_size > 0 and not overwrite:
            skipped += 1
            saved += 1
            t_last_iter_end = time.time()
            if saved % 5 == 0:
                _print_progress(saved, total, t0)
            continue

        t1 = time.time()
        err_msg, feats_dict = _forward_one_with_retry(
            batch, model, device, feat_indices, feat4_pool
        )
        t2 = time.time()
        t_gpu += t2 - t1

        if err_msg:
            print(f"  Error forward {fn}: {err_msg}")
            errors += 1
            t_last_iter_end = time.time()
            continue

        t3 = time.time()
        future = save_executor.submit(
            _save_feats_b2nd, feats_dict, pt, blosc2_threads
        )
        t_enq += time.time() - t3
        pending.append((fn, future, pt))
        saved += 1

        # Drain completed futures
        done = [(f, fut, p) for (f, fut, p) in pending if fut.done()]
        for f, fut, p in done:
            try:
                fut.result()
            except Exception as e:
                print(f"  Error save {f}: {e}")
                errors += 1
        pending = [p for p in pending if p not in done]

        # Throttle: cap pending queue (each ~400 MB with pool=2, ~1.2 GB without)
        max_pending = 8 if feat4_pool == 2 else 4
        if len(pending) >= max_pending:
            fn_old, fut_old, pt_old = pending.pop(0)
            try:
                fut_old.result()
            except Exception as e:
                print(f"  Error save {fn_old}: {e}")
                errors += 1

        n_timed += 1
        t_last_iter_end = time.time()

        if saved % 5 == 0:
            _print_progress(saved, total, t0)

        # Smoke test: stop after max_volumes (if set)
        if max_volumes > 0 and saved >= max_volumes:
            print(f"  [smoke] reached max_volumes={max_volumes}, stopping early")
            break

    # Wait for all pending saves
    for fn_pend, fut_pend, pt_pend in pending:
        try:
            fut_pend.result()
        except Exception as e:
            print(f"  Error save {fn_pend}: {e}")
            errors += 1
    pending.clear()
    save_executor.shutdown(wait=True)

    elapsed = time.time() - t0
    rate = saved / elapsed if elapsed > 0 else 0
    if n_timed > 0:
        print(f"  Per-volume: load={t_load/n_timed:.2f}s "
              f"gpu_fwd={t_gpu/n_timed:.2f}s enq={t_enq/n_timed:.2f}s "
              f"other={max(0,(elapsed-t_load-t_gpu-t_enq))/n_timed:.2f}s")
    return saved, skipped, errors, elapsed, rate


def _print_progress(saved, total, t0):
    elapsed = time.time() - t0
    rate = saved / elapsed if elapsed > 0 else 0
    print(f"  {saved}/{total} ({elapsed:.0f}s, {rate:.2f} vol/s)")


# ── Main ── #

def main():
    parser = argparse.ArgumentParser(
        description="Cache Brain-ID features (unified, .b2nd format)"
    )
    parser.add_argument("--config", required=True,
                        help="Path to FieldRF YAML config")
    parser.add_argument("--split", default=None,
                        help="Override data.split")
    parser.add_argument("--fields", nargs="+", default=None,
                        help="Override config fields (e.g. 0.1T 1.5T)")
    parser.add_argument("--modality", default=None,
                        help="Override config modality (e.g. T1W, T2W, T2FLAIR)")
    parser.add_argument("--feats", default="all",
                        help="Features to cache: 'all' (feat0-4) or comma-separated "
                             "levels (e.g. '3,4'). Default: all")
    parser.add_argument("--feat4_pool", type=int, default=None, choices=[1, 2],
                        help="feat4 pooling: 1=192^3, 2=96^3; default matches the configured ControlNet pyramid")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader num_workers (default 8 for GPU utilization)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch_factor (default 4)")
    parser.add_argument("--blosc2_threads", type=int, default=2,
                        help="Blosc2 compression threads per save worker")
    parser.add_argument("--save_workers", type=int, default=2,
                        help="Background save threads (1-2 recommended)")
    parser.add_argument("--max_volumes", type=int, default=0,
                        help="Max volumes per field (0=all, for smoke testing)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cache files (re-cache with different pool/feats)")
    args = parser.parse_args()

    import yaml

    fieldrf_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = fieldrf_root / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    split = args.split or data_cfg.get("split", "Training_retrospective")
    modalities = [args.modality] if args.modality else data_cfg.get("modalities", [data_cfg.get("modality", "T1W")])
    args.feat4_pool = _resolve_feat4_pool(cfg, args.feat4_pool)
    fields = args.fields or data_cfg.get("fields", [])
    if not fields:
        raise ValueError("No fields specified (config.fields or --fields)")

    # Parse feat indices
    if args.feats == "all":
        feat_indices = list(range(5))
    else:
        feat_indices = [int(x) for x in args.feats.split(",")]
    print(f"Config: feats={feat_indices}, feat4_pool={args.feat4_pool}, "
          f"num_workers={args.num_workers}, prefetch={args.prefetch_factor}")

    brainid_checkpoint = cfg.get("brainid", {}).get("ckpt_path")
    if not brainid_checkpoint or not Path(brainid_checkpoint).is_file():
        raise FileNotFoundError("Set brainid.ckpt_path to the downloaded Brain-ID checkpoint")
    if not (BRAINID_ROOT / "BrainID").is_dir():
        raise FileNotFoundError(
            "Initialize the Brain-ID submodule with git submodule update --init --recursive "
            "from the repository root, or set BRAINID_ROOT to another Brain-ID checkout"
        )
    # Import upstream utilities only during execution, so --help and native
    # cache serialization do not require an external Brain-ID installation.
    sys.path.insert(0, str(BRAINID_ROOT))
    from BrainID.models import build_feat_model
    from utils.checkpoint import load_checkpoint
    import utils.misc as brainid_utils

    # ── Brain-ID model ──
    print("Loading Brain-ID...")
    bd_cfg = BRAINID_ROOT / "cfgs"
    a = brainid_utils.preprocess_cfg([
        str(bd_cfg / "default_train.yaml"),
        str(bd_cfg / "default_dataset.yaml"),
        str(bd_cfg / "submit.yaml"),
    ])
    _, model, _, _, _ = build_feat_model(a, device=str(device))
    load_checkpoint(brainid_checkpoint, [model], None, ["model"], to_print=False)
    model.eval()

    # torch.compile for speedup
    try:
        print("Compiling feature extractor...")
        model.backbone.get_feature = torch.compile(
            model.backbone.get_feature, mode="default"
        )
        _warm = torch.zeros(1, 1, 192, 192, 192, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = model.backbone.get_feature(_warm)
        del _warm
        torch.cuda.synchronize()
        print("Brain-ID loaded + compiled.")
    except Exception as e:
        print(f"  torch.compile skipped ({e})")
        print("Brain-ID loaded (no compile).")

    for modality in modalities:
        # ── Output root (cache name from config, used verbatim) ──
        # The config's feat_cache_name should reflect the pooling (e.g.
        # brainid_feat_all_pool2 when feat4_pool=2). The script uses it as-is.
        cache_name = data_cfg.get("feat_cache_name", "brainid_feat")
        feat_root = Path(data_cfg["cache_root"]) / split / cache_name / modality
        feat_root.mkdir(parents=True, exist_ok=True)
        print(f"Output: {feat_root}")

        # ── DataLoader kwargs (GPU utilization: more workers + deeper prefetch) ──
        loader_kwargs = dict(batch_size=1, shuffle=False, num_workers=args.num_workers)
        if args.num_workers > 0:
            loader_kwargs["pin_memory"] = True
            loader_kwargs["prefetch_factor"] = args.prefetch_factor

        for field_val in fields:
            field_dir = feat_root / str(field_val)
            field_dir.mkdir(parents=True, exist_ok=True)

            ds = VolumeDataset(
                data_cfg["data_root"], split,
                modality, str(field_val),
            )
            print(f"\n[{field_val}] {len(ds)} vols -> {field_dir}")

            n_saved, n_skipped, n_errors, elapsed, rate = _run_field(
                field_val, field_dir, ds, model, device,
                loader_kwargs, feat_indices, args.feat4_pool,
                args.blosc2_threads, args.save_workers, args.max_volumes,
                overwrite=args.overwrite,
            )
            print(f"  Done: {n_saved}/{len(ds)} (skipped {n_skipped}, "
                  f"errors {n_errors}) in {elapsed:.0f}s @ {rate:.2f} vol/s")

        print("Done.")


if __name__ == "__main__":
    main()
