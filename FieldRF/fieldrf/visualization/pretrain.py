"""[FieldRF] Vis: eval-aligned (central 64-slice decode, axial [150,180) crop,
torch metrics matching Evaluation/evaluate.py, original-image comparison).

Per paired prospective sample (per modality):
  1. Euler ODE sample z_harm from feats_src, z_src, and target class.
  2. Unscale (z_harm is already unscaled from sample_x0).
  3. Decode 64 slices centered at z_center.
  4. Load original source + target nii.gz -> RAS 384 [0,1].
  5. Crop everything to the 30-slice eval window.
  6. Compute metrics: bridge, image.
"""

import time
from pathlib import Path

import numpy as np
import torch

from fieldrf.data.dataset import PairedVisualizationDataset
from fieldrf.utils.domains import get_class_label  # noqa: E402
from fieldrf.utils.visualization import (
    load_vae, decode_volume_centered, load_orig_ras_384, find_nifti_path,
    eval_window_in_384, compute_metrics_eval, render_evaluation_grid, init_metrics_csv,
)


@torch.no_grad()
def run_vis(
    bridge,
    scale_factor,
    cfg,
    epoch,
    device,
    endpoint_aligner=None,
):
    """RF eval-aligned vis for multi-modal field translation.

    Called from train.py each epoch. Generates per-modality PNG + CSV.
    """
    data_cfg = cfg["data"]
    is_real_paired_any_to_any = (
        data_cfg.get("split") == "Training_prospective"
        and bool(data_cfg.get("subjects"))
        and "cross_rf" in cfg
        and "inversion" in cfg
    )
    if is_real_paired_any_to_any:
        from fieldrf.visualization.paired import run_vis as run_paired_any_to_any

        return run_paired_any_to_any(
            bridge,
            scale_factor,
            cfg,
            epoch,
            device,
            endpoint_aligner=endpoint_aligner,
        )

    vis_cfg = cfg.get("vis", {})
    vae_cfg = cfg["vae"]
    inf_cfg = cfg.get("inference", {})
    # Training-time validation may use a cheaper deterministic Euler proxy
    # while keeping the deployment/inference contract unchanged.  Final
    # checkpoint comparisons should still use ``inference.n_infer_steps``.
    n_infer = vis_cfg.get(
        "n_infer_steps",
        inf_cfg.get("n_infer_steps", 20),
    )
    sampling_seed = int(vis_cfg.get("sampling_seed", 2026))
    amp_dtype = getattr(torch, vae_cfg.get("sliding_window", {}).get("amp_dtype", "float16"))
    eval_range = tuple(vis_cfg.get("eval_slice_range", [150, 180]))

    vis_split = vis_cfg.get("split", "Training_prospective")
    cache_root = data_cfg["cache_root"]
    feat_cache_name = data_cfg.get("feat_cache_name", "brainid_feat")
    source_fields = data_cfg.get("source_fields", [data_cfg.get("source_field")])
    target_fields = tuple(str(value) for value in data_cfg["target_fields"])
    if len(target_fields) != 1:
        raise ValueError("Task 1 validation requires exactly one target field")
    target_field = target_fields[0]
    n_per_field = vis_cfg.get("n_samples", 1)

    vae = load_vae(vae_cfg["checkpoint"], device)
    sw_cfg = vae_cfg.get("sliding_window", {"roi_size": [128, 128, 64]})

    output_dir = Path(cfg.get("output_dir", "runs/pretrain")) / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge.eval()

    for src_mod in data_cfg.get("modalities", [data_cfg.get("modality", "T1W")]):
        latent_root = f"{cache_root}/{vis_split}/latent/{src_mod}"
        feat_all_root = f"{cache_root}/{vis_split}/{feat_cache_name}/{src_mod}"

        ds = PairedVisualizationDataset(
            latent_root=latent_root,
            feat_all_root=feat_all_root,
            source_fields=source_fields,
            target_field=target_field,
            modality=src_mod,
            samples_per_field=n_per_field,
        )
        if len(ds) == 0:
            print(f"  [vis] No paired data for {src_mod}, skipping.", flush=True)
            continue

        target_class = get_class_label(src_mod, target_field)
        print(f"  [vis] {src_mod}: {len(ds)} pairs, Euler ODE {n_infer} steps, "
              f"target={src_mod}@{target_field}", flush=True)

        # Per-modality CSV
        metrics_csv = output_dir / f"vis_metrics_{src_mod}.csv"
        writer, csv_f = init_metrics_csv(
            metrics_csv,
            ["epoch", "source_field", "subject",
             "nrmse_vae", "ssim_vae", "psnr_vae",
             "nrmse_bridge", "ssim_bridge", "psnr_bridge",
             "nrmse_image", "ssim_image", "psnr_image"],
        )

        rows_data = []
        t0 = time.time()

        for i in range(len(ds)):
            batch = ds[i]
            z_src = batch["z_src"].unsqueeze(0).to(device)
            z_tgt = batch["z_tgt"].unsqueeze(0).to(device) * scale_factor
            feats = [f.unsqueeze(0).to(device) for f in batch["feats_src"]]
            sf = batch["source_field"]
            fn = batch["filename"]
            subj_id = fn.rsplit("_", 1)[-1]

            # Locate original source + target nii.gz
            tgt_nii = find_nifti_path(data_cfg["data_root"], vis_split, src_mod, target_field, subj_id)
            src_nii = find_nifti_path(data_cfg["data_root"], vis_split, src_mod, sf, subj_id)
            if tgt_nii is None or src_nii is None:
                print(f"  [vis] SKIP {fn}: nii not found "
                      f"(src={src_nii is not None}, tgt={tgt_nii is not None})", flush=True)
                continue

            # Original target + source in RAS 384 [0,1]
            orig_tgt_384, _, pad_before_z, crop_start_z = load_orig_ras_384(tgt_nii)
            orig_src_384, _, _, _ = load_orig_ras_384(src_nii)

            # Eval window in 384-space (30 slices, [150,180) mapped)
            z0_384, z1_384 = eval_window_in_384(pad_before_z, crop_start_z, eval_range)
            z_center = (z0_384 + z1_384) // 2

            endpoint_cfg = inf_cfg.get("endpoint_transport", {})
            if bool(endpoint_cfg.get("enabled", False)):
                source_class = get_class_label(src_mod, sf)
                source_endpoint = bridge.invert_to_endpoint(
                    z_src * scale_factor,
                    feats,
                    torch.tensor([source_class], device=device),
                    n_steps=int(endpoint_cfg.get("inversion_steps", 4)),
                    solver=str(endpoint_cfg.get("inversion_solver", "euler")),
                )
                if endpoint_aligner is not None:
                    source_endpoint, _ = endpoint_aligner(
                        source_endpoint.float(),
                        torch.tensor([source_class], device=device),
                    )
                z_harm = bridge.sample_x0_from_endpoint(
                    source_endpoint,
                    feats,
                    torch.tensor([target_class], device=device),
                    n_infer_steps=n_infer,
                )
            else:
                # Fixed per-pair noise across epochs makes ordinary Stage-1
                # conditional-generation comparisons deterministic.
                cuda_devices = [device.index] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(sampling_seed + i)
                    if device.type == "cuda":
                        torch.cuda.manual_seed(sampling_seed + i)
                    z_harm = bridge.sample_x0(
                        feats, torch.tensor([target_class], device=device),
                        n_infer_steps=n_infer, device=device, z_src=z_src,
                    )

            # z_harm is already unscaled from sample_x0
            harm_img = decode_volume_centered(vae, z_harm, sw_cfg, amp_dtype, z_center)
            tgt_img = decode_volume_centered(vae, z_tgt / scale_factor, sw_cfg, amp_dtype, z_center)

            harm_dec = torch.clamp((harm_img + 1) / 2, 0, 1)[0, 0].float().cpu().numpy()
            tgt_dec = torch.clamp((tgt_img + 1) / 2, 0, 1)[0, 0].float().cpu().numpy()

            # Crop to eval window
            win_start = round((z_center - 32) / 4) * 4
            c0 = max(0, z0_384 - win_start)
            c1 = min(harm_dec.shape[2], z1_384 - win_start)
            harm_slab = harm_dec[:, :, c0:c1]
            tgt_slab = tgt_dec[:, :, c0:c1]
            orig_tgt = orig_tgt_384[:, :, z0_384:z1_384]
            orig_src = orig_src_384[:, :, z0_384:z1_384]

            # Brain mask
            brain_mask = (orig_tgt > 0).astype(np.float32)
            harm_m = harm_slab * brain_mask
            tgt_dec_m = tgt_slab * brain_mask
            orig_tgt_m = orig_tgt * brain_mask

            # Metrics
            m_vae = compute_metrics_eval(tgt_dec_m, orig_tgt_m, slice_axis=2)
            m_bridge = compute_metrics_eval(harm_m, tgt_dec_m, slice_axis=2)
            m_image = compute_metrics_eval(harm_m, orig_tgt_m, slice_axis=2)

            writer.writerow([epoch + 1, sf, subj_id,
                             f"{m_vae['nrmse']:.4f}", f"{m_vae['ssim']:.4f}", f"{m_vae['psnr']:.2f}",
                             f"{m_bridge['nrmse']:.4f}", f"{m_bridge['ssim']:.4f}", f"{m_bridge['psnr']:.2f}",
                             f"{m_image['nrmse']:.4f}", f"{m_image['ssim']:.4f}", f"{m_image['psnr']:.2f}"])

            rows_data.append({
                "orig_src": orig_src, "harm": harm_m, "orig_tgt": orig_tgt_m,
                "tgt_decoded": tgt_dec_m,
                "source_field": sf, "target_field": target_field,
                "subject": subj_id,
                "metrics_vae": m_vae,
                "metrics_decoded": m_bridge, "metrics_orig": m_image,
            })

            print(f"  [{i+1}/{len(ds)}] {src_mod}@{sf}->{src_mod}@{target_field} {subj_id}: "
                  f"vae[SSIM={m_vae['ssim']:.3f}]  bridge[SSIM={m_bridge['ssim']:.3f}]  "
                  f"image[SSIM={m_image['ssim']:.3f}]", flush=True)

        csv_f.close()

        if rows_data:
            vis_path = output_dir / f"vis_epoch{epoch+1:03d}_{src_mod}.png"
            render_evaluation_grid(
                rows_data,
                f"FieldRF ({src_mod} -> {src_mod}@{target_field}) - Epoch {epoch+1}",
                vis_path, epoch,
                dec_label="bridge", orig_label="image",
            )

        print(f"  [vis] {src_mod} done in {time.time()-t0:.0f}s", flush=True)

    bridge.train()
    print("  [vis] All done", flush=True)
