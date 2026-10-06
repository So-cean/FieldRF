"""Reference-grid validation and rendering for decoder adaptation."""

from pathlib import Path
import numpy as np
import torch
from fieldrf.utils.visualization import (
    load_orig_ras_384, decode_volume_centered, eval_window_in_384,
    find_nifti_path, compute_metrics_eval, init_metrics_csv, render_vis_grid_vae,
)


@torch.no_grad()
def validate_vae(vae, cfg, epoch, device, output_dir):
    """RF-aligned validation: decode full latent, compare with 30-slice eval window.

    Iterates over ALL fields (from data.fields) × modalities, 1 subject each.
    Uses find_nifti_path, eval_window_in_384, render_vis_grid_vae (2-column).

    Outputs per epoch (per modality):
      vae_val_epoch{N}_{mod}.png   — 2-column grid (Orig | Decoded)
      vae_val_metrics_{mod}.csv    — epoch, field, subject, nrmse_vae, ssim_vae, psnr_vae
    """
    vis_cfg = cfg.get("vis", {})
    data_cfg = cfg["data"]
    vae_cfg = cfg["vae"]

    eval_range = tuple(vis_cfg.get("eval_slice_range", [150, 180]))
    vis_split = vis_cfg.get("split", "Validating_prospective")
    cache_root = data_cfg["cache_root"]
    fields = data_cfg.get("fields", ["0.1T", "1.5T", "3T", "5T", "7T"])
    modalities = data_cfg.get("modalities", ["T1W"])
    n_samples = vis_cfg.get("n_samples", 1)  # subjects per (modality, field)

    sw_cfg = vae_cfg.get("sliding_window", {"roi_size": [128, 128, 64]})
    amp_dtype = getattr(torch, vae_cfg.get("sliding_window", {}).get("amp_dtype", "float16"))

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    vae.eval()

    mod_rows = {}  # mod -> list of rows_data entries (for PNG grid: first subject only)
    all_ssim_list = []
    all_n = 0

    for mod in modalities:
        csv_path = vis_dir / f"vae_val_metrics_{mod}.csv"
        writer, csv_f = init_metrics_csv(
            csv_path,
            ["epoch", "field", "subject", "nrmse_vae", "ssim_vae", "psnr_vae"],
        )

        rows_data = []  # first subject per field (for PNG)
        for field in fields:
            latent_dir = Path(cache_root) / vis_split / "latent" / mod / field
            if not latent_dir.is_dir():
                continue
            pt_files = sorted(latent_dir.glob("*.pt"))
            if not pt_files:
                continue
            pt_files = pt_files[:n_samples]

            field_ssims = []
            for pi, pt in enumerate(pt_files):
                subj_id = pt.stem.rsplit("_", 1)[-1]
                nii_path = find_nifti_path(data_cfg["data_root"], vis_split, mod, field, subj_id)
                if nii_path is None:
                    continue

                z = torch.load(str(pt), map_location="cpu", weights_only=False)["latent"]
                orig_384, _, pad_before_z, crop_start_z = load_orig_ras_384(nii_path)

                z0, z1 = eval_window_in_384(pad_before_z, crop_start_z, eval_range)
                z_center = (z0 + z1) // 2

                recon_img = decode_volume_centered(
                    vae, z.unsqueeze(0).to(device), sw_cfg, amp_dtype, z_center
                )
                recon_np = torch.clamp((recon_img + 1) / 2, 0, 1)[0, 0].float().cpu().numpy()

                win_start = round((z_center - 32) / 4) * 4
                c0 = max(0, z0 - win_start)
                c1 = min(recon_np.shape[2], z1 - win_start)
                recon_slab = recon_np[:, :, c0:c1]
                orig_slab = orig_384[:, :, z0:z1]

                brain_mask = (orig_slab > 0).astype(np.float32)
                recon_m = recon_slab * brain_mask
                orig_m = orig_slab * brain_mask

                m = compute_metrics_eval(recon_m, orig_m, slice_axis=2)

                writer.writerow([epoch + 1, field, subj_id,
                                 f"{m['nrmse']:.4f}", f"{m['ssim']:.4f}", f"{m['psnr']:.2f}"])

                field_ssims.append(m['ssim'])
                all_ssim_list.append(m['ssim'])
                all_n += 1

                # First subject only → PNG grid
                if pi == 0:
                    rows_data.append({
                        "orig": orig_m, "decoded": recon_m,
                        "field": field, "subject": subj_id,
                        "metrics": m,
                    })

                print(f"    [{mod}@{field}/{subj_id}] ssim={m['ssim']:.4f} "
                      f"psnr={m['psnr']:.2f} nrmse={m['nrmse']:.4f}", flush=True)

            if field_ssims:
                avg_f = float(np.mean(field_ssims))
                print(f"    → {mod}@{field} avg_ssim={avg_f:.4f} ({len(field_ssims)} subjects)", flush=True)

        mod_rows[mod] = rows_data
        csv_f.close()

        if rows_data:
            vis_path = vis_dir / f"vae_val_epoch{epoch+1:03d}_{mod}.png"
            render_vis_grid_vae(
                rows_data,
                f"VAE Decoder Validation — {mod} — Epoch {epoch+1}",
                vis_path, epoch,
            )

    vae.train()

    if all_n > 0:
        avg_all = float(np.mean(all_ssim_list))
        print(f"  ALL: avg_ssim={avg_all:.4f} ({all_n} samples)", flush=True)
        return {"ssim": avg_all, "n": all_n}
    return {"ssim": 0, "n": 0}

