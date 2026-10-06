"""[FieldRF] Cache MAISI VAE latents — direct full-volume encode.

384³ input → VAE.encode (fully conv, handles any spatial size) → (4,96,96,96) latent.
No sliding window needed. Encoder is compressive (384³→96³). Peak ~28GB on A100.

Output: {cache_root}/{split}/latent/{modality}/{field}/{filename}.pt

Usage:
    python scripts/cache_latents.py --config configs/pretrain.yaml
    python scripts/cache_latents.py --config configs/pretrain.yaml --split Training_prospective
"""
import sys, argparse, time
from pathlib import Path

import torch
import yaml
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
_FIELDRF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FIELDRF))
from data.volume_dataset import VolumeDataset
from models.checkpoint import load_vae_checkpoint


def _save_latent(payload, path):
    """Publish a cache only after its complete write; retain any previous file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_maisi_vae(checkpoint, device):
    vae = AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
        num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
        norm_num_groups=32, norm_eps=1e-6,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        use_checkpointing=False, use_convtranspose=False,
        norm_float16=True, num_splits=4, dim_split=1,
    ).to(device)

    load_vae_checkpoint(vae, checkpoint, map_location=device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vae_checkpoint", default=None)
    parser.add_argument("--split", default=None, help="Override data.split (e.g. Training_prospective)")
    parser.add_argument("--modality", default=None, help="Cache one modality instead of all configured modalities")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing latent caches")
    args = parser.parse_args()

    # Resolve config path relative to FieldRF/ dir
    fieldrf_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = fieldrf_root / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    split = args.split or data_cfg["split"]
    vae_cfg = cfg["vae"]
    amp_dtype = getattr(torch, vae_cfg.get("sliding_window", {}).get("amp_dtype", "float16"))

    # ── VAE ──
    checkpoint_path = args.vae_checkpoint or vae_cfg["checkpoint"]
    ae = load_maisi_vae(checkpoint_path, device)
    latent_ch = vae_cfg["latent_channels"]
    print(f"MAISI VAE ({latent_ch}ch), direct encode: 384³ → ({latent_ch},96,96,96)")

    modalities = [args.modality] if args.modality else data_cfg.get("modalities", [data_cfg.get("modality", "T1W")])
    for modality in modalities:
        # ── Cache per field ──
        latent_root = Path(data_cfg["cache_root"]) / split / "latent" / modality
        fields = data_cfg.get("fields", [data_cfg.get("source_field"), data_cfg.get("target_field")])

        for field_val in fields:
            field_dir = latent_root / str(field_val)
            field_dir.mkdir(parents=True, exist_ok=True)

            ds = VolumeDataset(data_cfg["data_root"], split,
                                 modality, str(field_val))
            from torch.utils.data import DataLoader
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
            print(f"\n[{field_val}] {len(ds)} vols → {field_dir}")

            saved, t0 = 0, time.time()
            for batch in loader:
                img = batch["image"].to(device)
                fn = batch["filename"][0]
                pt_path = field_dir / f"{fn.replace('.nii.gz', '')}.pt"
                if pt_path.exists() and pt_path.stat().st_size > 0 and not args.overwrite:
                    saved += 1
                    continue

                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                        z_out = ae.encode(img)
                        z = z_out[0] if isinstance(z_out, tuple) else z_out

                _save_latent({"latent": z[0].detach().cpu().float(),
                            "field_strength": batch["field_strength"][0],
                            "modality": batch["modality"][0]}, pt_path)
                saved += 1
                if saved % 10 == 0:
                    print(f"  {saved}/{len(ds)} ({(time.time()-t0):.0f}s)")

        print(f"Done. Latents saved to {latent_root}")

        # ── Compute scale_factor ──
        all_pts = sorted(latent_root.rglob("*.pt"))
        if all_pts:
            sm = sv = sn = 0.0
            for f in all_pts:
                d = torch.load(f, map_location="cpu", weights_only=False)
                z = d["latent"].float().flatten()
                m, v, n = z.mean(), z.var(unbiased=False), z.numel()
                delta = m - sm
                sv += v * n + delta * delta * sn * n / (sn + n)
                sm = (sm * sn + m * n) / (sn + n)
                sn += n
            scale = 1.0 / (sv / sn).sqrt().clamp(min=1e-8)
            import json
            with open(latent_root / "meta.json", "w") as f:
                json.dump({"scale_factor": scale.item(), "n_voxels": int(sn), "n_files": len(all_pts)}, f)
            print(f"  scale_factor (manual ref): {scale.item():.4f}  → {latent_root}/meta.json")


if __name__ == "__main__":
    main()
