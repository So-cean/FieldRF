"""FieldRF: NIfTI inference — FieldRFModel, Rectified Flow Euler ODE.

Reads Brain-ID features from b2nd cache (Validating_prospective split).
Decodes full 384³ volumes via centered 64-slice decoding.
Output in Synapse submission sweep format for mrix2026 build.
For Task 3, one source inversion/alignment is reused for all four non-identity
target fields while the same shared RF and endpoint-aligner weights remain
loaded throughout.

Usage:
    python scripts/inference.py \
        --checkpoint runs/pretrain/fieldrf_epoch100.pt \
        --config configs/pretrain.yaml \
        --split Validating_prospective \
        --output_dir predictions_fieldrf

Endpoint-aligned inference additionally accepts a trained RF endpoint adapter:
    python scripts/inference.py \
        --checkpoint runs/pretrain/fieldrf_epoch90.pt \
        --endpoint_checkpoint \
          runs/endpoint_alignment/fieldrf_unpaired_alignment_best.pt \
        --modalities T1W
"""
import sys, argparse, time, gc
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import yaml

# FieldRF imports
_FIELDRF = Path(__file__).resolve().parents[2]
from fieldrf.models.model import FieldRFModel  # noqa: E402
from fieldrf.models.checkpoint import infer_resblock_updown, infer_include_fc
from fieldrf.utils.domains import (  # noqa: E402
    get_class_label,
    MODALITIES,
    FIELDS,
    NUM_CLASSES,
)
from fieldrf.models.endpoint_aligner import EndpointAligner
from fieldrf.data.dataset import _load_feats_b2nd, _load_latent
from fieldrf.utils.visualization import load_vae, decode_volume_centered, find_nifti_path

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged, EnsureTyped,
)
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation


def load_endpoint_aligner(checkpoint, device):
    """Restore the paired EMA aligner, or the selected legacy/best weights."""
    options = checkpoint.get("endpoint_aligner_config", {})
    aligner = EndpointAligner(
        endpoint_channels=options.get("endpoint_channels", 4),
        hidden_channels=options.get("hidden_channels", 24),
        domain_embed_dim=options.get("domain_embed_dim", 8),
        num_domains=options.get("num_domains", NUM_CLASSES),
        num_blocks=options.get("num_blocks", 2),
    )
    key = "endpoint_aligner_ema" if "endpoint_aligner_ema" in checkpoint else "endpoint_aligner"
    aligner.load_state_dict(checkpoint[key], strict=True)
    return aligner.to(device).float().eval(), key


def resolve_inversion(config, steps=None, solver=None):
    """Use the selected checkpoint's paired or unpaired inversion settings."""
    unpaired = config.get("unpaired_alignment", {})
    paired = config.get("inversion", {})
    transport = config.get("inference", {}).get("endpoint_transport", {})
    if steps is None:
        steps = unpaired.get("inversion_steps", paired.get("steps", transport.get("inversion_steps", 4)))
    if solver is None:
        solver = unpaired.get("inversion_solver", paired.get("solver", transport.get("inversion_solver", "euler")))
    if int(steps) < 1 or solver not in {"euler", "heun"}:
        raise ValueError(f"Invalid source inversion: steps={steps}, solver={solver}")
    return int(steps), solver


def map_to_original_space(harm_64, orig_nifti_path, roi_z, z_center):
    """Mask bg with source brain mask at z_center, then place 64 slices
    (centered at z_center) back into original image space.

    Creates a volume with the SAME shape + affine as orig_nifti_path,
    brain region filled with harmonized values, background = 0.
    """
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        SpatialPadd(keys=["image"], spatial_size=384, method="symmetric"),
        CenterSpatialCropd(keys=["image"], roi_size=384),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0),
        EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=True),
    ])
    item = transform({"image": str(orig_nifti_path)})
    vol_01 = torch.clamp((item["image"][0] + 1) / 2, 0, 1).numpy()
    z_start = round((z_center - roi_z // 2) / 4) * 4
    brain_mask = (vol_01[:, :, z_start:z_start + roi_z] > 0).astype(np.float32)
    harm_64 = harm_64 * brain_mask

    orig_img = nib.load(str(orig_nifti_path))
    orig_affine = orig_img.affine
    orig_shape = orig_img.shape[:3]
    orig_ornt = io_orientation(orig_affine)
    ras_ornt = axcodes2ornt(("R", "A", "S"))

    orig_data = orig_img.get_fdata().astype(np.float32)
    orig2ras = ornt_transform(orig_ornt, ras_ornt)
    ras_data = apply_orientation(orig_data, orig2ras)
    ras_shape = ras_data.shape

    pad_before = tuple(max(0, (384 - s) // 2) for s in ras_shape)
    pad_after = tuple(max(0, 384 - s - pb) for pb, s in zip(pad_before, ras_shape))
    padded_shape = tuple(max(s, 384) for s in ras_shape)
    crop_start = tuple((ps - 384) // 2 for ps in padded_shape)

    actual_img_start = round((z_center - roi_z // 2) / 4) * 4
    harm_384 = np.zeros((384, 384, 384), dtype=np.float32)
    harm_384[:, :, actual_img_start:actual_img_start + roi_z] = harm_64

    if any(ps > 384 for ps in padded_shape):
        full = np.zeros(padded_shape, dtype=np.float32)
        slices = tuple(slice(cs, cs + 384) for cs in crop_start)
        full[slices] = harm_384
        after_crop = full
    else:
        after_crop = harm_384

    if any(p > 0 for p in pad_before):
        slices = tuple(slice(pb, pb + rs) for pb, rs in zip(pad_before, ras_shape))
        after_pad = after_crop[slices]
    else:
        after_pad = after_crop

    ras2orig = ornt_transform(ras_ornt, orig_ornt)
    vol_orig = apply_orientation(after_pad, ras2orig)

    assert vol_orig.shape == orig_shape, f"{vol_orig.shape} != {orig_shape}"
    return vol_orig, orig_affine


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="FieldRF NIfTI Inference")
    parser.add_argument("--config", default=str(_FIELDRF / "configs/pretrain.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--vae_checkpoint",
        default=None,
        help=(
            "Optional VAE checkpoint override. This is useful for decoding "
            "with the decoder-only finetuned VAE while preserving the RF "
            "latent space. Defaults to vae.checkpoint in --config."
        ),
    )
    parser.add_argument(
        "--endpoint_checkpoint",
        default=None,
        help=(
            "Optional EndpointAligner checkpoint. When provided, inference "
            "starts from the source VAE latent, inverts it to an endpoint, "
            "aligns that endpoint, and reverses under the target condition."
        ),
    )
    parser.add_argument(
        "--source_endpoint_transport",
        action="store_true",
        help=(
            "Invert the real source VAE latent and reverse from that endpoint "
            "without requiring an external endpoint aligner. This is the "
            "deployment path for paired endpoint/RF fine-tuned checkpoints."
        ),
    )
    parser.add_argument(
        "--inversion_steps",
        type=int,
        default=None,
        help="Source inversion steps; defaults to the endpoint checkpoint config.",
    )
    parser.add_argument(
        "--inversion_solver",
        choices=["euler", "heun"],
        default=None,
        help="Source inversion solver; defaults to the endpoint checkpoint config.",
    )
    parser.add_argument("--split", default="Validating_prospective",
                        choices=["Training_prospective", "Validating_prospective"])
    parser.add_argument("--source_fields", nargs="+", choices=FIELDS, default=None,
                        help="Source fields to process.")
    parser.add_argument("--target_field", default="7T",
                        help="Target field (always 7T for task1).")
    parser.add_argument(
        "--target_fields",
        nargs="+",
        choices=FIELDS,
        default=None,
        help=(
            "Optional target-field subset. Task 3 defaults to all five "
            "fields; other tasks default to --target_field."
        ),
    )
    parser.add_argument("--output_dir", default=None,
                        help="Root output dir for sweep-format predictions.")
    parser.add_argument("--n_infer_steps", type=int, default=20)
    parser.add_argument("--max_volumes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoded_z_center", type=int, default=175,
                        help="RAS Z center for the 64 decoded slices in 384³ space.")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=None,
                        help="Modalities to process (default: configured modalities).")
    parser.add_argument("--method", default="fieldrf",
                        help="Method string for sweep directory.")
    parser.add_argument("--mode", default="stage1_3mod",
                        help="Mode string for sweep directory.")
    parser.add_argument("--epoch_tag", default="epoch100",
                        help="Epoch tag for sweep directory.")
    parser.add_argument(
        "--task",
        choices=["task1", "task2", "task3"],
        default="task1",
        help="Challenge task name for the sweep directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute predictions that already exist.",
    )
    args = parser.parse_args()

    fieldrf_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = fieldrf_root / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = cfg["data"]
    sw_cfg = cfg["vae"]["sliding_window"]
    amp_dtype = getattr(torch, sw_cfg.get("amp_dtype", "float16"))
    n_infer = args.n_infer_steps or cfg.get("inference", {}).get("n_infer_steps", 20)
    scale_factor = float(cfg["vae"]["scale_factor"])
    if args.task == "task3":
        source_fields = list(args.source_fields or FIELDS)
        target_fields = list(args.target_fields or FIELDS)
    else:
        source_fields = args.source_fields or ["0.1T", "1.5T", "3T", "5T"]
        target_fields = list(args.target_fields or [args.target_field])
    cache_root = data_cfg["cache_root"]
    data_root = data_cfg["data_root"]
    modalities = args.modalities or data_cfg.get("modalities", [data_cfg.get("modality", "T1W")])
    feature_cache_name = data_cfg.get("feat_cache_name", "brainid_feat")

    # Output root: sweep-format directory for mrix2026 build
    output_root = Path(args.output_dir or fieldrf_root / "predictions" / "pretrain")
    print(f"Output root: {output_root}")

    # ── VAE ── #
    vae_checkpoint = args.vae_checkpoint or cfg["vae"]["checkpoint"]
    print(f"Loading VAE checkpoint: {vae_checkpoint}")
    vae = load_vae(vae_checkpoint, device)
    print("VAE loaded.")

    # ── Bridge ── #
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", cfg)
    # Fine-tune configs contain the trainable architecture overrides but do
    # not repeat the immutable MAISI initialization path. Merge them over the
    # complete inference config instead of replacing that config wholesale.
    bc = dict(cfg["bridge"])
    bc.update(ckpt_cfg.get("bridge", {}))

    conditioning_channels = bc.get("conditioning_embedding_num_channels")
    if conditioning_channels is None:
        conditioning_channels = bc.get("controlnet", {}).get(
            "conditioning_embedding_num_channels", [64, 64]
        )
    bridge = FieldRFModel(
        unet_channels=tuple(bc.get("unet_channels", [64, 128, 256, 512])),
        num_res_blocks=tuple(bc.get("num_res_blocks", [2, 2, 2, 2])),
        attention_levels=tuple(bc.get("attention_levels", [False, False, True, True])),
        num_head_channels=tuple(bc.get("num_head_channels", [0, 0, 32, 32])),
        norm_num_groups=bc.get("norm_num_groups", 32),
        norm_eps=bc.get("norm_eps", 1e-6),
        num_class_embeds=15,
        feat4_channels=bc.get("feat4_channels", 64),
        conditioning_embedding_num_channels=tuple(conditioning_channels),
        num_train_timesteps=int(
            ckpt.get(
                "num_train_timesteps",
                bc.get("num_train_timesteps", 1000),
            )
        ),
        scale_factor=float(ckpt.get("scale_factor", scale_factor)),
        use_source_latent=bool(bc.get("use_source_latent", False)),
        source_injection=bc.get("source_injection"),
        source_adapter_channels=int(bc.get("source_adapter_channels", 16)),
        use_flash_attention=bc.get("use_flash_attention", True),
        resblock_updown=infer_resblock_updown(ckpt.get("bridge", {}), "unet."),
        include_fc=infer_include_fc(ckpt.get("bridge", {}), "unet."),
    )
    if "bridge" not in ckpt:
        raise RuntimeError(
            "FieldRF inference requires a full training checkpoint with "
            "a 'bridge' state"
        )
    # Load the complete trained bridge directly. Initializing from the MAISI
    # base UNet first is both unnecessary and unsafe here: architecture drift
    # can leave unmatched layers random before a permissive second load.
    if "ema" in ckpt:
        state = ckpt["ema"]
        weight_source = "EMA"
    else:
        state = ckpt["bridge"]
        weight_source = "bridge"
    state = FieldRFModel._adapt_source_channels(state, bridge)
    FieldRFModel._load_bridge_state(bridge, state)
    print(f"Loaded complete {weight_source} bridge state (strict architecture check).")
    print(f"Restored scale_factor={bridge.scale_factor:.6f}")
    del ckpt
    gc.collect()
    bridge.to(device).float().eval()
    print("Bridge ready.")

    # ── Optional endpoint alignment ── #
    endpoint_aligner = None
    inversion_steps = args.inversion_steps
    inversion_solver = args.inversion_solver
    if args.endpoint_checkpoint:
        print(f"Loading endpoint checkpoint: {args.endpoint_checkpoint}")
        endpoint_state = torch.load(
            args.endpoint_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        endpoint_aligner, aligner_weight_source = load_endpoint_aligner(endpoint_state, device)
        endpoint_cfg = endpoint_state.get("config", {})
        inversion_steps, inversion_solver = resolve_inversion(
            endpoint_cfg, inversion_steps, inversion_solver
        )
        print(
            "Endpoint aligner ready: "
            f"weights={aligner_weight_source}, "
            f"epoch={int(endpoint_state.get('epoch', -1)) + 1}, "
            f"inversion={inversion_solver}-{inversion_steps}, "
            f"validation_ssim="
            f"{float(endpoint_state.get('best_validation_ssim', float('nan'))):.6f}"
        )
        del endpoint_state
        gc.collect()

    if args.source_endpoint_transport and endpoint_aligner is None:
        if inversion_steps is None or inversion_solver is None:
            raise ValueError(
                "--source_endpoint_transport requires explicit "
                "--inversion_steps and --inversion_solver"
            )

    # ── Process each modality ── #
    zc = args.decoded_z_center
    roi_z = sw_cfg["roi_size"][2]  # 64

    for modality in modalities:
        print(f"\n{'='*60}")
        print(f"Modality: {modality}")
        print(f"{'='*60}")

        for sf in source_fields:
            active_targets = [
                target for target in target_fields
                if args.task != "task3" or target != sf
            ]
            if not active_targets:
                continue

            print(f"\n--- Source: {sf} -> {active_targets} ---")

            # List subjects from Brain-ID cache
            feat_dir = Path(cache_root) / args.split / feature_cache_name / modality / sf
            b2nd_files = sorted(feat_dir.glob("*.b2nd"))
            if args.max_volumes > 0:
                b2nd_files = b2nd_files[:args.max_volumes]

            if not b2nd_files:
                raise FileNotFoundError(f"No cached subjects found in {feat_dir}")

            print(f"  Found {len(b2nd_files)} subjects.")

            t0 = time.time()
            for idx, bl2_path in enumerate(b2nd_files):
                stem = bl2_path.stem  # e.g. P_T1W_0.1T_0001
                subject_sid = stem.rsplit("_", 1)[-1]
                subj_id = subject_sid

                # Load feats from b2nd cache (list of 5 tensors)
                feats_list = _load_feats_b2nd(str(bl2_path))
                feats = [f.unsqueeze(0).to(device) for f in feats_list]

                output_paths = {}
                for target_field in active_targets:
                    pair_name = f"{sf}_to_{target_field}"
                    sweep_dir = (
                        output_root
                        / f"{args.task}_{pair_name}_{modality}"
                        / args.method
                        / args.mode
                        / args.epoch_tag
                    )
                    output_paths[target_field] = (
                        sweep_dir
                        / f"P_{modality}_{target_field}_{subj_id}.nii.gz"
                    )
                pending_targets = [
                    target for target, path in output_paths.items()
                    if args.overwrite or not path.is_file()
                ]
                if not pending_targets:
                    print(
                        f"  [{idx+1}/{len(b2nd_files)}] {modality} {sf} "
                        "all targets already exist; skipped",
                        flush=True,
                    )
                    continue

                src_nii = find_nifti_path(data_root, args.split, modality, sf, subj_id)
                if src_nii is None:
                    raise FileNotFoundError(
                        f"Source NIfTI reference grid not found: {modality}/{sf}/{subj_id}"
                    )
                transport_endpoint = None
                if args.source_endpoint_transport or endpoint_aligner is not None:
                    # Transport from the real source latent. An optional
                    # endpoint aligner is applied only when one was supplied.
                    latent_path = (
                        Path(cache_root)
                        / args.split
                        / "latent"
                        / modality
                        / sf
                        / f"{stem}.pt"
                    )
                    if not latent_path.is_file():
                        raise FileNotFoundError(
                            f"Source latent cache not found: {latent_path}"
                        )
                    source_z = _load_latent(str(latent_path))
                    if source_z.ndim == 4:
                        source_z = source_z.unsqueeze(0)
                    source_z = source_z.to(device=device, dtype=torch.float32)
                    source_class = get_class_label(modality, sf)
                    source_class_tensor = torch.tensor(
                        [source_class],
                        device=device,
                        dtype=torch.long,
                    )
                    with torch.amp.autocast(
                        "cuda",
                        enabled=device.type == "cuda",
                        dtype=amp_dtype,
                    ):
                        source_endpoint = bridge.invert_to_endpoint(
                            source_z * bridge.scale_factor,
                            feats,
                            source_class_tensor,
                            n_steps=inversion_steps,
                            solver=inversion_solver,
                        )
                    if endpoint_aligner is None:
                        transport_endpoint = source_endpoint
                    else:
                        transport_endpoint, _ = endpoint_aligner(
                            source_endpoint.float(),
                            source_class_tensor,
                        )
                for target_field in pending_targets:
                    target_class_tensor = torch.tensor(
                        [get_class_label(modality, target_field)],
                        device=device,
                        dtype=torch.long,
                    )
                    with torch.amp.autocast(
                        "cuda",
                        enabled=device.type == "cuda",
                        dtype=amp_dtype,
                    ):
                        if transport_endpoint is None:
                            z_harm = bridge.sample_x0(
                                feats,
                                target_class_tensor,
                                n_infer_steps=n_infer,
                                device=device,
                            )
                        else:
                            z_harm = bridge.sample_x0_from_endpoint(
                                transport_endpoint,
                                feats,
                                target_class_tensor,
                                n_infer_steps=n_infer,
                            )

                    harm_img = decode_volume_centered(
                        vae, z_harm, sw_cfg, amp_dtype, zc
                    )
                    harm_np = torch.clamp(
                        (harm_img + 1) / 2, 0, 1
                    )[0, 0].float().cpu().numpy()

                    harm_orig, out_affine = map_to_original_space(
                        harm_np, src_nii, roi_z, zc
                    )

                    out_path = output_paths[target_field]
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(
                        nib.Nifti1Image(
                            harm_orig.astype(np.float32), out_affine
                        ),
                        out_path,
                    )
                    print(
                        f"  [{idx+1}/{len(b2nd_files)}] {modality} "
                        f"{sf}->{target_field} {stem}: saved",
                        flush=True,
                    )

            dt = time.time() - t0
            print(f"  Done {sf} in {dt:.0f}s ({dt/len(b2nd_files):.0f}s/subject)")

    print(f"\n{'='*60}")
    print(f"All done. Output: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
