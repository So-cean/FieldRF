"""FieldRF checkpoint loading, EMA restoration and source-channel adaptation.

The caller supplies the model class to keep checkpoint I/O independent of the
network definition. Parameter names and supported checkpoint layouts remain
unchanged when components move between source files.
"""

import torch

from utils.domains import NUM_CLASSES


def infer_resblock_updown(state_dict, prefix=""):
    """Recognize MAISI residual samplers or legacy convolutional samplers."""
    keys = [key for key in state_dict if key.startswith(prefix)]
    return any(
        ".downsampler.norm1." in key or ".upsampler.norm1." in key
        for key in keys
    )


def load_pretrained(
    model_class,
    ckpt_path,
    unet_channels=(64, 128, 256, 512),
    num_res_blocks=(2, 2, 2, 2),
    attention_levels=(False, False, True, True),
    num_head_channels=(0, 0, 32, 32),
    norm_num_groups=32,
    norm_eps=1e-6,
    num_class_embeds=NUM_CLASSES,
    feat4_channels=64,
    conditioning_embedding_num_channels=(64, 64),
    z_channels=4,
    num_train_timesteps=1000,
    scale_factor=1.0,
    use_source_latent=False,
    source_injection=None,
    source_adapter_channels=16,
    use_flash_attention=True,
    device="cpu",
    load_ema=False,
    resblock_updown=None,
    **kwargs,
):
    """Load pretrained UNet (4ch conv_in intact), attach fresh ControlNet.

    ControlNet is randomly initialized (DO NOT copy UNet weights).
    This avoids the oscillation observed in RF when ControlNet was
    initialized with copied UNet weights while the UNet was trainable.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "bridge" in sd:
        inferred_resampling = infer_resblock_updown(sd["bridge"], "unet.")
    else:
        if "state_dict" in sd:
            sd = sd["state_dict"]
        if "unet_state_dict" in sd:
            sd = sd["unet_state_dict"]
        sd = {key[7:] if key.startswith("module.") else key: value
              for key, value in sd.items()}
        inferred_resampling = infer_resblock_updown(sd)
    if resblock_updown is None:
        resblock_updown = inferred_resampling
    elif bool(resblock_updown) != inferred_resampling:
        raise RuntimeError(
            "[RF] UNet resampling architecture differs from checkpoint: "
            f"requested resblock_updown={resblock_updown}, "
            f"checkpoint requires {inferred_resampling}"
        )
    bridge = model_class(
        unet_channels=unet_channels,
        num_res_blocks=num_res_blocks,
        attention_levels=attention_levels,
        num_head_channels=num_head_channels,
        norm_num_groups=norm_num_groups,
        norm_eps=norm_eps,
        num_class_embeds=num_class_embeds,
        feat4_channels=feat4_channels,
        conditioning_embedding_num_channels=conditioning_embedding_num_channels,
        z_channels=z_channels,
        num_train_timesteps=num_train_timesteps,
        scale_factor=scale_factor,
        use_source_latent=use_source_latent,
        source_injection=source_injection,
        source_adapter_channels=source_adapter_channels,
        use_flash_attention=use_flash_attention,
        resblock_updown=resblock_updown,
    )

    # Handle Stage 1/2 training checkpoint format (full bridge state dict)
    if "bridge" in sd:
        raw_state = adapt_source_channels(sd["bridge"], bridge)
        load_bridge_state(bridge, raw_state)
        # Restore scale_factor from checkpoint (was missing — caused
        # vis/inference scripts that use from_pretrained directly to
        # silently use scale_factor=1.0, producing garbage VAE decodes).
        if "scale_factor" in sd:
            bridge.scale_factor = float(sd["scale_factor"])
            print(f"[RF] Restored scale_factor={bridge.scale_factor:.6f} from checkpoint")
        if "num_train_timesteps" in sd:
            bridge.num_train_timesteps = int(sd["num_train_timesteps"])
        if load_ema and "ema" in sd:
            ema_state = adapt_source_channels(sd["ema"], bridge)
            result = bridge.load_state_dict(ema_state, strict=False)
            float_keys = {k for k, v in bridge.state_dict().items()
                          if v.dtype.is_floating_point}
            missing_float = [
                k for k in result.missing_keys
                if k in float_keys and not k.startswith("source_adapter.")
            ]
            if missing_float:
                raise RuntimeError(
                    f"[RF] EMA checkpoint missing {len(missing_float)} float keys "
                    f"(first 5: {missing_float[:5]}). "
                    f"Checkpoint/model architecture mismatch."
                )
            print(f"[RF] Loaded EMA weights for inference/vis")
        else:
            print(f"[RF] Loaded full bridge from training checkpoint")
        bridge = bridge.to(device)
        return bridge

    cleaned = dict(sd)

    # Pretrained has 128 classes; we use 15 -> reinit class_embedding
    cleaned.pop("class_embedding.weight", None)

    # conv_in stays 4ch (pretrained intact)
    missing, unexpected = bridge.unet.load_state_dict(cleaned, strict=False)
    missing_real = [k for k in missing if "class_embedding" not in k]
    if missing_real or unexpected:
        raise RuntimeError(
            "[RF] Pretrained UNet checkpoint mismatch: "
            f"missing={missing_real[:5]}, unexpected={unexpected[:5]}. "
            "All pretrained parameters except the domain embedding must match."
        )

    # ControlNet: random init (no copy from UNet)
    bridge = bridge.to(device)
    print(f"[RF] Bridge ready: in_channels={bridge.in_channels}, "
          f"resblock_updown={bridge.resblock_updown}, "
          f"num_train_timesteps={num_train_timesteps}, "
          f"controlnet=on (feat4 {feat4_channels}ch, "
          f"conditioning_embedding="
          f"{list(bridge.conditioning_embedding_num_channels)}, random init), "
          f"Euler ODE inference")
    return bridge


def load_bridge_state(bridge, state_dict):
    """Load RF state, permitting only a new zero-init source adapter."""
    result = bridge.load_state_dict(state_dict, strict=False)
    missing = [
        k for k in result.missing_keys
        if not k.startswith("source_adapter.")
    ]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "[RF] Bridge checkpoint mismatch: "
            f"missing={missing[:5]}, unexpected={result.unexpected_keys[:5]}"
        )
    if result.missing_keys:
        print("[source-conditioned RF-adapter] Source adapter absent in checkpoint; kept zero initialized")


def adapt_source_channels(state_dict, bridge):
    """Zero-expand a identity RF UNet conv_in state for fair source-conditioned RF continuation."""
    state_dict = dict(state_dict)
    key = "unet.conv_in.conv.weight"
    if key not in state_dict:
        return state_dict
    old_weight = state_dict[key]
    new_weight = bridge.state_dict()[key]
    if old_weight.shape == new_weight.shape:
        return state_dict
    if (old_weight.shape[0] == new_weight.shape[0]
            and old_weight.shape[1] == bridge.z_channels
            and new_weight.shape[1] == bridge.z_channels * 2
            and old_weight.shape[2:] == new_weight.shape[2:]):
        expanded = torch.zeros_like(new_weight, device=old_weight.device)
        expanded[:, :bridge.z_channels].copy_(old_weight)
        state_dict[key] = expanded
        print("[source-conditioned RF] Expanded UNet conv_in 4->8; z_src weights zero-initialized")
        return state_dict
    raise RuntimeError(
        f"Cannot adapt {key} from {tuple(old_weight.shape)} to {tuple(new_weight.shape)}"
    )
