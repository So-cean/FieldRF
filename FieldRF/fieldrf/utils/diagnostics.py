"""Training-log diagnostics for ControlNet and source-latent injection."""


def controlnet_norms(bridge):
    """Compact per-scale zero_conv weight norm (injection-strength proxy).

    Returns a one-line string suitable for epoch logging.
    """
    cn = bridge.controlnet
    parts = []
    for k in range(4):
        idx_start = k * 3
        ns = []
        for j in range(3):
            block = cn.controlnet_down_blocks[idx_start + j]
            w = block.conv.weight if hasattr(block, 'conv') else block.weight
            ns.append(w.norm().item())
        parts.append(f"s{k}=[{ns[0]:.3f},{ns[1]:.3f},{ns[2]:.3f}]")
    mid_w = cn.controlnet_mid_block.conv.weight if hasattr(cn.controlnet_mid_block, 'conv') else cn.controlnet_mid_block.weight
    parts.append(f"mid=[{mid_w.norm().item():.3f}]")
    return " cn=" + " ".join(parts)


def source_channel_norms(bridge):
    """Report source-path norms for concat or adapter shortcut monitoring."""
    if not bridge.use_source_latent:
        return ""
    if bridge.source_injection == "adapter":
        first = bridge.source_adapter.net[0].weight.norm().item()
        zero = bridge.source_adapter.net[-1].weight.norm().item()
        base = bridge.unet.conv_in.conv.weight.norm().item()
        ratio = zero / max(base, 1e-12)
        return (
            f" src_adapter=[first={first:.4f},out={zero:.4f},"
            f"out/base={ratio:.4f}]"
        )
    weight = bridge.unet.conv_in.conv.weight
    base_norm = weight[:, :bridge.z_channels].norm().item()
    src_norm = weight[:, bridge.z_channels:].norm().item()
    ratio = src_norm / max(base_norm, 1e-12)
    return f" src_conv=[base={base_norm:.4f},src={src_norm:.4f},ratio={ratio:.4f}]"
