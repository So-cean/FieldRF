"""Rectified-flow, spatial supervision and paired latent training objectives.

Network prediction and RF path construction live in ``models.model``.
This module computes losses without owning model parameters or checkpoints.
"""

import torch
import torch.nn.functional as F


def spatial_axis_overlap_weights(
    axis_length,
    image_slice_range,
    image_axis_size,
    *,
    device=None,
    dtype=torch.float32,
):
    """Return exact image-to-latent overlap weights for one spatial axis.

    A latent cell represents a continuous interval in the preprocessed image.
    Boundary cells therefore receive fractional weights when an image-space
    evaluation slab is not divisible by the VAE downsampling factor.  This is
    preferable to rounding the 30-slice challenge slab to either 28 or 32
    supervised image slices.
    """
    axis_length = int(axis_length)
    image_axis_size = int(image_axis_size)
    if axis_length < 1 or image_axis_size < 1:
        raise ValueError("axis sizes must be positive")
    if len(image_slice_range) != 2:
        raise ValueError("image_slice_range must contain [start, end]")
    start, end = (float(value) for value in image_slice_range)
    if not 0.0 <= start < end <= float(image_axis_size):
        raise ValueError(
            "image_slice_range must lie inside the preprocessed image axis, "
            f"got [{start}, {end}) for size {image_axis_size}"
        )

    cell_width = float(image_axis_size) / axis_length
    cell_start = torch.arange(
        axis_length, device=device, dtype=torch.float32
    ) * cell_width
    cell_end = cell_start + cell_width
    overlap = (
        torch.minimum(cell_end, torch.tensor(end, device=device))
        - torch.maximum(cell_start, torch.tensor(start, device=device))
    ).clamp_min(0.0)
    weights = overlap / cell_width
    if not bool(weights.gt(0).any().item()):
        raise ValueError("image_slice_range does not overlap the latent axis")
    return weights.to(dtype=dtype)


def _weighted_axis_l1(prediction, target, weights, axis=-1):
    """L1 mean over one softly selected spatial slab."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} != target shape {target.shape}"
        )
    axis = int(axis) % prediction.ndim
    weights = torch.as_tensor(
        weights, device=prediction.device, dtype=torch.float32
    )
    if weights.ndim != 1 or weights.numel() != prediction.shape[axis]:
        raise ValueError(
            "spatial weights must be one-dimensional and match the selected "
            f"axis ({prediction.shape[axis]}), got {tuple(weights.shape)}"
        )
    if not bool(weights.ge(0).all().item()) or weights.sum().item() <= 0:
        raise ValueError("spatial weights must be non-negative with positive sum")

    view_shape = [1] * prediction.ndim
    view_shape[axis] = weights.numel()
    error = (prediction - target).abs().float()
    weighted_error = error * weights.view(view_shape)
    values_per_axis_cell = error.numel() / error.shape[axis]
    return weighted_error.sum() / (weights.sum() * values_per_axis_cell)


def rectified_flow_loss(
    bridge,
    z_0,
    feats,
    class_label,
    scale_factor=1.0,
    t=None,
    z_src=None,
    endpoint=None,
    endpoint_focus_probability=0.0,
    endpoint_min_fraction=0.7,
    spatial_axis_weights=None,
    spatial_axis=-1,
):
    """Rectified Flow loss: L1(v_pred, v_target).

    Args:
        bridge: FieldRFModel.
        z_0: (B, 4, 96³) clean target latent (unscaled).
        feats: list of 5 Brain-ID features.
        class_label: (B,) target class (0-14).
        scale_factor: latent scaling factor.
        t: optional pre-sampled timesteps (None = sample with transform).
        z_src: optional UNSCALED source condition. The standard Stage-1 anchor
            keeps this None; adapter Stage 2 also uses an active-source anchor.
        endpoint: optional externally supplied *scaled* RF endpoint.

        spatial_axis_weights: optional soft selection over one latent spatial
            axis. ``None`` preserves the historical full-volume objective.
        spatial_axis: tensor axis matched by ``spatial_axis_weights``.

    Returns:
        loss: L1 loss (scalar).
    """
    z_0_scaled = z_0 * scale_factor
    z_src_scaled = None if z_src is None else z_src * scale_factor
    prediction, target = bridge.forward_rf(
        z_0_scaled,
        feats,
        class_label,
        t=t,
        z_src=z_src_scaled,
        endpoint=endpoint,
        endpoint_focus_probability=endpoint_focus_probability,
        endpoint_min_fraction=endpoint_min_fraction,
    )
    if spatial_axis_weights is None:
        return F.l1_loss(prediction, target)
    return _weighted_axis_l1(
        prediction,
        target,
        spatial_axis_weights,
        axis=spatial_axis,
    )


def terminal_latent_loss(prediction, target, edge_weight):
    """Direct paired regression in the VAE latent plus spatial gradients."""
    prediction = prediction.float()
    target = target.float()
    l1 = F.l1_loss(prediction, target)
    edges = []
    for dimension in (-3, -2, -1):
        predicted_edge = torch.diff(prediction, dim=dimension)
        target_edge = torch.diff(target, dim=dimension)
        normalizer = target_edge.detach().abs().mean().clamp_min(0.05)
        edges.append(F.l1_loss(predicted_edge, target_edge) / normalizer)
    edge = torch.stack(edges).mean()
    return l1 + float(edge_weight) * edge, l1, edge
