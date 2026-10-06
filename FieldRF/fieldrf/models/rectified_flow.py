"""MAISI timestep remapping and rectified-flow training paths.

These functions use the model's velocity predictor and timestep count without
owning network parameters, losses, sampling solvers or checkpoint state.
"""

import torch


def timestep_transform(t, input_img_size_numel, base_img_size_numel=32768,
                       scale=1.0, num_train_timesteps=1000, spatial_dim=3):
    """MAISI timestep remap: resolution-driven sigmoid-like correction.

    Args:
        t: (B,) float timesteps in [0, num_train_timesteps).
        input_img_size_numel: total voxels in the latent (e.g. 96*96*96=884736).
        base_img_size_numel: reference voxel count (32*32*32=32768).
        scale: scaling factor for the remap (MAISI default 1.4).
        num_train_timesteps: max timestep (1000).
        spatial_dim: 3 for 3D.
    Returns:
        new_t: (B,) transformed timesteps, same range as input.
    """
    t = t / num_train_timesteps  # normalize to [0, 1]
    ratio_space = (input_img_size_numel / base_img_size_numel) ** (1.0 / spatial_dim)
    ratio = ratio_space * scale
    # Remap: new_t = ratio * t / (1 + (ratio - 1) * t)
    new_t = ratio * t / (1 + (ratio - 1) * t)
    new_t = new_t * num_train_timesteps
    return new_t


def sample_train_timesteps(
    model,
    z_0,
    endpoint_focus_probability=0.0,
    endpoint_min_fraction=0.7,
):
    """Sample transformed RF timesteps used by ``forward_rf``.

    ``endpoint_focus_probability`` optionally replaces part of the
    uniform raw-timestep batch with samples close to the noise endpoint.
    The returned values have already passed through MAISI's timestep
    transform and can be shared by paired source/target branches.
    """
    if not 0.0 <= endpoint_focus_probability <= 1.0:
        raise ValueError(
            "endpoint_focus_probability must be in [0, 1], got "
            f"{endpoint_focus_probability}"
        )
    if not 0.0 <= endpoint_min_fraction < 1.0:
        raise ValueError(
            "endpoint_min_fraction must be in [0, 1), got "
            f"{endpoint_min_fraction}"
        )

    batch = z_0.shape[0]
    raw_t = torch.rand((batch,), device=z_0.device)
    if endpoint_focus_probability > 0:
        use_endpoint = (
            torch.rand((batch,), device=z_0.device)
            < endpoint_focus_probability
        )
        endpoint_t = endpoint_min_fraction + (
            1.0 - endpoint_min_fraction
        ) * torch.rand((batch,), device=z_0.device)
        raw_t = torch.where(use_endpoint, endpoint_t, raw_t)
    raw_t = raw_t * model.num_train_timesteps

    input_img_size_numel = z_0.shape[2] * z_0.shape[3] * z_0.shape[4]
    return timestep_transform(
        raw_t,
        input_img_size_numel,
        base_img_size_numel=32768,
        scale=1.4,
        num_train_timesteps=model.num_train_timesteps,
        spatial_dim=3,
    )


def build_rf_path(model, z_0, endpoint, t):
    """Build a linear RF state and its data-minus-endpoint velocity.

    Args:
        z_0: Clean, scaled image latent.
        endpoint: Noise or another target-compatible RF endpoint.
        t: MAISI-transformed timestep returned by
            :meth:`sample_train_timesteps`.
    """
    if endpoint.shape != z_0.shape:
        raise ValueError(
            f"endpoint shape {endpoint.shape} != z_0 shape {z_0.shape}"
        )
    if t.ndim != 1 or t.shape[0] != z_0.shape[0]:
        raise ValueError(
            f"t must have shape ({z_0.shape[0]},), got {tuple(t.shape)}"
        )
    endpoint = endpoint.to(device=z_0.device, dtype=z_0.dtype)
    timepoints = 1.0 - t / model.num_train_timesteps
    tp = timepoints.view(-1, 1, 1, 1, 1)
    z_t = tp * z_0 + (1.0 - tp) * endpoint
    v_target = z_0 - endpoint
    return z_t, v_target


def forward_rf(
    model,
    z_0,
    feats,
    class_label,
    t=None,
    z_src=None,
    endpoint=None,
    endpoint_focus_probability=0.0,
    endpoint_min_fraction=0.7,
):
    """Construct the RF path and return predicted and target velocities.

    If t is None, samples uniformly with timestep_transform.

    Args:
        z_0: (B, 4, 96³) clean latent (scaled).
        feats: list of 5 Brain-ID features.
        class_label: (B,) target class.
        t: optional pre-sampled, already transformed timesteps.
        z_src: optional scaled source latent. Identity supervision leaves
            this unset so the new source-conditioned RF channels remain zero.
        endpoint: optional externally supplied RF endpoint. Passing the
            same deterministic endpoint to paired source/target branches
            establishes an explicit sample-wise coupling.
    Returns:
        v_pred: predicted velocity.
        v_target: ground-truth velocity.
    """
    if t is None:
        t = model.sample_train_timesteps(
            z_0,
            endpoint_focus_probability=endpoint_focus_probability,
            endpoint_min_fraction=endpoint_min_fraction,
        )

    if endpoint is None:
        endpoint = torch.randn_like(z_0)
    z_t, v_target = model.build_rf_path(z_0, endpoint, t)

    # Predict velocity
    v_pred = model.forward(z_t, t, class_label, feats, z_src=z_src)

    return v_pred, v_target
