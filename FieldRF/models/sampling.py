"""FieldRF Euler/Heun inversion, reverse sampling and differentiable grafts.

Callers provide a velocity model. These functions own numerical integration
and gradient truncation; they do not construct networks or load checkpoints.
"""

from contextlib import nullcontext

import numpy as np
import torch

from .rectified_flow import timestep_transform


def invert_to_endpoint(
    model,
    z_0,
    feats,
    class_label,
    n_steps=4,
    solver="euler",
    velocity_model=None,
    gradient_steps=None,
):
    """Integrate a scaled image latent toward its learned RF endpoint.

    This method intentionally remains differentiable. Callers that only
    need a detached source endpoint should invoke it under
    ``torch.no_grad()``. Short differentiable integrations can be used for
    an optional explicit endpoint-consistency loss.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if solver not in {"euler", "heun"}:
        raise ValueError(f"Unsupported solver: {solver}")
    if gradient_steps is None:
        gradient_steps = n_steps
    gradient_steps = int(gradient_steps)
    if not 0 <= gradient_steps <= n_steps:
        raise ValueError(
            "gradient_steps must lie in [0, n_steps], got "
            f"{gradient_steps} for n_steps={n_steps}"
        )
    predict_velocity = model.forward if velocity_model is None else velocity_model

    batch = z_0.shape[0]
    if isinstance(class_label, int):
        class_label = torch.full(
            (batch,),
            class_label,
            device=z_0.device,
            dtype=torch.long,
        )
    else:
        class_label = class_label.to(device=z_0.device, dtype=torch.long)

    raw_schedule = torch.linspace(
        0,
        model.num_train_timesteps - 1,
        n_steps + 1,
        device=z_0.device,
    )
    schedule = timestep_transform(
        raw_schedule,
        int(np.prod(z_0.shape[2:])),
        base_img_size_numel=32768,
        scale=1.4,
        num_train_timesteps=model.num_train_timesteps,
        spatial_dim=3,
    )

    state = z_0
    for index in range(n_steps):
        current_t = schedule[index].expand(batch)
        next_t = schedule[index + 1].expand(batch)
        delta = (
            schedule[index + 1] - schedule[index]
        ) / model.num_train_timesteps
        # Full deployment uses several ODE steps, but retaining every
        # 96^3 activation is prohibitively expensive.  ``gradient_steps``
        # implements truncated BPTT: the detached prefix follows the
        # actual online vector field and the final steps remain
        # differentiable.  The default preserves the historical fully
        # differentiable behavior.
        keep_gradient = index >= n_steps - gradient_steps
        context = nullcontext() if keep_gradient else torch.no_grad()
        with context:
            velocity = predict_velocity(
                state,
                current_t,
                class_label,
                feats,
            )
            prediction = state - velocity * delta
            if solver == "euler":
                state = prediction
            else:
                next_velocity = predict_velocity(
                    prediction,
                    next_t,
                    class_label,
                    feats,
                )
                state = state - 0.5 * (velocity + next_velocity) * delta
        if not keep_gradient:
            state = state.detach()
    return state


@torch.no_grad()
def sample_x0(model, feats, class_label, n_infer_steps=20, device=None, z_src=None):
    """Euler ODE sampling: deterministic reverse from noise to x0.

    Args:
        feats: list of 5 Brain-ID features.
        class_label: (B,) or int target class.
        n_infer_steps: number of Euler steps (default 20).
        z_src: optional UNSCALED source latent for source-conditioned RF conditioning.
    Returns:
        x0_pred: (B, 4, 96³) predicted target latent (UNSCALED, for VAE decode).
    """
    if device is None:
        device = next(model.parameters()).device

    B = feats[0].shape[0]
    if isinstance(class_label, int):
        class_label = torch.full((B,), class_label, device=device, dtype=torch.long)
    else:
        class_label = class_label.to(device)

    feat4 = feats[-1].float().to(device)
    z_src_scaled = None
    if z_src is not None:
        z_src_scaled = z_src.to(device).float() * model.scale_factor

    # Build inference timesteps: uniform from T-1 to 0
    # (T-1, not T, because training samples t ∈ [0, T) — querying at t=T
    # would be out-of-distribution. The model has seen t up to ~T-1.)
    N = n_infer_steps
    timesteps = torch.linspace(
        model.num_train_timesteps - 1, 0, N + 1, device=device
    )  # [T-1, ..., 0]

    # Apply timestep_transform to inference timesteps too (matching training).
    # CRITICAL: use LATENT spatial numel (96³), NOT feat4's 192³ — the transform
    # is a noise-schedule correction for the generation resolution, and the
    # generation happens at the latent's 96³, not feat4's 192³. Using 192³ here
    # would give ratio=8.4 vs training's 4.2, completely mismatching the
    # t-distribution the model was trained on.
    latent_shape = tuple(
        (length + model.conditioning_downsample_factor - 1)
        // model.conditioning_downsample_factor
        for length in feat4.shape[-3:]
    )
    input_img_size_numel = int(np.prod(latent_shape))
    timesteps = timestep_transform(
        timesteps, input_img_size_numel,
        base_img_size_numel=32768,
        scale=1.4,
        num_train_timesteps=model.num_train_timesteps,
        spatial_dim=3,
    )

    # Initial noise: sample from standard normal (t=T → pure noise)
    # At t=T: z_T ≈ noise (since timepoints ≈ 0)
    z_t = torch.randn(B, model.z_channels, *latent_shape, device=device)

    # Euler ODE: z_{t-dt} = z_t + v * dt
    for i in range(N):
        t_val = timesteps[i]  # scalar on device
        t_batch = t_val.expand(B)

        # Predict velocity
        v_pred = model.forward(
            z_t, t_batch, class_label, feats, z_src=z_src_scaled
        )

        # dt: difference to next timestep (in normalized units)
        dt = (timesteps[i] - timesteps[i + 1]) / model.num_train_timesteps

        # Euler step: z_{t-dt} = z_t + v * dt
        z_t = z_t + v_pred * dt

    # Final z_t ≈ z_0_pred (unscaled for VAE decode)
    return z_t / model.scale_factor


@torch.no_grad()
def sample_x0_from_endpoint(
    model,
    endpoint,
    feats,
    class_label,
    n_infer_steps=20,
):
    """Reverse the RF from a supplied scaled inversion endpoint.

    Unlike :meth:`sample_x0`, this method does not draw random noise.  It
    is used to compare an actual source inversion endpoint with its
    aligned counterpart under identical Brain-ID and target-label
    conditions.
    """
    if endpoint.ndim != 5:
        raise ValueError(
            "endpoint must have shape [B,C,H,W,D], got "
            f"{tuple(endpoint.shape)}"
        )
    if n_infer_steps < 1:
        raise ValueError(
            f"n_infer_steps must be >= 1, got {n_infer_steps}"
        )
    device = endpoint.device
    batch = endpoint.shape[0]
    if isinstance(class_label, int):
        class_label = torch.full(
            (batch,),
            class_label,
            device=device,
            dtype=torch.long,
        )
    else:
        class_label = class_label.to(device=device, dtype=torch.long)

    timesteps = torch.linspace(
        model.num_train_timesteps - 1,
        0,
        n_infer_steps + 1,
        device=device,
    )
    timesteps = timestep_transform(
        timesteps,
        int(np.prod(endpoint.shape[2:])),
        base_img_size_numel=32768,
        scale=1.4,
        num_train_timesteps=model.num_train_timesteps,
        spatial_dim=3,
    )
    state = endpoint
    for index in range(n_infer_steps):
        current_t = timesteps[index].expand(batch)
        velocity = model.forward(
            state,
            current_t,
            class_label,
            feats,
        )
        delta = (
            timesteps[index] - timesteps[index + 1]
        ) / model.num_train_timesteps
        state = state + velocity * delta
    return state / model.scale_factor


def generate_train(model, feats, class_label, z_src, source_dropout_p=0.1,
                   t=None, n_infer_steps=1, return_diagnostics=False):
    """Differentiable Stage-2 graft sample in scaled latent space.

    Legacy ``concat`` mode retains the random-t one-forward estimator for
    reproducibility. ``adapter`` mode mirrors deployment: it starts at pure
    noise and performs differentiable Euler steps, while z_src enters only
    through the adapter. This removes the old dual use of z_src as both RF
    starting point and direct condition.
    """
    B = feats[0].shape[0]
    device = feats[0].device
    z_src_scaled = z_src.to(device).float() * model.scale_factor

    z_src_cond = z_src_scaled
    keep = torch.ones((B,), device=device, dtype=z_src_scaled.dtype)
    if source_dropout_p > 0:
        keep = (torch.rand((B,), device=device) >= source_dropout_p).to(z_src_scaled.dtype)
        z_src_cond = z_src_cond * keep.view(-1, 1, 1, 1, 1)

    if model.source_injection == "adapter":
        if n_infer_steps < 1:
            raise ValueError(f"n_infer_steps must be >=1, got {n_infer_steps}")
        spatial_numel = int(np.prod(z_src_scaled.shape[2:]))
        timesteps = torch.linspace(
            model.num_train_timesteps - 1, 0, n_infer_steps + 1, device=device
        )
        timesteps = timestep_transform(
            timesteps, spatial_numel,
            base_img_size_numel=32768,
            scale=1.4,
            num_train_timesteps=model.num_train_timesteps,
            spatial_dim=3,
        )
        z_t = torch.randn_like(z_src_scaled)
        for i in range(n_infer_steps):
            t_batch = timesteps[i].expand(B)
            v_pred = model.forward(
                z_t, t_batch, class_label, feats, z_src=z_src_cond
            )
            dt = (timesteps[i] - timesteps[i + 1]) / model.num_train_timesteps
            z_t = z_t + v_pred * dt
        diagnostics = {
            "source_drop_fraction": (1.0 - keep).mean().detach(),
            "graft_start_source_fraction": z_t.new_zeros(()),
            "graft_steps": z_t.new_tensor(float(n_infer_steps)),
        }
        return (z_t, diagnostics) if return_diagnostics else z_t

    if t is None:
        t = torch.rand((B,), device=device) * model.num_train_timesteps
        spatial_numel = int(np.prod(z_src_scaled.shape[2:]))
        t = timestep_transform(
            t, spatial_numel,
            base_img_size_numel=32768,
            scale=1.4,
            num_train_timesteps=model.num_train_timesteps,
            spatial_dim=3,
        )
    else:
        t = t.to(device=device, dtype=torch.float)

    alpha = 1.0 - t / model.num_train_timesteps
    alpha_view = alpha.view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(z_src_scaled)
    z_t = alpha_view * z_src_scaled + (1.0 - alpha_view) * noise

    v_pred = model.forward(z_t, t, class_label, feats, z_src=z_src_cond)
    t_view = (t / model.num_train_timesteps).view(-1, 1, 1, 1, 1)
    x0_hat = z_t + t_view * v_pred
    diagnostics = {
        "source_drop_fraction": (1.0 - keep).mean().detach(),
        "graft_start_source_fraction": alpha.mean().detach(),
        "graft_steps": x0_hat.new_ones(()),
    }
    return (x0_hat, diagnostics) if return_diagnostics else x0_hat
