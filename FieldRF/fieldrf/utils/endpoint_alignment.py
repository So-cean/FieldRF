"""Shared losses and schedules for native RF endpoint alignment.

The endpoint is a dense 3D tensor, so every spatial patch supplies a sample
for distribution matching even when the volume batch size is one.  These
utilities deliberately contain no model code: callers decide whether the
reference endpoint is produced by an online, frozen, or EMA velocity field.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def patch_descriptors(endpoint: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Return patch mean/std descriptors as a ``[patches, 2*C]`` matrix."""
    spatial = endpoint.shape[2:]
    patch_size = int(patch_size)
    if patch_size < 1 or any(size % patch_size for size in spatial):
        raise ValueError(
            f"patch_size={patch_size} must divide endpoint shape {spatial}"
        )
    mean = F.avg_pool3d(endpoint, patch_size, stride=patch_size)
    second = F.avg_pool3d(
        endpoint.float().square(), patch_size, stride=patch_size
    )
    std = (second - mean.float().square()).clamp_min(1e-6).sqrt()
    descriptors = torch.cat([mean.float(), std], dim=1)
    return descriptors.permute(0, 2, 3, 4, 1).reshape(
        -1, descriptors.shape[1]
    )


def coral_loss(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Match descriptor means and covariances without paired observations."""
    if source.ndim != 2 or reference.ndim != 2:
        raise ValueError("CORAL inputs must be [N,D] matrices")
    if source.shape[1] != reference.shape[1]:
        raise ValueError("CORAL feature dimensions must match")
    source_centered = source - source.mean(dim=0, keepdim=True)
    reference_centered = reference - reference.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / max(
        1, source.shape[0] - 1
    )
    reference_cov = reference_centered.T @ reference_centered / max(
        1, reference.shape[0] - 1
    )
    mean_loss = F.mse_loss(source.mean(dim=0), reference.mean(dim=0))
    covariance_loss = F.mse_loss(source_cov, reference_cov)
    return mean_loss + covariance_loss


def multiscale_patch_coral(
    source: torch.Tensor,
    reference: torch.Tensor,
    patch_sizes,
) -> torch.Tensor:
    """Average patch-distribution CORAL across physical context scales."""
    losses = [
        coral_loss(
            patch_descriptors(source, int(size)),
            patch_descriptors(reference, int(size)),
        )
        for size in patch_sizes
    ]
    if not losses:
        raise ValueError("patch_sizes must not be empty")
    return torch.stack(losses).mean()


def endpoint_moment_loss(
    source: torch.Tensor,
    reference: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """Robustly match per-channel endpoint location and scale.

    Patch CORAL describes relative distribution geometry.  When its reference
    is allowed to move, both distributions can drift together without a large
    penalty.  This term matches the student to its stop-gradient teacher and
    avoids the quartic growth of a covariance-only correction.  An absolute
    RF-prior anchor is provided separately by
    :func:`standard_normal_endpoint_loss`.
    """
    if source.shape[1] != reference.shape[1]:
        raise ValueError("source/reference endpoint channels must match")
    reduce_dims = (0, *range(2, source.ndim))
    source_mean = source.float().mean(dim=reduce_dims)
    reference_mean = reference.float().mean(dim=reduce_dims)
    source_std = source.float().std(dim=reduce_dims, unbiased=False)
    reference_std = reference.float().std(dim=reduce_dims, unbiased=False)
    return F.smooth_l1_loss(
        source_mean,
        reference_mean,
        beta=float(beta),
    ) + F.smooth_l1_loss(
        source_std,
        reference_std,
        beta=float(beta),
    )


def standard_normal_endpoint_loss(
    endpoint: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """Anchor a rectified-flow endpoint to the training noise prior.

    The RF endpoint is sampled from ``N(0, I)`` during ordinary training. A
    moving EMA teacher is useful for semantic alignment, but cannot by itself
    prevent the teacher and student endpoint distributions from shrinking
    together. Matching per-channel mean and standard deviation to the known
    RF prior removes that jointly drifting solution without pairing subjects.
    """
    if endpoint.ndim < 3:
        raise ValueError("endpoint must have batch, channel, and spatial dims")
    reduce_dims = (0, *range(2, endpoint.ndim))
    endpoint_float = endpoint.float()
    mean = endpoint_float.mean(dim=reduce_dims)
    std = endpoint_float.std(dim=reduce_dims, unbiased=False)
    return F.smooth_l1_loss(
        mean,
        torch.zeros_like(mean),
        beta=float(beta),
    ) + F.smooth_l1_loss(
        std,
        torch.ones_like(std),
        beta=float(beta),
    )


def stable_endpoint_distribution_loss(
    source: torch.Tensor,
    reference: torch.Tensor,
    patch_sizes,
    *,
    coral_weight: float = 1.0,
    moment_weight: float = 1.0,
    moment_beta: float = 0.05,
    prior_weight: float = 0.0,
    prior_beta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine teacher matching with an optional absolute RF-prior anchor."""
    coral = multiscale_patch_coral(source, reference, patch_sizes)
    moments = endpoint_moment_loss(source, reference, beta=moment_beta)
    prior_weight = float(prior_weight)
    prior = (
        standard_normal_endpoint_loss(source, beta=prior_beta)
        if prior_weight != 0.0
        else source.new_zeros(())
    )
    total = (
        float(coral_weight) * coral
        + float(moment_weight) * moments
        + prior_weight * prior
    )
    components = {"coral": coral, "moments": moments}
    if prior_weight != 0.0:
        components["prior"] = prior
    return total, components


def alignment_weight(
    *,
    epoch: int,
    step: int,
    steps_per_epoch: int,
    maximum: float,
    start_epoch: float,
    ramp_epochs: float,
) -> float:
    """Linearly introduce alignment after the pretrained path has stabilized."""
    position = float(epoch) + float(step + 1) / max(1, int(steps_per_epoch))
    if position <= float(start_epoch):
        return 0.0
    if ramp_epochs <= 0:
        return float(maximum)
    progress = (position - float(start_epoch)) / float(ramp_epochs)
    return float(maximum) * min(1.0, max(0.0, progress))


def sparse_event_weight(
    per_step_weight: float,
    every_updates: int,
) -> float:
    """Return the unbiased weight for a periodically sampled loss event.

    ``per_step_weight`` is the desired coefficient in the full per-optimizer-
    step objective. When the loss is evaluated once every ``every_updates``
    steps, multiplying the event by that interval preserves the configured
    coefficient in expectation.
    """
    every_updates = int(every_updates)
    if every_updates < 1:
        raise ValueError("every_updates must be >= 1")
    if float(per_step_weight) < 0:
        raise ValueError("per_step_weight must be non-negative")
    return float(per_step_weight) * every_updates
