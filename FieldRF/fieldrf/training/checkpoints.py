"""Atomic training checkpoints and per-rank epoch-boundary random state."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def atomic_save(state, path):
    """Replace a checkpoint only after a successful complete write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rank_state(device, loader_generator=None):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    if loader_generator is not None:
        state["loader"] = loader_generator.get_state()
    return state


def collect_rank_states(device, loader_generator=None, extra=None):
    """Called on every rank, after any epoch-end validation has completed."""
    local = capture_rank_state(device, loader_generator)
    if extra is not None:
        local["extra"] = extra
    if not dist.is_initialized():
        return [local]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, local)
    return states


def restore_rank_state(states, rank, world_size, device, loader_generator=None):
    if len(states) != world_size:
        raise ValueError("Resume requires the checkpoint's original world size")
    state = states[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        if "cuda" not in state:
            raise ValueError("Checkpoint has no CUDA RNG state for this rank")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    if loader_generator is not None:
        loader_generator.set_state(state["loader"].cpu())


def validate_vae_resume(state, train_config=None, steps_per_epoch=None):
    required = {
        "state_dict", "optimizer", "scheduler", "epoch", "global_step",
        "ema_loss", "balance_step", "rng_states", "config", "steps_per_epoch",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(
            "--resume_ckpt requires a complete VAE training checkpoint; "
            f"missing {sorted(missing)}. Use --init_ckpt for weights-only initialization."
        )
    if train_config is not None:
        previous = state["config"]["train"]
        for key in ("batch_size", "accum_steps", "freeze_encoder", "seed", "num_workers",
                    "lr", "warmup_steps", "min_lr", "ssim_weight", "perceptual_weight", "loss_balance"):
            if previous.get(key) != train_config.get(key):
                raise ValueError(f"VAE resume requires the original train.{key}; use --init_ckpt for a new recipe")
    if steps_per_epoch is not None and steps_per_epoch != state["steps_per_epoch"]:
        raise ValueError("VAE resume requires the original number of batches per epoch")
