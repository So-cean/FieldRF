"""Native DDP execution while retaining the model's checkpoint namespace."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(enabled=False, device=None):
    """Initialize torchrun on CUDA/NCCL or CPU/Gloo; batch size stays per rank."""
    if enabled and not all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise ValueError("Launch --dist with torchrun (RANK/WORLD_SIZE/LOCAL_RANK required)")
    rank = int(os.environ["RANK"]) if enabled else 0
    world_size = int(os.environ["WORLD_SIZE"]) if enabled else 1
    local_rank = int(os.environ["LOCAL_RANK"]) if enabled else 0
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(selected)
    if selected.type == "cuda":
        selected = torch.device("cuda", local_rank if enabled else (selected.index or 0))
        torch.cuda.set_device(selected)
    if enabled and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if selected.type == "cuda" else "gloo",
            timeout=timedelta(minutes=30),
        )
    return rank, world_size, local_rank, selected


def wrap_module(module, device, find_unused_parameters=False):
    """Wrap a trainable submodule; all gradient-bearing calls use its forward."""
    return DDP(
        module,
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )


def wrap_rf_submodules(model, device):
    """Wrap RF predictors individually without re-registering their parameters.

    The raw UNet/ControlNet/adapter remain registered on FieldRFModel. DDP
    references live in an ordinary dictionary, so state_dict and EMA retain
    their existing keys. FieldRFModel routes training forwards through these
    wrappers; rank-zero no-grad visualization uses the raw modules.
    """
    model._ddp_predictors = {
        # Historical FieldRF UNets retain an unused registered middle block.
        # Discover those parameters rather than waiting for absent gradients;
        # changing the legacy forward here would change existing predictions.
        name: wrap_module(module, device, find_unused_parameters=(name == "unet"))
        for name in ("unet", "controlnet", "source_adapter")
        if (module := getattr(model, name)) is not None
        and any(parameter.requires_grad for parameter in module.parameters())
    }
    return model


def synchronize():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def teardown_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
