"""Task3 paired counterfactual fine-tuning for FieldRF.

Every update is a real same-subject, same-modality, cross-field pair.  The
source and target inversion endpoints are both routed to the same target
latent and target label.  Local RF losses train both routes; periodic
deployment-matched terminal rollouts make the source route agree with the
real target and the target-identity route.  BrainID remains frozen in cache;
the RF U-Net, ControlNet, and endpoint aligner are all learnable.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


FIELDRF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIELDRF_ROOT))

from data.dataset import (  # noqa: E402
    PairedFieldDataset,
    load_feat4,
    load_latent,
)
from models.model import (  # noqa: E402
    FieldRFModel,
    NUM_CLASSES,
    controlnet_norms,
    get_class_label,
    timestep_transform,
)
from models.endpoint_aligner import EndpointAligner  # noqa: E402
from utils.visualization import archive_metrics_csvs  # noqa: E402


class EMA:
    """Floating-point exponential moving average for checkpoint selection."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def apply_to(self, model):
        backup = {}
        state = model.state_dict()
        for key, value in self.shadow.items():
            if key in state and state[key].shape == value.shape:
                backup[key] = state[key].detach().clone()
                state[key].copy_(value)
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        state = model.state_dict()
        for key, value in backup.items():
            state[key].copy_(value)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        for key in self.shadow:
            if key in state and state[key].shape == self.shadow[key].shape:
                self.shadow[key].copy_(state[key])


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else FIELDRF_ROOT / value


def _build_bridge(config, checkpoint, device, load_ema):
    bridge = config["bridge"]
    return FieldRFModel.from_pretrained(
        checkpoint,
        unet_channels=tuple(bridge["unet_channels"]),
        num_res_blocks=tuple(bridge["num_res_blocks"]),
        attention_levels=tuple(bridge["attention_levels"]),
        num_head_channels=tuple(bridge["num_head_channels"]),
        norm_num_groups=int(bridge["norm_num_groups"]),
        norm_eps=float(bridge["norm_eps"]),
        num_class_embeds=NUM_CLASSES,
        feat4_channels=int(bridge["feat4_channels"]),
        conditioning_embedding_num_channels=tuple(
            bridge["conditioning_embedding_num_channels"]
        ),
        z_channels=4,
        num_train_timesteps=int(bridge["num_train_timesteps"]),
        scale_factor=float(config["vae"]["scale_factor"]),
        use_flash_attention=bool(bridge["use_flash_attention"]),
        use_source_latent=False,
        device=device,
        load_ema=load_ema,
    )


def _load_endpoint_aligner(checkpoint, base_checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "endpoint_aligner" not in state or "base_checkpoint" not in state:
        raise KeyError(
            "Endpoint alignment checkpoint must contain endpoint_aligner and "
            "base_checkpoint"
        )
    expected_base = _resolve(state["base_checkpoint"]).resolve()
    actual_base = _resolve(base_checkpoint).resolve()
    if expected_base != actual_base:
        raise ValueError(
            "Endpoint aligner was trained for a different RF base checkpoint: "
            f"{expected_base} != {actual_base}"
        )
    model_config = state["endpoint_aligner_config"]
    aligner = EndpointAligner(
        endpoint_channels=4,
        hidden_channels=int(model_config["hidden_channels"]),
        domain_embed_dim=int(model_config["domain_embed_dim"]),
        num_domains=NUM_CLASSES,
        num_blocks=int(model_config["num_blocks"]),
    ).to(device)
    aligner.load_state_dict(state["endpoint_aligner"], strict=True)
    aligner.train().requires_grad_(True)
    return aligner, model_config


def _make_loader(dataset, config):
    workers = int(config["num_workers"])
    kwargs = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(**kwargs)


def _terminal_latent_loss(prediction, target, edge_weight):
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


def _terminal_rollout(
    bridge,
    endpoint,
    feature,
    target_class,
    *,
    rollout_steps,
    gradient_steps,
):
    """Follow the deployed reverse ODE with truncated terminal gradients."""
    rollout_steps = int(rollout_steps)
    gradient_steps = int(gradient_steps)
    if rollout_steps < 1 or not 1 <= gradient_steps <= rollout_steps:
        raise ValueError(
            "rollout_steps must be positive and gradient_steps must be in range"
        )
    schedule = torch.linspace(
        bridge.num_train_timesteps - 1,
        0,
        rollout_steps + 1,
        device=endpoint.device,
    )
    schedule = timestep_transform(
        schedule,
        int(np.prod(endpoint.shape[2:])),
        base_img_size_numel=32768,
        scale=1.4,
        num_train_timesteps=bridge.num_train_timesteps,
        spatial_dim=3,
    )
    state = endpoint
    batch_size = endpoint.shape[0]
    for index in range(rollout_steps):
        keep_gradient = index >= rollout_steps - gradient_steps
        if keep_gradient:
            timestep = schedule[index].expand(batch_size)
            velocity = bridge(state, timestep, target_class, [feature])
            delta = (
                schedule[index] - schedule[index + 1]
            ) / bridge.num_train_timesteps
            state = state + velocity * delta
        else:
            with torch.no_grad():
                timestep = schedule[index].expand(batch_size)
                velocity = bridge(state, timestep, target_class, [feature])
                delta = (
                    schedule[index] - schedule[index + 1]
                ) / bridge.num_train_timesteps
                state = state + velocity * delta
            state = state.detach()
    return state


@torch.no_grad()
def _cache_epoch_endpoints(
    bridge,
    dataset,
    inversion,
    device,
    amp_dtype,
):
    """Invert each of the 45 real domains once with the current RF model."""
    bridge.eval()
    endpoints = {}
    started = time.time()
    for index, domain in enumerate(dataset.domains):
        latent = load_latent(domain["latent"]).unsqueeze(0).to(device)
        feature = load_feat4(domain["feature"]).unsqueeze(0).to(device)
        class_label = torch.tensor(
            [get_class_label(domain["modality"], domain["field"])],
            device=device,
            dtype=torch.long,
        )
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            endpoint = bridge.invert_to_endpoint(
                latent.float() * bridge.scale_factor,
                [feature],
                class_label,
                n_steps=int(inversion["steps"]),
                solver=str(inversion["solver"]),
                gradient_steps=0,
                velocity_model=bridge,
            )
        endpoints[domain["domain_key"]] = endpoint.float().cpu()
        del latent, feature, endpoint
        if (index + 1) % 15 == 0:
            print(
                f"  endpoint cache {index + 1}/{len(dataset.domains)}",
                flush=True,
            )
    bridge.train()
    print(
        f"  endpoint cache complete: {len(endpoints)} domains, "
        f"seconds={time.time() - started:.0f}",
        flush=True,
    )
    return endpoints


def _append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _atomic_checkpoint(state, path):
    """Atomically replace one checkpoint without retaining a full hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _save_training_checkpoint(state, latest_path):
    """Save the only full, optimizer-resumable training checkpoint."""
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_checkpoint(state, latest_path)


def _checkpoint_state(
    bridge,
    endpoint_aligner,
    endpoint_aligner_config,
    bridge_ema,
    endpoint_aligner_ema,
    optimizer,
    scheduler,
    config,
    epoch,
    global_step,
    best_task3_ssim,
):
    return {
        "bridge": bridge.state_dict(),
        "endpoint_aligner": endpoint_aligner.state_dict(),
        "endpoint_aligner_config": endpoint_aligner_config,
        "ema": bridge_ema.state_dict(),
        "endpoint_aligner_ema": endpoint_aligner_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "scale_factor": float(bridge.scale_factor),
        "num_train_timesteps": int(bridge.num_train_timesteps),
        "best_task3_ssim": float(best_task3_ssim),
        "training_objective": "paired_counterfactual_equality",
        "data_contract": "180_same_subject_same_modality_cross_field_pairs",
        "image_loss": False,
        "terminal_rollout": True,
        "counterfactual_identity_branch": True,
        "direct_endpoint_loss": False,
        "discriminator": False,
        "retrospective_replay": False,
    }


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_vis", action="store_true")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not torch.cuda.is_available():
        raise RuntimeError("RF paired fine-tuning requires a CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _seed_everything(int(config["train"]["seed"]))

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        archive_metrics_csvs(output_dir / "vis")
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    data = config["data"]
    dataset = PairedFieldDataset(
        cache_root=data["cache_root"],
        split=data["split"],
        modalities=data["modalities"],
        source_fields=data["source_fields"],
        target_fields=data["target_fields"],
        subjects=data["subjects"],
        feat_cache_name=data["feat_cache_name"],
        include_identity=data["include_identity"],
    )
    loader = _make_loader(dataset, config["train"])
    steps_per_epoch = len(loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)

    initialization = args.resume or config["finetune_from"]
    bridge = _build_bridge(
        config,
        _resolve(initialization),
        device,
        load_ema=args.resume is None and config["weight_source"] == "ema",
    )
    endpoint_aligner, endpoint_aligner_config = _load_endpoint_aligner(
        _resolve(config["endpoint_aligner_checkpoint"]),
        config["finetune_from"],
        device,
    )
    train = config["train"]
    unet_parameters = list(bridge.unet.parameters())
    controlnet_parameters = list(bridge.controlnet.parameters())
    endpoint_aligner_parameters = list(endpoint_aligner.parameters())
    grouped_parameters = {id(parameter) for parameter in unet_parameters}
    grouped_parameters.update(id(parameter) for parameter in controlnet_parameters)
    trainable_parameters = {
        id(parameter) for parameter in bridge.parameters() if parameter.requires_grad
    }
    if grouped_parameters != trainable_parameters:
        raise RuntimeError(
            "Paired RF optimizer must cover exactly the U-Net and ControlNet"
        )
    controlnet_lr = float(train["lr"]) * float(train["controlnet_lr_ratio"])
    endpoint_aligner_lr = float(train["lr"]) * float(
        train["endpoint_aligner_lr_ratio"]
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": unet_parameters, "lr": float(train["lr"]), "name": "unet"},
            {
                "params": controlnet_parameters,
                "lr": controlnet_lr,
                "name": "controlnet",
            },
            {
                "params": endpoint_aligner_parameters,
                "lr": endpoint_aligner_lr,
                "name": "endpoint_aligner",
            },
        ],
        weight_decay=float(train["weight_decay"]),
    )

    n_epochs = int(args.n_epochs or train["n_epochs"])
    total_steps = max(1, n_epochs * steps_per_epoch)
    warmup_steps = int(float(train["warmup_epochs"]) * steps_per_epoch)
    minimum_ratio = float(train["lr_min"]) / float(train["lr"])

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    bridge_ema = EMA(bridge, train["ema_decay"])
    endpoint_aligner_ema = EMA(endpoint_aligner, train["ema_decay"])
    start_epoch = 0
    global_step = 0
    if args.resume:
        state = torch.load(
            _resolve(args.resume), map_location="cpu", weights_only=False
        )
        endpoint_aligner.load_state_dict(state["endpoint_aligner"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        bridge_ema.load_state_dict(state["ema"])
        endpoint_aligner_ema.load_state_dict(state["endpoint_aligner_ema"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_task3_ssim = float(state["best_task3_ssim"])
    else:
        best_task3_ssim = -float("inf")

    inversion = config["inversion"]
    cross = config["cross_rf"]
    identity = config["identity_rf"]
    rollout = config["terminal_rollout"]
    amp_dtype = getattr(torch, train["mixed_precision"])

    print(
        "RF paired-only fine-tune: "
        f"subjects={data['subjects']} same_modality_pairs={len(dataset)} "
        f"steps_per_epoch={steps_per_epoch}",
        flush=True,
    )
    print(
        "Objectives: source-cross RF + target-identity RF + periodic deployed "
        "terminal equality; direct endpoint loss=0 cross-modality=0 "
        f"UNet_lr={float(train['lr']):.2e} ControlNet_lr={controlnet_lr:.2e} "
        f"Aligner_lr={endpoint_aligner_lr:.2e}",
        flush=True,
    )

    for epoch in range(start_epoch, n_epochs):
        bridge.train()
        endpoint_aligner.train()
        endpoint_cache = _cache_epoch_endpoints(
            bridge, dataset, inversion, device, amp_dtype
        )
        sums = {
            "total": 0.0,
            "cross_rf": 0.0,
            "identity_rf": 0.0,
            "terminal": 0.0,
            "terminal_l1": 0.0,
            "terminal_edge": 0.0,
            "terminal_consistency": 0.0,
            "identity_terminal_l1": 0.0,
        }
        modality_sums = {
            modality: {"cross_rf": 0.0, "identity_rf": 0.0, "count": 0}
            for modality in data["modalities"]
        }
        terminal_count = 0
        started = time.time()

        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            optimizer.zero_grad(set_to_none=True)

            source_feat4 = batch["source_feat4"].to(device, non_blocking=True)
            source_class = batch["source_class"].to(device, non_blocking=True)
            target_latent = batch["target_latent"].to(device, non_blocking=True)
            target_feat4 = batch["target_feat4"].to(device, non_blocking=True)
            target_class = batch["target_class"].to(device, non_blocking=True)
            target_scaled = target_latent.float() * bridge.scale_factor
            raw_source_endpoint = endpoint_cache[
                batch["source_domain_key"][0]
            ].to(device, non_blocking=True)
            raw_target_endpoint = endpoint_cache[
                batch["target_domain_key"][0]
            ].to(device, non_blocking=True)

            cross_t = bridge.sample_train_timesteps(
                target_scaled,
                endpoint_focus_probability=float(
                    cross["endpoint_focus_probability"]
                ),
                endpoint_min_fraction=float(cross["endpoint_min_fraction"]),
            )
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                source_endpoint, _ = endpoint_aligner(
                    raw_source_endpoint, source_class
                )
                cross_state, cross_velocity = bridge.build_rf_path(
                    target_scaled,
                    source_endpoint.to(dtype=target_scaled.dtype),
                    cross_t,
                )
                cross_prediction = bridge(
                    cross_state, cross_t, target_class, [source_feat4]
                )
                cross_loss = F.l1_loss(cross_prediction, cross_velocity)
                cross_objective = float(cross["weight"]) * cross_loss
            cross_objective.backward()

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                target_endpoint, _ = endpoint_aligner(
                    raw_target_endpoint, target_class
                )
                identity_state, identity_velocity = bridge.build_rf_path(
                    target_scaled,
                    target_endpoint.to(dtype=target_scaled.dtype),
                    cross_t,
                )
                identity_prediction = bridge(
                    identity_state, cross_t, target_class, [target_feat4]
                )
                identity_loss = F.l1_loss(identity_prediction, identity_velocity)
                identity_objective = float(identity["weight"]) * identity_loss
            identity_objective.backward()

            terminal_active = step % int(rollout["every_steps"]) == 0
            terminal_objective_value = 0.0
            terminal_loss_value = 0.0
            terminal_l1_value = 0.0
            terminal_edge_value = 0.0
            terminal_consistency_value = 0.0
            identity_terminal_l1_value = 0.0
            if terminal_active:
                with torch.no_grad(), torch.amp.autocast(
                    "cuda", dtype=amp_dtype
                ):
                    aligned_target_endpoint, _ = endpoint_aligner(
                        raw_target_endpoint, target_class
                    )
                    identity_terminal = _terminal_rollout(
                        bridge,
                        aligned_target_endpoint,
                        target_feat4,
                        target_class,
                        rollout_steps=int(rollout["steps"]),
                        gradient_steps=int(rollout["gradient_steps"]),
                    )
                with torch.no_grad(), torch.amp.autocast(
                    "cuda", dtype=amp_dtype
                ):
                    aligned_source_endpoint, _ = endpoint_aligner(
                        raw_source_endpoint, source_class
                    )
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    cross_terminal = _terminal_rollout(
                        bridge,
                        aligned_source_endpoint,
                        source_feat4,
                        target_class,
                        rollout_steps=int(rollout["steps"]),
                        gradient_steps=int(rollout["gradient_steps"]),
                    )
                    terminal_target, terminal_l1, terminal_edge = (
                        _terminal_latent_loss(
                            cross_terminal,
                            target_scaled,
                            float(rollout["edge_weight"]),
                        )
                    )
                    terminal_consistency = F.smooth_l1_loss(
                        cross_terminal.float(), identity_terminal.float()
                    )
                    terminal_loss = terminal_target + float(
                        rollout["consistency_weight"]
                    ) * terminal_consistency
                    terminal_objective = float(rollout["weight"]) * terminal_loss
                terminal_objective.backward()
                terminal_count += 1
                terminal_objective_value = float(terminal_objective.detach())
                terminal_loss_value = float(terminal_loss.detach())
                terminal_l1_value = float(terminal_l1.detach())
                terminal_edge_value = float(terminal_edge.detach())
                terminal_consistency_value = float(terminal_consistency.detach())
                identity_terminal_l1_value = float(
                    F.l1_loss(identity_terminal.float(), target_scaled).detach()
                )

            torch.nn.utils.clip_grad_norm_(
                bridge.parameters(), float(train["grad_clip"])
            )
            torch.nn.utils.clip_grad_norm_(
                endpoint_aligner.parameters(), float(train["grad_clip"])
            )
            optimizer.step()
            scheduler.step()
            bridge_ema.update(bridge)
            endpoint_aligner_ema.update(endpoint_aligner)
            global_step += 1

            total = (
                float(cross_objective.detach())
                + float(identity_objective.detach())
                + terminal_objective_value
            )
            cross_value = float(cross_loss.detach())
            identity_value = float(identity_loss.detach())
            sums["total"] += total
            sums["cross_rf"] += cross_value
            sums["identity_rf"] += identity_value
            if terminal_active:
                sums["terminal"] += terminal_loss_value
                sums["terminal_l1"] += terminal_l1_value
                sums["terminal_edge"] += terminal_edge_value
                sums["terminal_consistency"] += terminal_consistency_value
                sums["identity_terminal_l1"] += identity_terminal_l1_value

            modality = batch["source_modality"][0]
            if modality != batch["target_modality"][0]:
                raise RuntimeError("Cross-modality pair reached paired field trainer")
            modality_sums[modality]["cross_rf"] += cross_value
            modality_sums[modality]["identity_rf"] += identity_value
            modality_sums[modality]["count"] += 1

            if (step + 1) % int(train["log_every"]) == 0:
                print(
                    f"epoch={epoch + 1}/{n_epochs} step={step + 1}/"
                    f"{steps_per_epoch} total={total:.5f} "
                    f"cross={cross_value:.5f} identity={identity_value:.5f} "
                    f"terminal={terminal_loss_value:.5f} "
                    f"terminal_cons={terminal_consistency_value:.5f} "
                    f"modality={modality} "
                    f"lr_unet={optimizer.param_groups[0]['lr']:.2e} "
                    f"lr_controlnet={optimizer.param_groups[1]['lr']:.2e} "
                    f"lr_aligner={optimizer.param_groups[2]['lr']:.2e}",
                    flush=True,
                )

        count = max(1, steps_per_epoch)
        means = {
            "total": sums["total"] / count,
            "cross_rf": sums["cross_rf"] / count,
            "identity_rf": sums["identity_rf"] / count,
            **{
                key: sums[key] / max(1, terminal_count)
                for key in (
                    "terminal",
                    "terminal_l1",
                    "terminal_edge",
                    "terminal_consistency",
                    "identity_terminal_l1",
                )
            },
        }
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            **means,
            "terminal_updates": terminal_count,
            **{
                f"{modality}_{metric}": values[metric] / max(1, values["count"])
                for modality, values in modality_sums.items()
                for metric in ("cross_rf", "identity_rf")
            },
            "lr_unet": optimizer.param_groups[0]["lr"],
            "lr_controlnet": optimizer.param_groups[1]["lr"],
            "lr_endpoint_aligner": optimizer.param_groups[2]["lr"],
            "seconds": time.time() - started,
        }
        _append_csv(output_dir / "training_log.csv", row)
        print(
            f"Epoch {epoch + 1}: total={means['total']:.5f} "
            f"cross={means['cross_rf']:.5f} "
            f"identity={means['identity_rf']:.5f} "
            f"terminal={means['terminal']:.5f} "
            f"terminal_cons={means['terminal_consistency']:.5f} "
            f"identity_terminal_l1={means['identity_terminal_l1']:.5f} "
            f"seconds={row['seconds']:.0f}{controlnet_norms(bridge)}",
            flush=True,
        )
        del endpoint_cache

        state = _checkpoint_state(
            bridge,
            endpoint_aligner,
            endpoint_aligner_config,
            bridge_ema,
            endpoint_aligner_ema,
            optimizer,
            scheduler,
            config,
            epoch,
            global_step,
            best_task3_ssim,
        )
        latest_path = output_dir / "fieldrf_paired_latest.pt"
        _save_training_checkpoint(state, latest_path)
        print(f"Saved pre-validation checkpoint {latest_path}", flush=True)

        if not args.skip_vis and (epoch + 1) % int(config["vis"]["vis_every"]) == 0:
            from visualize import run_vis

            bridge_backup = bridge_ema.apply_to(bridge)
            aligner_backup = endpoint_aligner_ema.apply_to(endpoint_aligner)
            summary = run_vis(
                bridge,
                bridge.scale_factor,
                config,
                epoch,
                device,
                endpoint_aligner=endpoint_aligner,
            )
            task3_ssim = float(summary["same_modality_cross_field"]["ssim"])
            if task3_ssim > best_task3_ssim:
                best_task3_ssim = task3_ssim
                _atomic_checkpoint(
                    {
                        "bridge": bridge.state_dict(),
                        "endpoint_aligner": endpoint_aligner.state_dict(),
                        "endpoint_aligner_config": endpoint_aligner_config,
                        "config": config,
                        "epoch": epoch,
                        "global_step": global_step,
                        "task3_ssim": task3_ssim,
                        "training_objective": (
                            "paired_counterfactual_equality"
                        ),
                    },
                    output_dir / "fieldrf_paired_best_task3.pt",
                )
                print(
                    f"New Task3 best: SSIM={task3_ssim:.5f}", flush=True
                )
            endpoint_aligner_ema.restore(endpoint_aligner, aligner_backup)
            bridge_ema.restore(bridge, bridge_backup)
            if task3_ssim >= best_task3_ssim:
                state["best_task3_ssim"] = best_task3_ssim
                _save_training_checkpoint(state, latest_path)
                print(f"Updated best score in {latest_path}", flush=True)


if __name__ == "__main__":
    main()
