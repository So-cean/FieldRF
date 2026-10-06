"""Base trainer with native PyTorch DDP and AMP support."""

import json
import math
import os
import re
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from ..env import get_device
from ..losses.perceptual import PerceptualLoss
from ..losses.structure import SSIMLoss, StructureLoss
from ..utils_dist import get_dist_info, is_main_process


def _iter_unpaired(loader_src, loader_tgt):
    iter_src = iter(loader_src)
    iter_tgt = iter(loader_tgt)
    for _ in range(min(len(loader_src), len(loader_tgt))):
        yield next(iter_src), next(iter_tgt)


def _get_lr_lambda(n_epochs, n_epochs_decay):
    total = n_epochs + n_epochs_decay
    return lambda epoch: 1.0 - max(0, epoch - n_epochs) / max(1, total - n_epochs)


def _unwrap(module, use_dist):
    return module.module if use_dist else module


class BaseTrainer(ABC):
    """Base trainer class for all MRIxFields2026 methods.

    Handles:
        - DDP detection and device setup
        - Output directory setup
        - Automatic mixed precision (bf16 / fp16 / no)
        - Main-process-only logging
        - Resume from checkpoint
        - Unified checkpoint save/load
        - Validation + early stopping
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_cfg = cfg.get("training", {})
        self.mixed_precision = self.train_cfg.get("mixed_precision", "no")
        self.max_grad_norm = self.train_cfg.get("max_grad_norm", 0.0)
        self.use_dist = cfg.get("_runtime", {}).get("use_dist", False)

        if self.use_dist:
            self.rank, self.world_size = get_dist_info()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.rank, self.world_size = 0, 1
            self.device = torch.device(get_device())

        self.output_dir = Path(self.cfg["training"]["output_dir"])
        self.weights_dir = self.output_dir / "weights"
        self.visuals_dir = self.output_dir / "visuals"

        if self.is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            self.visuals_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.global_step = 0
        self._es_best = {}
        self._es_no_improve = 0

    def is_main_process(self) -> bool:
        return self.rank == 0

    def autocast_context(self):
        if self.mixed_precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        elif self.mixed_precision == "fp16":
            return torch.autocast(device_type=self.device.type, dtype=torch.float16)
        else:
            return nullcontext()

    def find_latest_checkpoint(self) -> Path | None:
        ckpts = sorted(self.weights_dir.glob("checkpoint_epoch*.pth"))
        if not ckpts:
            return None
        def _epoch(path):
            m = re.search(r"checkpoint_epoch(\d+)\.pth", path.name)
            return int(m.group(1)) if m else 0
        return max(ckpts, key=_epoch)

    # ------------------------------------------------------------------ #
    #  Checkpoint save / load
    # ------------------------------------------------------------------ #

    def _build_checkpoint_state(self, epoch, metrics=None, include_optimizer=True):
        state = dict(
            epoch=epoch,
            global_step=self.global_step,
            generator_config=self._generator_config(),
            es_best=self._es_best,
            es_no_improve=self._es_no_improve,
            metrics=metrics,
            _metadata=self._checkpoint_metadata(),
        )

        if hasattr(self, "model"):       # CUTPretrainTrainer
            state["model"] = self.model.state_dict()
            gen_sd = {k[5:]: v for k, v in state["model"].items() if k.startswith("netG.")}
            state["generator"] = gen_sd
            if hasattr(self.model, "ema_state") and self.model.ema_state is not None:
                state["ema_state"] = self.model.ema_state
            if include_optimizer:
                for k in ("optimizer_G", "optimizer_D", "optimizer_F",
                           "scheduler_G", "scheduler_D", "scheduler_F"):
                    v = getattr(self, k, None)
                    if v is not None:
                        state[k] = v.state_dict()

        elif hasattr(self, "generator"):  # CUTFineTuneTrainer
            state["model"] = self.generator.state_dict()
            state["generator"] = self.generator.state_dict()
            if include_optimizer:
                state["optimizer"] = self.optimizer.state_dict()
                if hasattr(self, "scheduler"):
                    state["scheduler"] = self.scheduler.state_dict()
        return state

    def _save_checkpoint(self, name, epoch, metrics=None):
        if not self.is_main_process():
            return
        path = self.weights_dir / name
        state = self._build_checkpoint_state(epoch, metrics=metrics)
        torch.save(state, path)
        print(f"Saved: {path}")

    def _try_resume(self):
        ckpt_path = self.find_latest_checkpoint()
        if not ckpt_path:
            if self.is_main_process():
                print("No checkpoint found to resume.")
            return
        if self.is_main_process():
            print(f"Resuming from {ckpt_path} ...")

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.start_epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)

        if hasattr(self, "model") and "model" in state:
            self.model.load_state_dict(state["model"])
        elif hasattr(self, "generator") and "model" in state:
            self.generator.load_state_dict(state["model"])

        for key in ["optimizer_G", "optimizer_D", "optimizer_F", "optimizer"]:
            if hasattr(self, key) and key in state:
                getattr(self, key).load_state_dict(state[key])

        for key in ["scheduler_G", "scheduler_D", "scheduler_F", "scheduler"]:
            if hasattr(self, key) and key in state:
                getattr(self, key).load_state_dict(state[key])

        if "ema_state" in state and state["ema_state"] is not None \
                and hasattr(self, "model") and hasattr(self.model, "ema_state"):
            self.model.ema_state = state["ema_state"]

        if "es_best" in state:
            self._es_best = state["es_best"]
        elif hasattr(self, "_es_best") and (self.weights_dir / "best_metrics.json").exists():
            bm = json.loads((self.weights_dir / "best_metrics.json").read_text())
            self._es_best = {}
            for v in bm.values():
                if not isinstance(v, dict):
                    continue
                for mk, mv in v.items():
                    if mk != "epoch":
                        cur = self._es_best.get(mk)
                        higher_better = mk.startswith("ssim_") or mk.startswith("psnr_")
                        if cur is None or (higher_better and mv > cur) or (not higher_better and mv < cur):
                            self._es_best[mk] = mv

        self._es_no_improve = state.get("es_no_improve", 0)

        if self.is_main_process():
            print(f"Resumed successfully. Starting from epoch {self.start_epoch}, step {self.global_step}")

    # ------------------------------------------------------------------ #
    #  Checkpoint metadata
    # ------------------------------------------------------------------ #

    def _generator_config(self) -> dict:
        """Override in subclass to return generator architecture dict."""
        raise NotImplementedError("Subclass must implement _generator_config()")

    def _checkpoint_metadata(self) -> dict:
        metadata = {
            "generator_config": self._generator_config(),
            "config_actual": self.cfg,
            "task_name": self.cfg.get("task_name", ""),
            "method": self.cfg.get("method", ""),
            "norm_stats": getattr(self, "_norm_stats", None),
        }
        # baseline metrics (identity mapping)
        if hasattr(self, "_baseline_metrics"):
            metadata["baseline_metrics"] = self._baseline_metrics
        try:
            import subprocess
            metadata["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            metadata["git_commit"] = "unknown"
        return metadata

    # ------------------------------------------------------------------ #
    #  Generator access
    # ------------------------------------------------------------------ #

    def _get_generator(self):
        if hasattr(self, "model"):
            return _unwrap(self.model.netG, self.use_dist)
        if hasattr(self, "generator"):
            return _unwrap(self.generator, self.use_dist)
        raise AttributeError("Trainer has no 'model' or 'generator' attribute")

    # ------------------------------------------------------------------ #
    #  Config validation
    # ------------------------------------------------------------------ #

    def _validate_config(self):
        data_cfg = self.cfg["data"]
        model_cfg = self.cfg["model"]
        if "modalities" not in data_cfg:
            raise ValueError("Config 'data' section is missing required key 'modalities'.")
        modalities = data_cfg["modalities"]
        if not isinstance(modalities, list) or len(modalities) == 0:
            raise ValueError("Config 'data.modalities' must be a non-empty list.")
        input_nc = model_cfg.get("input_nc", len(modalities))
        output_nc = model_cfg.get("output_nc", len(modalities))
        self.modalities = modalities
        self.input_nc = input_nc
        self.output_nc = output_nc
        return modalities

    # ------------------------------------------------------------------ #
    #  Validation + best checkpoint + early stopping
    # ------------------------------------------------------------------ #

    def _validate_and_early_stop(self, epoch: int, save_every: int):
        if self._evaluator is None:
            return False
        if self._val_every <= 0 or (epoch + 1) % self._val_every != 0:
            return False

        net_G = self._get_generator()
        was_training = net_G.training
        net_G.eval()

        # Use EMA weights for validation (matches what generator_final.pth saves)
        has_ema = (hasattr(self, "model") and hasattr(self.model, "ema_state")
                   and self.model.ema_state is not None)
        if has_ema:
            backup = net_G.state_dict()
            net_G.load_state_dict(self.model.ema_state, strict=False)

        metrics = self._evaluator.run(net_G)

        if has_ema:
            net_G.load_state_dict(backup)
        if was_training:
            net_G.train()

        if self.is_main_process():
            self._train_logger.log_val(epoch + 1, **metrics)

            _improved = False
            for k, v in metrics.items():
                is_lower_better = k.startswith("nrmse_") or k.startswith("lpips_")
                prev = self._es_best.get(k)
                if prev is None or (is_lower_better and v < prev) or (not is_lower_better and v > prev):
                    self._es_best[k] = v
                    _improved = True

            bm_path = self.weights_dir / "best_metrics.json"
            bm = json.loads(bm_path.read_text()) if bm_path.exists() else {}
            mod_suffixes = sorted({k.split("_", 1)[1] for k in metrics if "_" in k and not k.endswith("epoch")})
            for suffix in mod_suffixes:
                ssim_k = f"ssim_{suffix}"
                nrmse_k = f"nrmse_{suffix}"
                if ssim_k in metrics and metrics[ssim_k] == self._es_best.get(ssim_k):
                    self._save_checkpoint(f"best_ssim_{suffix}.pth", epoch + 1, metrics)
                    bm[f"best_ssim_{suffix}"] = {"epoch": epoch + 1, **metrics}
                if nrmse_k in metrics and metrics[nrmse_k] == self._es_best.get(nrmse_k):
                    self._save_checkpoint(f"best_nrmse_{suffix}.pth", epoch + 1, metrics)
                    bm[f"best_nrmse_{suffix}"] = {"epoch": epoch + 1, **metrics}
            bm_path.write_text(json.dumps(bm, indent=2))

            if (epoch + 1) >= self._es_min_epochs:
                self._es_no_improve = 0 if _improved else self._es_no_improve + 1

        _should_stop = False
        if self.is_main_process():
            _should_stop = (self._es_no_improve >= self._es_patience
                            and (epoch + 1) >= self._es_min_epochs)
        if self.use_dist:
            _stop_t = torch.tensor([_should_stop], device=self.device)
            torch.distributed.broadcast(_stop_t, src=0)
            _should_stop = _stop_t.item()
        if _should_stop:
            if self.is_main_process():
                stagnant = [f"{k}={v:.4f}" for k, v in self._es_best.items()]
                print(f"Early stopping: no metric improved for {self._es_patience} epochs "
                      f"(min {self._es_min_epochs} epochs reached). "
                      f"Best: {'  '.join(stagnant)}")
                self._save_checkpoint(f"checkpoint_epoch{epoch+1}.pth", epoch + 1)
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Baseline metrics (source vs target, no model)
    # ------------------------------------------------------------------ #

    def _save_baseline_metrics(self, baseline: dict, key: str = "baseline"):
        if not self.is_main_process():
            return
        if not hasattr(self, "_baseline_metrics"):
            self._baseline_metrics = {}
        self._baseline_metrics[key] = {"epoch": 0, **baseline}
        bm_path = self.weights_dir / "best_metrics.json"
        bm = json.loads(bm_path.read_text()) if bm_path.exists() else {}
        bm[key] = {"epoch": 0, **baseline}
        bm_path.write_text(json.dumps(bm, indent=2))

    # ------------------------------------------------------------------ #
    #  Training visualisation
    # ------------------------------------------------------------------ #

    def _save_training_visual(self, real_A, real_B, fake_B,
                              batch_info, batch_info_b=None,
                              idt_B=None, epoch=0):
        from ..utils.visualizer import save_training_visuals

        # Determine source field for per-field denorm
        fs = (batch_info.get("field_strength", [None]) or [None])[0] if batch_info else None

        if self.normalize:
            if hasattr(self, "_norm_stats") and self._norm_stats is not None:
                ns = self._norm_stats
                mu_s = torch.tensor(ns["mu_src"], device=real_A.device).view(1, -1, 1, 1)
                sg_s = torch.tensor(ns["sigma_src"], device=real_A.device).view(1, -1, 1, 1)
                mu_t = torch.tensor(ns["mu_tgt"], device=real_B.device).view(1, -1, 1, 1)
                sg_t = torch.tensor(ns["sigma_tgt"], device=real_B.device).view(1, -1, 1, 1)
                visuals = {
                    "real_A": real_A * sg_s + mu_s,
                    "real_B": real_B * sg_t + mu_t,
                    "fake_B": fake_B * sg_t + mu_t,
                }
                _idt_denorm = lambda x: x * sg_t + mu_t
            elif hasattr(self, "model") and hasattr(self.model, "has_norm") and self.model.has_norm():
                # Per-field denorm for multi-field training
                if fs and hasattr(self, "_field_stats") and self._field_stats and fs in self._field_stats:
                    s = self._field_stats[fs]
                    mu_s = torch.tensor(s["mu"], device=real_A.device).view(1, -1, 1, 1)
                    sg_s = torch.tensor(s["sigma"], device=real_A.device).view(1, -1, 1, 1)
                    real_A_denorm = real_A * sg_s + mu_s
                else:
                    real_A_denorm = self.model.denorm_src(real_A)
                visuals = {
                    "real_A": real_A_denorm,
                    "real_B": self.model.denorm_tgt(real_B),
                    "fake_B": self.model.denorm_tgt(fake_B),
                }
                _idt_denorm = self.model.denorm_tgt
            else:
                visuals = {"real_A": real_A, "real_B": real_B, "fake_B": fake_B}
                _idt_denorm = lambda x: x
            if idt_B is not None:
                visuals["idt_B"] = _idt_denorm(idt_B)
        else:
            visuals = {"real_A": real_A, "real_B": real_B, "fake_B": fake_B}
            if idt_B is not None:
                visuals["idt_B"] = idt_B

        save_training_visuals(visuals, self.visuals_dir, epoch=epoch,
                              slice_axis=self.cfg.get("data", {}).get("slice_axis", 2),
                              modalities=self.modalities,
                              batch_info=batch_info, batch_info_b=batch_info_b,
                              field_strength=fs or "")


    # ------------------------------------------------------------------ #
    #  Finetune losses helper
    # ------------------------------------------------------------------ #

    def _setup_finetune_losses(self, finetune_cfg):
        self.criterion_l1 = nn.L1Loss()
        self.criterion_lpips = PerceptualLoss().to(self.device)
        self.criterion_ssim = SSIMLoss().to(self.device)
        self.lambda_l1 = finetune_cfg.get("loss_l1", 1.0)
        self.lambda_lpips = finetune_cfg.get("loss_lpips", 0.1)
        self.lambda_ssim = finetune_cfg.get("loss_ssim", 0.0)
        self.lambda_edge = finetune_cfg.get("loss_edge", 0.0)
        if self.lambda_edge > 0:
            self.criterion_edge = StructureLoss(ssim_weight=0.0, edge_weight=1.0).to(self.device)


    def _compute_finetune_loss(self, pred, target):
        loss = self.lambda_l1 * self.criterion_l1(pred, target)
        # LPIPS/SSIM/Edge on [0,1] (same as official evaluation)
        if self._norm_stats is not None and (self.lambda_lpips > 0 or self.lambda_ssim > 0 or self.lambda_edge > 0):
            ns = self._norm_stats
            mu_t = torch.tensor(ns["mu_tgt"], device=target.device).view(1, -1, 1, 1)
            sg_t = torch.tensor(ns["sigma_tgt"], device=target.device).view(1, -1, 1, 1)
            pred_01 = (pred * sg_t + mu_t + 1) * 0.5
            tgt_01 = (target * sg_t + mu_t + 1) * 0.5
        elif not self.normalize and (self.lambda_lpips > 0 or self.lambda_ssim > 0 or self.lambda_edge > 0):
            # pred/target in [-1, 1]; map to [0, 1] for LPIPS/SSIM
            pred_01 = (pred + 1) * 0.5
            tgt_01 = (target + 1) * 0.5
        else:
            pred_01, tgt_01 = pred, target
        if self.lambda_lpips > 0:
            loss = loss + self.lambda_lpips * self.criterion_lpips(pred_01 * 2 - 1, tgt_01 * 2 - 1)
        if self.lambda_ssim > 0:
            loss = loss + self.lambda_ssim * self.criterion_ssim(pred_01, tgt_01)
        if self.lambda_edge > 0:
            loss = loss + self.lambda_edge * self.criterion_edge(pred_01, tgt_01)
        return loss

    # ------------------------------------------------------------------ #
    #  Abstract methods
    # ------------------------------------------------------------------ #

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def run(self):
        self.setup()
        if self.cfg.get("_runtime", {}).get("resume", False):
            self._try_resume()
        self.train()

    # ------------------------------------------------------------------ #
    #  Cosine warmup LR scheduler factory
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_warmup_lr(epoch, warmup, total):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))


__all__ = [
    "BaseTrainer",
    "_iter_unpaired",
    "_get_lr_lambda",
]
