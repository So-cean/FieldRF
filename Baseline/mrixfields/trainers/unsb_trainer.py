"""UNSB pretrain trainer with native DDP and AMP.

UNSB requires two independent source batches per iteration (real_A and real_A2)
for the Schrodinger Bridge energy loss.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..models.unsb_model import UNSBModel
from ..utils.visualizer import save_training_visuals
from ..models.net_init import init_net
from ..data.dataset import UnpairedMRIDataset
from ..data.cached_dataset import CachedUnpairedDataset
from ..data.unpaired_loader import UnpairedDataLoader
from ..data.utils import ABBR_TO_SPLIT as SPLIT_MAP


def _get_lr_lambda(n_epochs: int, n_epochs_decay: int):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
    return lambda_rule


def _iter_three_loaders(loader_src, loader_src2, loader_tgt):
    """Iterate three loaders in parallel."""
    iter_src = iter(loader_src)
    iter_src2 = iter(loader_src2)
    iter_tgt = iter(loader_tgt)
    n = min(len(loader_src), len(loader_src2), len(loader_tgt))
    for _ in range(n):
        yield next(iter_src), next(iter_src2), next(iter_tgt)


class UNSBPretrainTrainer(BaseTrainer):
    """UNSB pretraining with unpaired data (2D slices).

    Requires two independent source-domain batches per iteration for the
    Schrodinger Bridge energy loss.
    """

    def setup(self):
        model_cfg = self.cfg["model"]
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]

        # ---- Model ---------------------------------------------------------
        self.model = UNSBModel(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            ndf=model_cfg.get("ndf", 64),
            ngf=model_cfg.get("ngf", 64),
            n_blocks=model_cfg.get("n_blocks", 9),
            n_mlp=model_cfg.get("n_mlp", 3),
            z_dim=model_cfg.get("z_dim", 256),
            t_emb_dim=model_cfg.get("t_emb_dim", 256),
            num_timesteps=model_cfg.get("num_timesteps", 5),
            tau=model_cfg.get("tau", 0.01),
            std=model_cfg.get("std", 0.25),
            lambda_GAN=model_cfg.get("lambda_GAN", 1.0),
            lambda_NCE=model_cfg.get("lambda_NCE", 1.0),
            lambda_idt=model_cfg.get("lambda_idt", None),
            lambda_SB=model_cfg.get("lambda_SB", 1.0),
            nce_layers=model_cfg.get("nce_layers", [0, 4, 8, 12, 16]),
            nce_T=model_cfg.get("nce_T", 0.07),
            nce_idt=model_cfg.get("nce_idt", True),
            gan_mode=model_cfg.get("gan_mode", "lsgan"),
            lr=train_cfg.get("lr", 0.0002),
            beta1=train_cfg.get("beta1", 0.5),
            beta2=train_cfg.get("beta2", 0.999),
            patch_mode=model_cfg.get("patch_mode", "patchnce"),
            num_patches=model_cfg.get("num_patches", 256),
            batch_size=train_cfg.get("batch_size", 1),
        )
        self.model.to(self.device)
        self.model.init_weights()
        self.model.setup_optimizers()

        # ---- Datasets -------------------------------------------------------
        modality = data_cfg["modalities"][0]
        source_field = data_cfg["source_field"]
        target_field = data_cfg["target_field"]
        split = train_cfg.get("split", "retro_train")
        batch_size = train_cfg.get("batch_size", 1)
        num_workers = train_cfg.get("num_workers", 4)

        crop_size_raw = data_cfg.get("crop_size")
        crop_size = tuple(crop_size_raw) if crop_size_raw else None
        preprocessed_dir = data_cfg.get("preprocessed_dir")

        def _make_dataset(field):
            if preprocessed_dir:
                return CachedUnpairedDataset(
                    preprocessed_dir=preprocessed_dir, split=split,
                    modality=modality, field_strength=field, crop_size=crop_size,
                )
            else:
                split_full = SPLIT_MAP.get(split, split)
                return UnpairedMRIDataset(
                    data_root=data_cfg["data_dir"], split=split_full,
                    modality=modality, field_strength=field, crop_size=crop_size,
                )

        dataset_src = _make_dataset(source_field)
        dataset_src2 = _make_dataset(source_field)
        dataset_tgt = _make_dataset(target_field)

        if self.use_dist:
            sampler_src = DistributedSampler(dataset_src, shuffle=True)
            sampler_src2 = DistributedSampler(dataset_src2, shuffle=True)
            sampler_tgt = DistributedSampler(dataset_tgt, shuffle=True)
            kwargs = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=True)
            self.loader_src = DataLoader(dataset_src, sampler=sampler_src, **kwargs)
            self.loader_src2 = DataLoader(dataset_src2, sampler=sampler_src2, **kwargs)
            self.loader_tgt = DataLoader(dataset_tgt, sampler=sampler_tgt, **kwargs)
            self.sampler_src = sampler_src
            self.sampler_src2 = sampler_src2
            self.sampler_tgt = sampler_tgt
        else:
            kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
            self.loader_src = DataLoader(dataset_src, **kwargs)
            self.loader_src2 = DataLoader(dataset_src2, **kwargs)
            self.loader_tgt = DataLoader(dataset_tgt, **kwargs)
            self.sampler_src = None
            self.sampler_src2 = None
            self.sampler_tgt = None

        # ---- Data-dependent init (BEFORE DDP) ------------------------------
        init_batch_a = next(iter(self.loader_src))
        init_batch_a2 = next(iter(self.loader_src2))
        init_batch_b = next(iter(self.loader_tgt))
        self.model.data_dependent_initialize(
            init_batch_a["image"].to(self.device),
            init_batch_a2["image"].to(self.device),
            init_batch_b["image"].to(self.device),
        )

        # ---- DDP wrap sub-modules ------------------------------------------
        if self.use_dist:
            self.model.netG = DDP(self.model.netG, device_ids=[self.device])
            self.model.netD = DDP(self.model.netD, device_ids=[self.device])
            self.model.netF = DDP(
                self.model.netF, device_ids=[self.device],
                find_unused_parameters=True,
            )
            if self.model.lambda_SB > 0.0:
                self.model.netE = DDP(self.model.netE, device_ids=[self.device])

        # ---- LR schedulers --------------------------------------------------
        n_epochs = train_cfg.get("n_epochs", 100)
        n_epochs_decay = train_cfg.get("n_epochs_decay", 100)
        lr_lambda = _get_lr_lambda(n_epochs, n_epochs_decay)
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.model.optimizer_G, lr_lambda)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.model.optimizer_D, lr_lambda)
        self.scheduler_E = torch.optim.lr_scheduler.LambdaLR(self.model.optimizer_E, lr_lambda)
        netF_for_attr = self.model.netF.module if self.use_dist else self.model.netF
        if netF_for_attr.use_mlp:
            self.scheduler_F = torch.optim.lr_scheduler.LambdaLR(self.model.optimizer_F, lr_lambda)

        self.total_epochs = n_epochs + n_epochs_decay
        self.max_iters = train_cfg.get("max_iters_per_epoch", 0)
        self.loader_len = min(len(self.loader_src), len(self.loader_src2), len(self.loader_tgt))

        if self.is_main_process():
            print(f"Training UNSB: {source_field} -> {target_field}, "
                  f"{self.total_epochs} epochs, {self.loader_len} iters/epoch, T={self.model.num_timesteps}"
                  + (f" (capped at {self.max_iters})" if self.max_iters else "")
                  + (f" (DDP: {self.world_size} GPUs)" if self.use_dist else ""))

    def train(self):
        display_freq = self.cfg["training"].get("display_freq", 0)

        for epoch in range(self.start_epoch, self.total_epochs):
            self.model.train()
            if self.use_dist:
                self.sampler_src.set_epoch(epoch)
                self.sampler_src2.set_epoch(epoch)
                self.sampler_tgt.set_epoch(epoch)
                iterable = _iter_three_loaders(self.loader_src, self.loader_src2, self.loader_tgt)
            else:
                # For non-DDP, we can use UnpairedDataLoader-like logic but with 3 datasets
                iterable = _iter_three_loaders(self.loader_src, self.loader_src2, self.loader_tgt)

            pbar = tqdm(iterable, desc=f"Epoch {epoch+1}/{self.total_epochs}",
                        total=self.loader_len, disable=not self.is_main_process())

            for i, (batch_a, batch_a2, batch_b) in enumerate(pbar):
                if self.max_iters and i >= self.max_iters:
                    break
                real_A = batch_a["image"].to(self.device)
                real_A2 = batch_a2["image"].to(self.device)
                real_B = batch_b["image"].to(self.device)

                with self.autocast_context():
                    losses = self.model.optimize_parameters(
                        real_A, real_A2, real_B, max_grad_norm=self.max_grad_norm
                    )

                self.global_step += 1
                if self.is_main_process() and display_freq > 0 and self.global_step % display_freq == 0:
                    with torch.no_grad():
                        visuals = {
                            'real_A': real_A,
                            'real_B': real_B,
                            'fake_B': self.model.fake_B,
                        }
                        if hasattr(self.model, 'idt_B') and self.model.idt_B is not None:
                            visuals['idt_B'] = self.model.idt_B
                        save_training_visuals(visuals, self.visuals_dir, epoch=epoch+1, slice_axis=self.cfg["data"].get("slice_axis", 1))

                if self.is_main_process():
                    pbar.set_postfix(G=f"{losses['loss_G']:.4f}", D=f"{losses['loss_D']:.4f}", E=f"{losses['loss_E']:.4f}")

            self.scheduler_G.step()
            self.scheduler_D.step()
            self.scheduler_E.step()
            if hasattr(self, 'scheduler_F'):
                self.scheduler_F.step()

            if self.is_main_process() and (epoch + 1) % self.cfg["training"].get("save_every", 10) == 0:
                ckpt = self.weights_dir / f"checkpoint_epoch{epoch+1}.pth"
                save_state = {
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optimizer_G": self.model.optimizer_G.state_dict(),
                    "optimizer_D": self.model.optimizer_D.state_dict(),
                    "scheduler_G": self.scheduler_G.state_dict(),
                    "scheduler_D": self.scheduler_D.state_dict(),
                }
                if hasattr(self.model, 'optimizer_E'):
                    save_state["optimizer_E"] = self.model.optimizer_E.state_dict()
                if hasattr(self, 'scheduler_E'):
                    save_state["scheduler_E"] = self.scheduler_E.state_dict()
                if hasattr(self.model, 'optimizer_F'):
                    save_state["optimizer_F"] = self.model.optimizer_F.state_dict()
                if hasattr(self, 'scheduler_F'):
                    save_state["scheduler_F"] = self.scheduler_F.state_dict()
                torch.save(save_state, ckpt)
                print(f"Saved: {ckpt}")

        if self.is_main_process():
            net_G = self.model.netG.module if self.use_dist else self.model.netG
            torch.save(net_G.state_dict(), self.weights_dir / "generator_final.pth")
            print(f"UNSB training complete. Saved to: {self.output_dir}")

__all__ = [
    "UNSBPretrainTrainer",
]
