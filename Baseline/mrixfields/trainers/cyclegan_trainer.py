"""CycleGAN pretrain + finetune trainers with native DDP and AMP."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..models.cyclegan_model import CycleGANModel
from ..utils.visualizer import save_training_visuals
from ..utils.evaluator import Evaluator
from ..utils.train_logger import TrainLogger
from ..models.net_init import init_net
from ..models.networks_cyclegan import ResnetGenerator
from ..data import UnpairedMONAI2DDataset
from ..data.dataset import UnpairedMRIDataset, PairedMRIDataset
from ..data.cached_dataset import CachedUnpairedDataset, CachedPairedDataset
from ..data.unpaired_loader import UnpairedDataLoader
from ..data.utils import ABBR_TO_SPLIT as SPLIT_MAP
from ..losses.perceptual import PerceptualLoss
from ..losses.structure import SSIMLoss, StructureLoss
from ..inference.base_inference import extract_generator_state


def _get_lr_lambda(n_epochs: int, n_epochs_decay: int):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
    return lambda_rule


def _iter_unpaired(loader_src, loader_tgt):
    iter_src = iter(loader_src)
    iter_tgt = iter(loader_tgt)
    for _ in range(min(len(loader_src), len(loader_tgt))):
        yield next(iter_src), next(iter_tgt)


class CycleGANPretrainTrainer(BaseTrainer):
    """CycleGAN pretraining with unpaired data."""

    def setup(self):
        model_cfg = self.cfg["model"]
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]

        # ---- Model ---------------------------------------------------------
        self.model = CycleGANModel(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            ngf=model_cfg.get("ngf", 64),
            ndf=model_cfg.get("ndf", 64),
            n_blocks=model_cfg.get("n_blocks", 9),
            lambda_cycle=model_cfg.get("lambda_cycle", 10.0),
            lambda_idt=model_cfg.get("lambda_idt", 0.5),
            pool_size=model_cfg.get("pool_size", 50),
            gan_mode=train_cfg.get("gan_mode", model_cfg.get("gan_mode", "lsgan")),
            use_spectral_norm=model_cfg.get("use_spectral_norm", False),
            lr=train_cfg.get("lr", 0.0002),
            lr_D=train_cfg.get("lr_D", train_cfg.get("lr", 0.0002)),
            beta1=train_cfg.get("beta1", 0.5),
            real_label=train_cfg.get("real_label", 1.0),
            fake_label=train_cfg.get("fake_label", 0.0),
        ).to(self.device)
        self.model.init_weights()
        self.model.setup_optimizers()

        # ---- Datasets -------------------------------------------------------
        modality = data_cfg["modalities"][0]
        domain_a = data_cfg.get("domain_a") or data_cfg.get("source_field")
        domain_b = data_cfg.get("domain_b") or data_cfg.get("target_field")
        crop_size_raw = data_cfg.get("crop_size")
        crop_size = tuple(crop_size_raw) if crop_size_raw else None
        preprocessed_dir = data_cfg.get("preprocessed_dir")
        split = train_cfg.get("split", "retro_train")
        batch_size = train_cfg.get("batch_size", 4)
        num_workers = train_cfg.get("num_workers", 4)

        if data_cfg.get("use_monai"):
            split_full = SPLIT_MAP.get(split, split)
            slice_range = tuple(data_cfg.get("slice_range", [120, 210]))
            cache_dir = data_cfg.get("cache_dir")
            norm_flag = data_cfg.get("normalize", False)
            norm_nonzero = data_cfg.get("norm_nonzero", False)
            augment = data_cfg.get("augment")
            ds_kw = dict(
                data_root=data_cfg["data_dir"], split=split_full,
                modalities=data_cfg["modalities"],
                field_strength=domain_a,
                slice_range=slice_range,
                slice_axis=data_cfg.get("slice_axis", 2),
                crop_size=crop_size,
                normalize=norm_flag, norm_nonzero=norm_nonzero,
                augment=augment, cache_dir=cache_dir,
                spacing=data_cfg.get("spacing"),
                clip_percentile=data_cfg.get("clip_percentile"),
                auto_slice_range=data_cfg.get("auto_slice_range", False),
                auto_slice_min_frac=data_cfg.get("auto_slice_min_frac", 0.10),
                num_workers=num_workers,
            )
            dataset_a = UnpairedMONAI2DDataset(**ds_kw)
            ds_kw["field_strength"] = domain_b
            dataset_b = UnpairedMONAI2DDataset(**ds_kw)
        elif preprocessed_dir:
            dataset_a = CachedUnpairedDataset(
                preprocessed_dir=preprocessed_dir, split=split,
                modality=modality, field_strength=domain_a, crop_size=crop_size,
            )
            dataset_b = CachedUnpairedDataset(
                preprocessed_dir=preprocessed_dir, split=split,
                modality=modality, field_strength=domain_b, crop_size=crop_size,
            )
        else:
            split_full = SPLIT_MAP.get(split, split)
            dataset_a = UnpairedMRIDataset(
                data_root=data_cfg["data_dir"], split=split_full,
                modality=modality, field_strength=domain_a, crop_size=crop_size,
            )
            dataset_b = UnpairedMRIDataset(
                data_root=data_cfg["data_dir"], split=split_full,
                modality=modality, field_strength=domain_b, crop_size=crop_size,
            )

        if self.use_dist:
            sampler_a = DistributedSampler(dataset_a, shuffle=True)
            sampler_b = DistributedSampler(dataset_b, shuffle=True)
            self.loader_a = DataLoader(
                dataset_a, batch_size=batch_size, sampler=sampler_a,
                num_workers=num_workers, drop_last=True, pin_memory=True,
            )
            self.loader_b = DataLoader(
                dataset_b, batch_size=batch_size, sampler=sampler_b,
                num_workers=num_workers, drop_last=True, pin_memory=True,
            )
            self.sampler_a = sampler_a
            self.sampler_b = sampler_b
        else:
            self.loader_a = DataLoader(
                dataset_a, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, drop_last=True, pin_memory=True,
            )
            self.loader_b = DataLoader(
                dataset_b, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, drop_last=True, pin_memory=True,
            )
            self.sampler_a = None
            self.sampler_b = None

        # ---- DDP wrap sub-modules -------------------------------------------
        if self.use_dist:
            self.model.netG_AB = DDP(self.model.netG_AB, device_ids=[self.device])
            self.model.netG_BA = DDP(self.model.netG_BA, device_ids=[self.device])
            self.model.netD_A = DDP(self.model.netD_A, device_ids=[self.device])
            self.model.netD_B = DDP(self.model.netD_B, device_ids=[self.device])

        # ---- LR schedulers --------------------------------------------------
        n_epochs = train_cfg.get("n_epochs", 100)
        n_epochs_decay = train_cfg.get("n_epochs_decay", 100)
        lr_lambda = _get_lr_lambda(n_epochs, n_epochs_decay)
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.model.optimizer_G, lr_lambda)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.model.optimizer_D, lr_lambda)

        self.total_epochs = n_epochs + n_epochs_decay
        self.max_iters = train_cfg.get("max_iters_per_epoch", 0)
        self.loader_len = min(len(self.loader_a), len(self.loader_b))

        self._val_every = train_cfg.get("val_every", -1)
        self._evaluator = None
        self._train_logger = None
        self._es_patience = train_cfg.get("early_stop_patience", 10)
        self._es_min_epochs = train_cfg.get("early_stop_min_epochs", 50)

        if self._val_every > 0:
            if self.is_main_process():
                self._train_logger = TrainLogger(
                    self.output_dir / "logs",
                    resume=self.cfg.get("_runtime", {}).get("resume", False),
                    experiment=self.cfg.get("task_name", ""))
            self._evaluator = Evaluator(
                data_root=data_cfg.get("data_dir"),
                split=self.cfg.get("finetune", {}).get("split", "pro_train"),
                source_fields=[domain_a],
                target_field=domain_b,
                modalities=[modality],
                device=self.device,
                metric_names=self.cfg.get("evaluation", {}).get("metrics"),
                slice_axis=data_cfg.get("slice_axis", 2),
                crop_size=crop_size,
                batch_size=32,
                slice_range=self.cfg.get("evaluation", {}).get("slice_range"),
                is_3d=False,
            )
            baseline = self._evaluator.run(None)
            self._save_baseline_metrics(baseline)

        self.model.netG = self.model.netG_AB

        if self.is_main_process():
            print(f"Training CycleGAN: {domain_a} <-> {domain_b}, "
                  f"{self.total_epochs} epochs, {self.loader_len} iters/epoch"
                  + (f" (capped at {self.max_iters})" if self.max_iters else "")
                  + (f" (DDP: {self.world_size} GPUs)" if self.use_dist else ""))

    def train(self):
        display_freq = self.cfg["training"].get("display_freq", 0)

        for epoch in range(self.start_epoch, self.total_epochs):
            self.model.train()
            if self.use_dist:
                self.sampler_a.set_epoch(epoch)
                self.sampler_b.set_epoch(epoch)
                iterable = _iter_unpaired(self.loader_a, self.loader_b)
            else:
                iterable = UnpairedDataLoader(
                    self.loader_a.dataset, self.loader_b.dataset,
                    batch_size=self.loader_a.batch_size,
                    num_workers=self.loader_a.num_workers,
                )

            pbar = tqdm(iterable, desc=f"Epoch {epoch+1}/{self.total_epochs}",
                        total=self.loader_len, disable=not self.is_main_process())
            for i, (batch_a, batch_b) in enumerate(pbar):
                if self.max_iters and i >= self.max_iters:
                    break
                real_A = batch_a["image"].to(self.device)
                real_B = batch_b["image"].to(self.device)

                with self.autocast_context():
                    losses = self.model.optimize_parameters(
                        real_A, real_B, max_grad_norm=self.max_grad_norm
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
                        save_training_visuals(visuals, self.visuals_dir, epoch=epoch+1, slice_axis=self.cfg["data"].get("slice_axis", 1), batch_info=batch_a, batch_info_b=batch_b)

                if self.is_main_process():
                    pbar.set_postfix(G=f"{losses['loss_G']:.4f}", D=f"{losses['loss_D']:.4f}")

            self.scheduler_G.step()
            self.scheduler_D.step()

            if self._train_logger and self.is_main_process():
                self._train_logger.log_train(
                    epoch + 1, self.global_step,
                    losses.get("loss_D", 0), losses.get("loss_G", 0),
                    self.model.optimizer_G.param_groups[0]["lr"],
                    self.model.optimizer_D.param_groups[0]["lr"])

            val_save_every = self.cfg["training"].get("save_every", 10)
            if self._validate_and_early_stop(epoch, val_save_every):
                break

            if self.is_main_process() and (epoch + 1) % val_save_every == 0:
                ckpt = self.weights_dir / f"checkpoint_epoch{epoch+1}.pth"
                save_state = {
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optimizer_G": self.model.optimizer_G.state_dict(),
                    "optimizer_D": self.model.optimizer_D.state_dict(),
                    "scheduler_G": self.scheduler_G.state_dict(),
                    "scheduler_D": self.scheduler_D.state_dict(),
                }
                if hasattr(self.model, 'optimizer_F'):
                    save_state["optimizer_F"] = self.model.optimizer_F.state_dict()
                if hasattr(self, 'scheduler_F'):
                    save_state["scheduler_F"] = self.scheduler_F.state_dict()
                torch.save(save_state, ckpt)
                print(f"Saved: {ckpt}")

        if self.is_main_process():
            net_G_AB = self.model.netG_AB.module if self.use_dist else self.model.netG_AB
            net_G_BA = self.model.netG_BA.module if self.use_dist else self.model.netG_BA
            torch.save(net_G_AB.state_dict(), self.weights_dir / "generator_AB_final.pth")
            torch.save(net_G_BA.state_dict(), self.weights_dir / "generator_BA_final.pth")
            torch.save(net_G_AB.state_dict(), self.weights_dir / "generator_final.pth")
            print(f"CycleGAN training complete. Saved to: {self.output_dir}")
        if self._train_logger:
            self._train_logger.close()

    def _generator_config(self):
        return {"type": "ResnetGenerator", "kwargs": dict(ngf=64, n_blocks=9)}

    def _checkpoint_metadata(self):
        return {
            "task_name": self.cfg.get("task_name", ""),
            "method": self.cfg.get("method", ""),
            "mode": self.cfg.get("_runtime", {}).get("mode", ""),
            "config_actual": self.cfg,
        }


class CycleGANFineTuneTrainer(BaseTrainer):
    """Fine-tune pretrained CycleGAN generator with paired data."""

    def setup(self):
        model_cfg = self.cfg["model"]
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]
        pretrain_cfg = self.cfg.get("pretrain", {})
        finetune_cfg = self.cfg.get("finetune", {})

        # ---- Generator ------------------------------------------------------
        self.generator = ResnetGenerator(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            ngf=model_cfg.get("ngf", 64),
            n_blocks=model_cfg.get("n_blocks", 9),
        ).to(self.device)

        ckpt_path = pretrain_cfg.get("checkpoint")
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            gen_state = extract_generator_state(state_dict)
            self.generator.load_state_dict(gen_state)
            if self.is_main_process():
                print(f"Loaded pretrained generator from: {ckpt_path}")
        else:
            init_net(self.generator)
            if self.is_main_process():
                print("No pretrained checkpoint, training finetune from scratch")

        if self.use_dist:
            self.generator = DDP(self.generator, device_ids=[self.device])

        self.optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=train_cfg.get("lr", 5e-5),
            betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
        )
        self.criterion_l1 = nn.L1Loss()
        self.criterion_lpips = PerceptualLoss().to(self.device)
        self.criterion_ssim = SSIMLoss().to(self.device)

        self.lambda_l1 = finetune_cfg.get("loss_l1", 1.0)
        self.lambda_lpips = finetune_cfg.get("loss_lpips", 0.1)
        self.lambda_ssim = finetune_cfg.get("loss_ssim", 0.0)
        self.lambda_edge = finetune_cfg.get("loss_edge", 0.0)
        if self.lambda_edge > 0:
            self.criterion_edge = StructureLoss(ssim_weight=0.0, edge_weight=1.0).to(self.device)

        modality = data_cfg["modalities"][0]
        crop_size_raw = data_cfg.get("crop_size")
        crop_size = tuple(crop_size_raw) if crop_size_raw else None
        preprocessed_dir = data_cfg.get("preprocessed_dir")
        split = train_cfg.get("split") or data_cfg.get("split", "pro_train")
        batch_size = train_cfg.get("batch_size", 4)
        num_workers = train_cfg.get("num_workers", 4)

        if preprocessed_dir:
            dataset = CachedPairedDataset(
                preprocessed_dir=preprocessed_dir, split=split,
                modality=modality, source_field=data_cfg["source_field"],
                target_field=data_cfg["target_field"], crop_size=crop_size,
            )
        else:
            split_full = SPLIT_MAP.get(split, split)
            dataset = PairedMRIDataset(
                data_root=data_cfg["data_dir"], split=split_full,
                modality=modality, source_field=data_cfg["source_field"],
                target_field=data_cfg["target_field"], crop_size=crop_size,
            )

        if self.use_dist:
            sampler = DistributedSampler(dataset, shuffle=True)
            self.loader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=True,
            )
            self.sampler = sampler
        else:
            self.loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            )
            self.sampler = None

        self.epochs = train_cfg.get("epochs", 50)
        self.max_iters = train_cfg.get("max_iters_per_epoch", 0) or finetune_cfg.get("max_iters_per_epoch", 0)

        if self.is_main_process():
            print(f"CycleGAN fine-tuning: {data_cfg['source_field']} -> {data_cfg['target_field']}, "
                  f"{self.epochs} epochs, {len(self.loader)} iters/epoch"
                  + (f" (capped at {self.max_iters})" if self.max_iters else "")
                  + (f" (DDP: {self.world_size} GPUs)" if self.use_dist else ""))

    def train(self):
        display_freq = self.cfg["training"].get("display_freq", 0)

        for epoch in range(self.start_epoch, self.epochs):
            self.generator.train()
            if self.use_dist and self.sampler is not None:
                self.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}",
                        disable=not self.is_main_process())

            for i, batch in enumerate(pbar):
                if self.max_iters and i >= self.max_iters:
                    break
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)

                self.optimizer.zero_grad()
                with self.autocast_context():
                    pred = self.generator(source)
                    loss = self.lambda_l1 * self.criterion_l1(pred, target)
                    loss = loss + self.lambda_lpips * self.criterion_lpips(pred, target)
                    if self.lambda_ssim > 0:
                        loss = loss + self.lambda_ssim * self.criterion_ssim(pred, target)
                    if self.lambda_edge > 0:
                        loss = loss + self.lambda_edge * self.criterion_edge(pred, target)

                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.global_step += 1
                if self.is_main_process() and display_freq > 0 and self.global_step % display_freq == 0:
                    with torch.no_grad():
                        visuals = {
                            'real_A': source,
                            'real_B': target,
                            'fake_B': pred,
                        }
                        save_training_visuals(visuals, self.visuals_dir, epoch=epoch+1, slice_axis=self.cfg["data"].get("slice_axis", 1))

                epoch_loss += loss.item()
                if self.is_main_process():
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            if self.is_main_process():
                avg = epoch_loss / max(len(self.loader), 1)
                print(f"Epoch {epoch+1}: avg_loss={avg:.6f}")

            if self.is_main_process() and (epoch + 1) % self.cfg["training"].get("save_every", 10) == 0:
                save_state = {
                    "epoch": epoch + 1,
                    "generator": self.generator.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                if hasattr(self, "scheduler"):
                    save_state["scheduler"] = self.scheduler.state_dict()
                self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pth", save_state)

        if self.is_main_process():
            net_G = self.generator.module if self.use_dist else self.generator
            torch.save(net_G.state_dict(), self.weights_dir / "generator_final.pth")
            print(f"CycleGAN fine-tuning complete. Saved to: {self.output_dir}")


__all__ = [
    "CycleGANPretrainTrainer",
    "CycleGANFineTuneTrainer",
]
