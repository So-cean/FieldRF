"""StarGAN v2 trainers aligned with the unified MRIxFields pipeline.

The implementation keeps the official StarGAN v2 four-network objective,
while using the same MONAI data, validation, visualisation, checkpoint, resume,
and native DDP conventions as the CUT-family trainers.
"""

import itertools
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..data import MultiDomainMONAI2DDataset, PairedMONAI2DDataset
from ..data.cached_dataset import CachedMultiDomainDataset, CachedPairedDataset
from ..data.dataset import MultiDomainMRIDataset, PairedMRIDataset
from ..data.utils import (
    ABBR_TO_SPLIT as SPLIT_MAP,
    FIELD_TO_DOMAIN,
    get_joint_domain,
)
from ..losses.perceptual import PerceptualLoss
from ..losses.structure import SSIMLoss, StructureLoss
from ..models.starganv2_model import (
    build_stargan_v2,
    compute_d_loss,
    compute_g_loss,
    moving_average,
)
from ..utils.evaluator import Evaluator
from ..utils.train_logger import TrainLogger
from ..utils.visualizer import save_training_visuals


def _domain_index(modalities, modality, field_strength):
    """Return the label convention used by all StarGAN datasets."""
    if len(modalities) > 1:
        return get_joint_domain(modality, field_strength)
    return FIELD_TO_DOMAIN[field_strength]


class _PairedWithDomains(Dataset):
    """Attach StarGAN domain labels to a paired dataset."""

    def __init__(self, dataset, src_domain, tgt_domain,
                 src_modality, tgt_modality, src_field, tgt_field):
        self.dataset = dataset
        self.src_domain = src_domain
        self.tgt_domain = tgt_domain
        self.src_modality = src_modality
        self.tgt_modality = tgt_modality
        self.src_field = src_field
        self.tgt_field = tgt_field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["src_domain"] = self.src_domain
        item["tgt_domain"] = self.tgt_domain
        item["src_modality"] = self.src_modality
        item["tgt_modality"] = self.tgt_modality
        item["source_field"] = self.src_field
        item["target_field"] = self.tgt_field
        return item


class _StarGANValidationAdapter(nn.Module):
    """Expose deterministic multi-channel ``forward(x)`` for ``Evaluator``."""

    def __init__(self, generator, mapping_network, modalities, target_field,
                 latent_dim):
        super().__init__()
        self.generator = generator
        self.mapping_network = mapping_network
        self.modalities = list(modalities)
        self.target_field = target_field
        self.latent_dim = latent_dim

    def forward(self, x):
        outputs = []
        z = x.new_zeros((x.size(0), self.latent_dim))
        for channel, modality in enumerate(self.modalities):
            domain = _domain_index(self.modalities, modality, self.target_field)
            y = torch.full(
                (x.size(0),), domain, dtype=torch.long, device=x.device)
            style = self.mapping_network(z, y)
            outputs.append(self.generator(x[:, channel:channel + 1], style))
        return torch.cat(outputs, dim=1)


class _StarGANTrainerMixin:
    """Network, validation, normalisation, and checkpoint helpers."""

    def _network_kwargs(self):
        model_cfg = self.cfg["model"]
        data_cfg = self.cfg["data"]
        has_norm = data_cfg.get("normalize", False) or data_cfg.get("norm_nonzero", False)
        return {
            "img_size": model_cfg.get("img_size", 512),
            "style_dim": model_cfg.get("style_dim", 64),
            "latent_dim": model_cfg.get("latent_dim", 16),
            "num_domains": model_cfg.get("num_domains", 5),
            "max_conv_dim": model_cfg.get("max_conv_dim", 512),
            "input_nc": model_cfg.get("input_nc", 1),
            "use_tanh": not has_norm,
        }

    def _build_networks(self):
        self.nets, self.nets_ema = build_stargan_v2(**self._network_kwargs())
        for net in self.nets.values():
            net.to(self.device)
        for net in self.nets_ema.values():
            net.to(self.device).eval()

    def _wrap_ddp(self):
        if self.use_dist:
            for name, net in self.nets.items():
                self.nets[name] = DDP(net, device_ids=[self.device])

    def _unwrap_nets(self):
        return {
            name: (net.module if isinstance(net, DDP) else net)
            for name, net in self.nets.items()
        }

    def _generator_config(self):
        return {"kwargs": self._network_kwargs()}

    def _get_generator(self):
        return self._validation_adapter

    def _validate_domains(self, domain_labels):
        model_domains = self._network_kwargs()["num_domains"]
        required = max(domain_labels) + 1
        if model_domains < required:
            raise ValueError(
                f"model.num_domains={model_domains} cannot represent dataset label "
                f"{required - 1}; use at least {required} domains.")

    def _build_checkpoint_state(self, epoch, metrics=None, include_optimizer=True):
        nets = self._unwrap_nets()
        flat_model = {
            f"{name}.{key}": value
            for name, net in nets.items()
            for key, value in net.state_dict().items()
        }
        state = {
            "epoch": epoch,
            "step": self.global_step,
            "global_step": self.global_step,
            "model": flat_model,
            "generator": nets["generator"].state_dict(),
            "ema_state": self.nets_ema["generator"].state_dict(),
            "generator_config": self._generator_config(),
            "nets": {name: net.state_dict() for name, net in nets.items()},
            "nets_ema": {
                name: net.state_dict() for name, net in self.nets_ema.items()
            },
            "es_best": self._es_best,
            "es_no_improve": self._es_no_improve,
            "metrics": metrics,
            "_metadata": self._checkpoint_metadata(),
        }
        if include_optimizer:
            if hasattr(self, "optims"):
                state["optims"] = {
                    name: optim.state_dict() for name, optim in self.optims.items()
                }
                for name, optim in self.optims.items():
                    state[f"optimizer_{name}"] = optim.state_dict()
            if hasattr(self, "optimizer"):
                state["optimizer"] = self.optimizer.state_dict()
            if hasattr(self, "schedulers"):
                state["schedulers"] = {
                    name: scheduler.state_dict()
                    for name, scheduler in self.schedulers.items()
                }
                for name, scheduler in self.schedulers.items():
                    state[f"scheduler_{name}"] = scheduler.state_dict()
            if hasattr(self, "scheduler"):
                state["scheduler"] = self.scheduler.state_dict()
        return state

    def _load_network_state(self, state, load_optimizers=False):
        nets = self._unwrap_nets()
        if "nets" in state:
            for name, net in nets.items():
                if name in state["nets"]:
                    net.load_state_dict(state["nets"][name])
        elif "generator" in state:
            nets["generator"].load_state_dict(state["generator"])

        if "nets_ema" in state:
            for name, net in self.nets_ema.items():
                if name in state["nets_ema"]:
                    net.load_state_dict(state["nets_ema"][name])
        elif state.get("ema_state"):
            self.nets_ema["generator"].load_state_dict(state["ema_state"])

        if load_optimizers:
            if hasattr(self, "optims"):
                for name, optim in self.optims.items():
                    optim_state = state.get("optims", {}).get(name)
                    optim_state = optim_state or state.get(f"optimizer_{name}")
                    if optim_state:
                        optim.load_state_dict(optim_state)
            if hasattr(self, "optimizer") and state.get("optimizer"):
                self.optimizer.load_state_dict(state["optimizer"])
            if hasattr(self, "schedulers"):
                for name, scheduler in self.schedulers.items():
                    scheduler_state = state.get("schedulers", {}).get(name)
                    scheduler_state = scheduler_state or state.get(f"scheduler_{name}")
                    if scheduler_state:
                        scheduler.load_state_dict(scheduler_state)
            if hasattr(self, "scheduler") and state.get("scheduler"):
                self.scheduler.load_state_dict(state["scheduler"])

    def _try_resume(self):
        checkpoint = self.find_latest_checkpoint()
        if checkpoint is None:
            if self.is_main_process():
                print("No checkpoint found to resume.")
            return
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self._load_network_state(state, load_optimizers=True)
        self.start_epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", state.get("step", 0))
        self._es_best = state.get("es_best", {})
        self._es_no_improve = state.get("es_no_improve", 0)
        if self.is_main_process():
            print(f"Resumed from {checkpoint}: epoch={self.start_epoch}, "
                  f"step={self.global_step}")

    def _load_pretrained(self, checkpoint):
        if not checkpoint:
            if self.is_main_process():
                print("No pretrained checkpoint; training StarGAN v2 from scratch.")
            return
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self._load_network_state(state, load_optimizers=False)
        # Fine-tuning starts from the inference (EMA) weights.
        nets = self._unwrap_nets()
        for name in ("generator", "mapping_network", "style_encoder"):
            if name in self.nets_ema:
                nets[name].load_state_dict(self.nets_ema[name].state_dict())
        if self.is_main_process():
            print(f"Loaded pretrained StarGAN v2 from: {checkpoint}")

    def _write_norm_stats(self, domain_stats):
        self._domain_stats = domain_stats or {}
        norm_stats = {}
        for domain, stats in self._domain_stats.items():
            modality, field = self._domain_pairs[int(domain)]
            norm_stats[f"{modality}/{field}"] = dict(stats)
        self._norm_stats = norm_stats
        if self.is_main_process() and norm_stats:
            path = self.output_dir / "norm_stats.yaml"
            tmp = path.with_suffix(".yaml.tmp")
            tmp.write_text(yaml.safe_dump(norm_stats, sort_keys=True))
            tmp.replace(path)

    def _load_norm_stats(self, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        domain_stats = {}
        for domain, (modality, field) in self._domain_pairs.items():
            key = f"{modality}/{field}"
            if key in raw:
                domain_stats[domain] = raw[key]
        self._write_norm_stats(domain_stats)

    def _build_field_stats(self):
        field_stats = {}
        for field in self.domains:
            entries = []
            for modality in self.modalities:
                domain = _domain_index(self.modalities, modality, field)
                if domain not in self._domain_stats:
                    entries = []
                    break
                entries.append(self._domain_stats[domain])
            if entries:
                field_stats[field] = {
                    key: [entry[key] for entry in entries]
                    for key in ("mu", "sigma", "bg_norm")
                }
        return field_stats

    def _init_validation(self, field_stats):
        train_cfg = self.cfg["training"]
        eval_cfg = self.cfg.get("evaluation", {})
        self._val_every = train_cfg.get("val_every", -1)
        self._es_patience = train_cfg.get("early_stop_patience", 10)
        self._es_min_epochs = train_cfg.get("early_stop_min_epochs", 0)
        self._evaluator = None

        target_field = eval_cfg.get("target_field", self.domains[-1])
        source_fields = eval_cfg.get(
            "source_fields", [field for field in self.domains if field != target_field])
        self._validation_adapter = _StarGANValidationAdapter(
            self.nets_ema["generator"], self.nets_ema["mapping_network"],
            self.modalities, target_field, self.latent_dim).to(self.device).eval()

        if self._val_every <= 0:
            return
        target_stats = field_stats.get(target_field, {})
        self._evaluator = Evaluator(
            data_root=self.cfg["data"]["data_dir"],
            split=eval_cfg.get("split", self.cfg.get("finetune", {}).get("split", "pro_train")),
            source_fields=source_fields,
            target_field=target_field,
            field_stats=field_stats,
            modalities=self.modalities,
            device=self.device,
            metric_names=eval_cfg.get("metrics"),
            slice_axis=self.cfg["data"].get("slice_axis", 2),
            crop_size=tuple(self.cfg["data"]["crop_size"])
            if self.cfg["data"].get("crop_size") else None,
            batch_size=self.cfg.get("inference", {}).get("batch_size", 16),
            mu_tgt=target_stats.get("mu"),
            sigma_tgt=target_stats.get("sigma"),
            slice_range=eval_cfg.get("slice_range"),
        )
        baseline = self._evaluator.run(None)
        self._save_baseline_metrics(baseline)
        if self.normalize and field_stats:
            self._save_baseline_metrics(
                self._evaluator.run(None, force_norm=True), key="baseline_normed")

    def _denormalize_batch(self, tensor, domains):
        if not self.normalize or not self._domain_stats:
            return tensor
        output = []
        for sample, domain in zip(tensor, domains):
            stats = self._domain_stats[int(domain)]
            output.append(sample * float(stats["sigma"]) + float(stats["mu"]))
        return torch.stack(output)

    def _save_stargan_visual(self, real, fake, reference, source_domains,
                             target_domains, batch, epoch, prefix="epoch"):
        source_domain = int(source_domains[0])
        target_domain = int(target_domains[0])
        src_mod, src_field = self._domain_pairs[source_domain]
        tgt_mod, tgt_field = self._domain_pairs[target_domain]
        visuals = {
            "real_A": self._denormalize_batch(real, source_domains).detach(),
            "fake_B": self._denormalize_batch(fake, target_domains).detach(),
            "real_B": self._denormalize_batch(reference, target_domains).detach(),
        }
        batch_b = {
            "filename": batch.get("ref_filename", [""]),
            "slice_index": batch.get("ref_slice_index", [""]),
            "field_strength": batch.get("ref_field_strength", [tgt_field]),
        }
        save_training_visuals(
            visuals, self.visuals_dir, epoch=epoch, prefix=prefix,
            batch_info=batch, batch_info_b=batch_b,
            visual_domains={
                "real_A": {"modality": src_mod, "field_strength": src_field},
                "fake_B": {"modality": tgt_mod, "field_strength": tgt_field},
                "real_B": {"modality": tgt_mod, "field_strength": tgt_field},
            },
        )

    def _update_ema(self):
        nets = self._unwrap_nets()
        for name in ("generator", "mapping_network", "style_encoder"):
            moving_average(nets[name], self.nets_ema[name], self.ema_beta)


class StarGANv2Trainer(_StarGANTrainerMixin, BaseTrainer):
    """Unpaired StarGAN v2 pretraining with MONAI and native DDP."""

    def setup(self):
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]
        self.modalities = list(data_cfg["modalities"])
        self.domains = list(self.cfg.get("domains", data_cfg.get("source_fields", [])))
        if not self.domains:
            raise ValueError("StarGAN v2 config requires a non-empty 'domains' list.")
        if self._network_kwargs()["input_nc"] != 1:
            raise ValueError("Joint-domain StarGAN v2 expects model.input_nc=1.")
        self.normalize = data_cfg.get("normalize", False) or data_cfg.get("norm_nonzero", False)
        self.latent_dim = self._network_kwargs()["latent_dim"]
        self._domain_pairs = {
            _domain_index(self.modalities, modality, field): (modality, field)
            for modality in self.modalities for field in self.domains
        }

        self._build_networks()
        dataset = self._build_dataset()
        domain_labels = getattr(dataset, "domain_labels", None)
        if domain_labels is None:
            domain_labels = list(getattr(dataset, "domain_files",
                                         getattr(dataset, "domain_samples", {})))
        self._validate_domains(domain_labels)
        self._write_norm_stats(getattr(dataset, "domain_stats", {}))

        batch_size = train_cfg.get("batch_size", 4)
        num_workers = train_cfg.get("num_workers", 4)
        if self.use_dist:
            self.sampler = DistributedSampler(dataset, shuffle=True)
            self.loader = DataLoader(
                dataset, batch_size=batch_size, sampler=self.sampler,
                num_workers=num_workers, drop_last=True, pin_memory=True)
        else:
            self.sampler = None
            self.loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, drop_last=True, pin_memory=True)

        self._wrap_ddp()
        nets = self._unwrap_nets()
        lr = train_cfg.get("lr", 1e-4)
        f_lr = train_cfg.get("f_lr", 1e-6)
        betas = (train_cfg.get("beta1", 0.0), train_cfg.get("beta2", 0.99))
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.optims = {
            "generator": torch.optim.Adam(
                nets["generator"].parameters(), lr=lr, betas=betas,
                weight_decay=weight_decay),
            "mapping_network": torch.optim.Adam(
                nets["mapping_network"].parameters(), lr=f_lr, betas=betas,
                weight_decay=weight_decay),
            "style_encoder": torch.optim.Adam(
                nets["style_encoder"].parameters(), lr=lr, betas=betas,
                weight_decay=weight_decay),
            "discriminator": torch.optim.Adam(
                nets["discriminator"].parameters(), lr=lr, betas=betas,
                weight_decay=weight_decay),
        }

        n_epochs = train_cfg.get("n_epochs", 50)
        n_epochs_decay = train_cfg.get("n_epochs_decay", 50)
        self.total_epochs = n_epochs + n_epochs_decay

        def lr_lambda(epoch):
            if epoch < n_epochs:
                return 1.0
            return 1.0 - min(epoch - n_epochs, n_epochs_decay) / max(1, n_epochs_decay)

        self.schedulers = {
            name: torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
            for name, optim in self.optims.items()
        }
        self.lambda_ds = train_cfg.get("lambda_ds", 1.0)
        self.ds_iter = train_cfg.get("ds_iter", 100000)
        self.ema_beta = train_cfg.get("ema_beta", 0.999)
        self.max_iters = train_cfg.get("max_iters_per_epoch", 0)
        self.display_freq = train_cfg.get("display_freq", 0)
        self.print_every = train_cfg.get("print_every", 100)
        self.save_every = train_cfg.get("save_every", 10)
        self._train_logger = TrainLogger(
            self.output_dir / "logs",
            resume=self.cfg.get("_runtime", {}).get("resume", False),
            experiment=self.cfg.get("task_name", ""),
        ) if self.is_main_process() else None
        self._init_validation(self._build_field_stats())

        if self.is_main_process():
            print(f"Training StarGAN v2: {len(self._domain_pairs)} joint domains, "
                  f"{self.total_epochs} epochs, {len(self.loader)} iters/epoch"
                  + (f" (capped at {self.max_iters})" if self.max_iters else "")
                  + (f" (DDP: {self.world_size} GPUs)" if self.use_dist else ""))

    def _build_dataset(self):
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]
        split = train_cfg.get("split", "retro_train")
        crop_size = tuple(data_cfg["crop_size"]) if data_cfg.get("crop_size") else None
        if data_cfg.get("use_monai"):
            return MultiDomainMONAI2DDataset(
                data_root=data_cfg["data_dir"],
                split=SPLIT_MAP.get(split, split),
                modalities=self.modalities,
                field_strengths=self.domains,
                slice_range=tuple(data_cfg.get("slice_range", [120, 210])),
                slice_axis=data_cfg.get("slice_axis", 2),
                crop_size=crop_size,
                normalize=data_cfg.get("normalize", False),
                norm_nonzero=data_cfg.get("norm_nonzero", False),
                augment=data_cfg.get("augment"),
                cache_dir=data_cfg.get("cache_dir"),
                spacing=data_cfg.get("spacing"),
                clip_percentile=data_cfg.get("clip_percentile"),
                auto_slice_range=data_cfg.get("auto_slice_range", False),
                auto_slice_min_frac=data_cfg.get("auto_slice_min_frac", 0.10),
                num_workers=self.cfg["training"].get("num_workers", 4),
            )
        if data_cfg.get("preprocessed_dir"):
            return CachedMultiDomainDataset(
                preprocessed_dir=data_cfg["preprocessed_dir"], split=split,
                modalities=self.modalities, field_strengths=self.domains,
                crop_size=crop_size)
        return MultiDomainMRIDataset(
            data_root=data_cfg["data_dir"], split=SPLIT_MAP.get(split, split),
            modalities=self.modalities, field_strengths=self.domains,
            crop_size=crop_size, slice_axis=data_cfg.get("slice_axis", 2))

    @staticmethod
    def _set_requires_grad(module, requires_grad):
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)

    def train(self):
        last_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.total_epochs):
            last_epoch = epoch + 1
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for net in self.nets.values():
                net.train()

            sums = {"loss_D": 0.0, "loss_G": 0.0}
            component_sums = {}
            iterations = 0
            iterable = tqdm(
                self.loader, desc=f"Epoch {epoch + 1}/{self.total_epochs}",
                disable=not self.is_main_process())
            for iteration, batch in enumerate(iterable):
                if self.max_iters and iteration >= self.max_iters:
                    break
                x_real = batch["image"].to(self.device)
                y_org = batch["domain"].to(self.device).long()
                x_ref = batch["ref_image"].to(self.device)
                x_ref2 = batch.get("ref_image2", batch["ref_image"]).to(self.device)
                y_trg = batch["ref_domain"].to(self.device).long()
                z_trg = torch.randn(x_real.size(0), self.latent_dim, device=self.device)
                z_trg2 = torch.randn_like(z_trg)
                lambda_ds = self.lambda_ds if self.ds_iter <= 0 else (
                    self.lambda_ds * max(0.0, 1.0 - self.global_step / self.ds_iter))
                latent_guided = self.global_step % 2 == 0

                self._set_requires_grad(self.nets["discriminator"], True)
                for optim in self.optims.values():
                    optim.zero_grad(set_to_none=True)
                with self.autocast_context():
                    d_loss, d_parts = compute_d_loss(
                        self.nets, x_real, y_org, y_trg,
                        z_trg=z_trg if latent_guided else None,
                        x_ref=None if latent_guided else x_ref,
                        lambda_reg=self.cfg["training"].get("lambda_reg", 1.0))
                d_loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.nets["discriminator"].parameters(), self.max_grad_norm)
                self.optims["discriminator"].step()

                self._set_requires_grad(self.nets["discriminator"], False)
                for name in ("generator", "mapping_network", "style_encoder"):
                    self.optims[name].zero_grad(set_to_none=True)
                x_real = x_real.detach()
                with self.autocast_context():
                    g_loss, g_parts = compute_g_loss(
                        self.nets, x_real, y_org, y_trg,
                        z_trgs=[z_trg, z_trg2] if latent_guided else None,
                        x_refs=None if latent_guided else [x_ref, x_ref2],
                        lambda_sty=self.cfg["training"].get("lambda_sty", 1.0),
                        lambda_ds=lambda_ds,
                        lambda_cyc=self.cfg["training"].get("lambda_cyc", 1.0))
                g_loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.nets["generator"].parameters(),
                            self.nets["mapping_network"].parameters(),
                            self.nets["style_encoder"].parameters()),
                        self.max_grad_norm)
                for name in ("generator", "mapping_network", "style_encoder"):
                    self.optims[name].step()
                self._set_requires_grad(self.nets["discriminator"], True)
                self._update_ema()

                self.global_step += 1
                iterations += 1
                sums["loss_D"] += d_loss.item()
                sums["loss_G"] += g_loss.item()
                for name, value in {**d_parts, **g_parts}.items():
                    component_sums[name] = component_sums.get(name, 0.0) + value
                if self.is_main_process():
                    iterable.set_postfix(D=f"{d_loss.item():.4f}",
                                         G=f"{g_loss.item():.4f}")
                    if self.print_every and self.global_step % self.print_every == 0:
                        print(f"  step={self.global_step} D={d_loss.item():.4f} "
                              f"G={g_loss.item():.4f} lambda_ds={lambda_ds:.4f}")

                if (self.is_main_process() and self.display_freq > 0
                        and self.global_step % self.display_freq == 0):
                    with torch.no_grad():
                        style = self.nets_ema["style_encoder"](x_ref, y_trg)
                        fake = self.nets_ema["generator"](x_real, style)
                    self._save_stargan_visual(
                        x_real, fake, x_ref, y_org, y_trg, batch,
                        epoch=self.global_step, prefix="step")

            for scheduler in self.schedulers.values():
                scheduler.step()
            count = max(iterations, 1)
            if self._train_logger:
                self._train_logger.log_train(
                    epoch + 1, self.global_step,
                    sums["loss_D"] / count, sums["loss_G"] / count,
                    self.optims["generator"].param_groups[0]["lr"],
                    self.optims["discriminator"].param_groups[0]["lr"],
                    components={k: v / count for k, v in component_sums.items()})

            if self._validate_and_early_stop(epoch, self.save_every):
                break
            if self.is_main_process() and (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch{epoch + 1}.pth", epoch + 1)

        if self._train_logger:
            self._train_logger.close()
        self._save_checkpoint("generator_final.pth", last_epoch)


class StarGANv2FineTuneTrainer(_StarGANTrainerMixin, BaseTrainer):
    """Paired prospective fine-tuning for the shared StarGAN v2 model."""

    def setup(self):
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]
        self.modalities = list(data_cfg["modalities"])
        self.domains = list(self.cfg.get("domains", data_cfg.get("source_fields", [])))
        self.normalize = data_cfg.get("normalize", False) or data_cfg.get("norm_nonzero", False)
        self.latent_dim = self._network_kwargs()["latent_dim"]
        self._domain_pairs = {
            _domain_index(self.modalities, modality, field): (modality, field)
            for modality in self.modalities for field in self.domains
        }
        self._build_networks()
        pretrained = self.cfg.get("pretrain", {}).get("checkpoint")
        self._load_pretrained(pretrained)
        self._prepare_norm_stats(pretrained)
        dataset = self._build_paired_dataset()
        self._validate_domains(list(self._domain_pairs))

        batch_size = train_cfg.get("batch_size", 4)
        num_workers = train_cfg.get("num_workers", 4)
        if self.use_dist:
            self.sampler = DistributedSampler(dataset, shuffle=True)
            self.loader = DataLoader(
                dataset, batch_size=batch_size, sampler=self.sampler,
                num_workers=num_workers, drop_last=True, pin_memory=True)
        else:
            self.sampler = None
            self.loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, drop_last=True, pin_memory=True)

        self._wrap_ddp()
        nets = self._unwrap_nets()
        self.optimizer = torch.optim.Adam(
            itertools.chain(nets["generator"].parameters(),
                            nets["mapping_network"].parameters()),
            lr=train_cfg.get("lr", 5e-5),
            betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)))
        self.epochs = train_cfg.get("epochs", 50)
        warmup = train_cfg.get("warmup_epochs", 0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: self._cosine_warmup_lr(epoch, warmup, self.epochs))
        self.ema_beta = train_cfg.get("ema_beta", 0.999)
        self.display_freq = train_cfg.get("display_freq", 0)
        self.save_every = train_cfg.get("save_every", 10)
        self.criterion_l1 = nn.L1Loss()
        self.criterion_lpips = PerceptualLoss().to(self.device)
        self.criterion_ssim = SSIMLoss().to(self.device)
        self.lambda_l1 = train_cfg.get("loss_l1", 1.0)
        self.lambda_lpips = train_cfg.get("loss_lpips", 0.1)
        self.lambda_ssim = train_cfg.get("loss_ssim", 0.0)
        self.lambda_edge = train_cfg.get("loss_edge", 0.0)
        if self.lambda_edge > 0:
            self.criterion_edge = StructureLoss(
                ssim_weight=0.0, edge_weight=1.0).to(self.device)
        self._train_logger = TrainLogger(
            self.output_dir / "logs",
            resume=self.cfg.get("_runtime", {}).get("resume", False),
            experiment=self.cfg.get("task_name", ""),
        ) if self.is_main_process() else None
        self._init_validation(self._build_field_stats())

        if self.is_main_process():
            print(f"StarGAN v2 fine-tuning: {len(dataset)} paired slices, "
                  f"{self.epochs} epochs"
                  + (f" (DDP: {self.world_size} GPUs)" if self.use_dist else ""))

    def _prepare_norm_stats(self, pretrained):
        if not self.normalize:
            self._write_norm_stats({})
            return
        source_path = None
        if pretrained:
            candidate = Path(pretrained).parent.parent / "norm_stats.yaml"
            if candidate.exists():
                source_path = candidate
        if source_path:
            self._load_norm_stats(source_path)
            return
        if not self.cfg["data"].get("use_monai"):
            raise ValueError("Normalized StarGAN fine-tuning requires MONAI or pretrain norm_stats.yaml.")
        # pro_scratch: compute every domain once, persist, then reuse for pairs.
        stats_dataset = MultiDomainMONAI2DDataset(
            data_root=self.cfg["data"]["data_dir"],
            split=SPLIT_MAP.get(self.cfg["training"].get("split", "pro_train"),
                                self.cfg["training"].get("split", "pro_train")),
            modalities=self.modalities, field_strengths=self.domains,
            slice_range=tuple(self.cfg["data"].get("slice_range", [120, 210])),
            slice_axis=self.cfg["data"].get("slice_axis", 2),
            crop_size=tuple(self.cfg["data"]["crop_size"])
            if self.cfg["data"].get("crop_size") else None,
            normalize=self.cfg["data"].get("normalize", False),
            norm_nonzero=self.cfg["data"].get("norm_nonzero", False),
            cache_dir=self.cfg["data"].get("cache_dir"),
            num_workers=self.cfg["training"].get("num_workers", 4))
        self._write_norm_stats(stats_dataset.domain_stats)

    def _build_paired_dataset(self):
        data_cfg = self.cfg["data"]
        split = self.cfg["training"].get("split", "pro_train")
        crop_size = tuple(data_cfg["crop_size"]) if data_cfg.get("crop_size") else None
        norm_path = self.output_dir / "norm_stats.yaml"
        wrapped = []
        for modality in self.modalities:
            for source_field in self.domains:
                for target_field in self.domains:
                    if source_field == target_field:
                        continue
                    try:
                        if data_cfg.get("use_monai"):
                            dataset = PairedMONAI2DDataset(
                                data_root=data_cfg["data_dir"],
                                split=SPLIT_MAP.get(split, split),
                                modalities=[modality],
                                source_field=source_field,
                                target_field=target_field,
                                slice_range=tuple(data_cfg.get("slice_range", [120, 210])),
                                slice_axis=data_cfg.get("slice_axis", 2),
                                crop_size=crop_size,
                                normalize=data_cfg.get("normalize", False),
                                norm_nonzero=data_cfg.get("norm_nonzero", False),
                                norm_stats_path=norm_path if norm_path.exists() else None,
                                augment=data_cfg.get("augment"),
                                cache_dir=data_cfg.get("cache_dir"),
                                num_workers=self.cfg["training"].get("num_workers", 4))
                        elif data_cfg.get("preprocessed_dir"):
                            dataset = CachedPairedDataset(
                                preprocessed_dir=data_cfg["preprocessed_dir"], split=split,
                                modality=modality, source_field=source_field,
                                target_field=target_field, crop_size=crop_size)
                        else:
                            dataset = PairedMRIDataset(
                                data_root=data_cfg["data_dir"], split=SPLIT_MAP.get(split, split),
                                modality=modality, source_field=source_field,
                                target_field=target_field, crop_size=crop_size)
                    except (FileNotFoundError, ValueError):
                        continue
                    wrapped.append(_PairedWithDomains(
                        dataset,
                        _domain_index(self.modalities, modality, source_field),
                        _domain_index(self.modalities, modality, target_field),
                        modality, modality, source_field, target_field))
        if not wrapped:
            raise FileNotFoundError("No paired StarGAN v2 domain combinations were found.")
        return ConcatDataset(wrapped)

    def _image_range(self, tensor, domains):
        tensor = self._denormalize_batch(tensor, domains)
        return (tensor + 1.0) * 0.5

    def _loss(self, prediction, target, target_domains):
        loss = self.lambda_l1 * self.criterion_l1(prediction, target)
        if self.lambda_lpips or self.lambda_ssim or self.lambda_edge:
            pred_01 = self._image_range(prediction, target_domains)
            target_01 = self._image_range(target, target_domains)
            if self.lambda_lpips:
                loss = loss + self.lambda_lpips * self.criterion_lpips(
                    pred_01 * 2 - 1, target_01 * 2 - 1)
            if self.lambda_ssim:
                loss = loss + self.lambda_ssim * self.criterion_ssim(pred_01, target_01)
            if self.lambda_edge:
                loss = loss + self.lambda_edge * self.criterion_edge(pred_01, target_01)
        return loss

    def train(self):
        last_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs):
            last_epoch = epoch + 1
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            self.nets["generator"].train()
            self.nets["mapping_network"].train()
            total_loss = 0.0
            iterations = 0
            iterable = tqdm(
                self.loader, desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=not self.is_main_process())
            for batch in iterable:
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)
                src_domain = batch["src_domain"].to(self.device).long()
                tgt_domain = batch["tgt_domain"].to(self.device).long()
                z = torch.zeros(source.size(0), self.latent_dim, device=self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast_context():
                    style = self.nets["mapping_network"](z, tgt_domain)
                    prediction = self.nets["generator"](source, style)
                    loss = self._loss(prediction, target, tgt_domain)
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(self.nets["generator"].parameters(),
                                        self.nets["mapping_network"].parameters()),
                        self.max_grad_norm)
                self.optimizer.step()
                self._update_ema()
                self.global_step += 1
                iterations += 1
                total_loss += loss.item()
                if self.is_main_process():
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
                if (self.is_main_process() and self.display_freq > 0
                        and self.global_step % self.display_freq == 0):
                    visual_batch = dict(batch)
                    visual_batch["field_strength"] = batch["source_field"]
                    visual_batch["ref_filename"] = batch["filename"]
                    visual_batch["ref_slice_index"] = batch["slice_index"]
                    visual_batch["ref_field_strength"] = batch["target_field"]
                    self._save_stargan_visual(
                        source, prediction, target, src_domain, tgt_domain,
                        visual_batch, epoch=self.global_step, prefix="step")

            self.scheduler.step()
            average = total_loss / max(iterations, 1)
            if self._train_logger:
                self._train_logger.log_finetune(
                    epoch + 1, self.global_step, average,
                    self.optimizer.param_groups[0]["lr"])
            if self._validate_and_early_stop(epoch, self.save_every):
                break
            if self.is_main_process() and (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch{epoch + 1}.pth", epoch + 1)

        if self._train_logger:
            self._train_logger.close()
        self._save_checkpoint("generator_final.pth", last_epoch)


__all__ = ["StarGANv2Trainer", "StarGANv2FineTuneTrainer"]
