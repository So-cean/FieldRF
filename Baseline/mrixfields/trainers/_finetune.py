"""Base Fine-Tune Trainer — shared setup + train loop."""

import torch
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..data import PairedMONAI2DDataset, PairedMONAI3DDataset
from ..data import PairedMRIDataset, CachedPairedDataset
from ..data.utils import ABBR_TO_SPLIT as SPLIT_MAP
from ..utils.evaluator import Evaluator
from ..utils.train_logger import TrainLogger
from ..inference.base_inference import extract_generator_state


class BaseFineTuneTrainer(BaseTrainer):
    generator_cls = None
    method_name = ""
    _is_3d = False

    def __init__(self, cfg: dict, **kwargs):
        self.norm_stats = None
        dim = cfg.get("data", {}).get("dim")
        if dim is not None:
            self._is_3d = (dim == 3)
        super().__init__(cfg)

    def _generator_kwargs(self):
        mc = self.cfg.get("model", {})
        dc = self.cfg.get("data", {})
        has_norm = dc.get("normalize", False) or dc.get("norm_nonzero", False)
        return dict(input_nc=mc.get("input_nc", 1), output_nc=mc.get("output_nc", 1),
                    ngf=mc.get("ngf", 64), n_blocks=mc.get("n_blocks", 9),
                    use_tanh=not has_norm)

    def _generator_config(self):
        mc = self.cfg.get("model", {})
        dc = self.cfg.get("data", {})
        has_norm = dc.get("normalize", False) or dc.get("norm_nonzero", False)
        return {"kwargs": {"input_nc": mc["input_nc"], "output_nc": mc["output_nc"],
                "ngf": mc.get("ngf", 64), "n_blocks": mc.get("n_blocks", 9),
                "use_tanh": not has_norm}}

    def setup(self):
        mc = self.cfg["model"]; dc = self.cfg["data"]; tc = self.cfg["training"]
        modalities = self._validate_config()
        source_field = dc["source_field"]; target_field = dc["target_field"]
        self.normalize = dc.get("normalize", False) or dc.get("norm_nonzero", False)
        self.norm_nonzero = dc.get("norm_nonzero", False)
        self.modalities = modalities
        batch_size = tc.get("batch_size", 4); split = tc.get("split", "pro_train")
        num_workers = tc.get("num_workers", 4)

        self.generator = self.generator_cls(**self._generator_kwargs())
        self.generator.to(self.device)
        pretrained_ckpt = self.cfg.get("pretrain", {}).get("checkpoint")
        if pretrained_ckpt:
            sd = torch.load(pretrained_ckpt, map_location=self.device, weights_only=True)
            gs = extract_generator_state(sd); self.generator.load_state_dict(gs)
            if self.is_main_process(): print(f"Loaded pretrained generator from {pretrained_ckpt}")
        else:
            if self.is_main_process(): print("WARNING: No pretrained checkpoint")

        if self.use_dist: self.generator = DDP(self.generator, device_ids=[self.device])
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=tc.get("lr", 1e-5),
                                          betas=(tc.get("beta1", 0.5), tc.get("beta2", 0.999)))
        self._cosine_warmup_lr(tc); self._setup_finetune_losses(tc)

        crop_size_raw = dc.get("crop_size")
        crop_size = tuple(crop_size_raw) if crop_size_raw else None
        preprocessed_dir = dc.get("preprocessed_dir")

        if self._is_3d:
            split_full = SPLIT_MAP.get(split, split)
            kw = dict(data_root=dc["data_dir"], split=split_full, modalities=modalities,
                      cache_dir=dc.get("cache_dir"),
                      patch_size=tuple(dc.get("patch_size", [128, 128, 128])),
                      samples_per_volume=dc.get("samples_per_volume", 1))
            dataset = PairedMONAI3DDataset(source_field=source_field, target_field=target_field, **kw)
            self._norm_stats = getattr(dataset, "norm_stats", None)
        elif dc.get("use_monai"):
            split_full = SPLIT_MAP.get(split, split)
            cache_dir = dc.get("cache_dir")
            norm_flag = self.norm_nonzero if self.norm_nonzero else self.normalize
            dataset = PairedMONAI2DDataset(data_root=dc["data_dir"], split=split_full,
                modalities=modalities, source_field=source_field, target_field=target_field,
                slice_range=tuple(dc.get("slice_range", [120, 210])),
                slice_axis=dc.get("slice_axis", 2), crop_size=crop_size,
                normalize=norm_flag, norm_nonzero=self.norm_nonzero, cache_dir=cache_dir)
            # Build _norm_stats from dataset's per-field mu/sigma
            if self.normalize and dataset.src_mu is not None and dataset.tgt_mu is not None:
                self._norm_stats = {
                    "mu_src": dataset.src_mu,
                    "sigma_src": dataset.src_sigma,
                    "mu_tgt": dataset.tgt_mu,
                    "sigma_tgt": dataset.tgt_sigma,
                }
                if hasattr(self, "model") and hasattr(self.model, "set_norm_params") and not self._is_3d:
                    self.model.set_norm_params(
                        dataset.src_mu, dataset.src_sigma, dataset.src_bg_norm,
                        dataset.tgt_mu, dataset.tgt_sigma, dataset.tgt_bg_norm)
            else:
                self._norm_stats = None
        elif preprocessed_dir:
            dataset = CachedPairedDataset(preprocessed_dir=preprocessed_dir, split=split,
                modalities=modalities, source_field=source_field, target_field=target_field, crop_size=crop_size)
        else:
            split_full = SPLIT_MAP.get(split, split)
            dataset = PairedMRIDataset(data_root=dc["data_dir"], split=split_full,
                modalities=modalities, source_field=source_field, target_field=target_field, crop_size=crop_size)

        if self.use_dist:
            sampler = DistributedSampler(dataset, shuffle=True)
            self.loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                     num_workers=num_workers, drop_last=True, pin_memory=True)
            self.sampler = sampler
        else:
            self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, drop_last=True, pin_memory=True)
            self.sampler = None

        self._val_every = tc.get("val_every", -1)
        self._train_logger = self._evaluator = None
        self._es_patience = tc.get("early_stop_patience", 10)
        self._es_min_epochs = tc.get("early_stop_min_epochs", 100)

        if self._val_every > 0:
            if self.is_main_process():
                self._train_logger = TrainLogger(self.output_dir / "logs",
                    resume=self.cfg.get("_runtime", {}).get("resume", False),
                    experiment=self.cfg.get("task_name", ""))
            val_split = dc.get("val_split", "pro_train")
            mu_tgt = sigma_tgt = mu_src = sigma_src = None
            if self._norm_stats:
                if "mu_src" in self._norm_stats:
                    mu_src = self._norm_stats["mu_src"]
                    sigma_src = self._norm_stats["sigma_src"]
                    mu_tgt = self._norm_stats["mu_tgt"]
                    sigma_tgt = self._norm_stats["sigma_tgt"]
                elif self._norm_stats.get("src"):
                    mu_src = self._norm_stats["src"]["mu"]
                    sigma_src = self._norm_stats["src"]["sigma"]
                    mu_tgt = self._norm_stats["tgt"]["mu"]
                    sigma_tgt = self._norm_stats["tgt"]["sigma"]
            self._evaluator = Evaluator(data_root=dc.get("data_dir"), split=val_split,
                source_fields=[source_field], target_field=target_field, modalities=modalities,
                device=self.device, metric_names=self.cfg.get("evaluation", {}).get("metrics"),
                slice_axis=dc.get("slice_axis", 2), crop_size=crop_size,
                batch_size=self.cfg.get("inference", {}).get("batch_size", 32),
                mu_tgt=mu_tgt, sigma_tgt=sigma_tgt, mu_src=mu_src, sigma_src=sigma_src,
                slice_range=self.cfg.get("evaluation", {}).get("slice_range"), is_3d=self._is_3d)
            baseline = self._evaluator.run(None); self._save_baseline_metrics(baseline)
            if self.normalize:
                normed = self._evaluator.run(None, force_norm=True)
                self._save_baseline_metrics(normed, key="baseline_normed")

    def _cosine_warmup_lr(self, tc):
        epochs = tc.get("epochs", 50); warmup = tc.get("warmup_epochs", 5)
        def lr_fn(e):
            if e < warmup: return (e + 1) / max(warmup, 1)
            p = (e - warmup) / max(epochs - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_fn)

    def train(self):
        display_freq = self.cfg["training"].get("display_freq", 0)
        val_save_every = self.cfg["training"].get("save_every", 10)
        epochs = self.cfg["training"].get("epochs", 50)
        for epoch in range(self.start_epoch, epochs):
            self.generator.train()
            if self.use_dist and hasattr(self, "sampler") and self.sampler: self.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not self.is_main_process())
            for i, batch in enumerate(pbar):
                source, target = batch["source"].to(self.device), batch["target"].to(self.device)
                self.optimizer.zero_grad()
                with self.autocast_context():
                    pred = self.generator(source)
                    loss = self._compute_finetune_loss(pred, target)
                loss.backward()
                if self.max_grad_norm: torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
                self.optimizer.step(); self.global_step += 1
                if self.is_main_process() and display_freq > 0 and self.global_step % display_freq == 0:
                    with torch.no_grad(): self._save_training_visual(source, target, pred, batch_info=batch, epoch=epoch + 1)
                epoch_loss += loss.item()
                if self.is_main_process(): pbar.set_postfix(loss=f"{loss.item():.4f}")
            if self.is_main_process(): avg = epoch_loss / max(len(self.loader), 1); print(f"Epoch {epoch+1}: avg_loss={avg:.6f}")
            if self._train_logger and self.is_main_process():
                lr = self.optimizer.param_groups[0]["lr"]
                self._train_logger.log_finetune(epoch + 1, self.global_step, epoch_loss / max(len(self.loader), 1), lr)
            if hasattr(self, "scheduler"): self.scheduler.step()
            if self._validate_and_early_stop(epoch, val_save_every): break
            if self.is_main_process() and (epoch + 1) % val_save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch{epoch+1}.pth", epoch + 1)
        if self._train_logger: self._train_logger.close()
        self._save_checkpoint("generator_final.pth", epochs)
