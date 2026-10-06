"""Base Pretrain Trainer — shared setup + train loop for CUT-family methods.

Subclasses only need to define ``model_cls`` and override
``_model_kwargs()`` for method-specific parameters.
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..data import UnpairedMONAI2DDataset, UnpairedMONAI3DDataset
from ..data import UnpairedMRIDataset, CachedUnpairedDataset
from ..data import UnpairedDataLoader as _UnpairedDataLoader
from ..data.utils import ABBR_TO_SPLIT as SPLIT_MAP
from ..utils.evaluator import Evaluator
from ..utils.train_logger import TrainLogger


def _iter_unpaired(loader_src, loader_tgt):
    """Iterate over two DataLoaders jointly (zip)."""
    it_src = iter(loader_src)
    it_tgt = iter(loader_tgt)
    while True:
        try:
            src = next(it_src)
            tgt = next(it_tgt)
            yield src, tgt
        except StopIteration:
            break


def _get_lr_lambda(n_epochs, n_epochs_decay):
    """Linear decay after n_epochs."""
    def lr_lambda(epoch):
        if epoch < n_epochs:
            return 1.0
        return 1.0 - max(0.0, epoch - n_epochs) / max(1, n_epochs_decay)
    return lr_lambda


class BasePretrainTrainer(BaseTrainer):
    """Shared pretrain trainer for CUT / ITTR / Restormer / NAFNet.

    **Subclass attributes**::

        model_cls = CUTModel          # model class
        method_name = "cut"           # for logging

    **Override**: ``_model_kwargs()`` for method-specific constructor params.
    """

    model_cls = None      # set in subclass
    method_name = ""      # set in subclass
    _is_3d = False        # set in subclass (e.g. CUT3D)

    def __init__(self, cfg: dict, **kwargs):
        self.norm_stats = None
        # dim from config overrides class-level _is_3d
        dim = cfg.get("data", {}).get("dim")
        if dim is not None:
            self._is_3d = (dim == 3)
        super().__init__(cfg)

    # ── Subclass override points ─────────────────────────────────────

    def _model_kwargs(self) -> dict:
        """Return dict of kwargs for ``self.model_cls(**kw)``."""
        mc = self.cfg.get("model", {})
        dc = self.cfg.get("data", {})
        tc = self.cfg.get("training", {})
        has_norm = dc.get("normalize", False) or dc.get("norm_nonzero", False)
        use_tanh = not has_norm
        return dict(
            input_nc=mc.get("input_nc", 1),
            output_nc=mc.get("output_nc", 1),
            ngf=mc.get("ngf", 64),
            ndf=mc.get("ndf", 64),
            n_blocks=mc.get("n_blocks", 9),
            nce_layers=mc.get("nce_layers", [0, 4, 8, 12, 16]),
            nce_T=tc.get("nce_T", 0.07),
            num_patches=tc.get("num_patches", 256),
            lambda_GAN=tc.get("lambda_GAN", 1.0),
            lambda_NCE=tc.get("lambda_NCE", 1.0),
            lambda_idt=tc.get("lambda_idt", 0.0),
            lambda_idt_loss_type=tc.get("lambda_idt_loss_type", "smoothl1"),
            nce_idt=tc.get("nce_idt", True),
            gan_mode=tc.get("gan_mode", "lsgan"),
            netF_nc=mc.get("netF_nc", 256),
            lr=tc.get("lr", 0.0002),
            beta1=tc.get("beta1", 0.5),
            beta2=tc.get("beta2", 0.999),
            batch_size=tc.get("batch_size", 4),
            multiscale=mc.get("multiscale", False),
            num_D=mc.get("num_D", 2),
            n_layers_D=mc.get("n_layers_D", 3),
            use_spectral_norm=mc.get("use_spectral_norm", False),
            nce_layer_weights=tc.get("nce_layer_weights", None),
            foreground_sampling=tc.get("foreground_sampling", False),
            moco=tc.get("moco", False),
            moco_queue_size=tc.get("moco_queue_size", 4096),
            moco_momentum=tc.get("moco_momentum", 0.999),
            ema=tc.get("ema", False),
            ema_decay=tc.get("ema_decay", 0.999),
            nce_T_start=tc.get("nce_T_start", 0.07),
            use_tanh=use_tanh,
            lr_D=tc.get("lr_D"),
            lambda_LNCC=tc.get("lambda_LNCC", 0.0),
        )

    def _generator_config(self) -> dict:
        model_cfg = self.cfg.get("model", {})
        data_cfg = self.cfg.get("data", {})
        has_norm = data_cfg.get("normalize", False) or data_cfg.get("norm_nonzero", False)
        return {
            "kwargs": {
                "input_nc": model_cfg["input_nc"],
                "output_nc": model_cfg["output_nc"],
                "ngf": model_cfg.get("ngf", 64),
                "n_blocks": model_cfg.get("n_blocks", 9),
                "use_tanh": not has_norm,
            },
        }

    # ── Setup ────────────────────────────────────────────────────────

    def setup(self):
        model_cfg = self.cfg["model"]
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]

        self._input_nc = model_cfg.get("input_nc", 1)
        self._output_nc = model_cfg.get("output_nc", 1)
        modalities = self._validate_config()
        source_fields = data_cfg.get("source_fields",
                                      [data_cfg.get("source_field", "")])
        target_field = data_cfg["target_field"]
        self.normalize = data_cfg.get("normalize", False) or data_cfg.get("norm_nonzero", False)
        self.norm_nonzero = data_cfg.get("norm_nonzero", False)
        batch_size = train_cfg.get("batch_size", 4)

        # ── Model ──
        self.model = self.model_cls(**self._model_kwargs())
        self.model.to(self.device)
        self.model.init_weights()

        # ── Datasets ──
        split = train_cfg.get("split", "retro_train")
        num_workers = train_cfg.get("num_workers", 4)
        crop_size_raw = data_cfg.get("crop_size")
        crop_size = tuple(crop_size_raw) if crop_size_raw else None
        preprocessed_dir = data_cfg.get("preprocessed_dir")

        if self._is_3d:
            self._create_3d_datasets(data_cfg, split, modalities,
                                     source_fields, target_field)
            dataset_src = self._dataset_src
            dataset_tgt = self._dataset_tgt
        elif data_cfg.get("use_monai"):
            slice_range = tuple(data_cfg.get("slice_range", [120, 210]))
            split_full = SPLIT_MAP.get(split, split)
            cache_dir = data_cfg.get("cache_dir")
            norm_flag = self.norm_nonzero if self.norm_nonzero else self.normalize
            src_kw = dict(
                data_root=data_cfg["data_dir"], split=split_full,
                modalities=modalities,
                slice_range=slice_range,
                slice_axis=data_cfg.get("slice_axis", 2),
                crop_size=crop_size, normalize=norm_flag,
                norm_nonzero=self.norm_nonzero,
                augment=data_cfg.get("augment"), cache_dir=cache_dir,
                spacing=data_cfg.get("spacing"),
                clip_percentile=data_cfg.get("clip_percentile"),
                auto_slice_range=data_cfg.get("auto_slice_range", False),
                auto_slice_min_frac=data_cfg.get("auto_slice_min_frac", 0.10),
                num_workers=num_workers,
            )
            if len(source_fields) == 1:
                src_kw["field_strength"] = source_fields[0]
            else:
                src_kw["field_strengths"] = source_fields
            dataset_src = UnpairedMONAI2DDataset(**src_kw)
            dataset_tgt = UnpairedMONAI2DDataset(
                data_root=data_cfg["data_dir"], split=split_full,
                modalities=modalities, field_strength=target_field,
                slice_range=slice_range,
                slice_axis=data_cfg.get("slice_axis", 2),
                crop_size=crop_size, normalize=norm_flag,
                norm_nonzero=self.norm_nonzero,
                augment=data_cfg.get("augment"), cache_dir=cache_dir,
                spacing=data_cfg.get("spacing"),
                clip_percentile=data_cfg.get("clip_percentile"),
                auto_slice_range=data_cfg.get("auto_slice_range", False),
                auto_slice_min_frac=data_cfg.get("auto_slice_min_frac", 0.10),
                num_workers=num_workers,
            )
        elif preprocessed_dir:
            dataset_src = CachedUnpairedDataset(
                preprocessed_dir=preprocessed_dir, split=split,
                modalities=modalities, field_strength=source_fields[0],
                crop_size=crop_size,
            )
            dataset_tgt = CachedUnpairedDataset(
                preprocessed_dir=preprocessed_dir, split=split,
                modalities=modalities, field_strength=target_field,
                crop_size=crop_size,
            )
        else:
            split_full = SPLIT_MAP.get(split, split)
            dataset_src = UnpairedMRIDataset(
                data_root=data_cfg["data_dir"], split=split_full,
                modalities=modalities, field_strength=source_fields[0],
                crop_size=crop_size,
            )
            dataset_tgt = UnpairedMRIDataset(
                data_root=data_cfg["data_dir"], split=split_full,
                modalities=modalities, field_strength=target_field,
                crop_size=crop_size,
            )

        self._field_stats = getattr(dataset_src, "_field_stats", None)

        # ── Normalisation ──
        self._setup_norm_stats(data_cfg, dataset_src, dataset_tgt,
                               source_fields, target_field, modalities)

        # ── Dataloaders ──
        if self.use_dist:
            sampler_src = DistributedSampler(dataset_src, shuffle=True)
            sampler_tgt = DistributedSampler(dataset_tgt, shuffle=True)
            self.loader_src = DataLoader(
                dataset_src, batch_size=batch_size, sampler=sampler_src,
                num_workers=num_workers, drop_last=True, pin_memory=True)
            self.loader_tgt = DataLoader(
                dataset_tgt, batch_size=batch_size, sampler=sampler_tgt,
                num_workers=num_workers, drop_last=True, pin_memory=True)
            self.sampler_src = sampler_src
            self.sampler_tgt = sampler_tgt
        else:
            self.loader_src = DataLoader(
                dataset_src, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, drop_last=True, pin_memory=True)
            self.loader_tgt = DataLoader(
                dataset_tgt, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, drop_last=True, pin_memory=True)
            self.sampler_src = None
            self.sampler_tgt = None

        # ── Data-dependent init + optimizers ──
        init_a = next(iter(self.loader_src))
        init_b = next(iter(self.loader_tgt))
        self.model.data_dependent_initialize(
            init_a["image"].to(self.device),
            init_b["image"].to(self.device))
        self.model.setup_optimizers()

        # ── Validation ──
        self._init_validation(data_cfg, train_cfg, dataset_src,
                              dataset_tgt, source_fields, target_field,
                              modalities, crop_size)

        # ── EMA ──
        if hasattr(self.model, "init_ema") and self.model.ema_enabled:
            self.model.init_ema()

        # ── DDP ──
        if self.use_dist:
            self.model.netG = DDP(self.model.netG, device_ids=[self.device])
            self.model.netD = DDP(self.model.netD, device_ids=[self.device])
            self.model.netF = DDP(
                self.model.netF, device_ids=[self.device],
                find_unused_parameters=True)

        # ── LR schedulers ──
        n_epochs = train_cfg.get("n_epochs", 100)
        n_epochs_decay = train_cfg.get("n_epochs_decay", 100)
        lr_fn = _get_lr_lambda(n_epochs, n_epochs_decay)
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.model.optimizer_G, lr_fn)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(
            self.model.optimizer_D, lr_fn)
        self.scheduler_F = torch.optim.lr_scheduler.LambdaLR(
            self.model.optimizer_F, lr_fn)

        self.total_epochs = n_epochs + n_epochs_decay
        self.max_iters = train_cfg.get("max_iters_per_epoch", 0)
        self.loader_len = min(len(self.loader_src), len(self.loader_tgt))

        if self.is_main_process():
            mod_str = "+".join(modalities)
            src_str = "+".join(source_fields)
            dtag = "3D " if self._is_3d else ""
            print(f"Training {dtag}{self.method_name}: "
                  f"{src_str} -> {target_field}, "
                  f"modalities=[{mod_str}], "
                  f"{self.total_epochs} epochs, {self.loader_len} iters/epoch"
                  + (f" (capped at {self.max_iters})" if self.max_iters else ""))

    # ── Shared helpers ────────────────────────────────────────────────

    def _create_3d_datasets(self, data_cfg, split, modalities,
                            source_fields, target_field):
        split_full = SPLIT_MAP.get(split, split)
        kw = dict(
            data_root=data_cfg["data_dir"], split=split_full,
            modalities=modalities,
            cache_dir=data_cfg.get("cache_dir"),
            patch_size=tuple(data_cfg.get("patch_size", [128, 128, 128])),
            samples_per_volume=data_cfg.get("samples_per_volume", 1),
        )
        self._dataset_src = UnpairedMONAI3DDataset(
            field_strength=source_fields[0], **kw)
        self._dataset_tgt = UnpairedMONAI3DDataset(
            field_strength=target_field, **kw)

    def _setup_norm_stats(self, data_cfg, dataset_src, dataset_tgt,
                          source_fields, target_field, modalities):
        import yaml as _yaml
        ns_path = self.output_dir / "norm_stats.yaml"
        ns_data = {}
        do_norm = self.normalize or self.norm_nonzero
        if do_norm and data_cfg.get("use_monai"):
            for fs in source_fields:
                if self._field_stats and fs in self._field_stats:
                    s = self._field_stats[fs]
                    for c, mod in enumerate(modalities):
                        ns_data[f"{mod}/{fs}"] = {
                            k: s[k][c] for k in ("mu", "sigma", "bg_norm")}
            if dataset_tgt.mu is not None:
                for c, mod in enumerate(modalities):
                    ns_data[f"{mod}/{target_field}"] = {
                        "mu": dataset_tgt.mu[c],
                        "sigma": dataset_tgt.sigma[c],
                        "bg_norm": dataset_tgt.bg_norm[c],
                    }
            # set_norm_params for vis denorm (use first field's stats)
            if self._field_stats and source_fields[0] in self._field_stats:
                s = self._field_stats[source_fields[0]]
                self.model.set_norm_params(
                    s["mu"], s["sigma"], s["bg_norm"],
                    dataset_tgt.mu, dataset_tgt.sigma, dataset_tgt.bg_norm)
            elif dataset_src.mu is not None:
                self.model.set_norm_params(
                    dataset_src.mu, dataset_src.sigma, dataset_src.bg_norm,
                    dataset_tgt.mu, dataset_tgt.sigma, dataset_tgt.bg_norm)
            if self.is_main_process():
                tmp = ns_path.with_suffix(".yaml.tmp")
                tmp.write_text(_yaml.dump(ns_data, default_flow_style=None))
                tmp.rename(ns_path)
            self._norm_stats = ns_data   # for checkpoint embedding; also for visual code
            # Also store src/tgt keys in the format that _save_training_visual expects
            if dataset_tgt.mu is not None and dataset_src.mu is not None:
                self._norm_stats = dict(ns_data, **{
                    "mu_src": dataset_src.mu,
                    "sigma_src": dataset_src.sigma,
                    "mu_tgt": dataset_tgt.mu,
                    "sigma_tgt": dataset_tgt.sigma,
                })

    def _init_validation(self, data_cfg, train_cfg, dataset_src,
                         dataset_tgt, source_fields, target_field,
                         modalities, crop_size):
        self._val_every = train_cfg.get("val_every", -1)
        self._train_logger = None
        self._evaluator = None
        self._es_patience = train_cfg.get("early_stop_patience", 10)
        self._es_min_epochs = train_cfg.get("early_stop_min_epochs", 100)

        if self._val_every > 0:
            if self.is_main_process():
                self._train_logger = TrainLogger(
                    self.output_dir / "logs",
                    resume=self.cfg.get("_runtime", {}).get("resume", False),
                    experiment=self.cfg.get("task_name", ""))
            val_split = self.cfg.get("finetune", {}).get("split", "pro_train")
            mu_tgt = sigma_tgt = None
            if self.normalize and hasattr(self.model, "_tgt_mu"):
                mu_tgt = self.model._tgt_mu.cpu().tolist()
                sigma_tgt = self.model._tgt_sigma.cpu().tolist()
            elif self.normalize and dataset_tgt.mu is not None:
                mu_tgt = dataset_tgt.mu
                sigma_tgt = dataset_tgt.sigma
            self._evaluator = Evaluator(
                data_root=data_cfg.get("data_dir"),
                split=val_split,
                source_fields=source_fields,
                target_field=target_field,
                field_stats=getattr(dataset_src, "_field_stats", None),
                modalities=modalities,
                device=self.device,
                metric_names=self.cfg.get("evaluation", {}).get("metrics"),
                slice_axis=data_cfg.get("slice_axis", 2),
                crop_size=crop_size,
                batch_size=self.cfg.get("inference", {}).get("batch_size", 32),
                mu_tgt=mu_tgt, sigma_tgt=sigma_tgt,
                slice_range=self.cfg.get("evaluation", {}).get("slice_range"),
                is_3d=self._is_3d,
            )
            baseline = self._evaluator.run(None)
            self._save_baseline_metrics(baseline)
            if self.normalize:
                normed = self._evaluator.run(None, force_norm=True)
                self._save_baseline_metrics(normed, key="baseline_normed")

    # ── Train loop ────────────────────────────────────────────────────

    def train(self):
        display_freq = self.cfg["training"].get("display_freq", 0)
        val_save_every = self.cfg["training"].get("save_every", 10)

        for epoch in range(self.start_epoch, self.total_epochs):
            self.model.train()
            if hasattr(self.model, "set_nce_temperature"):
                self.model.set_nce_temperature(epoch, self.total_epochs)

            g_lr = self.model.optimizer_G.param_groups[0]["lr"]
            d_lr = self.model.optimizer_D.param_groups[0]["lr"]

            if self.use_dist:
                self.sampler_src.set_epoch(epoch)
                self.sampler_tgt.set_epoch(epoch)
                iterable = _iter_unpaired(self.loader_src, self.loader_tgt)
            else:
                iterable = _UnpairedDataLoader(
                    self.loader_src.dataset, self.loader_tgt.dataset,
                    batch_size=self.loader_src.batch_size,
                    num_workers=self.loader_src.num_workers)

            ema_d = ema_g = None

            pbar = tqdm(iterable,
                        desc=f"Epoch {epoch+1}/{self.total_epochs}",
                        total=self.loader_len,
                        disable=not self.is_main_process())

            for i, (batch_a, batch_b) in enumerate(pbar):
                if self.max_iters and i >= self.max_iters:
                    break
                real_A = batch_a["image"].to(self.device)
                real_B = batch_b["image"].to(self.device)

                with self.autocast_context():
                    losses = self.model.optimize_parameters(
                        real_A, real_B, max_grad_norm=self.max_grad_norm)
                self.global_step += 1

                if ema_d is None:
                    ema_d, ema_g = losses["loss_D"], losses["loss_G"]
                else:
                    ema_d = ema_d * 0.95 + losses["loss_D"] * 0.05
                    ema_g = ema_g * 0.95 + losses["loss_G"] * 0.05

                if (self.is_main_process() and display_freq > 0
                        and self.global_step % display_freq == 0):
                    with torch.no_grad():
                        idt_B = getattr(self.model, "idt_B", None)
                        self._save_training_visual(
                            real_A, real_B, self.model.fake_B,
                            batch_info=batch_a, batch_info_b=batch_b,
                            idt_B=idt_B, epoch=epoch + 1)

                if self.is_main_process():
                    pbar.set_postfix(
                        G=f"{losses['loss_G']:.4f}",
                        D=f"{losses['loss_D']:.4f}")

            # ── Epoch end ──
            if self._train_logger and self.is_main_process():
                self._train_logger.log_train(
                    epoch + 1, self.global_step,
                    ema_d or 0.0, ema_g or 0.0, g_lr, d_lr)

            self.scheduler_G.step()
            self.scheduler_D.step()
            self.scheduler_F.step()

            if self._validate_and_early_stop(epoch, val_save_every):
                break

            if self.is_main_process() and (epoch + 1) % val_save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch{epoch+1}.pth",
                                      epoch + 1)

        if self._train_logger:
            self._train_logger.close()

        self._save_checkpoint("generator_final.pth", self.total_epochs)
