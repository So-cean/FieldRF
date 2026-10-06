"""Base GAN Model for CUT-family methods.

Shared logic: GAN + NCE + MoCo + EMA + Norm + temperature annealing.
Subclasses override ``_build_generator`` and ``_build_discriminator``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_init import init_net
from ..losses.patchnce import PatchNCELoss, MemoryBank
from ..losses.adversarial import GANLoss
from ..losses.lncc import LNCCloss


class BaseGANModel(nn.Module):
    """Shared base for CUT / ITTR / Restormer / NAFNet / CUT3D.

    Subclasses must override:
        ``_build_generator(opt)``
        ``_build_discriminator(input_nc, opt)``

    Optionally override:
        ``_build_netF(opt)``
        ``_setup_losses(opt)``
        ``compute_G_loss(real_A, real_B)``       (Restormer adds FM/MSSSIM)
        ``calculate_NCE_loss(src, tgt, enqueue)``  (CUT3D: 3D mask)
    """

    def __init__(self, **opt):
        super().__init__()
        self._store_config(opt)
        self.ema_state = None
        self._setup_networks(opt)
        self._setup_losses(opt)
        if self.moco:
            self.memory_bank = MemoryBank(
                queue_size=opt.get("moco_queue_size", 65536),
                feat_dim=opt.get("netF_nc", 256),
                n_layers=len(self.nce_layers),
            )
        else:
            self.memory_bank = None

    # ── Config ──────────────────────────────────────────────────────

    def _store_config(self, opt):
        self.nce_layers = opt.get("nce_layers", [0, 4, 8, 12, 16])
        self.nce_layer_weights = opt.get("nce_layer_weights")
        nce_T = opt.get("nce_T", 0.07)
        self.nce_T = nce_T
        self.nce_T_start = opt.get("nce_T_start", nce_T)
        self.num_patches = opt.get("num_patches", 256)
        self.lambda_GAN = opt.get("lambda_GAN", 1.0)
        self.lambda_NCE = opt.get("lambda_NCE", 1.0)
        self.nce_idt = opt.get("nce_idt", True)
        self.lr = opt.get("lr", 0.0002)
        self.beta1 = opt.get("beta1", 0.5)
        self.beta2 = opt.get("beta2", 0.999)
        self.moco = opt.get("moco", False)
        self.moco_momentum = opt.get("moco_momentum", 0.999)
        self.foreground_sampling = opt.get("foreground_sampling", False)
        self.ema_enabled = opt.get("ema", False)
        self.ema_decay = opt.get("ema_decay", 0.999)
        self.lr_D = opt.get("lr_D") or self.lr
        self.lambda_LNCC = opt.get("lambda_LNCC", 0.0)
        self.lambda_seg = opt.get("lambda_seg", 0.0)
        self.lambda_seg_idt = opt.get("lambda_seg_idt", 0.0)
        self.seg_loss_ckpt = opt.get("seg_loss_ckpt", "")
        self.lambda_idt = opt.get("lambda_idt", 0.0)
        self.lambda_idt_loss_type = opt.get("lambda_idt_loss_type", "smoothl1")

    # ── Network construction ────────────────────────────────────────

    def _setup_networks(self, opt):
        nc_out = opt.get("output_nc", 1)
        self.netG = self._build_generator(opt)
        self.netD = self._build_discriminator(nc_out, opt)
        self.netF = self._build_netF(opt)
        self.netG_momentum = self._build_momentum_encoder(opt)

    def _build_generator(self, opt):
        raise NotImplementedError

    def _build_discriminator(self, input_nc, opt):
        raise NotImplementedError

    def _build_netF(self, opt):
        from .networks_cut import PatchSampleF
        return PatchSampleF(use_mlp=True, nc=opt.get("netF_nc", 256))

    def _build_momentum_encoder(self, opt):
        if not self.moco:
            return None
        import copy
        me = copy.deepcopy(self.netG)
        me.train(False)
        return me

    # ── Losses ──────────────────────────────────────────────────────

    def _setup_losses(self, opt):
        self.criterionGAN = GANLoss(mode=opt.get("gan_mode", "lsgan"))
        self.criterionNCE = nn.ModuleList([
            PatchNCELoss(batch_size=opt.get("batch_size", 1), nce_T=self.nce_T)
            for _ in self.nce_layers
        ])
        self.criterionLNCC = LNCCloss()
        self.criterionIdt = None
        if self.lambda_idt > 0:
            lt = self.lambda_idt_loss_type.lower()
            if lt == "l1":
                self.criterionIdt = nn.L1Loss()
            elif lt == "l2":
                self.criterionIdt = nn.MSELoss()
            else:
                self.criterionIdt = nn.SmoothL1Loss()
        self._seg_loss_fn = None
        seg_ckpt = opt.get("seg_loss_ckpt", "")
        if opt.get("lambda_seg", 0.0) > 0 and seg_ckpt:
            from ..losses.vinn_seg_loss import VINNSegLoss
            sf = opt.get("vinn_scale_factor", 2.0)
            self._seg_loss_fn = VINNSegLoss(seg_ckpt, scale_factor=sf)
            self.add_module("seg_loss", self._seg_loss_fn)

    # ── Init ────────────────────────────────────────────────────────

    def init_weights(self, init_type="normal", init_gain=0.02):
        init_net(self.netG, init_type, init_gain)
        init_net(self.netD, init_type, init_gain)

    def setup_optimizers(self):
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=float(self.lr_D), betas=(self.beta1, self.beta2))

    def data_dependent_initialize(self, real_A, real_B):
        self.forward(real_A, real_B)
        self.compute_D_loss(real_A, real_B).backward()
        self.compute_G_loss(real_A, real_B).backward()
        self._post_data_init()
        self.optimizer_F = torch.optim.Adam(
            self.netF.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.init_momentum_encoder()

    def _post_data_init(self):
        """Subclass hook: e.g. CUT3D clears GPU cache here."""
        pass

    # ── Forward ─────────────────────────────────────────────────────

    def forward(self, real_A, real_B):
        if self.nce_idt and self.training:
            real = torch.cat((real_A, real_B), dim=0)
        else:
            real = real_A
        fake = self.netG(real)
        self.fake_B = fake[:real_A.size(0)]
        if self.nce_idt and self.training:
            self.idt_B = fake[real_A.size(0):]
        return self.fake_B

    # ── Discriminator loss ──────────────────────────────────────────

    def compute_D_loss(self, real_A, real_B):
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        pred_real = self.netD(real_B)
        if isinstance(pred_fake[0], list):
            loss_D_fake = sum(self.criterionGAN(sf[-1], False).mean()
                              for sf in pred_fake)
            loss_D_real = sum(self.criterionGAN(sr[-1], True).mean()
                              for sr in pred_real)
            loss_D = (loss_D_fake + loss_D_real) * 0.5 / len(pred_fake)
        else:
            loss_D_fake = self.criterionGAN(pred_fake[-1], False).mean()
            loss_D_real = self.criterionGAN(pred_real[-1], True).mean()
            loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    # ── Generator loss ──────────────────────────────────────────────

    def compute_G_loss(self, real_A, real_B):
        fake = self.fake_B
        if self.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            if isinstance(pred_fake[0], list):
                loss_G_GAN = sum(self.criterionGAN.g_loss(pf[-1]).mean()
                                 for pf in pred_fake)
                loss_G_GAN = loss_G_GAN / len(pred_fake) * self.lambda_GAN
            else:
                loss_G_GAN = self.criterionGAN.g_loss(pred_fake[-1]).mean() * self.lambda_GAN
        else:
            loss_G_GAN = 0.0
        if self.lambda_NCE > 0.0:
            loss_NCE = self.calculate_NCE_loss(real_A, self.fake_B)
        else:
            loss_NCE = 0.0
        if self.nce_idt and self.lambda_NCE > 0.0 and self.training:
            loss_NCE_Y = self.calculate_NCE_loss(real_B, self.idt_B, enqueue=False, nce_weight=self.lambda_idt)
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE
        if self.lambda_LNCC > 0.0:
            loss_LNCC = self.criterionLNCC(self.fake_B, real_A) * self.lambda_LNCC
        else:
            loss_LNCC = 0.0
        if self._seg_loss_fn is not None and self.lambda_seg > 0.0:
            loss_seg = self._seg_loss_fn(self.fake_B, real_A) * self.lambda_seg
        else:
            loss_seg = 0.0
        if self._seg_loss_fn is not None and self.lambda_seg_idt > 0.0 and self.idt_B is not None:
            loss_seg_idt = self._seg_loss_fn(self.idt_B, real_B) * self.lambda_seg_idt
        else:
            loss_seg_idt = 0.0
        if self.criterionIdt is not None and self.idt_B is not None and self.training:
            loss_idt = self.criterionIdt(self.idt_B, real_B) * self.lambda_idt
        else:
            loss_idt = 0.0
        return loss_G_GAN + loss_NCE_both + loss_LNCC + loss_seg + loss_seg_idt + loss_idt

    # ── NCE ─────────────────────────────────────────────────────────

    def calculate_NCE_loss(self, src, tgt, enqueue=True, nce_weight=None):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = (self.netG_momentum(src, self.nce_layers, encode_only=True)
                  if self.moco else
                  self.netG(src, self.nce_layers, encode_only=True))
        mask = None
        if self.foreground_sampling:
            mask = self._compute_fg_mask(src)
        feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None, mask=mask)
        feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids, mask=mask)
        nce_w = self.lambda_NCE if nce_weight is None else nce_weight
        total_nce_loss = 0.0
        for i, (f_q, f_k, crit) in enumerate(
                zip(feat_q_pool, feat_k_pool, self.criterionNCE)):
            bank_i = (self.memory_bank._get_queue(i)
                      if self.moco and self.memory_bank.is_ready() else None)
            w = self.nce_layer_weights[i] if self.nce_layer_weights else 1.0
            total_nce_loss += (crit(f_q, f_k, bank=bank_i)
                               * nce_w * w).mean()
        if self.moco and enqueue and self.training:
            self._feat_k_for_enqueue = feat_k_pool
        return total_nce_loss / n_layers

    def _compute_fg_mask(self, src):
        if self.has_norm():
            bg = self._src_bg.view(1, -1, 1, 1)
        else:
            bg = -1.0
        mask = (src > bg).float()
        if mask.shape[1] > 1:
            mask = mask.amax(dim=1, keepdim=True)
        return mask

    # ── MoCo ────────────────────────────────────────────────────────

    def init_momentum_encoder(self):
        if self.netG_momentum is not None:
            self.netG_momentum.load_state_dict(self.netG.state_dict())

    def update_moco_momentum(self):
        if self.netG_momentum is None:
            return
        with torch.no_grad():
            for p_key, p_m in zip(self.netG.parameters(),
                                  self.netG_momentum.parameters()):
                p_m.data.mul_(self.moco_momentum).add_(
                    p_key.data, alpha=1.0 - self.moco_momentum)

    def enqueue_moco(self, feat_k_pool=None):
        """Push features into memory bank. Accepts optional explicit arg (CUT3D)."""
        pool = feat_k_pool if feat_k_pool is not None else getattr(self, "_feat_k_for_enqueue", None)
        if self.moco and pool is not None:
            self.memory_bank.enqueue(pool)
            self._feat_k_for_enqueue = None

    # ── EMA ─────────────────────────────────────────────────────────

    @staticmethod
    def _strip_module(key: str) -> str:
        return key.replace("module.", "", 1) if key.startswith("module.") else key

    def init_ema(self):
        self.ema_state = {}
        for k, v in self.netG.state_dict().items():
            key = self._strip_module(k)
            if all(s not in k for s in ("running_mean", "running_var",
                                         "num_batches_tracked")):
                self.ema_state[key] = v.detach().clone()

    def update_ema(self):
        if self.ema_state is None:
            return
        with torch.no_grad():
            for k, v in self.netG.state_dict().items():
                key = self._strip_module(k)
                if key in self.ema_state:
                    self.ema_state[key].mul_(self.ema_decay).add_(
                        v.detach(), alpha=1.0 - self.ema_decay)

    # ── NCE temperature ─────────────────────────────────────────────

    def set_nce_temperature(self, epoch: float, total_epochs: int):
        if self.nce_T_start == self.nce_T:
            return
        ratio = min(epoch / total_epochs, 1.0)
        t = self.nce_T_start + (self.nce_T - self.nce_T_start) * ratio
        for crit in self.criterionNCE:
            crit.nce_T = t

    # ── Optimize ────────────────────────────────────────────────────

    def optimize_parameters(self, real_A, real_B, max_grad_norm=None):
        self._feat_k_for_enqueue = None
        self.forward(real_A, real_B)
        for p in self.netD.parameters():
            p.requires_grad = True
        self.optimizer_D.zero_grad()
        loss_D = self.compute_D_loss(real_A, real_B)
        loss_D.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_grad_norm)
        self.optimizer_D.step()
        for p in self.netD.parameters():
            p.requires_grad = False
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        loss_G = self.compute_G_loss(real_A, real_B)
        loss_G.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.netF.parameters(), max_grad_norm)
        self.optimizer_G.step()
        self.optimizer_F.step()
        self.enqueue_moco()
        self.update_moco_momentum()
        if self.ema_state is not None:
            self.update_ema()
        return {"loss_D": loss_D.item(), "loss_G": loss_G.item()}

    # ── Norm ────────────────────────────────────────────────────────

    def set_norm_params(self, src_mu, src_sigma, src_bg,
                        tgt_mu, tgt_sigma, tgt_bg):
        device = next(self.parameters()).device
        self.register_buffer("_src_mu",
                             torch.tensor(src_mu, dtype=torch.float32, device=device))
        self.register_buffer("_src_sigma",
                             torch.tensor(src_sigma, dtype=torch.float32, device=device))
        self.register_buffer("_src_bg",
                             torch.tensor(src_bg, dtype=torch.float32, device=device))
        self.register_buffer("_tgt_mu",
                             torch.tensor(tgt_mu, dtype=torch.float32, device=device))
        self.register_buffer("_tgt_sigma",
                             torch.tensor(tgt_sigma, dtype=torch.float32, device=device))
        self.register_buffer("_tgt_bg",
                             torch.tensor(tgt_bg, dtype=torch.float32, device=device))

    def _denorm_view(self, t):
        """Reshape norm tensor to broadcast over spatial dims (2D or 3D)."""
        ndim = t.dim()
        return (1, -1) + (1,) * (ndim - 2) if ndim > 2 else (1, -1, 1, 1)

    def denorm_src(self, x):
        return x * self._src_sigma.view(self._denorm_view(x)) + self._src_mu.view(self._denorm_view(x))

    def denorm_tgt(self, x):
        return x * self._tgt_sigma.view(self._denorm_view(x)) + self._tgt_mu.view(self._denorm_view(x))

    def has_norm(self):
        return hasattr(self, "_src_mu")
