"""3D CUT model — ResnetGenerator3D + 3D discriminator + 3D NCE."""

import torch
import torch.nn as nn

from .base_gan_model import BaseGANModel
from .networks_cut3d import ResnetGenerator3D, NLayerDiscriminator3D, PatchSampleF3D
from .discriminators import MultiscaleDiscriminator3D
from ..losses.lncc import LNCCloss


class CUTModel3D(BaseGANModel):
    """3D CUT: 5D tensors, 3D Conv networks."""

    def _store_config(self, opt):
        super()._store_config(opt)
        self._use_checkpoint = opt.get("use_checkpoint", False)

    def _build_generator(self, opt):
        return ResnetGenerator3D(
            opt.get("input_nc", 1), opt.get("output_nc", 1),
            opt.get("ngf", 64), n_blocks=opt.get("n_blocks", 9),
            use_tanh=opt.get("use_tanh", True),
            skip_dropout=opt.get("skip_dropout", 0.0),
            use_checkpoint=self._use_checkpoint,
        )

    def _build_discriminator(self, input_nc, opt):
        ndf = opt.get("ndf", 64)
        nl = opt.get("n_layers_D", 3)
        sn = opt.get("use_spectral_norm", False)
        if opt.get("multiscale", False):
            return MultiscaleDiscriminator3D(input_nc, ndf, n_layers=nl, num_D=opt.get("num_D", 2), use_spectral_norm=sn)
        return NLayerDiscriminator3D(input_nc, ndf, n_layers=nl, use_spectral_norm=sn)

    def _build_netF(self, opt):
        return PatchSampleF3D(use_mlp=True, nc=opt.get("netF_nc", 256))

    def _setup_losses(self, opt):
        from ..losses.adversarial import GANLoss
        self.criterionGAN = GANLoss(mode=opt.get("gan_mode", "lsgan"))
        from ..losses.patchnce import PatchNCELoss
        self.criterionNCE = nn.ModuleList([PatchNCELoss(batch_size=opt.get("batch_size", 1), nce_T=self.nce_T) for _ in self.nce_layers])
        if self.lambda_LNCC > 0:
            self.criterionLNCC = LNCCloss()
        else:
            self.criterionLNCC = None

    def _post_data_init(self):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif hasattr(torch, "npu") and torch.npu.is_available(): torch.npu.empty_cache()

    # ── 3D overrides ────────────────────────────────────────────────

    def compute_G_loss(self, real_A, real_B):
        fake = self.fake_B
        loss_G_GAN = 0.0
        if self.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            if isinstance(pred_fake[0], list):
                loss_G_GAN = sum(self.criterionGAN.g_loss(pf[-1]).mean() for pf in pred_fake)
                loss_G_GAN = loss_G_GAN / len(pred_fake) * self.lambda_GAN
            else:
                loss_G_GAN = self.criterionGAN.g_loss(pred_fake[-1]).mean() * self.lambda_GAN
        loss_NCE = self.calculate_NCE_loss(real_A, fake) if self.lambda_NCE > 0.0 else 0.0
        if self.nce_idt and self.lambda_NCE > 0.0 and self.training:
            loss_NCE_Y = self.calculate_NCE_loss(real_B, self.idt_B, nce_weight=self.lambda_idt)
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE
        loss_LNCC_val = 0.0
        if self.lambda_LNCC > 0 and self.criterionLNCC is not None:
            fake_m11 = self.denorm_tgt(fake) if self.has_norm() else fake
            real_m11 = self.denorm_tgt(real_B) if self.has_norm() else real_B
            loss_LNCC_val = self.criterionLNCC(fake_m11, real_m11) * self.lambda_LNCC
        loss_idt = 0.0
        if self.criterionIdt is not None and self.idt_B is not None and self.training:
            loss_idt = self.criterionIdt(self.idt_B, real_B) * self.lambda_idt
        return loss_G_GAN + loss_NCE_both + loss_LNCC_val + loss_idt

    def calculate_NCE_loss(self, src, tgt, enqueue=True, nce_weight=None):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = (self.netG_momentum(src, self.nce_layers, encode_only=True) if self.moco else self.netG(src, self.nce_layers, encode_only=True))
        mask = None
        if self.foreground_sampling:
            bg_val = self._src_bg.mean().item() if self.has_norm() else -1.0
            mask = (src.amax(dim=1, keepdim=True) > bg_val).float()
        feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None, mask=mask)
        feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids, mask=mask)
        nce_w = self.lambda_NCE if nce_weight is None else nce_weight
        total_nce_loss = 0.0
        for i, (f_q, f_k, crit) in enumerate(zip(feat_q_pool, feat_k_pool, self.criterionNCE)):
            bank_i = self.memory_bank._get_queue(i) if (self.moco and self.memory_bank.is_ready()) else None
            w = self.nce_layer_weights[i] if self.nce_layer_weights else 1.0
            total_nce_loss += (crit(f_q, f_k, bank=bank_i) * nce_w * w).mean()
        if self.moco and enqueue:
            self.enqueue_moco(feat_k_pool)
        return total_nce_loss / n_layers

    def enqueue_moco(self, feat_k_pool=None):
        pool = feat_k_pool if feat_k_pool is not None else getattr(self, "_feat_k_for_enqueue", None)
        if self.memory_bank is not None and pool is not None:
            for i, f_k in enumerate(pool): self.memory_bank.enqueue(i, f_k)

    def update_moco_momentum(self):
        if not self.moco: return
        with torch.no_grad():
            for p, p_m in zip(self.netG.parameters(), self.netG_momentum.parameters()):
                p_m.data = self.moco_momentum * p_m.data + (1.0 - self.moco_momentum) * p.data

    def init_ema(self):
        self.ema_state = {}
        for k, v in self.netG.state_dict().items():
            if "running_" not in k and "num_batches" not in k:
                self.ema_state[k] = v.clone()

    def update_ema(self):
        if self.ema_state is None: return
        with torch.no_grad():
            for k, v in self.netG.state_dict().items():
                if k in self.ema_state and "running_" not in k and "num_batches" not in k:
                    self.ema_state[k] = self.ema_decay * self.ema_state[k] + (1.0 - self.ema_decay) * v

    def set_nce_temperature(self, epoch, n_epochs=None):
        if n_epochs is None: n_epochs = epoch
        if self.nce_T_start != self.nce_T and epoch < n_epochs:
            t = self.nce_T_start + (self.nce_T - self.nce_T_start) * epoch / max(1, n_epochs - 1)
            for crit in self.criterionNCE: crit.nce_T = t
