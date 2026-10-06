"""Restormer model: CUT framework with MDTA/GDFN transformer generator."""

import torch
import torch.nn as nn

from .base_gan_model import BaseGANModel
from .networks_restormer import RestormerGenerator, RestormerGeneratorNoSkip
from .discriminators import NLayerDiscriminator, MultiscaleDiscriminator
from ..losses.ms_ssim import MSSSIMLoss


class RestormerModel(BaseGANModel):
    """CUT framework with Restormer generator + optional FM/MSSSIM losses."""

    def _store_config(self, opt):
        super()._store_config(opt)
        self.lambda_FM = opt.get("lambda_FM", 0.0)
        self.lambda_MSSSIM = opt.get("lambda_MSSSIM", 0.0)

    def _build_generator(self, opt):
        cls = RestormerGenerator if opt.get("use_skip", True) else RestormerGeneratorNoSkip
        return cls(
            opt.get("input_nc", 1), opt.get("output_nc", 1),
            opt.get("ngf", 64),
            n_blocks=opt.get("n_blocks", 9),
            use_tanh=opt.get("use_tanh", True),
            encoder_blocks=opt.get("encoder_blocks", [4, 6, 6, 8]),
        )

    def _build_discriminator(self, input_nc, opt):
        ndf = opt.get("ndf", 64)
        nl = opt.get("n_layers_D", 3)
        sn = opt.get("use_spectral_norm", False)
        if opt.get("multiscale", False):
            return MultiscaleDiscriminator(input_nc, ndf, n_layers=nl,
                                           num_D=opt.get("num_D", 2), use_spectral_norm=sn)
        return NLayerDiscriminator(input_nc, ndf, n_layers=nl, use_spectral_norm=sn)

    def _setup_losses(self, opt):
        super()._setup_losses(opt)
        if self.lambda_MSSSIM > 0:
            from ..losses.ms_ssim import MSSSIMLoss
            self.criterionMSSSIM = MSSSIMLoss()

    # ── Generator loss with FM + MSSSIM ────────────────────────────

    def compute_G_loss(self, real_A, real_B):
        loss_G = super().compute_G_loss(real_A, real_B)

        # Feature matching loss
        if self.lambda_FM > 0.0:
            fake = self.fake_B
            with torch.no_grad():
                pred_fake = self.netD(fake)
                pred_real = self.netD(real_B)
            loss_G = loss_G + self._compute_FM_loss(pred_fake, pred_real)

        # Multi-scale SSIM loss
        if self.lambda_MSSSIM > 0.0:
            loss_G = loss_G + self.criterionMSSSIM(self.fake_B, real_A) * self.lambda_MSSSIM

        return loss_G

    def _compute_FM_loss(self, pred_fake, pred_real):
        """Feature matching loss between D intermediate features."""
        if isinstance(pred_fake[0], list):
            loss_fm = 0.0
            n = 0
            for sf, sr in zip(pred_fake, pred_real):
                for f_fake, f_real in zip(sf, sr):
                    loss_fm += nn.functional.l1_loss(f_fake, f_real.detach())
                    n += 1
            return loss_fm / max(n, 1) * self.lambda_FM
        else:
            loss_fm = 0.0
            n = 0
            for f_fake, f_real in zip(pred_fake, pred_real):
                loss_fm += nn.functional.l1_loss(f_fake, f_real.detach())
                n += 1
            return loss_fm / max(n, 1) * self.lambda_FM
