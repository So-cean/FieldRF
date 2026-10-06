"""NAFNet model: CUT framework with NAFNet generator backbone."""

from .base_gan_model import BaseGANModel
from .networks_nafnet import NAFNetGenerator
from .discriminators import NLayerDiscriminator, MultiscaleDiscriminator


class NAFNetModel(BaseGANModel):
    """CUT framework with NAFNet generator."""

    def _build_generator(self, opt):
        return NAFNetGenerator(
            img_channel=opt.get("input_nc", 1),
            img_channel_out=opt.get("output_nc", 1),
            width=opt.get("ngf", 32),
            enc_blk_nums=opt.get("enc_blk_nums", None),
            middle_blk_num=opt.get("middle_blk_num", 12),
            dec_blk_nums=opt.get("dec_blk_nums", None),
            DW_Expand=opt.get("DW_Expand", 2),
            FFN_Expand=opt.get("FFN_Expand", 2),
            drop_out_rate=opt.get("drop_out_rate", 0.),
            use_tanh=opt.get("use_tanh", True),
        )

    def _build_discriminator(self, input_nc, opt):
        ndf = opt.get("ndf", 64)
        nl = opt.get("n_layers_D", 3)
        sn = opt.get("use_spectral_norm", False)
        if opt.get("multiscale", False):
            return MultiscaleDiscriminator(input_nc, ndf, n_layers=nl,
                                           num_D=opt.get("num_D", 2), use_spectral_norm=sn)
        return NLayerDiscriminator(input_nc, ndf, n_layers=nl, use_spectral_norm=sn)
