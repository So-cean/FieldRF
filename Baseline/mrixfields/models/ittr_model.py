"""ITTR model: CUT framework with ITTRGenerator (HPB-based Transformer).

Reference: Zheng et al., "ITTR: Unpaired Image-to-Image Translation with
Transformers", 2022.
"""

from .base_gan_model import BaseGANModel
from .networks_ittr import ITTRGenerator, PatchSampleF
from .discriminators import NLayerDiscriminator, MultiscaleDiscriminator


class ITTRModel(BaseGANModel):
    """ITTR: CUT framework with HPB Transformer generator."""

    def _store_config(self, opt):
        super()._store_config(opt)
        self.nce_layers = opt.get("nce_layers", [0, 1, 2, 3, 7])

    def _build_generator(self, opt):
        return ITTRGenerator(
            opt.get("input_nc", 1), opt.get("output_nc", 1),
            opt.get("ngf", 64),
            n_blocks=opt.get("n_blocks", 9),
            dim_head=opt.get("dim_head", 32),
            heads=opt.get("heads", 8),
            attn_height_top_k=opt.get("attn_height_top_k", 16),
            attn_width_top_k=opt.get("attn_width_top_k", 16),
            attn_dropout=opt.get("attn_dropout", 0.0),
            ff_dropout=opt.get("ff_dropout", 0.0),
            ff_mult=opt.get("ff_mult", 4),
            use_tanh=opt.get("use_tanh", True),
            use_conv_transpose=opt.get("use_conv_transpose", False),
        )

    def _build_discriminator(self, input_nc, opt):
        ndf = opt.get("ndf", 64)
        nl = opt.get("n_layers_D", 3)
        sn = opt.get("use_spectral_norm", False)
        if opt.get("multiscale", False):
            return MultiscaleDiscriminator(input_nc, ndf, n_layers=nl,
                                           num_D=opt.get("num_D", 2), use_spectral_norm=sn)
        return NLayerDiscriminator(input_nc, ndf, n_layers=nl, use_spectral_norm=sn)

    def _build_netF(self, opt):
        """ITTR uses its own PatchSampleF."""
        return PatchSampleF(use_mlp=True, nc=opt.get("netF_nc", 256))
