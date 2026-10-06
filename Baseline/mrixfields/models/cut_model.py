"""CUT (Contrastive Unpaired Translation) model.

Thin subclass of ``BaseGANModel`` — only overrides ``_build_generator``
and ``_build_discriminator``.
"""

from .base_gan_model import BaseGANModel
from .networks_cut import ResnetGenerator
from .discriminators import NLayerDiscriminator, MultiscaleDiscriminator


class CUTModel(BaseGANModel):
    """CUT: 2D ResnetGenerator + NLayer/Multiscale Discriminator."""

    def _build_generator(self, opt):
        return ResnetGenerator(
            opt.get("input_nc", 1),
            opt.get("output_nc", 1),
            opt.get("ngf", 64),
            n_blocks=opt.get("n_blocks", 9),
            use_tanh=opt.get("use_tanh", True),
            skip_dropout=opt.get("skip_dropout", 0.0),
        )

    def _build_discriminator(self, input_nc, opt):
        ndf = opt.get("ndf", 64)
        n_layers = opt.get("n_layers_D", 3)
        sn = opt.get("use_spectral_norm", False)
        if opt.get("multiscale", False):
            return MultiscaleDiscriminator(
                input_nc, ndf, n_layers=n_layers,
                num_D=opt.get("num_D", 2),
                use_spectral_norm=sn,
            )
        return NLayerDiscriminator(
            input_nc, ndf, n_layers=n_layers,
            use_spectral_norm=sn,
        )
