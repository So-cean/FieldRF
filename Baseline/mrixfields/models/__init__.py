"""Models module."""

from .vinn_seg import FastSurferVINN2D
from .base_gan_model import BaseGANModel
from .cut_model import CUTModel
from .cut3d_model import CUTModel3D
from .cyclegan_model import CycleGANModel
from .discriminators import NLayerDiscriminator, MultiscaleDiscriminator
from .ittr_model import ITTRModel
from .net_init import init_net, init_weights
from .networks_cut import ResnetGenerator as CutGenerator, PatchSampleF
from .networks_cut3d import ResnetGenerator3D, NLayerDiscriminator3D, PatchSampleF3D
from .networks_cyclegan import ResnetGenerator as CycleGANGenerator
from .networks_ittr import ITTRGenerator
from .nafnet_model import NAFNetModel
from .networks_nafnet import NAFNetGenerator
from .networks_restormer import RestormerGenerator, RestormerGeneratorNoSkip
from .networks_unsb import ResnetGenerator_ncsn, NLayerDiscriminator_ncsn
from .restormer_model import RestormerModel
from .starganv2_model import (
    StarGANv2Generator, MappingNetwork, StyleEncoder,
    StarGANv2Discriminator, build_stargan_v2,
)
from .unsb_model import UNSBModel

__all__ = [
    "BaseGANModel",
    "CUTModel",
    "CutGenerator",
    "CycleGANGenerator",
    "NAFNetGenerator",
    "NAFNetModel",
    "CUTModel3D",
    "CycleGANModel",
    "FastSurferVINN2D",
    "ITTRGenerator",
    "ITTRModel",
    "MappingNetwork",
    "MultiscaleDiscriminator",
    "NLayerDiscriminator",
    "NLayerDiscriminator3D",
    "NLayerDiscriminator_ncsn",
    "PatchSampleF",
    "ResnetGenerator3D",
    "ResnetGenerator_ncsn",
    "RestormerGenerator",
    "RestormerGeneratorNoSkip",
    "RestormerModel",
    "StarGANv2Discriminator",
    "StarGANv2Generator",
    "StyleEncoder",
    "UNSBModel",
    "build_stargan_v2",
    "init_net",
    "init_weights",
]
