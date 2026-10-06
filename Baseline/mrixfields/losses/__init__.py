from .adversarial import GANLoss
from .patchnce import PatchNCELoss
from .perceptual import PerceptualLoss
from .sinkhorn import SinkhornLoss
from .structure import StructureLoss
from .tokennce import TokenNCELoss
from .ms_ssim import MSSSIMLoss
from .vinn_seg_loss import VINNSegLoss
from .lncc import LNCCloss

__all__ = [
    "GANLoss",
    "PatchNCELoss",
    "PerceptualLoss",
    "SinkhornLoss",
    "StructureLoss",
    "TokenNCELoss",
    "MSSSIMLoss",
    "VINNSegLoss",
    "LNCCloss",
]
