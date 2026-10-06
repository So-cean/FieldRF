"""Inference module."""

from .base_inference import (
    BaseInference, strip_module_prefix,
    extract_generator_state, predict_volume_slice_by_slice,
)
from .cut_inference import CUTInference
from .cut3d_inference import CUT3DInference
from .cyclegan_inference import CycleGANInference
from .ittr_inference import ITTRInference
from .nafnet_inference import NAFNetInference
from .restormer_inference import RestormerInference
from .stargan_inference import StarGANv2Inference
from .unsb_inference import UNSBInference

__all__ = [
    "BaseInference",
    "CUTInference",
    "CUT3DInference",
    "CycleGANInference",
    "ITTRInference",
    "NAFNetInference",
    "RestormerInference",
    "StarGANv2Inference",
    "UNSBInference",
    "extract_generator_state",
    "predict_volume_slice_by_slice",
    "strip_module_prefix",
]
