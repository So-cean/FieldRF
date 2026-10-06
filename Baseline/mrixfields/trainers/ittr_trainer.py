"""ITTR trainer — thin wrappers."""

from ._pretrain import BasePretrainTrainer
from ._finetune import BaseFineTuneTrainer
from ..models import ITTRModel
from ..models.networks_ittr import ITTRGenerator


class ITTRPretrainTrainer(BasePretrainTrainer):
    model_cls = ITTRModel
    method_name = "ittr"

    def _model_kwargs(self):
        kw = super()._model_kwargs()
        mc = self.cfg.get("model", {})
        kw["use_conv_transpose"] = mc.get("use_conv_transpose", False)
        return kw


class ITTRFineTuneTrainer(BaseFineTuneTrainer):
    generator_cls = ITTRGenerator
    method_name = "ittr"

    def _generator_kwargs(self):
        kw = super()._generator_kwargs()
        mc = self.cfg.get("model", {})
        kw["use_conv_transpose"] = mc.get("use_conv_transpose", False)
        return kw
