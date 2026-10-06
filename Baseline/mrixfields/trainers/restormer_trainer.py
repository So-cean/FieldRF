"""Restormer trainer — thin wrappers."""

from ._pretrain import BasePretrainTrainer
from ._finetune import BaseFineTuneTrainer
from ..models import RestormerModel
from ..models.networks_restormer import RestormerGenerator


class RestormerPretrainTrainer(BasePretrainTrainer):
    model_cls = RestormerModel
    method_name = "restormer"

    def _model_kwargs(self):
        kw = super()._model_kwargs()
        mc = self.cfg.get("model", {})
        tc = self.cfg.get("training", {})
        kw["encoder_blocks"] = mc.get("encoder_blocks", [4, 6, 6, 8])
        kw["use_skip"] = mc.get("use_skip", True)
        kw["lambda_FM"] = tc.get("lambda_FM", 0.0)
        kw["lambda_MSSSIM"] = tc.get("lambda_MSSSIM", 0.0)
        return kw

    def _generator_config(self):
        gc = super()._generator_config()
        mc = self.cfg.get("model", {})
        gc["kwargs"]["encoder_blocks"] = mc.get("encoder_blocks", [4, 6, 6, 8])
        return gc


class RestormerFineTuneTrainer(BaseFineTuneTrainer):
    generator_cls = RestormerGenerator
    method_name = "restormer"

    def _generator_kwargs(self):
        kw = super()._generator_kwargs()
        mc = self.cfg.get("model", {})
        kw["encoder_blocks"] = mc.get("encoder_blocks", [4, 6, 6, 8])
        kw["use_skip"] = mc.get("use_skip", True)
        return kw

    def _generator_config(self):
        gc = super()._generator_config()
        mc = self.cfg.get("model", {})
        gc["kwargs"]["encoder_blocks"] = mc.get("encoder_blocks", [4, 6, 6, 8])
        return gc
