"""CUT trainer (2D) — thin wrappers around BasePretrainTrainer / BaseFineTuneTrainer."""

from ._pretrain import BasePretrainTrainer
from ._finetune import BaseFineTuneTrainer
from ..models import CUTModel
from ..models.networks_cut import ResnetGenerator


class CUTPretrainTrainer(BasePretrainTrainer):
    model_cls = CUTModel
    method_name = "cut"

    def _model_kwargs(self):
        kw = super()._model_kwargs()
        mc = self.cfg.get("model", {})
        tc = self.cfg.get("training", {})
        kw["skip_dropout"] = mc.get("skip_dropout", 0.0)
        kw["seg_loss_ckpt"] = tc.get("seg_loss_ckpt", "")
        kw["lambda_seg"] = tc.get("lambda_seg", 0.0)
        kw["lambda_seg_idt"] = tc.get("lambda_seg_idt", 0.0)
        kw["vinn_scale_factor"] = tc.get("vinn_scale_factor", 2.0)
        return kw

    def _generator_config(self):
        gc = super()._generator_config()
        gc["kwargs"]["skip_dropout"] = self.cfg.get("model", {}).get("skip_dropout", 0.0)
        return gc


class CUTFineTuneTrainer(BaseFineTuneTrainer):
    generator_cls = ResnetGenerator
    method_name = "cut"

    def _generator_kwargs(self):
        kw = super()._generator_kwargs()
        mc = self.cfg.get("model", {})
        kw["skip_dropout"] = mc.get("skip_dropout", 0.0)
        return kw

    def _generator_config(self):
        gc = super()._generator_config()
        gc["kwargs"]["skip_dropout"] = self.cfg.get("model", {}).get("skip_dropout", 0.0)
        return gc
