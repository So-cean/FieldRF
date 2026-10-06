"""CUT3D trainer — independent (does NOT inherit from CUT trainer)."""

from ._pretrain import BasePretrainTrainer
from ._finetune import BaseFineTuneTrainer
from ..models import CUTModel3D
from ..models.networks_cut3d import ResnetGenerator3D


class CUT3DPretrainTrainer(BasePretrainTrainer):
    model_cls = CUTModel3D
    method_name = "cut3d"
    _is_3d = True

    def _model_kwargs(self):
        kw = super()._model_kwargs()
        mc = self.cfg.get("model", {})
        kw["ngf"] = mc.get("ngf", 32)
        kw["ndf"] = mc.get("ndf", 32)
        kw["skip_dropout"] = mc.get("skip_dropout", 0.0)
        kw["use_checkpoint"] = mc.get("use_checkpoint", False)
        return kw

    def _generator_config(self):
        gc = super()._generator_config()
        gc["kwargs"]["ngf"] = self.cfg.get("model", {}).get("ngf", 32)
        return gc


class CUT3DFineTuneTrainer(BaseFineTuneTrainer):
    generator_cls = ResnetGenerator3D
    method_name = "cut3d"
    _is_3d = True

    def _generator_kwargs(self):
        kw = super()._generator_kwargs()
        mc = self.cfg.get("model", {})
        kw["ngf"] = mc.get("ngf", 32)
        kw["use_checkpoint"] = mc.get("use_checkpoint", False)
        return kw

    def _generator_config(self):
        gc = super()._generator_config()
        gc["kwargs"]["ngf"] = self.cfg.get("model", {}).get("ngf", 32)
        return gc
