"""NAFNet trainer — thin wrappers."""

from ._pretrain import BasePretrainTrainer
from ._finetune import BaseFineTuneTrainer
from ..models import NAFNetModel
from ..models.networks_nafnet import NAFNetGenerator


class NAFNetPretrainTrainer(BasePretrainTrainer):
    model_cls = NAFNetModel
    method_name = "nafnet"

    def _model_kwargs(self):
        kw = super()._model_kwargs()
        mc = self.cfg.get("model", {})
        kw["ngf"] = mc.get("ngf", 32)
        kw["enc_blk_nums"] = mc.get("enc_blk_nums")
        kw["middle_blk_num"] = mc.get("middle_blk_num", 12)
        kw["dec_blk_nums"] = mc.get("dec_blk_nums")
        kw["DW_Expand"] = mc.get("DW_Expand", 2)
        kw["FFN_Expand"] = mc.get("FFN_Expand", 2)
        kw["drop_out_rate"] = mc.get("drop_out_rate", 0.0)
        return kw


class NAFNetFineTuneTrainer(BaseFineTuneTrainer):
    generator_cls = NAFNetGenerator
    method_name = "nafnet"

    def _generator_kwargs(self):
        kw = super()._generator_kwargs()
        mc = self.cfg.get("model", {})
        kw["width"] = mc.get("ngf", 32)
        kw["enc_blk_nums"] = mc.get("enc_blk_nums")
        kw["middle_blk_num"] = mc.get("middle_blk_num", 12)
        kw["dec_blk_nums"] = mc.get("dec_blk_nums")
        kw["DW_Expand"] = mc.get("DW_Expand", 2)
        kw["FFN_Expand"] = mc.get("FFN_Expand", 2)
        kw["drop_out_rate"] = mc.get("drop_out_rate", 0.0)
        return kw
