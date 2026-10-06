"""Self-contained StarGAN v2 inference for joint modality/field domains."""

import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from .base_inference import BaseInference, predict_volume_slice_by_slice, strip_module_prefix
from ..data.utils import FIELD_STRENGTHS, FIELD_TO_DOMAIN, MODALITIES, get_joint_domain
from ..models.starganv2_model import build_stargan_v2


class StarGANv2Inference(BaseInference):
    """Inference for the single shared Task 3 Any-to-Any model."""

    def __init__(self, cfg, checkpoint_path, device, target_field=None,
                 target_domain=None, source_field=None, source_modality=None,
                 target_modality=None):
        configured_modalities = list(cfg.get("data", {}).get("modalities", ["T1W"]))
        self._joint_domains = len(configured_modalities) > 1

        if target_domain is not None:
            if self._joint_domains:
                if target_domain < 0 or target_domain >= len(MODALITIES) * len(FIELD_STRENGTHS):
                    raise ValueError(f"Invalid joint target_domain: {target_domain}")
                inferred_modality = MODALITIES[target_domain // len(FIELD_STRENGTHS)]
                inferred_field = FIELD_STRENGTHS[target_domain % len(FIELD_STRENGTHS)]
            else:
                if target_domain < 0 or target_domain >= len(FIELD_STRENGTHS):
                    raise ValueError(f"Invalid field target_domain: {target_domain}")
                inferred_modality = configured_modalities[0]
                inferred_field = FIELD_STRENGTHS[target_domain]
            if target_modality is not None and target_modality != inferred_modality:
                raise ValueError(
                    f"target_domain={target_domain} maps to {inferred_modality}, "
                    f"not target_modality={target_modality}.")
            if target_field is not None and target_field != inferred_field:
                raise ValueError(
                    f"target_domain={target_domain} maps to {inferred_field}, "
                    f"not target_field={target_field}.")
            target_modality = target_modality or inferred_modality
            target_field = target_field or inferred_field
        else:
            if target_field is None:
                raise ValueError("StarGAN v2 inference requires --target_field.")
            if self._joint_domains:
                if target_modality is None:
                    raise ValueError(
                        "--target_modality is required for a multi-modality StarGAN v2 checkpoint.")
                target_domain = get_joint_domain(target_modality, target_field)
            else:
                target_modality = target_modality or configured_modalities[0]
                target_domain = FIELD_TO_DOMAIN[target_field]

        if source_modality is None:
            if len(configured_modalities) == 1:
                source_modality = configured_modalities[0]
            else:
                raise ValueError(
                    "--source_modality is required for a multi-modality StarGAN v2 checkpoint.")
        if source_modality not in configured_modalities:
            raise ValueError(
                f"source_modality={source_modality} is not in checkpoint modalities "
                f"{configured_modalities}.")
        if target_modality not in configured_modalities:
            raise ValueError(
                f"target_modality={target_modality} is not in checkpoint modalities "
                f"{configured_modalities}.")

        self.target_domain = int(target_domain)
        self.target_field = target_field
        self.target_modality = target_modality
        self.source_modality = source_modality

        # BaseInference loads per-channel stats.  StarGAN processes one input
        # modality at a time, so present a one-channel view of the full config.
        inference_cfg = copy.deepcopy(cfg)
        inference_cfg.setdefault("data", {})
        inference_cfg["data"]["modalities"] = [source_modality]
        inference_cfg["data"]["target_field"] = target_field
        super().__init__(
            inference_cfg, checkpoint_path, device, source_field=source_field)

        if self._has_norm:
            stats_path = Path(checkpoint_path).parent.parent / "norm_stats.yaml"
            norm_stats = yaml.safe_load(stats_path.read_text()) or {}
            target_stats = norm_stats[f"{target_modality}/{target_field}"]
            self._mu_tgt = np.asarray([target_stats["mu"]], dtype=np.float32)
            self._sigma_tgt = np.asarray([target_stats["sigma"]], dtype=np.float32)

    def load_generator(self):
        state = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint_kwargs = state.get("generator_config", {}).get("kwargs", {})
        model_cfg = self.cfg["model"]
        kwargs = {
            "img_size": checkpoint_kwargs.get("img_size", model_cfg.get("img_size", 512)),
            "style_dim": checkpoint_kwargs.get("style_dim", model_cfg.get("style_dim", 64)),
            "latent_dim": checkpoint_kwargs.get("latent_dim", model_cfg.get("latent_dim", 16)),
            "num_domains": checkpoint_kwargs.get("num_domains", model_cfg.get("num_domains", 5)),
            "max_conv_dim": checkpoint_kwargs.get(
                "max_conv_dim", model_cfg.get("max_conv_dim", 512)),
            "input_nc": checkpoint_kwargs.get("input_nc", model_cfg.get("input_nc", 1)),
            "use_tanh": checkpoint_kwargs.get(
                "use_tanh", not self.cfg.get("data", {}).get("normalize", False)),
        }
        nets, nets_ema = build_stargan_v2(**kwargs)

        if "nets_ema" in state:
            for name, net in nets_ema.items():
                if name in state["nets_ema"]:
                    net.load_state_dict(strip_module_prefix(state["nets_ema"][name]))
        else:
            if state.get("ema_state"):
                nets_ema["generator"].load_state_dict(
                    strip_module_prefix(state["ema_state"]))
            elif state.get("generator"):
                nets_ema["generator"].load_state_dict(
                    strip_module_prefix(state["generator"]))
            mapping_state = state.get("mapping_network")
            if mapping_state:
                nets_ema["mapping_network"].load_state_dict(
                    strip_module_prefix(mapping_state))

        # Legacy checkpoints may only have online nets.
        if "nets_ema" not in state and "nets" in state:
            for name, net in nets_ema.items():
                if name in state["nets"]:
                    net.load_state_dict(strip_module_prefix(state["nets"][name]))

        for net in nets_ema.values():
            net.to(self.device).eval()
        self.nets_ema = nets_ema
        self.latent_dim = kwargs["latent_dim"]
        print(f"Loaded StarGAN v2 from: {self.checkpoint_path}")
        return nets_ema["generator"]

    def predict(self, volume, slice_axis=2, crop_size=None, batch_size=1):
        was_3d = volume.ndim == 3
        if was_3d:
            volume = volume[np.newaxis, ...]
        if volume.shape[0] != 1:
            raise ValueError(
                "StarGAN v2 inference accepts one source modality per run; "
                "pass one --input_dir and select --source_modality.")

        def forward(tensor):
            z = tensor.new_zeros((tensor.size(0), self.latent_dim))
            y = torch.full(
                (tensor.size(0),), self.target_domain,
                dtype=torch.long, device=tensor.device)
            style = self.nets_ema["mapping_network"](z, y)
            return self.model(tensor, style)

        output = predict_volume_slice_by_slice(
            forward, volume, self.device,
            slice_axis=slice_axis, crop_size=crop_size, batch_size=batch_size,
            mu_src=self._mu_src, sigma_src=self._sigma_src,
            mu_tgt=self._mu_tgt, sigma_tgt=self._sigma_tgt)
        return output[0] if was_3d else output


__all__ = ["StarGANv2Inference"]
