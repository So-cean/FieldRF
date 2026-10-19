"""Read-only checks of the supplied FieldRF reproduction recipe before training.

Cache inspection reads tensor/container metadata, never full feature volumes.
It checks geometry and coverage; model restoration remains strict in the loaders.
"""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path

import yaml

from fieldrf.utils.artifacts import resolve_artifact, validate_rf_base
from fieldrf.utils.domains import MODALITIES, FIELDS

CONFIGS = {
    "pretrain": "pretrain.yaml", "endpoint": "endpoint_alignment.yaml",
    "vae": "vae_finetune.yaml", "paired": "finetune.yaml",
}
BRIDGE_KEYS = ("unet_channels", "num_res_blocks", "attention_levels",
               "num_head_channels", "norm_num_groups", "feat4_channels",
               "num_train_timesteps")
VAE_ENCODER_KEYS = ("spatial_dims", "in_channels", "latent_channels", "num_channels",
                    "num_res_blocks", "norm_num_groups", "norm_eps",
                    "attention_levels", "with_encoder_nonlocal_attn")


@dataclass
class CheckReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    domains: list = field(default_factory=list)

    @property
    def passed(self):
        return not self.errors

    def attempt(self, label, action):
        try:
            return action()
        except Exception as error:
            self.errors.append(f"{label}: {error}")
            return None


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    return path


def load_header(path):
    import torch
    # Memory-map tensor storage; no network construction or volume decoding.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def conditioning_channels(config):
    bridge = config.get("bridge", {})
    return bridge.get("conditioning_embedding_num_channels", bridge.get(
        "controlnet", {}).get("conditioning_embedding_num_channels", [64, 64]))


def _check_geometry(latent, feature, config, modality, field_name):
    payload = load_header(latent)
    shape = tuple(payload["latent"].shape)
    if shape != (4, 96, 96, 96):
        raise ValueError(f"Expected latent (4,96,96,96), got {shape}: {latent}")
    for key, expected in (("modality", modality), ("field_strength", field_name)):
        if key in payload and str(payload[key]) != expected:
            raise ValueError(f"Latent {key} label differs from its directory: {latent}")
    if feature is None:
        return
    import blosc2
    import numpy as np
    array = blosc2.open(str(feature), mode="r")
    shapes, keys = array.meta["shapes"], array.meta["keys"]
    factor = 2 ** (len(conditioning_channels(config)) - 1)
    expected = (64, *(size * factor for size in shape[1:]))
    if "feat4" not in keys or tuple(shapes.get("feat4", ())) != expected:
        raise ValueError(f"Expected Brain-ID feat4 {expected}, got {shapes}: {feature}")
    elements = sum(math.prod(shapes[key]) for key in keys)
    if array.shape != (elements,) or np.dtype(array.dtype) != np.dtype("float16"):
        raise ValueError(f"Invalid feature buffer length or dtype: {feature}")


def _domains(config, split, fields, report, *, root, features=True, caches=True,
             subjects=None):
    data = config["data"]
    modalities = data.get("modalities", [data.get("modality", "T1W")])
    data_root = resolve_artifact(data["data_root"], root)
    cache_root = resolve_artifact(data["cache_root"], root)
    fields = list(fields)
    if not fields:
        report.errors.append(f"{split}: field selection must be nonempty")
    for modality in modalities:
        for field_name in dict.fromkeys(str(value) for value in fields):
            label = f"{split}/{modality}/{field_name}"
            if modality not in MODALITIES or field_name not in FIELDS:
                report.errors.append(f"Unsupported MRI domain: {label}")
                continue
            raw_dir = data_root / split / modality / field_name
            raw = {p.name[:-7]: p for p in raw_dir.glob("*.nii.gz")}
            if subjects is not None:
                expected = {f"P_{modality}_{field_name}_{s}" for s in subjects}
                missing = sorted(expected - raw.keys())
                if missing:
                    report.errors.append(f"{label}: missing paired subjects: {missing}")
                raw = {k: v for k, v in raw.items() if k in expected}
            if not raw:
                report.errors.append(f"{label}: no source NIfTI volumes in {raw_dir}")
            report.domains.append({"domain": label, "volumes": len(raw)})
            for source in raw.values():
                report.attempt(str(source), lambda: require_file(source))
            if not caches:
                continue
            latent_dir = cache_root / split / "latent" / modality / field_name
            feature_dir = cache_root / split / data.get("feat_cache_name", "brainid_feat") / modality / field_name
            # Extra latents would be read by the unpaired/VAE datasets too.
            if subjects is None:
                extra = {p.stem for p in latent_dir.glob("*.pt")} - raw.keys()
                if extra:
                    report.errors.append(f"{label}: latent caches without raw inputs: {sorted(extra)}")
            for stem, source in sorted(raw.items()):
                latent = latent_dir / f"{stem}.pt"
                feature = feature_dir / f"{stem}.b2nd" if features else None
                if report.attempt(str(latent), lambda: require_file(latent)) is None:
                    continue
                if feature is not None and report.attempt(str(feature), lambda: require_file(feature)) is None:
                    continue
                report.attempt(label, lambda: _check_geometry(latent, feature, config, modality, field_name))


def _artifact(config, key, report, root, *, rf=False):
    value = config
    for part in key.split("."):
        value = value.get(part, {})
    if not value or isinstance(value, dict):
        raise ValueError(f"Missing configuration setting: {key}")
    path = require_file(resolve_artifact(value, root))
    state = load_header(path)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{key} is not a nonempty checkpoint dictionary: {path}")
    if rf:
        if "bridge" not in state:
            raise ValueError(f"{key} must be a full trained RF checkpoint: {path}")
        old = state.get("config", {}).get("bridge", {})
        for setting in BRIDGE_KEYS:
            current = config.get("bridge", {}).get(setting)
            if setting in old and current is not None and current != old[setting]:
                raise ValueError(f"RF checkpoint architecture differs at bridge.{setting}")
        if "scale_factor" in state and not math.isclose(
            float(state["scale_factor"]), float(config["vae"]["scale_factor"]), rel_tol=1e-6
        ):
            raise ValueError("RF checkpoint and configuration VAE scale factors differ")
    return path


def check_stage(config, stage, *, root):
    """Validate one stage; paths follow the documented FieldRF working directory."""
    report = CheckReport()
    data = config["data"]
    if stage not in {*CONFIGS, "cache"}:
        raise ValueError(f"Unknown training stage: {stage}")
    if not data.get("modalities", [data.get("modality")]):
        report.errors.append("data.modalities must be nonempty")
    expected_split = "Training_prospective" if stage == "paired" else "Training_retrospective"
    if stage != "cache" and data.get("split") != expected_split:
        report.errors.append(f"{stage} optimization requires data.split={expected_split}")
    _artifact(config, "vae.checkpoint", report, root)
    if stage == "cache":
        _artifact(config, "brainid.ckpt_path", report, root)
        brainid = Path(os.environ.get("BRAINID_ROOT", str(Path(root).parent / "third_party/Brain-ID")))
        require_file(brainid / "BrainID/__init__.py")
        for split in ("Training_retrospective", "Training_prospective", "Validating_prospective"):
            _domains(config, split, data["fields"], report, root=root, caches=False)
        return report
    if stage == "pretrain":
        _artifact(config, "bridge.ckpt_path", report, root)
        fields = data["fields"]
    elif stage == "endpoint":
        _artifact(config, "finetune_from", report, root, rf=True)
        alignment = config["unpaired_alignment"]
        fields = [*alignment["source_fields"], *alignment["reference_fields"]]
    elif stage == "vae":
        if not config["train"].get("freeze_encoder", True):
            report.errors.append("Cached-latent VAE adaptation requires train.freeze_encoder=true")
        fields = data["fields"]
    else:
        base = _artifact(config, "finetune_from", report, root, rf=True)
        endpoint = _artifact(config, "endpoint_aligner_checkpoint", report, root)
        state = load_header(endpoint)
        if "endpoint_aligner" not in state or "endpoint_aligner_config" not in state:
            report.errors.append("Endpoint checkpoint is missing aligner weights/configuration")
        report.attempt("RF/aligner association", lambda: validate_rf_base(state, base, root=root))
        if "base_checkpoint_sha256" not in state:
            report.warnings.append("Legacy aligner identity uses its original RF path; migrate before moving the files")
        if not data.get("subjects"):
            report.errors.append("Paired training requires nonempty data.subjects")
        fields = [*data["source_fields"], *data["target_fields"]]
    _domains(config, data["split"], fields, report, root=root,
             features=stage != "vae", subjects=data.get("subjects") if stage == "paired" else None)
    validation = config.get("validation" if stage == "endpoint" else "vis", {})
    if validation.get("split") and validation.get("vis_every", validation.get("every_epochs", 1)) > 0:
        _domains(config, validation["split"], fields, report, root=root, features=stage != "vae")
    return report


def check_pipeline(config_dir, stage="all", *, root):
    """Check shared configuration contracts and the selected ready-to-run stages."""
    report = CheckReport()
    configs = {}
    for name, filename in CONFIGS.items():
        def read_config(filename=filename):
            with (Path(config_dir) / filename).open(encoding="utf-8") as handle:
                value = yaml.safe_load(handle)
            if not isinstance(value, dict):
                raise ValueError("Expected a YAML mapping")
            return value
        value = report.attempt(filename, read_config)
        if value is not None:
            configs[name] = value
    if len(configs) != len(CONFIGS):
        return report
    initial = configs["pretrain"]
    for name, config in configs.items():
        def compare(config=config):
            for key in ("data_root", "cache_root"):
                if resolve_artifact(config["data"][key], root).resolve() != resolve_artifact(initial["data"][key], root).resolve():
                    raise ValueError(f"All stages must share data.{key}")
            if config["data"].get("feat_cache_name", "brainid_feat") != initial["data"].get("feat_cache_name", "brainid_feat"):
                raise ValueError("All stages must share data.feat_cache_name")
            if float(config["vae"]["scale_factor"]) != float(initial["vae"]["scale_factor"]):
                raise ValueError("All stages must share vae.scale_factor")
            for key in VAE_ENCODER_KEYS:
                if key in config["vae"] and key in initial["vae"] and config["vae"][key] != initial["vae"][key]:
                    raise ValueError(f"All stages must share VAE encoder architecture: vae.{key}")
            for key in BRIDGE_KEYS:
                if key in config.get("bridge", {}) and key in initial["bridge"] and config["bridge"][key] != initial["bridge"][key]:
                    raise ValueError(f"All RF stages must share bridge.{key}")
            if "bridge" in config and conditioning_channels(config) != conditioning_channels(initial):
                raise ValueError("All RF stages must share ControlNet conditioning geometry")
        report.attempt(name, compare)
    # These two stages must initialize from identical RF bytes, even after renaming.
    if stage in {"all", "endpoint", "paired"}:
        from fieldrf.utils.artifacts import checkpoint_sha256
        def same_base():
            a = resolve_artifact(configs["endpoint"]["finetune_from"], root)
            b = resolve_artifact(configs["paired"]["finetune_from"], root)
            if a.resolve() != b.resolve() and checkpoint_sha256(a) != checkpoint_sha256(b):
                raise ValueError("Endpoint and paired stages select different RF base checkpoints")
        report.attempt("Stage association", same_base)
    selected = list(CONFIGS) if stage == "all" else [stage]
    for name in selected:
        config = configs["pretrain" if name == "cache" else name]
        result = report.attempt(name, lambda: check_stage(config, name, root=root))
        if result is not None:
            report.errors.extend(result.errors)
            report.warnings.extend(result.warnings)
            report.domains.extend(result.domains)
    report.errors = list(dict.fromkeys(report.errors))
    report.warnings = list(dict.fromkeys(report.warnings))
    return report
