"""Unified training entry point for MRIxFields2026 baselines (native DDP).

Usage:
    # Single GPU
    python scripts/train.py --config configs/task1/cut/0.1T_to_7T_T1W.yaml --mode retro_scratch

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/task1/cut/0.1T_to_7T_T1W.yaml \
        --mode retro_scratch --dist
"""

import argparse
import torch
from pathlib import Path
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NPU initialization (if available)
import shutil
import importlib.util
if shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False

    # Ascend CANN optimization flags (auto-set for 910B + CANN 8.x)
    import os
    os.environ.setdefault("ACL_OP_COMPILER_CACHE_MODE", "enable")
    os.environ.setdefault("ACL_OP_COMPILER_CACHE_DIR", "/tmp/ascend_op_cache")
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    
# DEBUG
torch.autograd.set_detect_anomaly(True)

from mrixfields.env import load_env, get_data_dir, get_preprocessed_dir, get_output_dir
from mrixfields.utils_dist import init_dist, cleanup_dist
import mrixfields.trainers as T


METHOD_MAP = {
    "cut": "CUT",
    "cut3d": "CUT3D",
    "cyclegan": "CycleGAN",
    "nafnet": "NAFNet",
    "starganv2": "StarGANv2",
    "ittr": "ITTR",
    "restormer": "Restormer",
    "unsb": "UNSB",
}

_SUPPORT_FINETUNE = {"cut", "cut3d", "cyclegan", "nafnet", "starganv2", "ittr", "restormer"}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _save_actual_config(cfg: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Disable YAML anchors/aliases to avoid &id001 / *id001 references
    yaml.Dumper.ignore_aliases = lambda *args: True
    with open(output_dir / "config_actual.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _build_pretrain_cfg(cfg: dict, output_dir: str) -> dict:
    pretrain_cfg = cfg["pretrain"]
    return {
        **cfg,
        "training": {
            **pretrain_cfg,
            "output_dir": output_dir,
        },
    }





def _build_finetune_cfg(cfg: dict, output_dir: str, pretrain_ckpt: str) -> dict:
    ft_cfg = cfg["finetune"]
    return {
        **cfg,
        "pretrain": {"method": cfg["method"], "checkpoint": pretrain_ckpt},
        "model": {**cfg.get("model", {})},
        "training": {
            **ft_cfg,
            "output_dir": output_dir,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="MRIxFields2026 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="retro_scratch",
                        choices=["retro_scratch", "pro_scratch", "pro_pretrained"],
                        help="Training mode")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Path to pretrained generator checkpoint (skip pretrain step)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint in the output dir")
    parser.add_argument("--dist", action="store_true",
                        help="Use PyTorch DDP (launch with torchrun)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Inject paths from .env into config (only if not already set by config YAML)
    load_env()
    cfg.setdefault("data", {})
    cfg["data"].setdefault("data_dir", get_data_dir())
    cfg["data"].setdefault("preprocessed_dir", get_preprocessed_dir())

    method = cfg["method"]
    task_name = cfg.get("task_name", f"{method}_unknown")

    # Output dirs with mode suffix
    output_dir = cfg.get("output_dir") or get_output_dir()
    output_base = Path(output_dir) / task_name / method / args.mode

    # Initialize DDP if requested
    if args.dist:
        init_dist()

    # Record runtime info for trainers
    cfg["_runtime"] = {
        "mode": args.mode,
        "config_path": str(Path(args.config).resolve()),
        "pretrain_ckpt": args.pretrain_ckpt,
        "resume": args.resume,
        "use_dist": args.dist,
    }

    # ------------------------------------------------------------------ #
    #  Mode: retro_scratch — pretrain only
    # ------------------------------------------------------------------ #
    if args.mode == "retro_scratch":
        run_cfg = _build_pretrain_cfg(cfg, str(output_base))

        prefix = METHOD_MAP[method]
        trainer_name = "StarGANv2Trainer" if method == "starganv2" else f"{prefix}PretrainTrainer"
        trainer_cls = getattr(T, trainer_name)
        trainer = trainer_cls(run_cfg)

        _save_actual_config(run_cfg, output_base)
        trainer.run()

    # ------------------------------------------------------------------ #
    #  Mode: pro_pretrained — finetune a pretrained model
    # ------------------------------------------------------------------ #
    elif args.mode == "pro_pretrained":
        pretrain_ckpt = args.pretrain_ckpt
        if pretrain_ckpt is None:
            default_ckpt = output_base.parent / "retro_scratch" / "weights" / "generator_final.pth"
            if not default_ckpt.exists():
                raise FileNotFoundError(
                    f"No pretrain checkpoint found at {default_ckpt}. "
                    f"Run --mode retro_scratch first, or use --pretrain_ckpt to specify."
                )
            pretrain_ckpt = str(default_ckpt)

        if method not in _SUPPORT_FINETUNE:
            raise NotImplementedError(f"{method} does not support pro_pretrained yet. Use retro_scratch.")

        prefix = METHOD_MAP[method]
        run_cfg = _build_finetune_cfg(cfg, str(output_base), pretrain_ckpt)
        trainer_name = ("StarGANv2FineTuneTrainer" if method == "starganv2"
                        else f"{prefix}FineTuneTrainer")
        trainer_cls = getattr(T, trainer_name)
        trainer = trainer_cls(run_cfg)

        trainer.run()
        _save_actual_config(run_cfg, output_base)

    # ------------------------------------------------------------------ #
    #  Mode: pro_scratch — supervised training from scratch
    # ------------------------------------------------------------------ #
    elif args.mode == "pro_scratch":
        if method not in _SUPPORT_FINETUNE:
            raise NotImplementedError(f"{method} does not support pro_scratch yet. Use retro_scratch.")

        prefix = METHOD_MAP[method]
        run_cfg = _build_finetune_cfg(cfg, str(output_base), None)
        if method == "starganv2":
            trainer_cls = getattr(T, "StarGANv2FineTuneTrainer")
            trainer = trainer_cls(run_cfg)
        else:
            trainer_cls = getattr(T, f"{prefix}FineTuneTrainer")
            trainer = trainer_cls(run_cfg)

        trainer.run()
        _save_actual_config(run_cfg, output_base)

    if args.dist:
        cleanup_dist()


if __name__ == "__main__":
    main()
