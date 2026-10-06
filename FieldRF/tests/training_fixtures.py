"""Small real MONAI networks and stochastic inputs for training regressions."""

from pathlib import Path
from types import SimpleNamespace
import random

import torch
import yaml

from fieldrf.training.vae import build_vae


RF_OPTIONS = dict(
    unet_channels=(8, 16), num_res_blocks=(1, 1),
    attention_levels=(False, False), num_head_channels=(4, 4),
    norm_num_groups=4, feat4_channels=4,
    conditioning_embedding_num_channels=(4, 4), use_flash_attention=False,
)


class RandomPatchDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 5

    def __getitem__(self, index):
        # Consume Python and torch randomness, exercising both saved streams.
        value = random.random() + index / 10
        return {
            "latent": torch.randn(4, 4, 4, 4) + value,
            "image": torch.randn(1, 8, 8, 8).tanh(),
            "modality": "T1W" if index % 2 else "T2W", "field": "7T",
        }


class RandomRFDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"z": torch.randn(4, 8, 8, 8) + random.random(),
                "feats": [torch.randn(4, 16, 16, 16)], "class_label": index % 2}


def make_vae_config(directory):
    root = Path(directory)
    config = {
        "data": {"data_root": str(root), "cache_root": str(root),
                 "split": "Training_retrospective", "modalities": ["T1W", "T2W"], "fields": ["7T"]},
        "vae": {"num_channels": [8, 8], "num_res_blocks": [1, 1],
                "norm_num_groups": 4, "attention_levels": [False, False],
                "norm_float16": False, "num_splits": 1, "checkpoint": str(root / "initial.pt")},
        "train": {"lr": 1e-3, "min_lr": 1e-6, "epochs": 2, "batch_size": 1,
                  "num_workers": 0, "freeze_encoder": True, "accum_steps": 2,
                  "warmup_steps": 1, "save_every": 1, "ssim_weight": 0,
                  "perceptual_weight": 0, "seed": 2026,
                  "loss_balance": {"enabled": True, "warmup_steps": 0}},
        "vis": {"vis_every": 0}, "output_dir": str(root / "continuous"),
    }
    torch.manual_seed(17)
    torch.save(build_vae(config["vae"]).state_dict(), root / "initial.pt")
    path = root / "vae.yaml"
    path.write_text(yaml.safe_dump(config))
    return config, path


def vae_args(path, distributed=False, resume=None):
    return SimpleNamespace(config=str(path), max_volumes=0, epochs=None,
                           resume_ckpt=resume, init_ckpt=None, dist=distributed, device="cpu")


def assert_nested_equal(testcase, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        testcase.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_nested_equal(testcase, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        testcase.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(testcase, a, b)
    else:
        testcase.assertEqual(left, right)
