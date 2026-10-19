"""Preflight catches missing domains, invalid native caches and stage drift."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import blosc2
import numpy as np
import torch
import yaml

from fieldrf.utils.preflight import CONFIGS, check_pipeline, check_stage


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        weights = self.root / "pretrained.pt"
        torch.save({"weight": torch.ones(1)}, weights)
        self.config = {
            "data": {"data_root": str(self.root / "data"),
                     "cache_root": str(self.root / "cache"),
                     "split": "Training_retrospective", "modalities": ["T1W"],
                     "fields": ["7T"]},
            "vae": {"checkpoint": str(weights), "scale_factor": 1.0},
            "bridge": {"ckpt_path": str(weights), "unet_channels": [8, 16]},
            "train": {"freeze_encoder": True},
        }
        self.stem = "P_T1W_7T_0001"
        raw_dir = self.root / "data/Training_retrospective/T1W/7T"
        raw_dir.mkdir(parents=True)
        (raw_dir / f"{self.stem}.nii.gz").write_bytes(b"raw volume placeholder")
        self.latent = self.root / f"cache/Training_retrospective/latent/T1W/7T/{self.stem}.pt"
        self.latent.parent.mkdir(parents=True)
        torch.save({"latent": torch.zeros(4, 96, 96, 96), "modality": "T1W",
                    "field_strength": "7T"}, self.latent)
        self.feature = self.root / f"cache/Training_retrospective/brainid_feat/T1W/7T/{self.stem}.b2nd"
        self.feature.parent.mkdir(parents=True)
        # Native sparse uninitialized chunks let us test full-volume headers
        # without allocating/decompressing a 64x192x192x192 feature tensor.
        blosc2.empty((64 * 192**3,), dtype=np.float16, chunks=(1024*1024,),
                     blocks=(256*1024,), urlpath=str(self.feature), mode="w",
                     meta={"keys": ["feat4"], "shapes": {"feat4": [64,192,192,192]}})

    def test_native_cache_headers_pass_without_feature_decompression(self):
        report = check_stage(self.config, "pretrain", root=self.root)
        self.assertTrue(report.passed, report.errors)
        self.assertEqual(report.domains[0]["volumes"], 1)

    def test_missing_domains_and_incomplete_caches_are_reported(self):
        self.config["data"]["fields"].append("1.5T")
        self.feature.unlink()
        report = check_stage(self.config, "pretrain", root=self.root)
        self.assertFalse(report.passed)
        self.assertTrue(any("Missing or empty" in error for error in report.errors))
        self.assertTrue(any("1.5T" in error and "no source" in error for error in report.errors))

    def test_wrong_latent_shape_or_label_is_rejected(self):
        torch.save({"latent": torch.zeros(4, 8, 8, 8)}, self.latent)
        report = check_stage(self.config, "pretrain", root=self.root)
        self.assertTrue(any("Expected latent" in error for error in report.errors))
        torch.save({"latent": torch.zeros(4, 96, 96, 96), "modality": "T2W"}, self.latent)
        report = check_stage(self.config, "pretrain", root=self.root)
        self.assertTrue(any("label differs" in error for error in report.errors))

    def test_incompatible_brainid_geometry_is_rejected(self):
        self.config["bridge"]["conditioning_embedding_num_channels"] = [64]
        report = check_stage(self.config, "pretrain", root=self.root)
        self.assertTrue(any("Expected Brain-ID feat4" in error for error in report.errors))

    def test_stage_configs_cannot_silently_change_cache_or_architecture(self):
        config_dir = self.root / "configs"
        config_dir.mkdir()
        for name, filename in CONFIGS.items():
            config = deepcopy(self.config)
            if name == "paired":
                config["data"]["cache_root"] = str(self.root / "wrong_cache")
                config["bridge"]["unet_channels"] = [16, 32]
            (config_dir / filename).write_text(yaml.safe_dump(config))
        report = check_pipeline(config_dir, "pretrain", root=self.root)
        self.assertTrue(any("share data.cache_root" in error for error in report.errors))
        self.assertFalse(report.passed)
        paired_path = config_dir / CONFIGS["paired"]
        paired = yaml.safe_load(paired_path.read_text())
        paired["data"]["cache_root"] = self.config["data"]["cache_root"]
        paired_path.write_text(yaml.safe_dump(paired))
        report = check_pipeline(config_dir, "pretrain", root=self.root)
        self.assertTrue(any("share bridge.unet_channels" in error for error in report.errors))

    def test_paired_stage_rejects_validation_split_and_missing_subjects(self):
        config = deepcopy(self.config)
        config["data"]["split"] = "Validating_prospective"
        config["data"]["source_fields"] = ["7T"]
        config["data"]["target_fields"] = ["7T"]
        config["data"]["subjects"] = ["0009"]
        base = self.root / "base.pt"
        torch.save({"bridge": {"weight": torch.ones(1)}}, base)
        config["finetune_from"] = str(base)
        aligner = self.root / "aligner.pt"
        torch.save({"endpoint_aligner": {}, "endpoint_aligner_config": {},
                    "base_checkpoint": str(base)}, aligner)
        config["endpoint_aligner_checkpoint"] = str(aligner)
        report = check_stage(config, "paired", root=self.root)
        self.assertTrue(any("requires data.split=Training_prospective" in e for e in report.errors))
        self.assertTrue(any("missing paired subjects" in e for e in report.errors))
