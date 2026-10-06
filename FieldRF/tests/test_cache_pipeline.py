"""Regression checks for cache generation and the real MONAI RF network."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

FIELDRF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIELDRF_ROOT))
from data.dataset import _load_feats_b2nd, _load_latent
from models.model import FieldRFModel

spec = importlib.util.spec_from_file_location(
    "fieldrf_feature_cache", FIELDRF_ROOT / "scripts/cache_brainid_feat.py"
)
feature_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_cache)
from scripts.cache_latents import _save_latent


class CachePipelineTests(unittest.TestCase):
    def test_failed_latent_write_preserves_previous_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            _save_latent({"latent": torch.ones(1)}, path)
            previous = path.read_bytes()

            def interrupted_save(payload, temporary):
                Path(temporary).write_bytes(b"partial")
                raise OSError("interrupted write")

            with patch("scripts.cache_latents.torch.save", side_effect=interrupted_save):
                with self.assertRaisesRegex(OSError, "interrupted write"):
                    _save_latent({"latent": torch.zeros(1)}, path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_async_feature_save_failure_is_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = [{"filename": "subject.nii.gz"}]
            with patch.object(feature_cache, "_forward_one_with_retry", return_value=(None, {"feat4": torch.ones(1)})), \
                 patch.object(feature_cache, "_save_feats_b2nd", side_effect=OSError("disk full")):
                saved, skipped, errors, _, _ = feature_cache._run_field(
                    "7T", Path(directory), ds, None, "cpu",
                    {"batch_size": 1, "num_workers": 0}, [4], 1, 1, 1,
                )
            self.assertEqual((saved, skipped, errors), (0, 0, 1))

    def test_feature_cache_limit_includes_already_cached_volumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first.b2nd").write_bytes(b"existing")
            ds = [{"filename": "first.nii.gz"}, {"filename": "second.nii.gz"}]
            with patch.object(feature_cache, "_forward_one_with_retry") as forward:
                saved, skipped, errors, _, _ = feature_cache._run_field(
                    "7T", root, ds, None, "cpu",
                    {"batch_size": 1, "num_workers": 0}, [4], 1, 1, 1, max_volumes=1,
                )
            forward.assert_not_called()
            self.assertEqual((saved, skipped, errors), (1, 1, 0))

    def test_cache_entrypoints_help_without_brainid(self):
        env = dict(os.environ, BRAINID_ROOT="/nonexistent/Brain-ID", PYTHONPATH="")
        for name in ["cache_latents.py", "cache_brainid_feat.py"]:
            with self.subTest(entrypoint=name):
                result = subprocess.run(
                    [sys.executable, str(FIELDRF_ROOT / "scripts" / name), "--help"],
                    cwd=FIELDRF_ROOT, env=env, capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--modality", result.stdout)

    def test_cache_pool_rejects_incompatible_controlnet_geometry(self):
        cfg = yaml.safe_load((FIELDRF_ROOT / "configs/pretrain.yaml").read_text())
        self.assertEqual(feature_cache._resolve_feat4_pool(cfg), 1)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            feature_cache._resolve_feat4_pool(cfg, 2)
        cfg["bridge"]["conditioning_embedding_num_channels"] = [64]
        self.assertEqual(feature_cache._resolve_feat4_pool(cfg), 2)

    def test_native_cache_round_trip_trains_and_reloads_real_rf(self):
        torch.manual_seed(2026)
        kwargs = dict(
            unet_channels=(8, 16), num_res_blocks=(1, 1),
            attention_levels=(False, False), num_head_channels=(8, 8),
            norm_num_groups=8, feat4_channels=64,
            conditioning_embedding_num_channels=(8, 8), use_flash_attention=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latent = torch.randn(4, 8, 8, 8)
            feat4 = torch.randn(64, 16, 16, 16).half()
            torch.save({"latent": latent}, root / "latent.pt")
            feature_cache._save_feats_b2nd({"feat4": feat4}, root / "features.b2nd", 1)
            z = _load_latent(root / "latent.pt").unsqueeze(0)
            feats = [value.unsqueeze(0) for value in _load_feats_b2nd(root / "features.b2nd")]
            torch.testing.assert_close(z[0], latent)
            torch.testing.assert_close(feats[-1][0], feat4, rtol=0, atol=0)
            model = FieldRFModel(**kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            label = torch.tensor([14])
            t = torch.tensor([500.0])
            endpoint = torch.randn_like(z)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            pred, target = model.forward_rf(z, feats, label, t=t, endpoint=endpoint)
            loss = torch.nn.functional.l1_loss(pred, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))
            optimizer.step()
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()))
            checkpoint = root / "trained.pt"
            torch.save({"bridge": model.state_dict()}, checkpoint)
            restored = FieldRFModel.from_pretrained(checkpoint, **kwargs)
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(z, t, label, feats), restored(z, t, label, feats), rtol=0, atol=0)
                torch.manual_seed(1)
                generated = restored.sample_x0(feats, label, n_infer_steps=2, device="cpu")
            self.assertEqual(generated.shape, z.shape)
            self.assertTrue(torch.isfinite(generated).all())


if __name__ == "__main__":
    unittest.main()
