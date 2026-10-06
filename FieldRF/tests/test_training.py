"""Actual epoch loops must resume with the same optimizer, schedule and weights."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from fieldrf.training import vae
from fieldrf.training import pretrain
from fieldrf.models.model import FieldRFModel
from fieldrf.training.checkpoints import validate_vae_resume
from fieldrf.models.vae import prepare_decoder
from training_fixtures import make_vae_config, vae_args, RandomPatchDataset, RandomRFDataset, RF_OPTIONS, assert_nested_equal


class TrainingStateTests(unittest.TestCase):
    def test_rf_epoch_resume_matches_uninterrupted_training(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialization = root / "initial.pt"
            torch.save({"bridge": FieldRFModel(**RF_OPTIONS).state_dict()}, initialization)
            config = {
                "bridge": {key: list(value) if isinstance(value, tuple) else value for key, value in RF_OPTIONS.items()} | {"ckpt_path": str(initialization)},
                "vae": {"scale_factor": 1.0},
                "data": {"cache_root": str(root), "modality": "T1W", "modalities": ["T1W"], "fields": ["7T"]},
                "train": {"n_epochs": 2, "lr": 1e-3, "warmup_epochs": 0, "num_workers": 0, "batch_size": 1,
                          "save_every": 1, "ema_decay": 0.9}, "vis": {"vis_every": 0},
                "output_dir": str(root / "continuous"),
            }
            path = root / "rf.yaml"
            path.write_text(yaml.safe_dump(config))
            args = SimpleNamespace(config=str(path), dist=False, device="cpu", resume=None,
                                   pretrain_ckpt=None, n_epochs=None, max_volumes=0)
            with patch.object(pretrain, "MultiFieldCacheDataset", RandomRFDataset):
                pretrain.run(args)
                full = torch.load(root / "continuous" / "fieldrf_epoch2.pt", weights_only=False)
                args.resume = str(root / "continuous" / "fieldrf_epoch1.pt")
                config["output_dir"] = str(root / "resumed")
                path.write_text(yaml.safe_dump(config))
                pretrain.run(args)
            resumed = torch.load(root / "resumed" / "fieldrf_epoch2.pt", weights_only=False)
            for key in ("bridge", "optimizer", "scheduler", "ema", "epoch"):
                assert_nested_equal(self, full[key], resumed[key])

    def test_vae_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path = make_vae_config(directory)
            initial = torch.load(config["vae"]["checkpoint"], weights_only=False)
            with patch.object(vae, "LatentImageDataset", RandomPatchDataset):
                vae.run(vae_args(path))
                full = torch.load(Path(config["output_dir"]) / "vae_finetune_epoch2.pt", weights_only=False)
                first = Path(config["output_dir"]) / "vae_finetune_epoch1.pt"
                config["output_dir"] = str(Path(directory) / "resumed")
                path.write_text(yaml.safe_dump(config))
                vae.run(vae_args(path, resume=str(first)))
            resumed = torch.load(Path(config["output_dir"]) / "vae_finetune_epoch2.pt", weights_only=False)
            for key in ("state_dict", "optimizer", "scheduler", "epoch", "global_step", "ema_loss", "balance_step"):
                assert_nested_equal(self, full[key], resumed[key])
            self.assertEqual(full["global_step"], 6)  # 5 minibatches -> 3 updates per epoch
            self.assertTrue(full["optimizer"]["state"])
            for name, value in initial.items():
                if name.startswith(("encoder.", "quant_conv_mu.", "quant_conv_log_sigma.")):
                    torch.testing.assert_close(value, full["state_dict"]["autoencoder." + name], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, full["state_dict"]["autoencoder." + name])
                                for name, value in initial.items() if name.startswith("decoder.")))

    def test_resume_rejects_weights_only_artifacts(self):
        with self.assertRaisesRegex(ValueError, "Use --init_ckpt"):
            validate_vae_resume({"state_dict": {}})

    def test_decode_wrapper_matches_maisi_and_freezes_encoding_path(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = make_vae_config(directory)
            model = vae.build_vae(config["vae"])
            keys = list(model.state_dict())
            decoder = prepare_decoder(model)
            latent = torch.randn(1, 4, 4, 4, 4)
            torch.testing.assert_close(decoder(latent), model.decode(latent), rtol=0, atol=0)
            self.assertEqual(keys, list(model.state_dict()))
            names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            self.assertTrue(names)
            self.assertTrue(all(name.startswith(("decoder.", "post_quant_conv.")) for name in names))


if __name__ == "__main__":
    unittest.main()
