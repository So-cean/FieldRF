"""Checkpoint contracts for the decomposed FieldRF model."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import FieldRFModel  # noqa: E402
from models.endpoint_aligner import EndpointAligner
from scripts.inference import load_endpoint_aligner, resolve_inversion
from models.checkpoint import load_vae_checkpoint


MODEL_OPTIONS = dict(
    unet_channels=(8, 16, 16, 16),
    num_res_blocks=(1, 1, 1, 1),
    attention_levels=(False, False, False, False),
    num_head_channels=(4, 4, 4, 4),
    norm_num_groups=4,
    feat4_channels=4,
    conditioning_embedding_num_channels=(4, 4),
    z_channels=1,
    use_flash_attention=False,
)


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "checkpoint.pt"
        self.model = FieldRFModel(**MODEL_OPTIONS)

    def load(self, payload, **options):
        torch.save(payload, self.checkpoint)
        return FieldRFModel.from_pretrained(
            self.checkpoint, **(MODEL_OPTIONS | options)
        )

    def test_full_checkpoint_restores_ema_and_runtime_metadata(self):
        state = self.model.state_dict()
        ema = {
            name: value + 0.125
            for name, value in state.items()
            if value.dtype.is_floating_point
        }
        loaded = self.load(
            {"bridge": state, "ema": ema, "scale_factor": 0.8,
             "num_train_timesteps": 500},
            load_ema=True,
        )
        self.assertEqual(loaded.scale_factor, 0.8)
        self.assertEqual(loaded.num_train_timesteps, 500)
        self.assertEqual(list(state), list(loaded.state_dict()))
        for name, value in ema.items():
            torch.testing.assert_close(loaded.state_dict()[name], value)

    def test_concat_continuation_preserves_old_channels_and_zeros_source(self):
        state = self.model.state_dict()
        loaded = self.load({"bridge": state}, source_injection="concat")
        actual = loaded.state_dict()["unet.conv_in.conv.weight"]
        torch.testing.assert_close(actual[:, :1], state["unet.conv_in.conv.weight"])
        torch.testing.assert_close(actual[:, 1:], torch.zeros_like(actual[:, 1:]))

    def test_adapter_continuation_keeps_new_residual_zero(self):
        loaded = self.load(
            {"bridge": self.model.state_dict()},
            source_injection="adapter", source_adapter_channels=4,
        )
        source = torch.randn(1, 1, 4, 4, 4)
        torch.testing.assert_close(
            loaded.source_adapter(source), torch.zeros(1, 8, 4, 4, 4)
        )

    def test_full_checkpoint_rejects_missing_network_parameters(self):
        state = dict(self.model.state_dict())
        state.pop("unet.conv_in.conv.weight")
        with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
            self.load({"bridge": state})

    def test_ema_rejects_missing_network_parameters(self):
        state = self.model.state_dict()
        ema = dict(state)
        ema.pop("unet.conv_in.conv.weight")
        with self.assertRaisesRegex(RuntimeError, "EMA checkpoint missing"):
            self.load({"bridge": state, "ema": ema}, load_ema=True)

    def test_raw_pretrained_unet_strips_ddp_prefix_and_reinitializes_labels(self):
        raw = {
            "module." + name: value
            for name, value in self.model.unet.state_dict().items()
        }
        raw["module.class_embedding.weight"] = torch.ones(128, 32)
        loaded = self.load({"unet_state_dict": raw})
        for name, value in self.model.unet.state_dict().items():
            if name != "class_embedding.weight":
                torch.testing.assert_close(loaded.unet.state_dict()[name], value)
        self.assertEqual(loaded.unet.class_embedding.weight.shape[0], 15)

    def test_residual_sampling_pretrained_weights_load_and_continue(self):
        residual_model = FieldRFModel(**MODEL_OPTIONS, resblock_updown=True)
        loaded = self.load({"unet_state_dict": residual_model.unet.state_dict()})
        self.assertTrue(loaded.resblock_updown)
        for name, value in residual_model.unet.state_dict().items():
            if name != "class_embedding.weight":
                torch.testing.assert_close(loaded.unet.state_dict()[name], value, rtol=0, atol=0)
        continued = self.load({"bridge": loaded.state_dict()})
        self.assertTrue(continued.resblock_updown)
        for name, value in loaded.state_dict().items():
            torch.testing.assert_close(continued.state_dict()[name], value, rtol=0, atol=0)

    def test_raw_pretrained_rejects_incomplete_weights(self):
        state = dict(self.model.unet.state_dict())
        state.pop("conv_in.conv.weight")
        with self.assertRaisesRegex(RuntimeError, "Pretrained UNet checkpoint mismatch"):
            self.load({"unet_state_dict": state})

    def test_legacy_checkpoint_keeps_convolutional_sampling(self):
        loaded = self.load({"bridge": self.model.state_dict()})
        self.assertFalse(loaded.resblock_updown)

    def test_explicit_resampling_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "resampling architecture differs"):
            self.load({"bridge": self.model.state_dict()}, resblock_updown=True)

    def test_paired_inference_restores_aligner_ema(self):
        options = dict(endpoint_channels=4, hidden_channels=4, domain_embed_dim=4, num_blocks=1)
        state = EndpointAligner(**options).state_dict()
        ema = {key: value + 0.125 for key, value in state.items()}
        restored, selected = load_endpoint_aligner(
            {"endpoint_aligner_config": options, "endpoint_aligner": state, "endpoint_aligner_ema": ema}, "cpu"
        )
        self.assertEqual(selected, "endpoint_aligner_ema")
        for key, value in ema.items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_inference_keeps_legacy_or_selected_aligner_weights(self):
        options = dict(endpoint_channels=4, hidden_channels=4, domain_embed_dim=4, num_blocks=1)
        state = EndpointAligner(**options).state_dict()
        restored, selected = load_endpoint_aligner(
            {"endpoint_aligner_config": options, "endpoint_aligner": state}, "cpu"
        )
        self.assertEqual(selected, "endpoint_aligner")
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_pretrained_attention_projection_is_restored(self):
        options = MODEL_OPTIONS | {"attention_levels": (False, False, True, True)}
        pretrained = FieldRFModel(**options, include_fc=True, resblock_updown=True)
        loaded = self.load(
            {"unet_state_dict": pretrained.unet.state_dict()},
            attention_levels=options["attention_levels"],
        )
        self.assertTrue(loaded.include_fc)
        self.assertTrue(loaded.resblock_updown)
        for name, value in pretrained.unet.state_dict().items():
            if name != "class_embedding.weight":
                torch.testing.assert_close(loaded.unet.state_dict()[name], value, rtol=0, atol=0)
        continued = self.load({"bridge": loaded.state_dict()}, attention_levels=options["attention_levels"])
        self.assertTrue(continued.include_fc)
        for name, value in loaded.state_dict().items():
            torch.testing.assert_close(continued.state_dict()[name], value, rtol=0, atol=0)


class VAEAndInversionTests(unittest.TestCase):
    def test_vae_restores_raw_wrapped_and_prefixed_weights(self):
        source = torch.nn.Linear(3, 2)
        state = source.state_dict()
        payloads = [state, {"state_dict": state}, {"model": {
            "module.autoencoder." + key: value for key, value in state.items()
        }}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vae.pt"
            for payload in payloads:
                with self.subTest(layout=list(payload)):
                    torch.save(payload, path)
                    restored = torch.nn.Linear(3, 2)
                    load_vae_checkpoint(restored, path)
                    for key, value in state.items():
                        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_vae_rejects_partial_and_unexpected_weights(self):
        model = torch.nn.Linear(3, 2)
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vae.pt"
            for payload in [{"weight": state["weight"]}, state | {"other": torch.ones(1)}]:
                with self.subTest(keys=list(payload)):
                    torch.save(payload, path)
                    with self.assertRaisesRegex(RuntimeError, "MAISI VAE checkpoint mismatch"):
                        load_vae_checkpoint(model, path)

    def test_inference_uses_checkpoint_inversion_and_explicit_overrides(self):
        paired = {"inversion": {"steps": 20, "solver": "euler"}}
        unpaired = {"unpaired_alignment": {"inversion_steps": 4, "inversion_solver": "heun"}}
        self.assertEqual(resolve_inversion(paired), (20, "euler"))
        self.assertEqual(resolve_inversion(unpaired), (4, "heun"))
        self.assertEqual(resolve_inversion(paired, steps=12, solver="heun"), (12, "heun"))
        with self.assertRaisesRegex(ValueError, "Invalid source inversion"):
            resolve_inversion(paired, steps=0)


if __name__ == "__main__":
    unittest.main()
