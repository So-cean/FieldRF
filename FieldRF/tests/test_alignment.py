"""CPU unit tests for RF alignment data and RF coupling utilities."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

FIELDRF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIELDRF_ROOT))
sys.path.insert(0, str(FIELDRF_ROOT / "scripts"))

from data.alignment_dataset import PairedAlignmentCacheDataset
from models.endpoint_aligner import EndpointAligner
from models.model import FieldRFModel
from train_alignment import deterministic_pair_noise
from utils.endpoint_alignment import (
    alignment_weight,
    endpoint_moment_loss,
    multiscale_patch_coral,
    patch_descriptors,
    stable_endpoint_distribution_loss,
)


class _RFStub:
    num_train_timesteps = 1000


class _InvertStub(torch.nn.Module):
    num_train_timesteps = 1000

    def __init__(self):
        super().__init__()
        self.velocity = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, state, timestep, class_label, feats):
        return torch.ones_like(state) * self.velocity


class AlignmentUtilityTests(unittest.TestCase):
    def test_endpoint_aligner_is_identity_at_initialization(self):
        aligner = EndpointAligner(hidden_channels=8, num_blocks=1)
        endpoint = torch.randn(2, 4, 8, 8, 8)
        aligned, delta = aligner(endpoint, torch.tensor([0, 14]))

        torch.testing.assert_close(aligned, endpoint)
        torch.testing.assert_close(delta, torch.zeros_like(delta))

    def test_patch_coral_is_zero_for_identical_endpoints_and_has_gradient(self):
        endpoint = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        descriptors = patch_descriptors(endpoint, 4)
        self.assertEqual(tuple(descriptors.shape), (8, 8))
        loss = multiscale_patch_coral(endpoint, endpoint, [4])
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        loss.backward()
        self.assertIsNotNone(endpoint.grad)

    def test_stable_endpoint_loss_anchors_scale(self):
        reference = torch.randn(1, 4, 8, 8, 8)
        identical, components = stable_endpoint_distribution_loss(
            reference,
            reference,
            [4],
        )
        torch.testing.assert_close(identical, torch.zeros_like(identical))
        self.assertGreater(
            endpoint_moment_loss(3 * reference, reference).item(),
            0.0,
        )
        self.assertEqual(set(components), {"coral", "moments"})

    def test_four_step_truncated_inversion_matches_forward_value(self):
        initial = torch.randn(1, 1, 4, 4, 4)
        labels = torch.zeros(1, dtype=torch.long)
        full_model = _InvertStub()
        truncated_model = _InvertStub()
        truncated_model.load_state_dict(full_model.state_dict())
        full = FieldRFModel.invert_to_endpoint(
            full_model,
            initial,
            [],
            labels,
            n_steps=4,
            gradient_steps=4,
        )
        truncated = FieldRFModel.invert_to_endpoint(
            truncated_model,
            initial,
            [],
            labels,
            n_steps=4,
            gradient_steps=1,
        )
        torch.testing.assert_close(full, truncated)
        truncated.mean().backward()
        self.assertIsNotNone(truncated_model.velocity.grad)
        self.assertNotEqual(truncated_model.velocity.grad.item(), 0.0)

    def test_alignment_weight_warms_up_and_ramps(self):
        kwargs = {
            "steps_per_epoch": 10,
            "maximum": 5.0,
            "start_epoch": 1.0,
            "ramp_epochs": 1.0,
        }
        self.assertEqual(alignment_weight(epoch=0, step=9, **kwargs), 0.0)
        self.assertAlmostEqual(
            alignment_weight(epoch=1, step=4, **kwargs), 2.5
        )
        self.assertEqual(alignment_weight(epoch=2, step=0, **kwargs), 5.0)

    def test_rf_path_uses_data_minus_endpoint_velocity(self):
        clean = torch.full((2, 1, 2, 2, 2), 3.0)
        endpoint = torch.full_like(clean, -1.0)
        timesteps = torch.tensor([0.0, 1000.0])

        state, velocity = FieldRFModel.build_rf_path(
            _RFStub(), clean, endpoint, timesteps
        )

        torch.testing.assert_close(state[0], clean[0])
        torch.testing.assert_close(state[1], endpoint[1])
        torch.testing.assert_close(velocity, clean - endpoint)

    def test_supplied_endpoint_reverse_rejects_invalid_step_count(self):
        endpoint = torch.randn(1, 4, 8, 8, 8)
        with self.assertRaisesRegex(ValueError, "n_infer_steps"):
            FieldRFModel.sample_x0_from_endpoint(
                _RFStub(),
                endpoint,
                [],
                torch.tensor([0]),
                n_infer_steps=0,
            )

    def test_pair_noise_is_stable_and_pair_specific(self):
        reference = torch.empty(2, 1, 3, 3, 3)
        first = deterministic_pair_noise(["a", "b"], reference, 2026)
        second = deterministic_pair_noise(["a", "b"], reference, 2026)
        changed = deterministic_pair_noise(["a", "c"], reference, 2026)

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[0], changed[0])
        self.assertFalse(torch.equal(first[1], changed[1]))

    def test_manifest_filters_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filenames = (
                "source.pt",
                "source.b2nd",
                "target.pt",
                "target.b2nd",
            )
            for filename in filenames:
                (root / filename).touch()
            manifest = root / "pairs.csv"
            columns = (
                "pair_id",
                "modality",
                "source_field",
                "target_field",
                "source_latent",
                "source_feat",
                "target_latent",
                "target_feat",
            )
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "T2W:subject:0.1T->7T",
                        "modality": "T2W",
                        "source_field": "0.1T",
                        "target_field": "7T",
                        "source_latent": filenames[0],
                        "source_feat": filenames[1],
                        "target_latent": filenames[2],
                        "target_feat": filenames[3],
                    }
                )

            dataset = PairedAlignmentCacheDataset(
                manifest=manifest,
                modalities=["T2W"],
                source_fields=["0.1T"],
                target_fields=["7T"],
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(
                dataset.records[0]["pair_id"], "T2W:subject:0.1T->7T"
            )
            self.assertTrue(
                Path(dataset.records[0]["source_latent"]).is_absolute()
            )


if __name__ == "__main__":
    unittest.main()
