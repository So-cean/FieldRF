"""Tests for evaluation-slab RF supervision in the original FieldRF path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

FIELDRF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIELDRF_ROOT))

from fieldrf.models.losses import _weighted_axis_l1, spatial_axis_overlap_weights  # noqa: E402


class SpatialLossTests(unittest.TestCase):
    def test_thirty_image_slices_map_to_fractional_latent_boundary(self):
        weights = spatial_axis_overlap_weights(96, (160, 190), 384)
        active = torch.nonzero(weights > 0, as_tuple=False).flatten()
        self.assertEqual(active.tolist(), list(range(40, 48)))
        torch.testing.assert_close(weights[40:47], torch.ones(7))
        torch.testing.assert_close(weights[47], torch.tensor(0.5))
        self.assertEqual(weights.sum().item(), 7.5)

    def test_weighted_loss_ignores_predictions_outside_eval_slab(self):
        prediction = torch.zeros(1, 1, 2, 2, 4)
        target = torch.zeros_like(prediction)
        prediction[..., 0] = 9.0
        prediction[..., 2] = 2.0
        weights = torch.tensor([0.0, 0.0, 1.0, 0.0])

        loss = _weighted_axis_l1(prediction, target, weights)

        torch.testing.assert_close(loss, torch.tensor(2.0))

    def test_weighted_loss_rejects_wrong_axis_length(self):
        prediction = torch.zeros(1, 1, 2, 2, 4)
        with self.assertRaisesRegex(ValueError, "match the selected axis"):
            _weighted_axis_l1(prediction, prediction, torch.ones(3))


if __name__ == "__main__":
    unittest.main()
