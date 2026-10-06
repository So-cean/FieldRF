"""Numerical and autograd contracts for RF sampling and inversion."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fieldrf.models.model import FieldRFModel  # noqa: E402


class ConstantVelocity(torch.nn.Module):
    """A constant vector field makes forward/reverse integration exact."""

    num_train_timesteps = 1000
    scale_factor = 0.8

    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, state, timestep, class_label, feats):
        return torch.ones_like(state) * self.rate


class SamplingTests(unittest.TestCase):
    def test_euler_and_heun_inversion_reverse_to_original_unscaled_latent(self):
        initial = torch.full((1, 1, 4, 4, 4), 2.0)
        for solver in ("euler", "heun"):
            with self.subTest(solver=solver):
                model = ConstantVelocity()
                endpoint = FieldRFModel.invert_to_endpoint(
                    model, initial, [], 3, n_steps=4, solver=solver
                )
                self.assertTrue(bool((endpoint < initial).all()))
                recovered = FieldRFModel.sample_x0_from_endpoint(
                    model, endpoint, [], 3, n_infer_steps=4
                )
                torch.testing.assert_close(recovered, initial / model.scale_factor)
                self.assertFalse(recovered.requires_grad)

    def test_full_inversion_propagates_parameter_and_input_gradients(self):
        model = ConstantVelocity()
        initial = torch.ones(1, 1, 4, 4, 4, requires_grad=True)
        endpoint = FieldRFModel.invert_to_endpoint(
            model, initial, [], 0, n_steps=3, solver="heun"
        )
        endpoint.sum().backward()
        torch.testing.assert_close(initial.grad, torch.ones_like(initial))
        self.assertLess(model.rate.grad.item(), 0.0)

    def test_zero_gradient_steps_detaches_inversion(self):
        model = ConstantVelocity()
        initial = torch.ones(1, 1, 4, 4, 4, requires_grad=True)
        endpoint = FieldRFModel.invert_to_endpoint(
            model, initial, [], 0, n_steps=3, gradient_steps=0
        )
        self.assertFalse(endpoint.requires_grad)

    def test_supplied_endpoint_reverse_does_not_draw_random_noise(self):
        model = ConstantVelocity()
        endpoint = torch.ones(1, 1, 4, 4, 4)
        rng_before = torch.get_rng_state()
        FieldRFModel.sample_x0_from_endpoint(model, endpoint, [], 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))


if __name__ == "__main__":
    unittest.main()
