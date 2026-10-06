"""Two-rank real MONAI RF and VAE training regressions on CPU/Gloo."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from fieldrf.models.model import FieldRFModel
from fieldrf.models.losses import rectified_flow_loss
from fieldrf.training.distributed import wrap_rf_submodules
from fieldrf.training import vae
from training_fixtures import RF_OPTIONS, make_vae_config, vae_args, RandomPatchDataset, assert_nested_equal


def _assert_ranks_match(model):
    vector = torch.cat([value.detach().float().flatten() for value in model.state_dict().values()])
    values = [torch.empty_like(vector) for _ in range(2)]
    dist.all_gather(values, vector)
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)


def _rf_inputs(rank):
    generator = torch.Generator().manual_seed(100 + rank)
    return (torch.randn(1, 4, 8, 8, 8, generator=generator),
            [torch.randn(1, 4, 16, 16, 16, generator=generator)],
            torch.tensor([rank]), torch.tensor([400.0]),
            torch.randn(1, 4, 8, 8, 8, generator=generator))


def _distributed_worker(rank, directory):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    root = Path(directory)
    dist.init_process_group("gloo", init_method="file://" + str(root / "rf_group"), rank=rank, world_size=2)
    try:
        torch.manual_seed(2026)
        model = FieldRFModel(**RF_OPTIONS)
        reference = FieldRFModel(**RF_OPTIONS)
        reference.load_state_dict(model.state_dict())
        keys = list(model.state_dict())
        wrap_rf_submodules(model, torch.device("cpu"))
        assert list(model.state_dict()) == keys
        _assert_ranks_match(model)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            z, feats, label, t, endpoint = _rf_inputs(rank)
            rectified_flow_loss(model, z, feats, label, t=t, endpoint=endpoint).backward()
            for other_rank in range(2):
                z, feats, label, t, endpoint = _rf_inputs(other_rank)
                (rectified_flow_loss(reference, z, feats, label, t=t, endpoint=endpoint) / 2).backward()
            for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7, msg=lambda message: name + "\n" + message)
            optimizer.step()
            reference_optimizer.step()
            _assert_ranks_match(model)

        # Exercise the conditional adapter, multiple ODE forwards and sequential backward calls.
        model = FieldRFModel(**RF_OPTIONS, source_injection="adapter", source_adapter_channels=4)
        wrap_rf_submodules(model, torch.device("cpu"))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            z, feats, label, t, endpoint = _rf_inputs(rank)
            rectified_flow_loss(model, z, feats, label, t=t, endpoint=endpoint).backward()
            graft = model.generate_train(feats, label, z, source_dropout_p=0, n_infer_steps=2)
            graft.square().mean().backward()
            rectified_flow_loss(model, z, feats, label, t=t, z_src=z, endpoint=endpoint).backward()
            optimizer.step()
            _assert_ranks_match(model)
        # Main-rank validation must not enter DDP collectives.
        model.eval()
        if rank == 0:
            with torch.no_grad():
                model(z, t, label, feats, z_src=z)
        dist.barrier()
    finally:
        dist.destroy_process_group()

    config_path = root / "vae.yaml"
    trained_models = []
    original_prepare = vae.prepare_decoder

    def capture_model(model):
        trained_models.append(model)
        return original_prepare(model)

    with patch.object(vae, "LatentImageDataset", RandomPatchDataset), \
         patch.object(vae, "prepare_decoder", side_effect=capture_model):
        dist.init_process_group("gloo", init_method="file://" + str(root / "vae_group"), rank=rank, world_size=2)
        vae.run(vae_args(config_path, distributed=True))
        config = yaml.safe_load(config_path.read_text())
        config["output_dir"] = str(root / "resumed")
        # Each process uses its own config file; published recipes remain untouched.
        resumed_config = root / f"resume_rank{rank}.yaml"
        resumed_config.write_text(yaml.safe_dump(config))
        dist.init_process_group("gloo", init_method="file://" + str(root / "resume_group"), rank=rank, world_size=2)
        _assert_ranks_match(trained_models[-1])
        vae.run(vae_args(resumed_config, distributed=True,
                         resume=str(root / "continuous" / "vae_finetune_epoch1.pt")))
    dist.init_process_group("gloo", init_method="file://" + str(root / "final_group"), rank=rank, world_size=2)
    try:
        _assert_ranks_match(trained_models[-1])
    finally:
        dist.destroy_process_group()
    if rank == 0:
        testcase = unittest.TestCase()
        full = torch.load(root / "continuous" / "vae_finetune_epoch2.pt", weights_only=False)
        resumed = torch.load(root / "resumed" / "vae_finetune_epoch2.pt", weights_only=False)
        for key in ("state_dict", "optimizer", "scheduler", "global_step", "ema_loss", "balance_step"):
            assert_nested_equal(testcase, full[key], resumed[key])
        assert len(full["rng_states"]) == 2
        assert full["global_step"] == 4
        assert all(state["extra"]["ema_loss"] for state in full["rng_states"])


class DistributedTrainingTests(unittest.TestCase):
    def test_two_rank_rf_gradients_and_vae_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            make_vae_config(directory)
            mp.spawn(_distributed_worker, args=(directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
