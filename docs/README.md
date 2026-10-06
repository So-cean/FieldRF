# Setup and reproducibility

## Python dependencies

Use Python 3.10 or later. Configure PyTorch and torchvision for your hardware
first, then install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e Baseline/
```

`requirements.txt` covers FieldRF, the comparison baselines and submission
utilities. It specifies no PyTorch/torchvision version, CUDA runtime version or
CUDA wheel index. A compatible PyTorch installation can be reused. MONAI must
provide the MAISI autoencoder/UNet and ControlNet components; blosc2 must provide
its native NDArray containers, `asarray`, `open` and metadata support.

The FieldRF package also declares its direct dependencies in
[`FieldRF/pyproject.toml`](../FieldRF/pyproject.toml). Baseline dependencies are
listed in [`Baseline/setup.py`](../Baseline/setup.py). Source entry points locate
the local packages directly. To register the submission CLI in the configured
environment:

```bash
python -m pip install --no-deps -e mrix2026-cli/
```

Brain-ID source is pinned by the `third_party/Brain-ID` Git submodule. Initialize
it from the repository root with `git submodule update --init --recursive`.
The root requirements include `iopath`, `simplejson`, `tabulate`, `pytz` and
`visdom`, which its frozen feature-extraction import path needs. When installing
only the FieldRF package, select its `brainid` extra as well. Use this
repository's requirements for feature extraction: Brain-ID's own requirements
describe its older full training environment, including PyTorch version pins.
Brain-ID pretrained weights still require a separate download. SynthSeg segmentation
uses a separately configured installation and executable, with its own
TensorFlow/Keras environment. See the [CLI guide](../mrix2026-cli/README.md).
NPU installations use their platform-specific PyTorch adapter separately.

## CPU tests

Install the test dependencies and run the committed test suite:

```bash
python -m pip install -r requirements.txt
python -m pytest -q FieldRF/tests
```

The FieldRF suite contains 34 tests in six files, covering data pairing, RF/alignment
losses, spatial supervision, numerical integration/autograd and checkpoint/EMA
loading. It uses synthetic tensors, small networks and temporary cache/weight
files. No MRI dataset, Brain-ID checkout or released model checkpoint is needed.
See the [test inventory](../FieldRF/README.md#cpu-regression-tests) and
[test sources](../FieldRF/tests/).
Run `python -m pytest -q FieldRF/tests mrix2026-cli/tests` to include the two
submission ZIP regression tests. See the [independent-checkout audit](../REPRODUCIBILITY.md)
for the actual tested versions and verification scope.

## Data and checkpoints

MRI data, latent/feature caches, MAISI weights, Brain-ID weights and trained
FieldRF checkpoints must be obtained separately. The repository does not supply
trained model binaries. Replace `/path/to/` placeholders before running.

For FieldRF, configure `data.data_root`, `data.cache_root`, `vae.checkpoint`,
`bridge.ckpt_path` and adaptation checkpoint paths as applicable.
The cache writer defaults to the pinned `third_party/Brain-ID` checkout;
`BRAINID_ROOT` is an optional override for a different source checkout.
Set `brainid.ckpt_path` in the pretraining configuration before feature caching.
The [FieldRF reproduction guide](../README.md#reproduction-pipeline)
lists weight downloads, all configuration fields, cache-generation commands
for all three splits, and the subsequent artifact dependencies. It also covers
the command-scoped compatibility setting for loading Brain-ID's legacy
checkpoint with PyTorch 2.6 or later.
For baselines, configure `.env` from `.env.example`, plus any cache or auxiliary
checkpoint paths specified by the selected YAML.

## Full experiments

Experiment reproduction uses the external data and weights above, followed by
cache generation, training, adaptation, inference and reference evaluation.
CPU test results verify code contracts; experimental metrics come from running
this complete pipeline on the MRI dataset.

Run each method from the working directory stated in its README. Baseline
multi-device training uses native PyTorch DDP through `torchrun` with `--dist`;
YAML batch sizes are per device. Configure scheduling for your own platform.

Follow [FieldRF](../FieldRF/README.md) for the shared RF reproduction pipeline
or [Baseline](../Baseline/README.md) for comparison methods.
Use [Evaluation](../Evaluation/README.md) for reference metrics and
[Submission](../Submission/README.md) for validation packaging.
Original projects and citation instructions are listed in [REFERENCES.md](../REFERENCES.md).
