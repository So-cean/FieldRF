# Setup and reproducibility

## Python dependencies

The baseline package requires Python 3.10 or later. FieldRF requires a compatible
PyTorch/MONAI installation with MAISI autoencoder, rectified-flow and ControlNet
components, plus NumPy, NiBabel, PyYAML, blosc2, matplotlib and scikit-image.
Baseline methods additionally use SciPy, LPIPS, SimpleITK, h5py and tqdm;
segmentation-based objectives require their separately configured pretrained models.
Use `pytest` for the lightweight contract checks.

The optional root `environment.yml` describes a baseline CUDA environment;
it is not a complete FieldRF/MAISI environment specification. Install the
appropriate PyTorch stack for your hardware in a new environment. Reuse an
existing compatible installation when one is available.

Entry points locate the local Python sources directly. To register the baseline
package in an existing configured environment without reinstalling dependencies:

```bash
python -m pip install --no-deps -e Baseline/
```

The [submission CLI](../mrix2026-cli/README.md) can run through `PYTHONPATH` or
be installed separately with `--no-deps`. SynthSeg requires its own supported
installation and executable; see the CLI guide.

## Data and checkpoints

MRI data, latent/feature caches, MAISI weights, Brain-ID weights and trained
FieldRF checkpoints must be obtained separately. The repository does not supply
trained model binaries. Replace `/path/to/` placeholders before running.

For FieldRF, configure `data.data_root`, `data.cache_root`, `vae.checkpoint`,
`bridge.ckpt_path` and adaptation checkpoint paths as applicable.
Set `BRAINID_ROOT` to the external Brain-ID checkout.
For baselines, configure `.env` from `.env.example`, plus any cache or auxiliary
checkpoint paths specified by the selected YAML.

## Execution

Run each method from the working directory stated in its README. Baseline
multi-device training uses native PyTorch DDP through `torchrun` with `--dist`;
YAML batch sizes are per device. Configure scheduling for your own platform.

Follow [FieldRF](../FieldRF/README.md) for the shared RF reproduction pipeline
or [Baseline](../Baseline/README.md) for comparison methods.
Use [Evaluation](../Evaluation/README.md) for reference metrics and
[Submission](../Submission/README.md) for validation packaging.
