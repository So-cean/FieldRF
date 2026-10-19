# Setup and reproducibility

## Project questions

### What does FieldRF translate?

FieldRF translates MRI volumes between field strengths within the same modality.
One shared 3D rectified flow model covers T1W, T2W and T2FLAIR at 0.1T, 1.5T,
3T, 5T and 7T. Brain-ID features condition anatomy; modality and field labels
specify the requested appearance. The output is a NIfTI volume on the source
image grid. The [paper](https://papers.miccai.org/miccai-2026-sat/MRIxFields2026_026.html)
describes the method; the [code guide](../FieldRF/README.md#code-layout) maps its implementation.

### Are Brain-ID and both caches required?

The supplied configurations use MAISI latent caches and Brain-ID feature caches
during training and inference. Generate both for each required split with the
[cache commands](../README.md#3-generate-both-caches-for-every-required-split).
Brain-ID source and pretrained weights are needed for feature extraction;
subsequent stages read the saved features. Inference also reads raw source
volumes to restore their image grids.

### Where do the data splits and trained weights come from?

Use the dataset and split definitions provided by the
[official MRIxFields2026 benchmark](https://github.com/MRIxFields/MRIxFields2026).
The [reproduction pipeline](../README.md#reproduction-pipeline) lists the upstream
MAISI/Brain-ID weight downloads and trains the FieldRF stage checkpoints.
FieldRF checkpoint reproduction currently requires running these training stages.
The [artifact table](../README.md#4-train-and-connect-the-stage-artifacts) specifies
how the selected training outputs connect to later stages.

### What has been verified?

The [verification record](../REPRODUCIBILITY.md) separates CPU regression tests,
short real-data GPU runs of the full pipeline, and retained-paper-checkpoint
loading/inference checks. CPU/Gloo tests cover two-rank DDP and RF/VAE resume.
Multi-GPU CUDA/NCCL execution and full benchmark metric reproduction remain
outside that recorded validation scope.

### Where are project and citation metadata?

[CITATION.cff](../CITATION.cff) identifies the preferred conference-paper citation,
including its authors, proceedings, publisher and year.
[codemeta.json](../codemeta.json) describes the software and its paper using
CodeMeta/Schema.org. [llms.txt](../llms.txt) is a short text index linking the
setup, method, evidence and references. Python package metadata links back to
this repository, the documentation, the paper and the issue tracker.

## Python dependencies

Use Python 3.10 or later. Configure PyTorch and torchvision for your hardware
first, then install the Python dependencies from the repository root:

```bash
python -m pip install --upgrade 'pip>=25.3'
python -m pip install -r requirements.txt --build-constraint build-constraints.txt
python -m pip install --no-deps -e FieldRF/ -e Baseline/
```

`requirements.txt` covers FieldRF, the comparison baselines and submission
utilities. It specifies no PyTorch/torchvision version, CUDA runtime version or
CUDA wheel index. A compatible PyTorch installation can be reused. MONAI must
provide the MAISI autoencoder/UNet and ControlNet components; blosc2 must provide
its native NDArray containers, `asarray`, `open` and metadata support.

`build-constraints.txt` limits setuptools only in isolated source builds because
Brain-ID's `visdom` dependency still imports `pkg_resources` in its build script.
The build constraint does not select the runtime setuptools or PyTorch version.

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
python -m pip install --upgrade 'pip>=25.3'
python -m pip install -r requirements.txt --build-constraint build-constraints.txt
python -m pip install --no-deps -e FieldRF/
python -m pytest -q FieldRF/tests
```

The FieldRF suite contains 59 tests in eleven files, covering data pairing, RF/alignment
losses, spatial supervision, numerical integration/autograd and checkpoint/EMA
loading. It uses synthetic tensors, small networks and temporary cache/weight
files. No MRI dataset, Brain-ID checkout or released model checkpoint is needed.
The native DDP regression uses two CPU/Gloo processes. RF/VAE epoch-boundary
resume is compared against uninterrupted optimization. GitHub Actions installs
the package and runs these tests on Python 3.10/3.12; its CPU Torch/MONAI
versions are specific to CI, while runtime requirements retain hardware freedom.
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

## Pretrained artifacts

The [homepage](../README.md#1-obtain-source-dependencies-and-pretrained-weights)
links the exact Brain-ID, MAISI VAE and brain-RF weight revisions used by the
recipe. Brain-ID source is pinned at
[`5d8a2d3`](https://github.com/jhuldr/Brain-ID/tree/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253).
The linked MAISI releases provide these SHA-256 checksums:

```text
1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3  autoencoder_v1.pt
b49aa3162ff94b1852f81e520ed4165c5656000ba176c033063c35350b3305e9  diff_unet_3d_rflow-mr-brain_v0.pt
```

The RF loader detects the pretrained UNet architecture and loads its complete
weights, replacing the original domain embedding with the MRI-domain embedding.

## Cache format

Generate both caches for every required volume using the
[homepage commands](../README.md#3-generate-both-caches-for-every-required-split).
The supplied three-modality, five-field recipe uses:

| Split | Consumer |
|---|---|
| `Training_retrospective` | RF pretraining, endpoint alignment and VAE adaptation |
| `Training_prospective` | Paired RF adaptation and training diagnostics |
| `Validating_prospective` | VAE validation and inference |

A preprocessed `384³` image yields a `(4,96,96,96)` MAISI latent. Brain-ID
processes the image at `192³` in `[0,1]`. The configured `[64,64]` ControlNet
requires native `(64,192,192,192)` `feat4`, with `feat4_pool=1`. Each input stem
has these two files:

```text
<cache_root>/<split>/latent/<modality>/<field>/<stem>.pt
<cache_root>/<split>/brainid_feat/<modality>/<field>/<stem>.b2nd
```

The writers skip existing outputs. Changing feature geometry requires cache
regeneration with `--overwrite`. Check the summaries, including Brain-ID's
`errors` count. The writers' `--help` lists modality, field and overwrite controls.
Training and inference use the caches; Brain-ID runs during feature extraction.

## Distributed training and resume

RF pretraining, endpoint alignment and VAE adaptation support native PyTorch
DDP. YAML batch sizes are per device. From `FieldRF/`:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/pretrain.yaml --dist
torchrun --standalone --nproc_per_node=2 scripts/finetune_vae.py --config configs/vae_finetune.yaml --dist
```

Paired RF adaptation runs on a single CUDA device. The committed DDP regressions
use CPU/Gloo; the [verification record](../REPRODUCIBILITY.md#remaining-limits)
states the tested execution scope.

Resume RF pretraining with `--resume <fieldrf_epochN.pt>` and VAE adaptation
with `--resume_ckpt <vae_finetune_epochN.pt>`. New VAE epoch checkpoints restore
weights, optimizer, learning-rate plan, epoch/global step, loss balancing and
per-rank random/data-loader state. Keep the original training recipe and rank
count. Use `--init_ckpt` for older VAE files containing only weights.

## Preflight checks

From `FieldRF/`, run the checker before the corresponding operation:

```bash
python scripts/check_pipeline.py --stage cache
python scripts/check_pipeline.py --stage pretrain
# Run these as the preceding stages produce their checkpoints:
python scripts/check_pipeline.py --stage endpoint
python scripts/check_pipeline.py --stage vae
python scripts/check_pipeline.py --stage paired
```

The default `--config-dir configs` reads all four YAMLs and checks shared data/cache
roots, feature names, VAE encoder geometry/scale and RF/ControlNet architecture.
It checks selected-stage weight files, RF/aligner content identity, every configured
modality/field and required prospective subject. Raw input stems must have their
required caches; latent headers must be `(4,96,96,96)` and Brain-ID headers must
match the ControlNet geometry and buffer length. VAE adaptation requires latents
and raw images; RF stages require both caches. Training and visualization splits
are checked separately. The checker reports missing domains rather than silently
reducing the training set.

Use `--stage all` after all stage inputs exist. `--config-dir /path/to/configs`
selects another complete recipe; `--config /path/to/stage.yaml --stage paired`
checks only that file. Relative artifact paths use the documented `FieldRF/`
working directory. `--report /path/to/report.json` optionally saves the findings.
The checker exits with status 1 on errors. It reads cache/checkpoint headers using
memory mapping and Brain-ID metadata, without decoding volumes or constructing
networks. Full tensor integrity and numerical behavior are covered by the actual
loaders and training/inference tests.

## Checkpoint selection and inference

Follow the [stage-artifact table](../README.md#4-train-and-connect-the-stage-artifacts)
to connect pretraining, alignment, VAE adaptation and paired training.
New endpoint-alignment checkpoints record `base_checkpoint_sha256`. Paired
initialization, endpoint resume and unpaired-aligner inference verify the actual
RF file content. Files can be moved or renamed while remaining byte-identical;
the recorded base path serves as provenance. Full RF/VAE loading uses the file
paths supplied at runtime. Joint paired resume restores RF and aligner from the
same complete checkpoint, without reading their old initialization files.

For a legacy unpaired aligner, create a portable copy while the original RF base
is still accessible. From `FieldRF/`:

```bash
python scripts/migrate_checkpoint.py \
  --checkpoint /old/artifacts/endpoint_aligner.pt \
  --base-checkpoint /new/artifacts/rf.pt \
  --legacy-root /old/FieldRF \
  --output /new/artifacts/endpoint_aligner_portable.pt
```

The command compares the original and selected RF files, records their SHA-256
in the new aligner copy and preserves all weights and optimizer state. Existing
output files are refused. Set `endpoint_aligner_checkpoint` and `finetune_from`
to the selected new files, then run the paired preflight check. A legacy file
whose original base is unavailable and whose checksum was never recorded needs
that original base restored for verification; matching filenames provide no
content identity.

VAE adaptation freezes the encoder, keeping the latent caches valid. Paired
adaptation freezes the VAE and reads the cached Brain-ID features. RF,
ControlNet and the endpoint aligner share parameters across fields and modalities.

`fieldrf_paired_latest.pt` after epoch 20 contains joint RF/aligner EMA.
`fieldrf_paired_best_task3.pt` is selected by validation. For either choice,
pass the same file as `--checkpoint` and `--endpoint_checkpoint`, set its actual
epoch tag, and use the epoch-1 adapted VAE decoder.

For source transport without the endpoint aligner, omit `--endpoint_checkpoint`
and add `--source_endpoint_transport`, retaining explicit inversion steps and
solver. Task 3 translates all non-identity field pairs with one model. Inference
reads both caches and the raw source MRI grid; TTA defaults to none.
Use `--help` for source/target subsets and the
[submission guide](../mrix2026-cli/README.md) for packaging.

For an isolated one-epoch run, follow the
[small-data instructions](../REPRODUCIBILITY.md#run-an-isolated-small-data-check).
