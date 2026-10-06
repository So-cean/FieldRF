# Reproducibility check

## Scope

On 2026-10-06, `So-cean/FieldRF` was checked in independent clones of remote
`main`. The initial CPU audit started at `6634c88`. A subsequent real-data GPU
pipeline used a fresh recursive clone, the pinned public Brain-ID submodule,
official pretrained weights and a separate, newly generated cache/output tree.
No private development code or modified upstream source was used as a source
dependency. MRI inputs and pretrained artifacts were supplied externally;
temporary orchestration and smoke configurations remained outside the clone.

The check used a separate virtual environment that reused an existing
PyTorch installation and its installed scientific libraries through
`--system-site-packages`. Baseline and the submission CLI were installed into
that virtual environment with `--no-deps --no-build-isolation`. This checks the
released source and installation entry points, rather than a new dependency
resolution from an empty environment. The PyTorch stack was unchanged.

## Tested environment

These are observed versions, not required CUDA/PyTorch pins:

| Component | Version |
|---|---|
| Python | 3.12.9 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0 |
| GPU for real-data pipeline | NVIDIA A100 80GB PCIe; single GPU |
| MONAI | 1.5.1 |
| NumPy | 2.2.6 |
| blosc2 | 4.7.0 |
| nibabel | 5.3.2 |
| pytest | 9.0.2 |

The runtime dependencies listed in `requirements.txt` were present. Hardware
specific PyTorch/torchvision installation remains the user's choice.

## Verification

```bash
python -m pip install --no-deps --no-build-isolation -e Baseline/ -e mrix2026-cli/
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q FieldRF/tests mrix2026-cli/tests
```

At `5db5262`, all **36 tests** passed (34 FieldRF and 2 submission CLI), with
4 subtests. The initial CPU audit passed 29 tests, before the seven loading/EMA
regressions were added; its 21 source/scorer/CLI entry points passed `--help`.
The current tests check:

- Cache/script entry points without an external Brain-ID checkout.
- Native `.pt` / `.b2nd` serialization and feature geometry with the real
  MONAI MAISI UNet and ControlNet, a CPU optimization step, checkpoint reload
  with matching predictions and finite RF sampling.
- Subject/modality pairing, RF/alignment/spatial losses, EMA loading and
  Euler/Heun integration and gradients.
- Complete pretrained UNet loading, residual resampling and attention
  projection restoration, legacy architecture detection, rejection of partial
  initialization, and consistent paired RF/aligner EMA selection.
- ZIP modality roots, relative output paths and nonzero CLI failure status.

A subsequent release review passed **42 tests with 9 subtests** (40 FieldRF
and 2 submission CLI), including strict VAE restoration, interrupted latent
writes, asynchronous feature-save failures, cache limits and checkpoint-derived
inversion settings. These are CPU regressions; the GPU record below remains
tied to its stated source revisions.

The independent integration check also runs a reduced CUT optimization step,
loads its embedded configuration through the public inference script and
verifies NIfTI shape/affine preservation. Synthetic complete Task 1 and Task 3
prediction trees exercise dry-run, build, source-to-target filename mapping,
axial cropping and ZIP packaging. These checks use synthetic volumes and
reduced network widths; they do not measure MRI image quality.

## Real-data GPU pipeline

The follow-up ran every stage through the public scripts in an independent
recursive clone, with the full MAISI UNet `[64,128,256,512]`, two-level
`[64,64]` ControlNet conditioning, pretrained Brain-ID and MAISI VAE.
Inputs were 45 real `384³` benchmark volumes: one per split/modality/field,
including the same prospective subject across all fields and modalities.
The three splits were retrospective training, prospective training and
prospective validation. Caches were generated afresh for all 45 inputs.

| Stage | Work completed | Script elapsed time |
|---|---|---:|
| Latent generation | 15 volumes per split, 45 total; `(4,96,96,96)` | 389.50 s |
| Brain-ID extraction | 15 volumes per split, 45 total; native `(64,192,192,192)` `feat4` | 768.70 s |
| RF pretraining | 1 epoch, 15 optimization steps | 64.38 s |
| Unpaired endpoint alignment | 1 epoch, 12 optimization steps; one T1W `0.1T` to `7T` paired diagnostic | 110.75 s |
| VAE decoder adaptation | 1 epoch, 15 minibatches, accumulation 8, 2 optimizer steps; 15 validation volumes before/after | 179.01 s |
| Paired RF adaptation | 1 epoch, all 60 ordered same-subject cross-field pairs; 15 terminal-rollout updates | 399.37 s |
| Joint EMA inference | All five source fields and three modalities into `0.1T`/`7T`, excluding identity; 24 NIfTI outputs | 384.20 s |
| Plain source transport | T1W `0.1T` to `7T`, without the endpoint aligner; 1 NIfTI output | 35.87 s |

The VAE used its configured L1, SSIM and SqueezeNet perceptual losses.
Paired adaptation retained 20-step Euler inversion and terminal rollouts,
including the configured two gradient-carrying rollout steps. Both inference
branches used 20 Euler inversion steps and 20 generation steps, with the newly
adapted decoder. Joint inference restored RF EMA and aligner EMA from the same
paired checkpoint. The endpoint diagnostic reads the prospective training split
without using its paired images in the unpaired optimizer.

The final validation ran inside the GPU allocation and passed:

- All 45 latent files and every value of all 45 compressed native feature files
  had the expected geometry and finite values.
- RF, aligner and paired checkpoints contained nonempty optimizer state and
  finite model/EMA tensors.
- The adapted VAE encoder was bitwise unchanged; decoder parameters changed.
- Exactly 24 aligned and 1 unaligned inference outputs were generated.
- Every output was finite, nonempty, within `[0,1]`, and preserved the source
  shape and affine (absolute affine tolerance `1e-6`).

Caches were generated with source `e4a7c8b`; corrected pretraining, alignment
and VAE adaptation ran at `d9af9ee`. The final complete paired epoch,
both inference branches and CPU regression suite ran at `5db5262`.
The GPU job completed successfully after the final checks. Earlier partial
initialization runs were discarded from the accepted result and their local
artifacts retained separately. No original experiment caches were reused.

The [homepage README](README.md#reproduction-pipeline) includes both the full
recipe and instructions for an isolated small-data check. The smoke run changed
input/output paths, subject count, epoch count, workers and diagnostic frequency;
it retained full network widths, native image/cache sizes and training losses.

## Required artifacts for full experiments

| Artifact | Source / configuration |
|---|---|
| MRI dataset and split definitions | [MRIxFields2026](https://github.com/MRIxFields/MRIxFields2026); set `data.data_root` |
| Brain-ID source and pretrained weights | Initialize `third_party/Brain-ID` at `5d8a2d3eef507180b0cbff4cfe418cdfc5c55253`; obtain [Brain-ID weights](https://github.com/jhuldr/Brain-ID/blob/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253/README.md#playing-with-brain-id-synthesizer-and-feature-extractor) and set `brainid.ckpt_path`; `BRAINID_ROOT` is an optional source override |
| MAISI VAE and brain RF weights | Shared VAE from [NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT/blob/4156648ce44f5561b75d96b12600fe5afe6b6e8f/models/autoencoder_v1.pt); brain RF from [NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain/blob/67ba36a4d3aae5854d7f2f56638e26a641684433/models/diff_unet_3d_rflow-mr-brain_v0.pt); set `vae.checkpoint` and `bridge.ckpt_path` |
| Latent / feature caches | Generate all configured modalities/fields for retrospective, prospective and inference splits; set `data.cache_root` |
| Adapted decoder, endpoint aligner and paired RF | Produced by the [FieldRF stages](README.md#reproduction-pipeline); use the configured epoch selections |
| Task 1/2 segmentations | Separate [SynthSeg](https://github.com/BBillot/SynthSeg) installation; configure `SYNTHSEG_BIN` |

The RF initialization in the supplied recipe is
`diff_unet_3d_rflow-mr-brain_v0.pt`, with `autoencoder_v1.pt`. Record the exact
upstream weight revision and checksums used. Replacing the RF initialization
with another upstream revision changes the experiment. Adaptation and inference
use the same adapted epoch-1 VAE decoder and the selected paired epoch-20 EMA.

The pinned public Brain-ID source was used without private patches. Its upstream
checkpoint loader requires the documented command-scoped
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` setting under PyTorch 2.6. The GPU check
loaded the actual pretrained weights and extracted full-resolution features.

Follow the [complete reproduction procedure](README.md#reproduction-pipeline)
for submodule initialization, weight acquisition, YAML paths, both cache writers,
the split requirements and downstream checkpoint selection.

## Run an isolated small-data check

Before a full run, make local copies of the four YAML files under `tmp/` and use
separate data, cache and output directories. Keep the full architecture and
official pretrained weights. Use one real volume per modality and field in each
of the three splits (45 inputs total), with the same prospective subject ID
across all fields for paired training. Set `data.subjects` in the paired config
to that subject; preserve the other split roles described above.

Generate both caches with the [homepage cache commands](README.md#3-generate-both-caches-for-every-required-split), using the smoke pretraining config. Set
`finetune_from` in both downstream RF configs to the smoke run's
`fieldrf_epoch1.pt`, set the aligner path to the smoke alignment artifact, and
set the paired VAE path to the smoke decoder's `vae_finetune_epoch1.pt`.
Use small worker counts, set pretraining `vis.vis_every` above 1 to skip its
optional image rendering, and set VAE validation `vis.n_samples: 1`. From the
method directory, run the public entry points:

```bash
python scripts/train.py --config ../tmp/smoke/pretrain.yaml \
  --n_epochs 1 --max_volumes 1
python scripts/train_endpoint_aligner.py \
  --config ../tmp/smoke/endpoint_alignment.yaml --n_epochs 1 --max_volumes 1
python scripts/finetune_vae.py --config ../tmp/smoke/vae_finetune.yaml \
  --epochs 1 --max_volumes 1
python scripts/finetune.py --config ../tmp/smoke/finetune.yaml \
  --n_epochs 1 --skip_vis
```

Use the [homepage inference command](README.md#inference) with these smoke artifacts and `--epoch_tag epoch1`. Add `--target_fields 0.1T 7T` for 24 non-identity outputs covering all
five source fields and all three modalities. Also test the optional
`--source_endpoint_transport` branch separately. Verify that every cache is
present and finite, the checkpoints contain optimizer state, and the decoded
NIfTI files are finite, nonempty and preserve the source shape and affine.
Check that the adapted VAE encoder is unchanged and its decoder has updated.
Keep smoke artifacts separate from the full experiment.


## Remaining limits

The short run checks execution and artifact consistency on real inputs. It
does not reproduce full-dataset, full-duration training or published challenge
metrics. NPU/DDP execution, all five target-field inference sweeps, and GPU
SynthSeg/submission evaluation were not included in this pipeline check.
A source review found that `scripts/train.py` computes the RF loss through
`bridge.module`, and `scripts/finetune_vae.py` decodes through `vae.module` after
wrapping their models in DDP. Both paths bypass `DDP.forward`. A two-process
CPU reproducer demonstrated divergent weights when bypassing the wrapper,
and matching weights when using its forward call. FieldRF RF/VAE distributed
training is therefore currently unsupported; use a single device until the
distributed paths are corrected and validated. This finding concerns FieldRF,
not the separate baseline trainers.

The environment reused an existing PyTorch stack rather than resolving all
dependencies from an empty environment. Reference evaluation implementations
are unchanged.

Strict loading uncovered that the previous raw MAISI initialization silently
omitted residual resampling and attention output-projection parameters.
The corrected loader restores those pretrained parameters, so a new run's
initialization differs from the previous partially initialized recipe.
Existing trained FieldRF checkpoints retain their architecture through
state-based detection and strict restoration. Passing these checks does not
establish that a new training run matches older reported metrics.

Project attribution and upstream citation instructions are in
[REFERENCES.md](REFERENCES.md).
