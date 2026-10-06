# FieldRF

Anatomy-conditioned 3D rectified flow with a shared model for three modalities
and five field strengths. The maintained implementation uses rectified flow and
ControlNet, unpaired pretraining and paired cross-field adaptation.

## Code layout

| File | Responsibility |
|---|---|
| `data/dataset.py` | Cached unpaired volumes, paired field adaptation and visualization pairs |
| `data/volume_dataset.py` | Raw NIfTI volumes and preprocessing for cache generation |
| `data/alignment_dataset.py` | Unpaired and synthetic endpoint-alignment datasets |
| `models/model.py` | `FieldRFModel`, network assembly and velocity prediction |
| `models/networks.py` | Gradient-safe MAISI UNet and source-latent adapter |
| `models/rectified_flow.py` | Timestep remapping, RF paths and training velocity targets |
| `models/sampling.py` | Euler/Heun inversion, reverse sampling and differentiable graft generation |
| `models/checkpoint.py` | Pretrained/full RF loading, EMA restoration and source-channel adaptation |
| `models/losses.py` | RF velocity loss, spatially weighted supervision and paired latent/gradient loss |
| `models/endpoint_aligner.py` | `EndpointAligner`, shared domain-conditioned endpoint correction |
| `utils/domains.py` | Modality/field labels shared by data and models |
| `utils/diagnostics.py` | ControlNet and source-path training-log diagnostics |
| `utils/visualization.py` | VAE decoding, reference-grid metrics and rendering |

Modules, class names, entry points and configuration names describe their roles.
Experiment epochs and checkpoint filenames identify artifacts independently
of the source-code layout. `FieldRFModel` retains its existing training, sampling
and loading methods as thin delegates. The UNet, ControlNet, source adapter and
spacing buffer retain their state-dict names, so existing weights remain usable.
Data modules import domain labels without loading the model architecture.

## Dataset roles

`dataset.py` is a module containing several dataset classes, rather than a
single dataset type. RF and endpoint-aligner training read cached MAISI
latents and Brain-ID features. `VolumeDataset` supplies raw NIfTI volumes
to the cache writers. VAE decoder adaptation has a separate `LatentImageDataset`
in `scripts/finetune_vae.py`, which matches cached latents to raw image patches.

| Class | Input and sampling | Consumer |
|---|---|---|
| `VolumeDataset` | Raw MRI volumes, standardized preprocessing | Latent and Brain-ID cache writers |
| `MultiFieldCacheDataset` | Individual cached volumes from multiple modalities and fields; no anatomical pairing required | Unpaired RF pretraining |
| `PairedFieldDataset` | Real same-subject, same-modality source/target field pairs | Final paired RF adaptation |
| `PairedVisualizationDataset` | Matched prospective cases for rendering and validation | Visualization helpers |
| `UnpairedEndpointAlignmentDataset` | Independent real source/reference caches in the same modality | Endpoint alignment with distribution losses |
| `PairedAlignmentCacheDataset` | Same-anatomy synthetic-source / real-target cache pairs, optionally specified by a manifest | Auxiliary synthetic-alignment trainer |

Independent samples from `UnpairedEndpointAlignmentDataset` are suitable for
distribution matching, not voxel-wise reconstruction. Real paired adaptation
uses `Training_prospective`; the auxiliary synthetic-alignment stage is a
separate experiment.

## Dependencies and paths

See [setup](../docs/README.md) for Python dependencies. MAISI pretrained weights,
Brain-ID source and weights, MRI volumes and feature/latent caches are external
artifacts. Set their locations in the YAML configurations before running.
The Brain-ID feature writer reads `BRAINID_ROOT`; set it to your checkout:

```bash
export BRAINID_ROOT=/path/to/Brain-ID
```

`data.data_root` points to the dataset, with splits such as
`Training_retrospective`, `Training_prospective` and `Validating_prospective`.
`data.cache_root` points to cached latents and Brain-ID features. Placeholder
paths under `/path/to/` must be replaced. Run the commands below from `FieldRF/`.

## Reproduction pipeline

| Stage | Entry point | Configuration |
|---|---|---|
| MAISI latent cache | `scripts/cache_latents.py` | `configs/pretrain.yaml` |
| Brain-ID feature cache | `scripts/cache_brainid_feat.py` | same pretraining configuration |
| Shared unpaired RF pretraining | `scripts/train.py` | `configs/pretrain.yaml` |
| Unpaired endpoint alignment | `scripts/train_endpoint_aligner.py` | `configs/endpoint_alignment.yaml` |
| Shared VAE decoder adaptation | `scripts/finetune_vae.py` | `configs/vae_finetune.yaml` |
| Paired RF adaptation | `scripts/finetune.py` | `configs/finetune.yaml` |
| Inference | `scripts/inference.py` | checkpoint and configured inference paths |

```bash
PYTHONPATH=. python scripts/cache_latents.py --config configs/pretrain.yaml
PYTHONPATH=. python scripts/cache_brainid_feat.py --config configs/pretrain.yaml
python scripts/train.py --config configs/pretrain.yaml
python scripts/train_endpoint_aligner.py --config configs/endpoint_alignment.yaml
python scripts/finetune_vae.py --config configs/vae_finetune.yaml
python scripts/finetune.py --config configs/finetune.yaml
```

Before paired adaptation, set `finetune_from`, `endpoint_aligner_checkpoint` and
`vae.checkpoint` to the selected artifacts from the preceding stages.
The manuscript inference uses the epoch-20 paired EMA and the shared epoch-1
adapted VAE decoder. Use that decoder consistently in adaptation and inference.
The original MAISI decoder gives a different protocol.

Paired adaptation uses `Training_prospective` only. Brain-ID and the VAE remain
frozen during paired RF adaptation. The RF, ControlNet and endpoint adapter
are shared across fields and modalities. The synthetic-alignment trainer is
retained as a helper covered by regression tests, rather than the final recipe.


## Inference

For shared Any-to-Any inference:

```bash
python scripts/inference.py \
  --config configs/finetune.yaml \
  --checkpoint /path/to/paired_rf_checkpoint.pt \
  --vae_checkpoint /path/to/adapted_vae_decoder.pt \
  --source_endpoint_transport \
  --task task3 --source_fields 0.1T 1.5T 3T 5T 7T \
  --output_dir /path/to/predictions
```

Use `--help` for target-field subsets, source inversion and output options.
Inference reuses the same shared model for all requested translations and
performs no test-time augmentation by default. Outputs use the sweep layout
consumed by [the packaging tools](../mrix2026-cli/README.md).
Existing checkpoints can retain their original filenames: the loader accepts
explicit paths, and parameter layouts and state-dict keys are preserved.

## CPU regression tests

The committed suite contains 25 tests in five files:

| File | Tests | Coverage |
|---|---:|---|
| [test_alignment.py](tests/test_alignment.py) | 9 | Alignment losses, pair manifests, RF paths and truncated inversion |
| [test_paired_dataset.py](tests/test_paired_dataset.py) | 3 | Same-subject/modality pairing, domain completeness and cross-field record counts |
| [test_spatial_loss.py](tests/test_spatial_loss.py) | 3 | Fractional evaluation-slab weights and weighted L1 supervision |
| [test_model_loading.py](tests/test_model_loading.py) | 6 | Full/EMA and raw UNet loading, source adaptation and invalid checkpoints |
| [test_sampling.py](tests/test_sampling.py) | 4 | Euler/Heun direction, latent scaling, gradient propagation and deterministic endpoint reversal |

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q FieldRF/tests
```

Tests use synthetic tensors, small networks and temporary cache/checkpoint
files. They run on CPU without MRI data, a Brain-ID checkout or trained FieldRF
weights. Full experiment reproduction follows the pipeline above and requires
the external data, caches and pretrained weights listed under dependencies.
