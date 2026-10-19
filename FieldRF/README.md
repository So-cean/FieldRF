# FieldRF

Anatomy-conditioned 3D rectified flow with a shared model for three modalities
and five field strengths. The maintained implementation uses rectified flow and
ControlNet, unpaired pretraining and paired cross-field adaptation.

## Code layout

| File | Responsibility |
|---|---|
| `fieldrf/data/dataset.py` | Cached unpaired volumes, paired field adaptation and visualization pairs |
| `fieldrf/data/volume_dataset.py` | Raw NIfTI volumes and preprocessing for cache generation |
| `fieldrf/data/vae_dataset.py` | Cached latent/raw image patches for decoder adaptation |
| `fieldrf/data/alignment_dataset.py` | Real unpaired samples for endpoint-distribution alignment |
| `fieldrf/models/model.py` | `FieldRFModel`, network assembly and velocity prediction |
| `fieldrf/models/networks.py` | Gradient-safe MAISI UNet and source-latent adapter |
| `fieldrf/models/rectified_flow.py` | Timestep remapping, RF paths and training velocity targets |
| `fieldrf/models/sampling.py` | Euler/Heun inversion, reverse sampling and differentiable graft generation |
| `fieldrf/models/checkpoint.py` | Strict VAE/pretrained/full RF loading, EMA restoration and source-channel adaptation |
| `fieldrf/models/losses.py` | RF velocity loss, spatially weighted supervision and paired latent/gradient loss |
| `fieldrf/models/vae.py` | Trainable MAISI decode path, including latent projection |
| `fieldrf/training/` | Stage loops, submodule DDP, EMA, atomic checkpoints and resumable LR scheduling |
| `fieldrf/cli/` | Argument parsing and training dispatch |
| `fieldrf/visualization/` | Stage-specific validation and rendering |
| `fieldrf/models/endpoint_aligner.py` | `EndpointAligner`, shared domain-conditioned endpoint correction |
| `fieldrf/utils/domains.py` | Modality/field labels shared by data and models |
| `fieldrf/utils/diagnostics.py` | ControlNet and source-path training-log diagnostics |
| `fieldrf/utils/visualization.py` | VAE decoding, reference-grid metrics and rendering |

Modules, class names, entry points and configuration names describe their roles.
Experiment epochs and checkpoint filenames identify artifacts independently
of the source-code layout. `FieldRFModel` retains its existing training, sampling
and loading methods as thin delegates. The UNet, ControlNet, source adapter and
spacing buffer retain their state-dict names, so existing weights remain usable.
Data modules import domain labels without loading the model architecture.

The installed Python package is `fieldrf`. Public scripts remain compatible
entry points; `fieldrf.cli` parses arguments and `fieldrf.training` owns the
training stages. Stage visualization lives in `fieldrf.visualization`.

## Dataset roles

`dataset.py` is a module containing several dataset classes, rather than a
single dataset type. RF and endpoint-aligner training read cached MAISI
latents and Brain-ID features. `VolumeDataset` supplies raw NIfTI volumes
to the cache writers. VAE decoder adaptation has a separate `LatentImageDataset`
in `fieldrf/data/vae_dataset.py`, which matches cached latents to raw image patches.

| Class | Input and sampling | Consumer |
|---|---|---|
| `VolumeDataset` | Raw MRI volumes, standardized preprocessing | Latent and Brain-ID cache writers |
| `MultiFieldCacheDataset` | Individual cached volumes from multiple modalities and fields; no anatomical pairing required | Unpaired RF pretraining |
| `PairedFieldDataset` | Real same-subject, same-modality source/target field pairs | Final paired RF adaptation |
| `PairedVisualizationDataset` | Matched prospective cases for rendering and validation | Visualization helpers |
| `UnpairedEndpointAlignmentDataset` | Independent real source/reference caches in the same modality | Endpoint alignment with distribution losses |

Independent samples from `UnpairedEndpointAlignmentDataset` are suitable for
distribution matching, not voxel-wise reconstruction. Real paired adaptation
uses `Training_prospective`.

## Dependencies and paths

The supplied configurations require MAISI latent caches and frozen Brain-ID
feature caches. Brain-ID source is pinned in `third_party/Brain-ID`; its
pretrained weights and the MAISI weights are downloaded separately. Both caches
must exist for training and inference. Brain-ID source/weights are only needed
while generating features.

## Reproduction pipeline

The complete procedure is maintained directly in the
[repository homepage README](../README.md#reproduction-pipeline), including:

1. Recursive clone, Brain-ID submodule and Python dependencies.
2. Exact upstream weight downloads/checksums and all four YAML path requirements.
3. Latent and Brain-ID cache commands for retrospective, prospective and
   validation splits, with the native feature geometry and output layout.
4. Unpaired RF pretraining, endpoint alignment, VAE decoder adaptation and
   paired RF adaptation, with the checkpoint dependencies between stages.
5. Full inference commands, inversion settings and the endpoint-aligner choice.

Run setup from the repository root, and cache/training/inference commands from
its `FieldRF/` subdirectory as shown in that README.

## Inference

See the [homepage inference commands](../README.md#inference). They state the
selected RF/VAE checkpoints, cache split, inversion solver/steps and output
layout explicitly. Existing checkpoint filenames remain accepted; no test-time
augmentation is used by default.

## CPU regression tests

The committed suite contains 46 tests in nine files:

| File | Tests | Coverage |
|---|---:|---|
| [test_cache_pipeline.py](tests/test_cache_pipeline.py) | 6 | Atomic writes, asynchronous failures, cache limits, entry points, feature geometry and native cache → real RF training → checkpoint reload → sampling |
| [test_alignment.py](tests/test_alignment.py) | 8 | Unpaired reference sampling, endpoint losses, RF paths and truncated inversion |
| [test_paired_dataset.py](tests/test_paired_dataset.py) | 3 | Same-subject/modality pairing, domain completeness and cross-field record counts |
| [test_spatial_loss.py](tests/test_spatial_loss.py) | 3 | Fractional evaluation-slab weights and weighted L1 supervision |
| [test_model_loading.py](tests/test_model_loading.py) | 16 | Full/EMA and raw UNet loading, strict VAE restoration, source adaptation, invalid checkpoints and inversion configuration |
| [test_sampling.py](tests/test_sampling.py) | 4 | Euler/Heun direction, latent scaling, gradient propagation and deterministic endpoint reversal |
| [test_training.py](tests/test_training.py) | 4 | RF/VAE full epoch-boundary resume, frozen encoding path and complete training state |
| [test_distributed.py](tests/test_distributed.py) | 1 | Two-rank real RF gradient averaging, adapter rollouts, VAE synchronization and resume |
| [test_namespace.py](tests/test_namespace.py) | 1 | Installed imports coexist with unrelated top-level utils |

From the repository root:

```bash
python -m pip install --upgrade 'pip>=25.3'
python -m pip install -r requirements.txt --build-constraint build-constraints.txt
python -m pip install --no-deps -e FieldRF/
python -m pytest -q FieldRF/tests
```

Tests use synthetic tensors, small networks and temporary cache/checkpoint
files. They run on CPU without MRI data, a Brain-ID checkout or trained FieldRF
weights. Full experiment reproduction follows the pipeline above and requires
the external data, caches and pretrained weights listed under dependencies.

See [project citations](../REFERENCES.md), [tested environment](../REPRODUCIBILITY.md)
and [MAISI brain-MRI model downloads](https://huggingface.co/nvidia/NV-Generate-MR-Brain).
