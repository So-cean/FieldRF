# FieldRF

Anatomy-conditioned 3D rectified flow with a shared model for three modalities
and five field strengths. The maintained implementation uses rectified flow and
ControlNet, unpaired pretraining and paired cross-field adaptation.

## Code layout

| File | Responsibility |
|---|---|
| `data/dataset.py` | Cached unpaired volumes, paired field adaptation and visualization pairs |
| `data/volume_dataset.py` | Raw NIfTI volumes and preprocessing for cache generation |
| `data/alignment_dataset.py` | Real unpaired samples for endpoint-distribution alignment |
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

Independent samples from `UnpairedEndpointAlignmentDataset` are suitable for
distribution matching, not voxel-wise reconstruction. Real paired adaptation
uses `Training_prospective`.

## Dependencies and paths

**The supplied recipe is Brain-ID conditioned.** Every RF training stage reads
both a MAISI latent and a Brain-ID `feat4` cache for each volume. Brain-ID source
and its pretrained checkpoint are required to **generate** the feature caches;
RF training and inference read those caches without running Brain-ID again.
MONAI supplies the MAISI networks. Its original VAE weights are required for
latent generation, and a decoder checkpoint is required for visualization and
inference.

The CPU tests use synthetic caches and therefore do not establish that external
weights or full-volume feature extraction work. See the
[tested environment and verification scope](../REPRODUCIBILITY.md).

## Reproduction pipeline

### 1. Obtain source, dependencies and pretrained weights

Run this setup from the **repository root**, which contains `requirements.txt`:

```bash
git submodule update --init --recursive
# Configure PyTorch/torchvision for your hardware first.
python -m pip install -r requirements.txt
```

The `third_party/Brain-ID` submodule pins the public
[Brain-ID source at `5d8a2d3`](https://github.com/jhuldr/Brain-ID/tree/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253).
The feature writer finds this path automatically. To use another checkout, set
`BRAINID_ROOT` to its absolute path; doing so changes the source dependency.
Cloning this repository with `--recurse-submodules` initializes it automatically;
downloading a GitHub source ZIP does not fetch submodules.

The root requirements include the frozen extractor's import dependencies
(`iopath`, `simplejson`, `tabulate`, `pytz`, `visdom`, plus the shared imaging
libraries). Use these requirements for this pipeline; the upstream Brain-ID
requirements describe its older full training stack and pin PyTorch versions.
See [setup](../docs/README.md) for package and CLI installation options.

Download the following pretrained files separately. Their filenames identify
upstream weight releases, independently of the FieldRF source layout.

| Artifact | Download source | Configuration |
|---|---|---|
| `brain_id_pretrained.pth` | The weights folder linked in the [Brain-ID feature-extractor demo](https://github.com/jhuldr/Brain-ID/blob/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253/README.md#playing-with-brain-id-synthesizer-and-feature-extractor) | `brainid.ckpt_path` in `pretrain.yaml` |
| `autoencoder_v1.pt` | [Shared MAISI VAE, NVIDIA release `4156648`](https://huggingface.co/nvidia/NV-Generate-CT/blob/4156648ce44f5561b75d96b12600fe5afe6b6e8f/models/autoencoder_v1.pt) | `vae.checkpoint` in pretraining, endpoint alignment and VAE adaptation |
| `diff_unet_3d_rflow-mr-brain_v0.pt` | [Brain-MRI RF, NVIDIA release `67ba36a`](https://huggingface.co/nvidia/NV-Generate-MR-Brain/blob/67ba36a4d3aae5854d7f2f56638e26a641684433/models/diff_unet_3d_rflow-mr-brain_v0.pt) | `bridge.ckpt_path` in `pretrain.yaml` |

Record the download revisions and checksums. Replacing the RF initialization
with a newer upstream release changes this recipe. MRI volumes, pretrained
weights, generated caches and trained FieldRF checkpoints are not bundled.
The VAE is distributed in NVIDIA's shared CT model repository; the brain-MRI
repository supplies the RF checkpoint. The two linked upstream releases publish
these SHA-256 values for checking your downloads:

```text
1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3  autoencoder_v1.pt
b49aa3162ff94b1852f81e520ed4165c5656000ba176c033063c35350b3305e9  diff_unet_3d_rflow-mr-brain_v0.pt
```

### 2. Configure data and artifact paths

Obtain the challenge data through
[MRIxFields2026](https://github.com/MRIxFields/MRIxFields2026). Set
`data.data_root` to the directory containing the split folders, not their parent:

```text
/path/to/MRIxFields2026/
  Training_retrospective/{T1W,T2W,T2FLAIR}/{0.1T,1.5T,3T,5T,7T}/*.nii.gz
  Training_prospective/{T1W,T2W,T2FLAIR}/{0.1T,1.5T,3T,5T,7T}/*.nii.gz
  Validating_prospective/{T1W,T2W,T2FLAIR}/{0.1T,1.5T,3T,5T,7T}/*.nii.gz
```

Set the same absolute `data.data_root` and `data.cache_root` in **all four**
configuration files. Replace every `/path/to/` placeholder. The pretraining
configuration must include:

```yaml
data:
  data_root: /path/to/MRIxFields2026
  cache_root: /path/to/fieldrf_cache
  modalities: [T1W, T2W, T2FLAIR]
  fields: [0.1T, 1.5T, 3T, 5T, 7T]
  feat_cache_name: brainid_feat
brainid:
  ckpt_path: /path/to/pretrained/brain_id_pretrained.pth
vae:
  checkpoint: /path/to/pretrained/autoencoder_v1.pt
bridge:
  ckpt_path: /path/to/pretrained/diff_unet_3d_rflow-mr-brain_v0.pt
```

This is a path checklist, not a replacement for the full network/training YAML.
Keep the supplied architecture settings. Complete the downstream paths as their
artifacts are produced:

| Configuration | Required artifact paths |
|---|---|
| `pretrain.yaml` | Original VAE, original MAISI RF and Brain-ID weights |
| `endpoint_alignment.yaml` | `finetune_from: runs/pretrain/fieldrf_epoch90.pt`; original VAE |
| `vae_finetune.yaml` | Original VAE to adapt; same raw data and latent caches |
| `finetune.yaml` | Same epoch-90 RF base; `endpoint_aligner_checkpoint: runs/endpoint_alignment/fieldrf_unpaired_alignment_best.pt`; `vae.checkpoint: runs/vae_finetune/vae_finetune_epoch1.pt` |

The endpoint aligner records its RF base checkpoint path; paired adaptation
checks that it matches `finetune_from`. Preserve that relationship when relocating
artifacts. The paired config's subject IDs must match the released prospective
training subjects. Validation cases do not enter the optimizer.

For **all subsequent commands**, enter the method directory from the repository
root:

```bash
cd FieldRF
```

### 3. Generate both caches for every required split

Cache generation processes all three `data.modalities` and all five
`data.fields` by default. The three splits have different roles:

| Split | Cache consumers |
|---|---|
| `Training_retrospective` | Unpaired RF pretraining, endpoint alignment and VAE decoder adaptation |
| `Training_prospective` | Paired RF adaptation and training-time visualization/diagnostics |
| `Validating_prospective` | VAE validation and challenge validation inference |

Run the two writers for **each** split on a suitably sized CUDA GPU:

```bash
for split in Training_retrospective Training_prospective Validating_prospective; do
  python scripts/cache_latents.py \
    --config configs/pretrain.yaml --split "$split" || exit 1
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python scripts/cache_brainid_feat.py \
    --config configs/pretrain.yaml --split "$split" --feats 4 || exit 1
done
```

Brain-ID's upstream loader uses the older `torch.load` behavior for a checkpoint
containing configuration objects. On PyTorch 2.6 or later, the command-scoped
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` setting restores that behavior for the
**official pretrained checkpoint**. It does not change your PyTorch installation
or persist beyond that feature-cache command.

MAISI encodes a preprocessed `384³` image as a `(4,96,96,96)` latent. Brain-ID
receives the same image resized to `192³` and mapped to `[0,1]`. The current
`[64,64]` ControlNet conditioning pyramid requires native
`(64,192,192,192)` `feat4`, so use `--feats 4` and the automatically derived
`feat4_pool=1`. A single-level `[64]` conditioning pyramid would require `pool=2`.
Existing pooled caches must match the selected architecture; changing cache
geometry requires a different cache location or explicit regeneration.

Each volume produces the following two files, preserving its NIfTI stem:

```text
/path/to/fieldrf_cache/
  Training_retrospective/
    latent/T1W/0.1T/R_T1W_0.1T_0001.pt
    brainid_feat/T1W/0.1T/R_T1W_0.1T_0001.b2nd
  Training_prospective/
    latent/T1W/0.1T/P_T1W_0.1T_0006.pt
    brainid_feat/T1W/0.1T/P_T1W_0.1T_0006.b2nd
  Validating_prospective/
    latent/{modality}/{field}/{NIfTI_stem}.pt
    brainid_feat/{modality}/{field}/{NIfTI_stem}.b2nd
```

The `.pt` contains the `latent` tensor; the `.b2nd` contains compressed float16
features and shape metadata. Check the writer summaries and ensure every input
stem has **both** output files before training. Brain-ID reports per-volume
errors in its summaries; a completed command with nonzero `errors` does not
mean a complete cache. The writers skip existing outputs. Use
`--modality T1W` to generate a single modality when diagnosing extraction.
The Brain-ID writer also supports `--fields`, `--max_volumes` and `--overwrite`;
use `--help` for these controls. Generate the full caches for the actual run.

### 4. Train and connect the stage artifacts

Run these stages in order after completing the caches and YAML paths:

```bash
python scripts/train.py --config configs/pretrain.yaml
python scripts/train_endpoint_aligner.py --config configs/endpoint_alignment.yaml
python scripts/finetune_vae.py --config configs/vae_finetune.yaml
python scripts/finetune.py --config configs/finetune.yaml
```

| Stage | Selected artifact / behavior |
|---|---|
| Unpaired RF pretraining | Runs for 100 epochs; subsequent stages use `runs/pretrain/fieldrf_epoch90.pt` |
| Unpaired endpoint alignment | Produces `runs/endpoint_alignment/fieldrf_unpaired_alignment_best.pt`, tied to that epoch-90 base |
| Decoder adaptation | Runs for 5 epochs; the supplied paired/inference recipe selects `runs/vae_finetune/vae_finetune_epoch1.pt` |
| Paired RF adaptation | Runs for 20 epochs; `runs/paired_finetune/fieldrf_paired_latest.pt` is the latest full checkpoint, with RF EMA; `fieldrf_paired_best_task3.pt` is a separate validation-selected artifact |

The VAE adaptation freezes its encoder and updates only the decoder. Original
latent caches remain valid, so no latent regeneration is needed afterward.
Paired adaptation uses only real same-subject, same-modality
`Training_prospective` pairs. The VAE is frozen during paired adaptation;
Brain-ID is already represented by its frozen caches. RF, ControlNet and the
endpoint aligner share parameters across fields and modalities.

An epoch-20 checkpoint and the best validation checkpoint are different
selection rules. After the configured 20-epoch run, `fieldrf_paired_latest.pt`
contains the epoch-20 RF EMA. Use the epoch-1 adapted VAE consistently for
paired adaptation and inference; the original MAISI decoder changes the
protocol. These commands describe the released recipe; full benchmark results
have not been remeasured by the CPU audit.

## Inference

For shared Any-to-Any **source endpoint transport** using the epoch-20 RF EMA:

```bash
python scripts/inference.py \
  --config configs/finetune.yaml \
  --checkpoint runs/paired_finetune/fieldrf_paired_latest.pt \
  --vae_checkpoint runs/vae_finetune/vae_finetune_epoch1.pt \
  --source_endpoint_transport --inversion_steps 20 --inversion_solver euler \
  --n_infer_steps 20 --task task3 --split Validating_prospective \
  --mode pro_pretrained --epoch_tag epoch20 \
  --output_dir /path/to/predictions
```

This invocation transports the source endpoint directly. To apply the learned
endpoint aligner as well, use the validation-selected paired artifact as **both**
`--checkpoint runs/paired_finetune/fieldrf_paired_best_task3.pt` and
`--endpoint_checkpoint runs/paired_finetune/fieldrf_paired_best_task3.pt`.
That file stores the RF and endpoint-aligner weights together under EMA
validation; set `--epoch_tag` to the actual selected epoch. Passing the original
unpaired aligner would use its earlier weights instead of the jointly adapted
aligner. Keep this inference choice explicit when comparing experiments.

Task 3 translates all five source fields into each of the other four fields
with one shared model. Inference requires both caches for the chosen split, the
raw MRI/reference grid and the adapted VAE decoder. It does not require the
Brain-ID checkout or Brain-ID checkpoint once the caches exist. No test-time
augmentation is used by default. Outputs follow the sweep layout consumed by
[the packaging tools](../mrix2026-cli/README.md). Use `--help` for target subsets
and other output options.

Existing checkpoints can retain their original filenames: the loader accepts
explicit paths, and parameter layouts and state-dict keys are preserved.

## CPU regression tests

The committed suite contains 27 tests in six files:

| File | Tests | Coverage |
|---|---:|---|
| [test_cache_pipeline.py](tests/test_cache_pipeline.py) | 3 | Cache entry points, feature geometry and native cache → real RF training → checkpoint reload → sampling |
| [test_alignment.py](tests/test_alignment.py) | 8 | Unpaired reference sampling, endpoint losses, RF paths and truncated inversion |
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

See [project citations](../REFERENCES.md), [tested environment](../REPRODUCIBILITY.md)
and [MAISI brain-MRI model downloads](https://huggingface.co/nvidia/NV-Generate-MR-Brain).
