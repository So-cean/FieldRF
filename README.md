# FieldRF

Anatomy-conditioned 3D rectified flow for cross-field MRI harmonization.
FieldRF uses one shared model across **T1W, T2W and T2FLAIR** and five field
strengths: **0.1T, 1.5T, 3T, 5T and 7T**.

The pipeline combines a MAISI latent representation, frozen Brain-ID anatomical
features, a shared rectified-flow network and paired cross-field adaptation.
**The supplied configurations require both MAISI latent caches and Brain-ID
feature caches.** Brain-ID source is a pinned Git submodule; its pretrained
weights and the MAISI weights must be downloaded separately. See the
[complete cache and training procedure](#reproduction-pipeline).

| Directory | Contents |
|---|---|
| [FieldRF/](FieldRF/README.md) | Models, data loaders, configurations, training, inference and tests |
| [Baseline/](Baseline/README.md) | CUT, CycleGAN, StarGANv2 and other comparison methods |
| [Evaluation/](Evaluation/README.md) | Reference evaluation implementation |
| [Submission/](Submission/README.md) | Validation submission format and packaging |
| [mrix2026-cli/](mrix2026-cli/README.md) | Segmentation, building, checking and ZIP utilities |

## Getting started

```bash
git clone --recurse-submodules https://github.com/So-cean/FieldRF.git
cd FieldRF
# For an existing clone, or a clone made without --recurse-submodules:
git submodule update --init --recursive
```

1. Configure PyTorch/torchvision for your hardware, then install
   `requirements.txt`; see [setup](docs/README.md). It includes the Python
   libraries needed by the frozen Brain-ID feature extractor.
2. Obtain the MRI dataset, the original MAISI VAE/RF weights and the
   [Brain-ID pretrained weights](https://github.com/jhuldr/Brain-ID/blob/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253/README.md#playing-with-brain-id-synthesizer-and-feature-extractor).
3. Set the paths in the four `FieldRF/configs/` files as described in the
   [configuration checklist](#2-configure-data-and-artifact-paths).
4. Generate **both caches for all three dataset splits**, then run pretraining,
   endpoint alignment, decoder adaptation and paired adaptation in the
   [reproduction guide](#reproduction-pipeline).

For baseline runs, copy `.env.example` to `.env`, configure local paths, and
follow the [baseline guide](Baseline/README.md).

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
[tested environment and verification scope](REPRODUCIBILITY.md).

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
See [setup](docs/README.md) for package and CLI installation options.

Download the following pretrained files separately. Their filenames identify
upstream weight releases, independently of the FieldRF source layout.

| Artifact | Download source | Configuration |
|---|---|---|
| `brain_id_pretrained.pth` | The weights folder linked in the [Brain-ID feature-extractor demo](https://github.com/jhuldr/Brain-ID/blob/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253/README.md#playing-with-brain-id-synthesizer-and-feature-extractor) | `brainid.ckpt_path` in `pretrain.yaml` |
| `autoencoder_v1.pt` | [Shared MAISI VAE, NVIDIA release `4156648`](https://huggingface.co/nvidia/NV-Generate-CT/blob/4156648ce44f5561b75d96b12600fe5afe6b6e8f/models/autoencoder_v1.pt) | `vae.checkpoint` in pretraining, endpoint alignment and VAE adaptation |
| `diff_unet_3d_rflow-mr-brain_v0.pt` | [Brain-MRI RF, NVIDIA release `67ba36a`](https://huggingface.co/nvidia/NV-Generate-MR-Brain/blob/67ba36a4d3aae5854d7f2f56638e26a641684433/models/diff_unet_3d_rflow-mr-brain_v0.pt) | `bridge.ckpt_path` in `pretrain.yaml` |

The released brain RF checkpoint uses residual down/up-sampling and learned attention output projections.
The loader infers these architecture settings from its parameters, restores every pretrained UNet
parameter except the intentionally replaced MRI-domain embedding, and rejects
missing or unexpected network parameters. Legacy FieldRF checkpoints retain
their original convolutional resampling through the same inference mechanism.
New checkpoints record the resolved architecture. A run that only prints
parameter-mismatch warnings is not a successful pretrained initialization.

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

For shared Any-to-Any transport using the jointly adapted epoch-20 RF and
endpoint-aligner EMA:

```bash
python scripts/inference.py \
  --config configs/finetune.yaml \
  --checkpoint runs/paired_finetune/fieldrf_paired_latest.pt \
  --vae_checkpoint runs/vae_finetune/vae_finetune_epoch1.pt \
  --endpoint_checkpoint runs/paired_finetune/fieldrf_paired_latest.pt \
  --inversion_steps 20 --inversion_solver euler \
  --n_infer_steps 20 --task task3 --split Validating_prospective \
  --mode pro_pretrained --epoch_tag epoch20 \
  --output_dir /path/to/predictions
```

The full paired checkpoint stores the RF and endpoint-aligner EMA together.
Both are selected automatically when present. To evaluate source endpoint
transport without the learned aligner, omit `--endpoint_checkpoint` and add
`--source_endpoint_transport`; keep the explicit inversion steps and solver.
This is a different inference choice, so record it when comparing experiments.
For validation-selected inference, use `fieldrf_paired_best_task3.pt` as both
checkpoint arguments and set `--epoch_tag` to its actual selected epoch.
That artifact already stores the jointly selected EMA weights. A separate
unpaired aligner checkpoint contains its earlier weights and changes the
protocol.

Task 3 translates all five source fields into each of the other four fields
with one shared model. Inference requires both caches for the chosen split, the
raw MRI/reference grid and the adapted VAE decoder. It does not require the
Brain-ID checkout or Brain-ID checkpoint once the caches exist. No test-time
augmentation is used by default. Outputs follow the sweep layout consumed by
[the packaging tools](mrix2026-cli/README.md). Use `--help` for target subsets
and other output options.

Existing checkpoints can retain their original filenames: the loader accepts
explicit paths, and parameter layouts and state-dict keys are preserved.

## Tests and experiment reproduction

The [committed CPU test suite](FieldRF/tests/) contains 34 tests across six
files. Install its dependencies and run it from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q FieldRF/tests mrix2026-cli/tests
```

Two additional CLI tests check submission ZIP layout. The tests cover data
pairing, RF/spatial losses, sampling/autograd, cache generation and a real MONAI
network's training/checkpoint round trip with synthetic fixtures. See the
[test inventory](FieldRF/README.md#cpu-regression-tests). Full experiment
reproduction follows the [method guide](FieldRF/README.md).

Training and inference require the external data, cached representations and
weights described in the method guide. A pretrained checkpoint can be loaded
by specifying its existing filename; model parameter keys are unchanged.
Inference uses no test-time augmentation by default.

See [the method guide](FieldRF/README.md) for the pipeline and
[reference evaluation](Evaluation/README.md) for metric definitions.

## Acknowledgements and citations

The benchmark, baseline and evaluation framework originates from
[MRIxFields/MRIxFields2026](https://github.com/MRIxFields/MRIxFields2026).
FieldRF uses [MONAI](https://github.com/Project-MONAI/MONAI),
[Brain-ID](https://github.com/jhuldr/Brain-ID) and
[MAISI / MAISI-v2](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR).
Please cite the corresponding original works listed in [REFERENCES.md](REFERENCES.md).

The [reproducibility check](REPRODUCIBILITY.md) records the tested environment,
verification scope and external artifacts needed for full experiments.

## License

See [LICENSE](LICENSE).
