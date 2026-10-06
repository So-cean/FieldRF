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
| `models/model.py` | `FieldRFModel`, RF integration and training loss |
| `models/endpoint_aligner.py` | `EndpointAligner`, shared domain-conditioned endpoint correction |
| `utils/visualization.py` | VAE decoding, reference-grid metrics and rendering |

Modules, class names, entry points and configuration names describe their roles.
Experiment epochs and checkpoint filenames identify artifacts independently
of the source-code layout.

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

## Tests

From the repository root:

```bash
python -m pytest -q FieldRF/tests
```
