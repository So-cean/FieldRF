# FieldRF

FieldRF is an anatomy-conditioned 3D rectified flow method for cross-field MRI
harmonization. One shared model translates volumes between field strengths
within each modality: **T1W, T2W, T2FLAIR** across **0.1T, 1.5T, 3T, 5T, 7T**.

[Paper (MICCAI 2026)](https://papers.miccai.org/miccai-2026-sat/paper/MRIxFields2026_026.pdf) ·
[Code guide](FieldRF/README.md) · [Setup](docs/README.md) ·
[Baselines](Baseline/README.md) · [Evaluation](Evaluation/README.md) ·
[Submission](mrix2026-cli/README.md)

![FieldRF pipeline for anatomy-conditioned volumetric MRI harmonization](assets/fieldrf_pipeline.svg)

| Project facts | Details |
|---|---|
| Input → output | Source MRI volume and target field → target-field NIfTI volume on the source image grid |
| Conditioning | Cached MAISI latents, frozen Brain-ID anatomy features and modality/field labels |
| Training | Unpaired RF pretraining, endpoint alignment, VAE decoder adaptation, then paired RF adaptation |
| Artifacts | Official dataset and upstream weights are obtained separately; train FieldRF checkpoints with the recipe below |

[Project questions](docs/README.md#project-questions) · [Text index](llms.txt) ·
[Software metadata](codemeta.json)

## Reproduction pipeline

### 1. Obtain source, dependencies and pretrained weights

Use Python 3.10+ and install PyTorch/torchvision for your hardware, then run:

```bash
git clone --recurse-submodules https://github.com/So-cean/FieldRF.git
cd FieldRF
python -m pip install --upgrade 'pip>=25.3'
python -m pip install -r requirements.txt --build-constraint build-constraints.txt
python -m pip install --no-deps -e FieldRF/
```

Brain-ID is pinned in `third_party/Brain-ID`; existing clones can initialize it
with `git submodule update --init --recursive`. Obtain the
[official dataset](https://github.com/MRIxFields/MRIxFields2026) and these weights:

| Weights | Download | Configuration setting |
|---|---|---|
| `brain_id_pretrained.pth` | [Brain-ID](https://github.com/jhuldr/Brain-ID/blob/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253/README.md#playing-with-brain-id-synthesizer-and-feature-extractor) | `brainid.ckpt_path` |
| `autoencoder_v1.pt` | [MAISI VAE](https://huggingface.co/nvidia/NV-Generate-CT/blob/4156648ce44f5561b75d96b12600fe5afe6b6e8f/models/autoencoder_v1.pt) | `vae.checkpoint` |
| `diff_unet_3d_rflow-mr-brain_v0.pt` | [MAISI brain RF](https://huggingface.co/nvidia/NV-Generate-MR-Brain/blob/67ba36a4d3aae5854d7f2f56638e26a641684433/models/diff_unet_3d_rflow-mr-brain_v0.pt) | `bridge.ckpt_path` |

### 2. Configure data and artifact paths

Edit `pretrain.yaml`, `endpoint_alignment.yaml`, `vae_finetune.yaml` and
`finetune.yaml` in `FieldRF/configs/`. Set absolute data/cache/weight paths,
use the same cache root throughout, and match paired `data.subjects` to the
prospective training IDs. Keep the supplied architecture and official splits:

```text
<data_root>/Training_retrospective/<modality>/<field>/*.nii.gz
<data_root>/Training_prospective/<modality>/<field>/*.nii.gz
<data_root>/Validating_prospective/<modality>/<field>/*.nii.gz
```

### 3. Generate both caches for every required split

From the method directory, cache MAISI latents and Brain-ID features for all
three modalities and five fields. Subsequent stages read these caches.

```bash
cd FieldRF
for split in Training_retrospective Training_prospective Validating_prospective; do
  python scripts/cache_latents.py \
    --config configs/pretrain.yaml --split "$split" || exit 1
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python scripts/cache_brainid_feat.py \
    --config configs/pretrain.yaml --split "$split" --feats 4 || exit 1
done
```

The environment setting supports Brain-ID's legacy checkpoint on PyTorch 2.6+.
Keep `feat_cache_name: brainid_feat`; check writer summaries for errors.
[Cache geometry and paths](docs/README.md#cache-format) are documented in setup.

### 4. Train and connect the stage artifacts

Run the stages in order, setting the selected artifacts in downstream configs:

```bash
python scripts/train.py --config configs/pretrain.yaml
python scripts/train_endpoint_aligner.py --config configs/endpoint_alignment.yaml
python scripts/finetune_vae.py --config configs/vae_finetune.yaml
python scripts/finetune.py --config configs/finetune.yaml
```

| Stage | Selected artifact / downstream setting |
|---|---|
| RF pretraining, 100 epochs | `finetune_from: runs/pretrain/fieldrf_epoch90.pt` in alignment and paired configs |
| Endpoint alignment, 20 epochs | `endpoint_aligner_checkpoint: runs/endpoint_alignment/fieldrf_unpaired_alignment_best.pt` in paired config |
| VAE adaptation, 5 epochs | `vae.checkpoint: runs/vae_finetune/vae_finetune_epoch1.pt` in paired config and inference |
| Paired RF adaptation, 20 epochs | `runs/paired_finetune/fieldrf_paired_latest.pt`, with joint RF/aligner EMA |

Use the same RF base for alignment and paired adaptation. VAE adaptation freezes
the encoder, keeping latent caches valid. Paired training uses same-subject,
same-modality prospective pairs.
See [DDP and resume](docs/README.md#distributed-training-and-resume) for multi-GPU
commands and checkpoint restoration.

## Inference

Use the paired checkpoint for both RF and aligner, with the epoch-1 decoder:

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

Inference uses both caches and source MRI grids, with TTA set to none.
See [checkpoint selection](docs/README.md#checkpoint-selection-and-inference)
for best-checkpoint and source-transport options, and the
[submission guide](mrix2026-cli/README.md) for packaging.

## Verification

From the repository root:

```bash
python -m pytest -q FieldRF/tests mrix2026-cli/tests
```

The suite passes **48 tests and 9 subtests**, including two-rank CPU/Gloo DDP
and epoch-boundary resume. [CI](https://github.com/So-cean/FieldRF/actions/workflows/ci.yml)
runs on Python 3.10/3.12. [Reproducibility records](REPRODUCIBILITY.md) cover the
real-data GPU pipeline, retained-paper-weight checks and small-run instructions.

## License

[MIT](LICENSE).

## Citation

Please cite FieldRF, presented at the MRIxFields2026 satellite event at MICCAI 2026.
[CITATION.cff](CITATION.cff) provides the machine-readable conference citation:

```bibtex
@inproceedings{SonHai_FieldRF_MICCAISAT2026,
  author = {Song, Haiyu and Xiang, Lei and Yang, Zongjian and Wu, Yang and Shen, Dinggang and Wang, Qian},
  title = {{FieldRF: Anatomy-Conditioned 3D Rectified Flow for Unified Cross-Field MRI Harmonization}},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Workshops and Challenges},
  series = {Lecture Notes in Computer Science},
  volume = {17274},
  publisher = {Springer Nature Switzerland},
  year = {2026},
  url = {https://papers.miccai.org/miccai-2026-sat/MRIxFields2026_026.html}
}
```

FieldRF builds on [MRIxFields2026](https://github.com/MRIxFields/MRIxFields2026),
[MONAI](https://github.com/Project-MONAI/MONAI),
[Brain-ID](https://github.com/jhuldr/Brain-ID) and
[MAISI / MAISI-v2](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR).
Citations for these projects and the comparison methods are in
[REFERENCES.md](REFERENCES.md).
