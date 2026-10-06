# MRIxFields2026 — Evaluation

Official evaluation code for the MRIxFields2026 challenge.

## Prerequisites

Metric evaluation and SynthSeg segmentation use separate environments. See
[main README](../README.md#3-set-up-environment) for the metric environment.

```bash
conda activate mf  # Evaluation/evaluate.py
```

## SynthSeg Setup

SynthSeg is required for Dice and Volume metrics. Keep TensorFlow isolated from
the project's PyTorch environments by creating the dedicated environment:

```bash
conda env create -f Evaluation/environment_synthseg.yml
conda activate synthseg
```

On the MRIxFields cluster, install the supplied activation hooks so the
environment's CUDA 12.2/cuDNN libraries take precedence over CUDA paths inherited
from FreeSurfer or the login shell:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cp Evaluation/conda/activate.d/synthseg_tf215_cuda.sh "$CONDA_PREFIX/etc/conda/activate.d/"
cp Evaluation/conda/deactivate.d/synthseg_tf215_cuda.sh "$CONDA_PREFIX/etc/conda/deactivate.d/"
conda deactivate
conda activate synthseg
```

The environment pins TensorFlow 2.15.1 and Keras 2.15.0 and does not install
PyTorch.

### Step 1: Clone Repository

```bash
git clone https://github.com/BBillot/SynthSeg.git /path/to/SynthSeg
```

### Step 2: Add the Model Weights

The `.npy` label/class files ship with the repo (under `data/labels_classes_priors/`).
Only the model weights `synthseg_2.0.h5` need to be obtained separately:

- Download from [SynthSeg releases](https://github.com/BBillot/SynthSeg/releases), **or**
- Copy from a FreeSurfer 7.x install: `$FREESURFER_HOME/python/packages/SynthSeg/models/synthseg_2.0.h5`

Place it under `SynthSeg/models/`. Final layout:

```
SynthSeg/
├── models/
│   └── synthseg_2.0.h5                              # downloaded / copied
└── data/labels_classes_priors/
    ├── synthseg_segmentation_labels_2.0.npy        # already in the repo
    ├── synthseg_denoiser_labels_2.0.npy
    └── synthseg_topological_classes_2.0.npy
```

### Step 3: Configure Path

```bash
# Add to .env (repo root)
echo 'SYNTHSEG_DIR=/path/to/SynthSeg' >> .env
```

### Step 4: Verify Installation

Run segmentation on a single NIfTI to confirm the install works end-to-end:

```bash
conda activate synthseg
mkdir -p /tmp/synthseg_test_in
cp "$(ls $DATA_DIR/Validating_prospective/T1W/7T/*.nii.gz | head -1)" /tmp/synthseg_test_in/

python Evaluation/segment.py \
    --input_dir /tmp/synthseg_test_in/ \
    --output_dir /tmp/synthseg_test_out/
ls /tmp/synthseg_test_out/    # should contain one *_seg.nii.gz
```

If the script errors with missing weights, double-check `SYNTHSEG_DIR` and that
`synthseg_2.0.h5` is under `$SYNTHSEG_DIR/models/`.

## Pipeline

```text
1. Inference (Baseline)      → $INFERENCE_DIR/*.nii.gz
2. Segmentation (this step)  → ${INFERENCE_DIR}_seg/*.nii.gz
3. Evaluation (this step)    → results.csv
```

## Segmentation

SynthSeg segmentations are required only for **Task 1 / Task 2** (Dice / Volume metrics). **Task 3 is evaluated on voxel-level metrics only and does not require any segmentation step.**

```bash
# Task 1 / Task 2 only — segment predictions
python Evaluation/segment.py \
    --input_dir $INFERENCE_DIR/ \
    --output_dir ${INFERENCE_DIR}_seg/

# Task 1 / Task 2 only — segment ground truth (needed for Dice/Volume evaluation)
# Example for Task 1 (target: 7T)
python Evaluation/segment.py \
    --input_dir $DATA_DIR/Validating_prospective/T1W/7T/ \
    --output_dir $DATA_DIR/target_seg/
```

## Evaluation

```bash
# Task 3 (voxel-level only) — required path for Task 3
# Example: 0.1T -> 7T translation under task3/
python Evaluation/evaluate.py \
    --pred_dir $INFERENCE_DIR/ \
    --target_dir $DATA_DIR/Validating_prospective/T1W/7T/ \
    --metrics nrmse ssim lpips

# Task 1 / Task 2 — voxel-level metrics only (skip Dice/Volume)
# Example for Task 1: 0.1T -> 7T
python Evaluation/evaluate.py \
    --pred_dir $INFERENCE_DIR/ \
    --target_dir $DATA_DIR/Validating_prospective/T1W/7T/ \
    --metrics nrmse ssim lpips

# Task 1 / Task 2 — full evaluation with all 5 metrics
python Evaluation/evaluate.py \
    --pred_dir $INFERENCE_DIR/ \
    --target_dir $DATA_DIR/Validating_prospective/T1W/7T/ \
    --pred_seg_dir ${INFERENCE_DIR}_seg/ \
    --target_seg_dir $DATA_DIR/target_seg/ \
    --metrics nrmse ssim lpips dice volume \
    --output_csv results.csv
```

## Metrics

| # | Metric | Direction | Applies to | Description |
|---|--------|-----------|------------|-------------|
| 1 | **nRMSE** | Lower is better | Task 1 / 2 / 3 | Normalized Root Mean Square Error |
| 2 | **SSIM** | Higher is better | Task 1 / 2 / 3 | Structural Similarity Index |
| 3 | **LPIPS** | Lower is better | Task 1 / 2 / 3 | Learned Perceptual Image Patch Similarity (AlexNet) |
| 4 | **Dice** | Higher is better | Task 1 / 2 | Overlap on 14 deep gray matter structures (via SynthSeg) |
| 5 | **Volume** | Higher is better | Task 1 / 2 | Normalized volume consistency per DGM structure |

**Normalization**: Both predictions and targets are in `[0, 1]`. Ground truth volumes are pre-normalized; model outputs are mapped via `tanh*0.5+0.5`. All voxel-level metrics (nRMSE / SSIM / LPIPS) are computed on the **full volume without a brain mask** for a consistent evaluation scope.

**SynthSeg resampling** (Task 1 / 2 only): SynthSeg 2.0 internally resamples volumes to 1mm isotropic resolution before segmentation. Input volumes at 0.5mm (364x436x364) produce segmentation maps at 1mm (182x218x182). Dice and volume consistency are computed in the segmentation's 1mm space. This does not affect metric comparisons since both prediction and target segmentations are in the same space.

**Ranking**: Per-metric rank summed across the applicable metrics — **5 metrics for Task 1 / Task 2**, **3 metrics for Task 3** (nRMSE / SSIM / LPIPS only). Lowest total = best.

## Dependencies

Metric dependencies are included in the `mf` environment. The standalone
segmentation dependencies are pinned in `Evaluation/environment_synthseg.yml`:

```text
mf:               numpy, scipy, scikit-image, torch, lpips, nibabel
synthseg:         tensorflow==2.15.1, keras==2.15.0, nibabel
```
