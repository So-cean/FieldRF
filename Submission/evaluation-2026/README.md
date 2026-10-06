# Local validation scorer

`score.py` validates a per-task submission directory and reports the challenge
metrics. It is also used by `mrix2026 check`.

## Inputs

The submission directory ends in `task1`, `task2` or `task3` and contains:

```text
task1/T1W/0.1T_to_7T/pred/P_T1W_7T_0001.nii.gz
task1/T1W/0.1T_to_7T/seg/P_T1W_7T_0001_seg.nii.gz
```

The ground-truth directory mirrors the task layout, using `gt/` and `gt_seg/`.
Its files define the expected subjects for each modality and field pair.
Task 3 evaluates voxel metrics and uses no segmentation branch.

## Running

From the repository root:

```bash
python Submission/evaluation-2026/score.py \
  --input /path/to/submission/task1 \
  --gt_dir /path/to/ground_truth/task1 \
  --task task1 --output /path/to/scores
```

Use `--help` for output and execution options. Task 1/2 report nRMSE, SSIM,
LPIPS, Dice and volume metrics; Task 3 reports nRMSE, SSIM and LPIPS. Missing
or invalid predictions are handled by the retained evaluator's validation rules.

The scoring implementation is retained unchanged. Packaging is documented in
[Submission](../README.md), and reference metrics in
[Evaluation](../../Evaluation/README.md).
