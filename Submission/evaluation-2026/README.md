# MRIxFields2026 — Validation Scorer

Drop-in scoring script for the test platform. Consumes an extracted submission
directory (the platform unzips before invoking — `score.py` does not extract),
compares predictions against ground truth, and produces `Result/results.json`
plus a per-sample `result_{task}.xlsx`.

> This is the retained scorer reference. The internal `OPERATIONS.md` is not
> included in this repository; this document does not provide its deployment runbook.

## Files in this directory (standalone library)

```
score.py            # platform scorer — entry point; self-contained
evaluate.py         # standalone per-pair evaluator (mirrors ../../Evaluation/evaluate.py)
segment.py          # SynthSeg wrapper for producing GT segmentations
                    # (mirrors ../../Evaluation/segment.py)
requirements.txt    # pip deps for all three scripts
README.md           # this file (technical reference)
```

`score.py`, `evaluate.py`, and `segment.py` are independent — no cross-imports,
no shared helpers. The repo's `Evaluation/` copies are the original; this
directory contains verbatim duplicates so the test-platform image only needs to
pull from `Submission/evaluation-2026/`.

## Tasks & metrics

| Task  | Pairs                                                                    | # subtasks | Metrics                          | Needs seg/ |
|-------|--------------------------------------------------------------------------|------------|----------------------------------|------------|
| task1 | 4 pairs (Any → 7T)                                                       | 12         | nRMSE, SSIM, LPIPS, Dice, Volume | yes        |
| task2 | 4 pairs (0.1T → Higher)                                                  | 12         | nRMSE, SSIM, LPIPS, Dice, Volume | yes        |
| task3 | 20 pairs (5 fields × 4 other fields, **incl. downfield**)                | 60         | nRMSE, SSIM, LPIPS               | no         |

A **subtask** = one `(pair, modality)` combination. `task3` is voxel-level only;
any `seg/` placed under `task3/` is ignored. task1 and task2 share the
`0.1T_to_7T` pair; that's intentional — they are independent evaluation queues.

### Terminology hierarchy

```
task        — a single score.py invocation (task1 / task2 / task3)
└── pair    — a (source_field, target_field) translation, e.g. 0.1T_to_7T
    └── modality — T1W / T2W / T2FLAIR
        └── subject — defined by the GT tree (whatever IDs sit under that
                       pair's gt/ directory)

subtask = pair × modality (one row in the JSON's per-subtask summary)
```

## Submission layout (`--input`)

`--input` is a **directory** (no archive extraction — unzip ahead of time)
whose **last path component is `task{N}`** and that contains modality
subdirectories directly:

```
{input}/                         ← directory name MUST be task1 / task2 / task3
├── {T1W,T2W,T2FLAIR}/{pair}/pred/P_{MOD}_{TGT}_{ID:04d}.nii.gz
└── {T1W,T2W,T2FLAIR}/{pair}/seg/ P_{MOD}_{TGT}_{ID:04d}_seg.nii.gz   # task1/task2 only
```

Filenames must follow `P_{MOD}_{TGT}_{ID:04d}.nii.gz` exactly — the strict
pre-validator rejects any file that doesn't match (see "Failure path"
below).

## Ground-truth layout (`--gt_dir`)

`--gt_dir` is the **per-task** GT directory whose **last path component is
`task{N}`** and that mirrors the submission layout one-to-one — `gt/` plays
the role of the submission's `pred/`, `gt_seg/` the role of `seg/`.
The **set of expected IDs per `(modality, pair)` is whatever exists under
that pair's `gt/` directory** — the GT tree itself defines a complete
submission.

```
{gt_dir}/                        ← directory name MUST equal --task
├── {T1W,T2W,T2FLAIR}/{pair}/gt/    P_{MOD}_{TGT}_{ID:04d}.nii.gz
└── {T1W,T2W,T2FLAIR}/{pair}/gt_seg/P_{MOD}_{TGT}_{ID:04d}_seg.nii.gz   # task1/task2 only
```

`task3` GT trees carry `gt/` only (no `gt_seg/`). For task1/task2 a missing
`gt_seg/` file is treated as an evaluator-side error (the GT tree is owned
by the platform, so any gap there means re-build the GT tree).

The three CLI inputs cross-validate each other: `Path(--input).name`,
`Path(--gt_dir).name`, and `--task` must all be identical
(e.g. all three are `task1`). A mismatch returns
`INVALID` with an `ERR_TASK_MISMATCH` prefix.

### Building the GT tree

Take the private full-paired cohort and filter it down to the IDs each pair's
source-field input release exposes. `segment.py` produces a flat
`P_{MOD}_{FIELD}_{ID}_seg.nii.gz` tree; copy or symlink the relevant subset
into the per-pair `gt_seg/` slot:

```bash
# Set SYNTHSEG_DIR (or put it in $REPO_ROOT/.env)
export SYNTHSEG_DIR=/path/to/SynthSeg

# Step 1: run SynthSeg per (modality, field) into a flat staging directory.
for MOD in T1W T2W T2FLAIR; do
  for FIELD in 0.1T 1.5T 3T 5T 7T; do
    python Submission/evaluation-2026/segment.py \
        --input_dir  $DATA_DIR/Validating_prospective/$MOD/$FIELD/ \
        --output_dir $DATA_DIR/target_seg_flat/
  done
done

# Step 2: assemble the task-organized GT tree by copying / linking the
# per-pair subset into {gt_dir}/{task}/{mod}/{pair}/{gt,gt_seg}/.
# (assembly script lives outside score.py — see release-build tooling.)
```

## Usage

```bash
conda activate mf

# -i, -g, and -t all carry the same task name. -i and -g are
# directories (already extracted) whose last path component is task{N}.
python Submission/evaluation-2026/score.py \
    -i /path/to/submissions/task3 \
    -t task3 \
    -g /path/to/Validating_prospective_pack1_ground_truth/task3 \
    -o /path/to/evaluation/task3

python Submission/evaluation-2026/score.py \
    -i /path/to/submissions/task1 \
    -t task1 \
    -g /path/to/Validating_prospective_pack1_ground_truth/task1 \
    -o /path/to/evaluation/task1
```

If the participant uploaded a `task1.zip` archive, unzip it first
(`unzip task1.zip -d /tmp/sub/task1/`) and pass the resulting directory.

Outputs land in the directory passed to `-o` (in the example above,
`/path/to/evaluation/task{N}/`):

- `Result/results.json` — main file consumed by the platform
- `Result/result_{task}.xlsx` — per-sample detail + flattened summary
- `better_log.zip` — bundle of the entire `Result/` directory (parity with the
  reference scorer; convenient for admins to download a complete log artifact)

## `results.json` shape

All three tasks ship a **per-modality** summary (averaged across pairs within
each modality). The schema is identical for task1 / task2 / task3 — only the
metric set differs (task3 has no `Dice` / `Volume`).

```json
{
  "submission_status": "SCORED",
  "primary_metric": "SSIM",
  "primary_score": 0.9311,
  "Num_Files": "36/36",
  "Avg_T1W_Num": "12/12",
  "Avg_T1W_nRMSE_adj": 0.0512,
  "Avg_T1W_SSIM_adj":  0.9421,
  "Avg_T1W_LPIPS_adj": 0.0834,
  "Avg_T1W_Dice_adj":  0.8714,
  "Avg_T1W_Volume_adj":0.9123,
  "Avg_T2W_Num": "12/12",
  "Avg_T2W_nRMSE_adj": 0.0584,
  ...
  "Mean_of_all_subtasks_nRMSE_adj":  0.0598,
  "Mean_of_all_subtasks_SSIM_adj":   0.9311,
  "Mean_of_all_subtasks_LPIPS_adj":  0.0921,
  "Mean_of_all_subtasks_Dice_adj":   0.8602,
  "Mean_of_all_subtasks_Volume_adj": 0.9015
}
```

- Sizes: task1 / task2 → 27 keys (5 metrics × 3 modalities + 5 global + 4 header).
  task3 → 19 keys (3 metrics × 3 modalities + 3 global + 4 header).
- `primary_metric` / `primary_score` are the platform's ranking handle:
  always `"SSIM"`, the value is `Mean_of_all_subtasks_SSIM_adj`. Read these
  two fields directly instead of picking a metric out of the per-modality block.
- `Avg_{T1W,T2W,T2FLAIR}_*` aggregates over every (pair, subject) sample
  belonging to that modality (task1/task2: 4 pairs × 3 subjects = 12 samples;
  task3: 20 pairs × 3 subjects = 60 samples).
- `Mean_of_all_subtasks_*` averages over every expected sample (sum across modalities).
- For task1/task2 submissions that omit `seg/` entirely, every `*_Dice_adj`
  and `*_Volume_adj` field (per-modality and `Mean_of_all_subtasks_*`) is
  reported as `null`. `primary_score` is unaffected because SSIM doesn't
  depend on segmentation.
- `result_{task}.xlsx` (the companion Excel file) still ships the
  per-(pair, modality) breakdown on its summary sheet, for inspection.

### Failure path

The scorer is strict: any participant-side defect produces `INVALID` and a
non-zero exit code, *before* any metric is computed. `results.json` is still
written so the platform always has a JSON to read:

```json
{
  "submission_status": "INVALID",
  "submission_errors": "<ERR_PREFIX>: <details>"
}
```

`<ERR_PREFIX>` is one of:

| Prefix                 | Meaning                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `TASK_MISMATCH_ERROR`  | `Path(--input).name` didn't equal `--task` (or the input path isn't a directory at all — this scorer doesn't extract archives). |
| `FILE_TREE_ERROR`      | A `(modality, pair)`'s pred (or seg) ID set didn't match the GT — missing or extra subjects. |
| `FORMAT_ERROR`         | A file's location/name violated `{modality}/{pair}/{pred\|seg}/P_{MOD}_{FIELD}_{ID:04d}.nii.gz`, or `nibabel` couldn't load it. |
| `SIZE_ERROR`           | A pred (or seg) shape didn't match the corresponding GT.                |
| `NAN_ERROR`            | A pred (or seg) volume contained `NaN`.                                 |

Evaluator-side errors (missing `--gt_dir`, GT path's last component
doesn't equal `--task`, CUDA OOM during LPIPS, etc.) also write
`INVALID`, but the `submission_errors` string is prefixed with the
Python exception type (e.g. `RuntimeError: CUDA out of memory`,
`FileNotFoundError: --gt_dir not a directory`) instead of one of the
five prefixes above. Use that distinction to decide whether to re-queue
(evaluator side, re-queue) or surface to the participant (`ERR_*`
prefix, do not re-queue).

## Missing-submission handling

A submission must be complete and well-formed: the pred ID set under each
`(modality, pair)` has to match the GT ID set exactly (no missing, no
extras), every file has to use the canonical filename, every volume has
to have the same shape as its GT counterpart, and no volume may contain
`NaN`. Anything else returns `INVALID` per the failure-path table above —
no worst-case fill, no partial scoring.

The single intentional concession is **task-level seg omission for
task1/task2**: if a submission contains *zero* `_seg.nii.gz` files,
`Dice` / `Volume` (per-subtask and global mean) are reported as `null`
and the submission still scores `SSIM` / `nRMSE` / `LPIPS` normally. Note
this is all-or-nothing: as soon as the submission contains *any* seg file,
seg is treated as required and every `(modality, pair)` must have a seg
file for every subject the GT lists. task3 submissions never carry seg,
so this concession doesn't apply.

Per-sample records still carry a `Status` field (`OK` on the happy path),
plus `Comments` annotating any task-level seg omission. Status values
other than `OK` would indicate an evaluator-side bug — pre-validation
should have rejected the submission before it reached this code path.

## CLI reference

```
score.py
  -i, --input       submission directory; last path component MUST be task{N}
                    (no archive extraction — unzip ahead of time)
  -t, --task        task1 | task2 | task3 (cross-validates -i and -g)
  -g, --gt_dir      per-task GT directory; last path component MUST be task{N}
  -o, --output      output directory (default ./)
  --device          torch device for LPIPS (default cuda; falls back to cpu)
```

## Dependencies

Listed in `requirements.txt`. The `mf` conda env covers everything except
`rarfile` (only needed for `.rar` submissions). LPIPS pulls AlexNet weights
on first run; cache them in the docker image to avoid runtime downloads.

Note: `segment.py` (SynthSeg) additionally needs TensorFlow 2.15 and a
checkout of [SynthSeg](https://github.com/BBillot/SynthSeg) plus the
`synthseg_2.0.h5` weights. It looks for `SYNTHSEG_DIR` either in `$REPO_ROOT/.env`
(two levels up from this directory it falls back to the env var, so simply
setting `export SYNTHSEG_DIR=...` works fine inside the platform image).

## Companion scripts

- `evaluate.py` — same evaluator participants run locally to compare a `pred_dir`
  against a `target_dir`. Useful for offline sanity checks; not used by `score.py`.
- `segment.py` — SynthSeg wrapper. Used **only** to pre-compute GT segmentations
  before scoring task1 / task2. The scorer never invokes it at evaluation time.
