# Baseline configurations

Standard MRIxFields2026 configuration templates for the retained baseline
methods. Replace `/path/to/` cache and weight placeholders and configure the
root `.env` paths before running.

| Directory | YAML files |
|---|---:|
| `task1/cut/` | 12 |
| `task1/cut3d/` | 12 |
| `task1/cyclegan/` | 12 |
| `task1/ittr/` | 4 |
| `task1/nafnet/` | 12 |
| `task1/restormer/` | 12 |
| `task1/unsb/` | 1 |
| `task2/cut/` | 12 |
| `task2/cyclegan/` | 12 |
| `task3/starganv2/` | 1 |

Task 1 translates each lower source field to 7T. Task 2 translates 0.1T to
higher fields. Task 3 uses one shared StarGAN v2 model across fields/modalities.
Task 1/2 templates select T1W, T2W or T2FLAIR individually. ITTR provides T1W
examples and UNSB provides a Task 1 T1W pretraining example.

`method` selects the trainer, `model` the architecture, `data` the loading
pipeline, and `pretrain` / `finetune` the training phases. Choose the phase with
`train.py --mode`. UNSB currently supports pretraining only; other retained
methods support the three modes described in [training](../README.md).
