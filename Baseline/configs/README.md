# Baseline configurations

Public configuration templates for the retained baseline methods. Replace
`/path/to/` cache and pretrained-weight placeholders before running. Dataset
and output defaults also use the root `.env` configuration.

| Directory | YAML files |
|---|---:|
| `pvwmi/` | 4 |
| `task1/cut/` | 17 |
| `task1/cut3d/` | 12 |
| `task1/cut_baseline/` | 12 |
| `task1/cut_baseline_nonzero/` | 12 |
| `task1/cut_multifield/` | 3 |
| `task1/cut_seg/` | 12 |
| `task1/cut_seg_best/` | 12 |
| `task1/cyclegan/` | 12 |
| `task1/ittr/` | 4 |
| `task1/nafnet/` | 12 |
| `task1/restormer/` | 15 |
| `task1/unsb/` | 1 |
| `task2/cut/` | 12 |
| `task2/cyclegan/` | 12 |
| `task3/starganv2/` | 1 |

Task 1 translates source fields to 7T; Task 2 translates 0.1T to higher fields;
Task 3 uses one shared Any-to-Any model. `pvwmi/` contains multi-site templates.

`method` selects the trainer, `model` defines the architecture, `data` defines
data loading and transforms, and `pretrain` / `finetune` specify training phases.
Choose the phase with `train.py --mode`. See [training](../README.md).
