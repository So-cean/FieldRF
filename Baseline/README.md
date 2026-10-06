# MRIxFields2026 Baseline

CUT、CycleGAN、StarGANv2 及其他保留基线的训练与推理实现。
配置与模型入口见 [配置清单](configs/README.md)。

## 环境与路径

依赖与路径设置见 [setup](../docs/README.md)。
根目录 `requirements.txt` 列出当前 Python 依赖；PyTorch/torchvision 按硬件配置，
文件不固定其版本或 CUDA 版本。SynthSeg 使用单独配置的环境。
各入口脚本会定位本目录的 `mrixfields`，通常无需重新安装本包。

根目录 `.env.example` 是路径模板，`.env` 保留在本机。检查 `DATA_DIR`、
`OUTPUT_DIR`、`INFERENCE_DIR`、`PREPROCESSED_DIR`，同时检查 YAML 中的
cache 路径。Python 自动加载 `.env`；shell 中使用 `$DATA_DIR` 前需加载环境变量。

## 训练

从 `Baseline/` 执行：

```bash
python scripts/train.py --config configs/task1/cut/0.1T_to_7T_T1W.yaml --mode retro_scratch

python scripts/train.py --config configs/task1/cut/0.1T_to_7T_T1W.yaml \
  --mode pro_pretrained --pretrain_ckpt /absolute/path/generator_final.pth

# 原生 PyTorch DDP；卡数按硬件配置
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/task1/cut/0.1T_to_7T_T1W.yaml --mode retro_scratch --dist
```

`retro_scratch` 为非配对预训练，`pro_scratch` 为配对从零训练，
`pro_pretrained` 为配对微调。每卡 batch size 按 YAML 执行。
MONAI 路线直接加载体数据；仅需要切片缓存的路线使用 `scripts/preprocess.py`，
先查看其 `--help`。

## 推理

配置已嵌入 checkpoint，推理不传 `--config`：

```bash
python scripts/inference.py \
  --checkpoint /absolute/path/generator_final.pth \
  --input_dir /absolute/path/Validating_prospective/T1W/0.1T \
  --output_dir /absolute/path/predictions
```

多场强 checkpoint 另加 `--source_field 0.1T`。StarGANv2 另加
`--source_field 0.1T --target_field 7T --source_modality T1W --target_modality T1W`。
保留 checkpoint 旁的标准化统计。训练时增强不表示推理使用 TTA。

## 分割、提交与检查

统一使用 [mrix2026 CLI](../mrix2026-cli/README.md) 的 `segment/build/zip/check`；
参考评估说明见 [Evaluation](../Evaluation/README.md)。配置数据、缓存、
权重和输出路径后，按通用验证集格式打包预测结果。
