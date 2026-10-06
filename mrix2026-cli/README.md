# mrix2026 提交工具

提供 `segment`、`build`、`zip`、`check` 等命令。使用已有环境，在项目根目录执行，
无需为了查看或运行入口重新安装依赖：

```bash
export PYTHONPATH="$PWD/mrix2026-cli/src${PYTHONPATH:+:$PYTHONPATH}"
python -m mrix2026_cli --help
python -m mrix2026_cli build --help
```

## 验证集打包

用实际任务的目录和 method/mode/epoch 替换示例：

```bash
python -m mrix2026_cli segment \
  --input-dir /absolute/predictions --output-dir /absolute/segmentations --dry-run

python -m mrix2026_cli build \
  --predictions-dir /absolute/predictions \
  --predictions-seg-dir /absolute/segmentations \
  --output-dir /absolute/submission --tasks task1 \
  --task1-method cut --task1-mode retro_scratch --task1-epoch epoch100 --dry-run

python -m mrix2026_cli zip \
  --submission-dir /absolute/submission --output-dir /absolute/zips --tasks task1
```

`--dry-run` 只核对路径，不生成分割或打包结果；确认路径和选择后再去掉该参数。
分割可执行文件按以下顺序解析：`SYNTHSEG_BIN` 显式路径、
`$FREESURFER_HOME/bin/mri_synthseg`、PATH 中的 `mri_synthseg`。例如：

```bash
export SYNTHSEG_BIN=/path/to/freesurfer/bin/mri_synthseg
```

分割保留 `--keepgeom`；输出 shape/affine 必须与预测一致。

验证集 builder 使用 [constants.py](src/mrix2026_cli/constants.py) 中的
`Z_CLIP_RANGE=(150,180)`。Task 1/2 的 pred 和 seg 应处于相同网格；Task 3
不需要 segmentation。ZIP 内从模态目录开始，不含 `task1/` 等外层目录。


旧 [standalone builder](../Submission/build_submission/build_submission.py)
保留兼容入口，共用 CLI 的切片常量。
