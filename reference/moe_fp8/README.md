# moe_fp8 归档

当前主线 family：`moe_fp8`

## 默认目标

- `definition`: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- `baseline_solution`: `flashinfer_wrapper_9sdjf3`
- `comparison_denominator`: `official_baseline`
- `baseline_dataset_group`: `moe`

## 官方 baseline 来源

- 数据集目录：
  `/mnt/d/Agent/flashinfer-trace/solutions/baseline/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/flashinfer_wrapper_9sdjf3.json`
- 当前数据集 baseline 形态：`python wrapper`
- `spec.entry_point`: `main.py::run`

说明：该 baseline 是官方 baseline 数据集项，可直接作为当前闭环的官方分母。它记录的是官方方案入口，而不是你仓库里的候选实现。

## 当前状态

- 仅完成 baseline 归档初始化
- 尚未建立 accepted anchor
- 尚未记录任何 round

## 归档文件

- `baseline.json`
- `TRAPS.md`
- `solutions.jsonl`
- `variants/`
