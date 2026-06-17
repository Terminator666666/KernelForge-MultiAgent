# gdn_prefill 归档

当前主线 family：`gdn_prefill`

## 默认目标

- `definition`: `gdn_prefill_qk4_v8_d128_k_last`
- `baseline_solution`: `flashinfer_wrapper_123ca6`
- `comparison_denominator`: `official_baseline`
- `baseline_dataset_group`: `gdn`

## 官方 baseline 来源

- 数据集目录：
  `/mnt/d/Agent/flashinfer-trace/solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/flashinfer_wrapper_123ca6.json`
- 当前数据集 baseline 形态：`python wrapper + Python package sources`
- `spec.entry_point`: `main.py::run`

说明：该 baseline 是数据集中的官方 baseline 方案。它除了 `main.py` 之外，还内嵌了 `gdn_blackwell/...` 等源码文件，因此比单 wrapper 更接近完整实现，但仍应视作“官方 baseline 方案包”，不是你仓库里的自研起点。

## 当前状态

- 仅完成 baseline 归档初始化
- 尚未建立 accepted anchor
- 尚未记录任何 round

## 归档文件

- `baseline.json`
- `TRAPS.md`
- `solutions.jsonl`
- `variants/`
