# dsa_topk_indexer 归档

当前主线 family：`dsa_topk_indexer`

## 默认目标

- `definition`: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
- `baseline_solution`: `flashinfer_deepgemm_wrapper_2ba145`
- `comparison_denominator`: `official_baseline`
- `baseline_dataset_group`: `dsa`

## 官方 baseline 来源

- 数据集目录：
  `/mnt/d/Agent/flashinfer-trace/solutions/baseline/dsa/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/flashinfer_deepgemm_wrapper_2ba145.json`
- 当前数据集 baseline 形态：`python wrapper`
- `spec.entry_point`: `main.py::run`

说明：该 baseline 属于官方 baseline 数据集项，命名中带 `deepgemm_wrapper`，但在你当前闭环里它仍然只作为官方分母与语义锚点。

## 当前状态

- 仅完成 baseline 归档初始化
- 尚未建立 accepted anchor
- 尚未记录任何 round

## 归档文件

- `baseline.json`
- `TRAPS.md`
- `solutions.jsonl`
- `variants/`
