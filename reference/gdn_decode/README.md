# gdn_decode 归档

当前主线 family：`gdn_decode`

## 默认目标

- `definition`: `gdn_decode_qk4_v8_d128_k_last`
- `baseline_solution`: `flashinfer_wrapper_9b7f1e`
- `comparison_denominator`: `official_baseline`
- `baseline_dataset_group`: `gdn`

## 官方 baseline 来源

- 数据集目录：
  `/mnt/d/Agent/flashinfer-trace/solutions/baseline/gdn/gdn_decode_qk4_v8_d128_k_last/flashinfer_wrapper_9b7f1e.json`
- 当前数据集 baseline 形态：`python wrapper`
- `spec.entry_point`: `main.py::run`

说明：该 baseline 是官方数据集基线方案。它记录的是官方 baseline 的可执行入口，不代表你已经拿到了最终底层 CUDA kernel；若要追到底层实现，仍需继续沿 FlashInfer 安装包或依赖库定位。

## 当前状态

- 仅完成 baseline 归档初始化
- 尚未建立 accepted anchor
- 尚未记录任何 round

## 归档文件

- `baseline.json`
- `TRAPS.md`
- `solutions.jsonl`
- `variants/`
