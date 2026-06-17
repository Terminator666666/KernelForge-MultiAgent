# dsa_sparse_attention 归档

当前主线 family：`dsa_sparse_attention`

## 默认目标

- `definition`: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
- `baseline_solution`: `flashinfer_wrapper_5af199`
- `comparison_denominator`: `official_baseline`
- `baseline_dataset_group`: `dsa`

## 官方 baseline 来源

- 数据集目录：
  `/mnt/d/Agent/flashinfer-trace/solutions/baseline/dsa/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/flashinfer_wrapper_5af199.json`
- 当前数据集 baseline 形态：`python wrapper`
- `spec.entry_point`: `main.py::run`

说明：这个 official baseline 确实来自 `flashinfer-trace/solutions/baseline/...`，但它是官方 baseline 方案文件，不等于“裸 CUDA kernel 源文件”。如果后续要继续深挖底层 CUDA 实现，需要沿 wrapper 再追到底层 FlashInfer 安装包或依赖库实现。

## 当前状态

- 仅完成 baseline 归档初始化
- 尚未建立 accepted anchor
- 尚未记录任何 round

## 归档文件

- `baseline.json`: 主线 family 配置与 baseline 口径
- `TRAPS.md`: 闭环过程中积累的陷阱
- `solutions.jsonl`: 方案 DAG
- `variants/`: 接收的候选归档
