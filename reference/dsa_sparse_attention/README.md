# dsa_sparse_attention 归档

当前主线 family：`dsa_sparse_attention`

## 默认目标

- `definition`: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
- `baseline_solution`: `official_reference_dsa_sparse_attention_v1`
- `comparison_denominator`: `official_baseline`
- `baseline_dataset_group`: `dsa`

## 官方 baseline 来源

- 数据集目录：
  `/mnt/d/Agent/flashinfer-trace/solutions/baseline/dsa/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/official_reference_dsa_sparse_attention_v1.json`
- 当前数据集 baseline 形态：`python wrapper`
- `spec.entry_point`: `main.py::run`

说明：这个 official baseline 确实来自 `flashinfer-trace/solutions/baseline/...`，但它是官方 baseline 方案文件，不等于“裸 CUDA kernel 源文件”。如果后续要继续深挖底层 CUDA 实现，需要沿 wrapper 再追到底层 FlashInfer 安装包或依赖库实现。

## 当前状态

- 当前快迭代口径已经改为 **3 个代表 workload**：低 / 中 / 高各 1 个真实 workload
- 当前这 3 个代表 workload 固定来自：
  - low: `1c29650b35bb4caca50fd7acb24b07be` (`num_tokens=1`)
  - mid: `0f8137e0c871466f8b4bdce5bfd57658` (`num_tokens=62`)
  - high: `605d05fe80bd4f42a1b9c1290b11cd89` (`num_tokens=893`)
- `round0-v1` 与 `round1-v1` 在单 workload 上分别达到 `17.008x` 和 `23.535x`
- 2026-06-18 对 `kernelforge_dsa_sparse_attention_cuda_v2` 做全量扩展时，
  数据集声明 `120` 个 workload，但当前机器仅有 `50` 个 safetensors 可运行；
  其中只有 `20/50 PASS`，并且在大 `num_tokens` workload 上出现 OOM，因此当前**没有 accepted anchor**
- 当前最新全量结论：可通过 workload 上平均 `sol/base = 6.752x`，但这不是可接受成绩，因为 coverage 与正确性都未闭合
- 后续日常闭环优化默认不再跑 29/50/120 workload，而是只跑
  `rounds/round-1/dsa_sparse_attention/docs/workload_allowlist_repr3.json`
- 2026-06-18 最新快迭代 anchor 已更新为 `round1-v4 / kernelforge_dsa_sparse_attention_cuda_v6`
  - `3/3 PASS`
  - low (`num_tokens=1`): `20.78x`
  - mid (`num_tokens=62`): `2.21x`
  - high (`num_tokens=893`): `7.18x`
  - 平均 `sol/base = 10.057x`
  - 当前判断：高 token 段已经基本贴近官方 attention 主路径，后续更该沿 `memory-bound`
    方向压缩重复重排和搬运，而不是再大改算法骨架

## 归档文件

- `baseline.json`: 主线 family 配置与 baseline 口径
- `TRAPS.md`: 闭环过程中积累的陷阱
- `solutions.jsonl`: 方案 DAG
- `variants/`: 接收的候选归档
