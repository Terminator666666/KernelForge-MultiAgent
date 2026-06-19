# gdn_prefill 归档

当前主线 family：`gdn_prefill`

## 默认目标

- `definition`: `gdn_prefill_qk4_v8_d128_k_last`
- `baseline_solution`: `official_reference_gdn_prefill_v1`
- `comparison_denominator`: `official_baseline`
- `baseline_dataset_group`: `gdn`

## 官方 baseline 来源

- 数据集目录：
  `/mnt/d/Agent/flashinfer-trace/solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/official_reference_gdn_prefill_v1.json`
- 当前数据集 baseline 形态：`python reference`
- `spec.entry_point`: `main.py::run`

说明：该 baseline 是数据集里的官方 reference 方案，后续所有候选都必须从它的语义与接口派生。

## 当前状态

- 当前快迭代口径固定为 **3 个代表 workload**：low / mid / high
- 3 个真实 workload 固定为：
  - low: `77daf91d-0660-4c4b-8c32-336a69281cd9` (`total_seq_len=6`, `num_seqs=1`)
  - mid: `25d9c14d-90ad-442d-8b0f-9452ad198832` (`total_seq_len=132`, `num_seqs=2`)
  - high: `06f21bb1-6cbd-4d55-b620-fb4d62181a71` (`total_seq_len=8192`, `num_seqs=56`)
- 当前 accepted anchor：`round1-v2`
- 当前最佳候选：`kernelforge_gdn_prefill_cuda_v2`
- `round1-v2` 真实结果（3 个代表 workload，warmup=3，iters=10）：
  - low: `2.3316 ms` vs baseline `3.1932 ms`，`1.37x`
  - mid: `23.4187 ms` vs baseline `47.0738 ms`，`2.01x`
  - high: `1385.1130 ms` vs baseline `2978.9100 ms`，`2.15x`
  - 平均：solution `470.2878 ms`，baseline `1009.7257 ms`，平均 `sol/base = 1.843x`
- 当前 high-workload NCU 结论：
  - baseline 与 candidate 都以 `elementwise_kernel` / `vectorized_elementwise_kernel` / 通用 `kernel` 为主要热点
  - 说明当前 Python 语义实现仍是 memory-bound / pointwise-heavy，不是单一大 GEMM 主导
- 下一步：若继续优化 `gdn_prefill`，优先继续减少索引、gate、state 相关的重复预处理与点算子链路，再考虑更激进的 chunk 化重写

## 归档文件

- `baseline.json`
- `TRAPS.md`
- `solutions.jsonl`
- `variants/`
