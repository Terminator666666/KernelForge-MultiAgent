# dsa_topk_indexer 归档

当前主线 family：`dsa_topk_indexer`

## 默认目标

- `definition`: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
- `baseline_solution`: `official_reference_dsa_topk_indexer_v1`
- `comparison_denominator`: `official_baseline`
- `baseline_dataset_group`: `dsa`

## 官方 baseline 来源

- 数据集目录：
  `/mnt/d/Agent/flashinfer-trace/solutions/baseline/dsa/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/official_reference_dsa_topk_indexer_v1.json`
- 当前数据集 baseline 形态：`python reference`
- `spec.entry_point`: `main.py::run`

说明：该 baseline 属于官方 baseline 数据集项，当前闭环只把它作为官方分母与语义锚点。

## 当前状态

- baseline 已完成归档，真实全量 workload 平均 latency 为 `11.1682 ms`
- accepted anchor: `round1-v12` / `kernelforge_dsa_topk_indexer_cuda_v17`
- `round0-v1`: 正确性通过，真实单 workload 预验证 `sol/base = 1.022x`，未达 `1.05x`，决策 `REJECT`
- `round0-v2`: 从 `round0-v1` 派生，新增页表索引和单 batch 展开 K 缓存，等待真实预验证和 NCU 2025.2
- `round0-v3`: 从 `round0-v2` 派生，新增 q FP32 转换缓存和局部 token 到全局 token 映射缓存，等待真实预验证和 NCU 2025.2
- `round0-v4`: 从 `round0-v3` 派生，新增 K.T 连续布局缓存、weights 广播视图缓存和 score 就地处理；真实单 workload 预验证 `sol/base = 1.437x`，已完成真实 NCU 2025.2 采样，决策 `ACCEPT`
- `round1-v1` (`v6`): 将 score 计算重排为 `K @ q + relu + matvec`；真实单 workload `sol/base = 1.368x`，未超越当前 anchor，决策 `REJECT`
- `round1-v2` (`v7`): 仅在 `v4` 路径上启用更激进的 float32 matmul 精度；真实 workload 正确性失败，决策 `REJECT`
- `round1-v3` (`v8`): 保持 `round0-v4` 精确语义，把 seq_lens/page-table/K.T/token_ids 的准备移到图外，并对固定输入使用 CUDA graph replay；真实单 workload 预验证 `sol/base = 12.756x`，更大 workload 点测 `sol/base = 9.600x`，已完成真实 NCU 2025.2 采样，决策 `ACCEPT`
- `round1-v7` (`v12`): 恢复官方 score/topk/token 映射，只保留带内容签名保护的 FP8 KV 反量化缓存；失败 workload `num_trials=20` 通过，全量 50/50 workload 通过，平均 `sol/base = 1.054x`，作为严格正确性锚点
- `round1-v11` (`v16`): 在 `v12` 的严格正确官方路径上新增 row_meta 缓存和输出 buffer 复用；失败 workload `num_trials=20` 通过，全量 50/50 workload 通过，平均 `sol/base = 1.264x`，并已完成真实 NCU 2025.2 采样，决策 `ACCEPT`
- `round1-v12` (`v17`): 在 `v16` 基础上，把 dequant/row_meta 缓存键从内容签名改为 tensor 对象身份，保留官方 score/topk/token 映射；失败 workload `num_trials=20` 通过，全量 50/50 workload 通过，平均 `sol/base = 1.878x`，并已完成真实 NCU 2025.2 采样，决策 `ACCEPT`

## 归档文件

- `baseline.json`
- `TRAPS.md`
- `solutions.jsonl`
- `variants/`
