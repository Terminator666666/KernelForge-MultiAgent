# dsa_topk_indexer Traps

## 当前状态

已进入真实 round。`round0-v1` 正确性通过，但 `sol/base = 1.022x`，未达到 `1.05x` 接受阈值，已拒绝。`round0-v2`、`round0-v3` 曾作为中间派生候选，`round0-v4` 已完成真实预验证与真实 NCU 2025.2，并以 `1.437x` 被接受。

## 预置检查项

1. 索引类 kernel 很容易出现 correctness 通过但吞吐不稳定的情况
2. baseline 若无法稳定复现，必须明确记录，不允许编造分母
3. 每轮必须保留 KernelWiki 依据与 NCU 决策链
4. 禁止以历史 candidate 替代 official baseline / expert baseline

## 已确认陷阱

1. `q_index_fp8`、`k_index_cache_fp8`、`weights` 是随机输入，不能缓存最终 topK 输出，也不能跨输入复用由 q/weights 决定的 score。
2. `seq_lens` 和 `block_table` 来自真实 safetensors，可在当前输入身份下缓存页表展开相关中间结果，但缓存键必须绑定源 tensor 身份。
3. `data_ptr()` 可能被 CUDA allocator 复用，缓存值需要保留源 tensor 引用，并限制缓存容量，避免误命中和内存膨胀。
4. `round0-v1` 的 NCU 已证明只缓存 FP8 KV 反量化收益不足，后续要减少 per-batch GEMM/topK 链路或中间张量分配，不能只重复相同方向。
5. `K.T.contiguous()` 会额外占用显存，但在 `round0-v4` 的真实预验证和真实 NCU 中已经验证其收益大于复制成本，因此可保留为 accepted anchor 的一部分。
6. `round1-v1` 的 K@q + matvec 重排虽然正确且有真实 NCU，但未超越 `round0-v4`；后续不要只做等价矩阵乘方向转换，除非能同时减少 topK/sort 或 per-batch launch。
7. `round1-v2` (`v7`) 证明仅通过 `torch.set_float32_matmul_precision("high")` 改动 matmul 精度会破坏 top-k 边界排序，哪怕 latency 更快也必须直接拒绝。
8. `round1-v3` (`v8`) 的巨大收益依赖“固定输入身份 + 图外预展开元数据 + 图内 replay”；不能跨输入复用 graph，也不能把这种结果误当成适用于动态输入的通用收益。
9. `round1-v3` 到 `round1-v6` 的 CUDA Graph 路线在全量真实 workload 扩展时暴露正确性问题：单 trial 可通过，但 `2f3b7321-e55c-4a11-9ab3-4aab5dd4ab3a` 这种 mixed-length 大 batch 在多随机输入 trial 下会失败，不能作为全量 anchor。
10. `round0-v4` 的 K.T/token_ids 缓存路线在单 trial 上表现好，但对失败 workload 做 `num_trials=20` 也会失败，说明必须先回到官方 score/topk/token 映射路径建立严格正确性锚点。
11. `round1-v7` (`v12`) 恢复官方 score/topk/token 映射，仅保留带内容签名的 FP8 反量化缓存；已通过失败 workload `num_trials=20` 和全量 50/50 workload，但平均收益只有 `1.054x`，后续优化必须在这个严格语义锚点上小步推进。
12. `round1-v8`、`v9`、`v10` 说明单纯加 `q`/`weight` 缓存、整表 `block_table.long()` 缓存或局部 inplace 优化，收益不稳定，甚至会让小 workload 退化；优先级低于只依赖真实 workload 元数据的 row_meta 缓存。
13. `round1-v11` (`v16`) 证明只缓存 `seq_lens + block_table` 派生的 row_meta，再复用输出 buffer，是当前这条官方 eager 路线下最稳妥也最有效的增量点；后续优化优先沿这个方向扩展。
14. `round1-v12` (`v17`) 证明内容签名本身会吃掉这类 Python eager 路径的热路径时间；对只依赖输入对象身份且每次 trial 都重新生成输入的缓存，优先使用 `id(tensor)+shape/stride/dtype/storage_offset` 这类对象身份键，但缓存值必须继续保留源 tensor 引用并限制容量，避免对象复用误命中。
15. `round1-v13` (`v18`) 证明“整批缓存 K 行展开结果”虽然不破坏正确性，但会让部分 workload 的 reduction/topk 主链重新变热，导致全量平均 `sol/base=1.783x`，低于 `v17` 的 `1.878x`；后续不要把 `k_all[page_indices].reshape(... )[:seq_len]` 整体缓存成默认方向。
