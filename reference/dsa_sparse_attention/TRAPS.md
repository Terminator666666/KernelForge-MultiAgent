# dsa_sparse_attention Traps

## 当前状态

已做过单 workload 探路，但尚未形成全量真实 workload anchor。

## 预置检查项

1. 不允许把 `sol/ref` 当最终成绩
2. 每轮必须同时补齐 candidate 与 baseline 的真实 NCU
3. 每轮必须记录实际参考的 KernelWiki 页面
4. 只允许使用 `/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu`
5. `dsa_sparse_attention` 的单 workload 超高加速比不能直接当成结论；当前快迭代阶段至少要覆盖固定的低/中/高 3 个代表 workload，再决定 ACCEPT/REJECT
6. 当前机器上这条 definition 的数据集声明 `120` 个 workload，但只有 `50` 个 safetensors 可运行；coverage 必须按 `50/120` 明确记录
7. `kernelforge_dsa_sparse_attention_cuda_v2` 在小 `num_tokens` 段收益极高，但从 `num_tokens=124/131/150` 开始已出现慢于 baseline 的样本，后续不能只拿 decode 极小 token workload 决策
8. `kernelforge_dsa_sparse_attention_cuda_v2` 在更大 `num_tokens` 段会 OOM（如 `893+` token），说明“合并 GEMM + 缓存稀疏 KV 子集”这条路线当前不具备全量稳定性
9. 当前快迭代代表集的中段 workload 是 `num_tokens=62`；如果候选在这个点上掉到 `sol/base < 1.0`，说明小 token 快路径阈值可能过高，必须先回收阈值，再谈更激进的融合
10. high 代表 workload 的成对真实 NCU 已显示 candidate 与 baseline 共用 `index_elementwise + gemmSN_TN / gemmSN_NN + softmax_warp_forward` 主链路；当高段已经接近官方路径时，不要继续大改算法骨架，优先做 memory-bound 路径的重复重排/搬运收敛
