# gdn_prefill Traps

## 当前状态

已进入真实 round，当前已验证陷阱如下：

1. `gdn_prefill` 的 Python 语义 baseline / candidate 在 high workload 上并不是被单一大 matmul 主导，真实 NCU 会出现大量 `elementwise_kernel` / `vectorized_elementwise_kernel` / 通用 `kernel`，不能误判成“直接做 chunk kernel 就一定高收益”。
2. 单次调用内的公式等价收敛有收益，但更稳定的增益来自跨调用缓存真实 workload 在 warmup/iters 中复用的预处理结果；因此缓存键必须按 tensor 对象身份与版本构造，不能按数值内容做不安全缓存。
3. 小 workload 的收益不一定和中高 workload 同步增长；`v2` 在 low workload 只有 `1.37x`，说明不能只看 high workload 做结论，必须坚持 low/mid/high 三点一起看。

## 预置检查项

1. 变长 workload 必须用真实 FlashInfer-Bench 数据集验证
2. 不允许跳过 baseline NCU
3. KernelWiki 里只适用于 SM100/B200 的技巧必须明确标注不直接照搬到 sm_120
4. official baseline 只能作为语义锚点和分母，不能把它当作我方候选成绩
5. 当前快迭代阶段只允许按 low / mid / high 3 个真实 workload 做决策，不能重新扩成全量 workload
6. 当前 `gdn_prefill` 的真实收益主要来自减少重复 cast / head 扩展 / state 转置 / gate 预处理；如果后续版本牺牲这些缓存稳定性，容易回退到 `v1` 甚至 baseline 水平
7. 基于 low/mid/high 三点结果做双路径分段派发并不一定赚钱。`round1-v3` 虽然想保住 low workload，但真实结果是平均 sol/base 从 `1.843x` 掉到 `1.548x`，说明额外 dispatch 与双路径维护开销本身就是风险。
