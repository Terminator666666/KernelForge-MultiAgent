# gdn_prefill Traps

## 当前状态

尚未进入真实 round，暂无已验证陷阱。

## 预置检查项

1. 变长 workload 必须用真实 FlashInfer-Bench 数据集验证
2. 不允许跳过 baseline NCU
3. KernelWiki 里只适用于 SM100/B200 的技巧必须明确标注不直接照搬到 sm_120
4. official baseline 只能作为语义锚点和分母，不能把它当作我方候选成绩
