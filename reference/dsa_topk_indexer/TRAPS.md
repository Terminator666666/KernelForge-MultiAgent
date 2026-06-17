# dsa_topk_indexer Traps

## 当前状态

尚未进入真实 round，暂无已验证陷阱。

## 预置检查项

1. 索引类 kernel 很容易出现 correctness 通过但吞吐不稳定的情况
2. baseline 若无法稳定复现，必须明确记录，不允许编造分母
3. 每轮必须保留 KernelWiki 依据与 NCU 决策链
4. 禁止以历史 candidate 替代 official baseline / expert baseline
