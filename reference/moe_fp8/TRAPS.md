# moe_fp8 Traps

## 当前状态

尚未进入真实 round，暂无已验证陷阱。

## 预置检查项

1. MoE 的分母必须固定到同一个 official baseline
2. routing / block-scale / output accumulation 的 correctness 必须先过，再谈速度
3. KernelWiki 依据里若引用 SM100/B200 特性，必须注明对 sm_120 的迁移边界
4. baseline 只跑一次的前提是 definition、batch_size、device、baseline_solution、NCU 版本完全一致
