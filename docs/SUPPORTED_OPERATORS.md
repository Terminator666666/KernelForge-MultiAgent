# 当前聚焦的算子类型

KernelForge-MultiAgent 当前的主动闭环优化**只聚焦 6 类高收益算子**。这里的“聚焦”指：

- 会作为主线 campaign 持续迭代
- 会要求官方 baseline 派生
- 会要求真实 NCU + KernelWiki 证据闭环
- 会优先消耗工程时间与 GPU profile 预算

其余算子默认不进入主线排期，除非用户明确要求重开。

---

## 6 类主线算子

| 主线类型 | 代表 definition / 家族 | 说明 |
|---------|----------------------|------|
| `dsa_sparse_attention` | `dsa_sparse_attention_*` | DeepSeek 稀疏注意力主算子，通常收益空间最大 |
| `gdn_prefill` | `gdn_prefill_*` | Gated Delta Net 预填充，变长扫描与状态更新优化空间大 |
| `gdn_decode` | `gdn_decode_*` | Gated Delta Net 解码，小 kernel，对调度和寄存器预算非常敏感 |
| `dsa_topk_indexer` | `dsa_topk_indexer_*` | DeepSeek 稀疏索引器，访存/索引布局/图捕获是核心 |
| `paged_attention` | `gqa_paged_decode_*`、`mla_paged_decode_*`、`mla_paged_prefill_*` | 分页注意力家族，统一视作一个主线优化方向 |
| `moe_fp8` | `moe_fp8_*` | FP8 block-scale MoE，重点在 routing / fused GEMM / quant 路径 |

---

## 主线优先级

### P0

1. `dsa_sparse_attention`
2. `gdn_prefill`
3. `gdn_decode`

### P1

4. `dsa_topk_indexer`
5. `paged_attention`
6. `moe_fp8`

---

## 非主线说明

下面这些内容可以存在于仓库中，保留兼容目录、旧样板或历史实验，但**默认不再作为主动闭环目标**：

- `rmsnorm`
- `gemm`
- `rope`
- `sampling`
- `gqa_ragged`

其中：

- `rmsnorm` 保留为官方 baseline 派生、真实 NCU 驱动的流程样板；
- 其余算子若未来重开，仍必须遵守当前仓库的统一闭环规则。

---

## 选择依据

之所以收敛到这 6 类，是因为对照外部项目与本地研究资料后，收益分布很清楚：

- `mlsys2026-flashinfer-contest-main` 的竞赛主轴主要就是 DSA / GDN / MoE 这类高收益算子；
- `AKO4X-main` 的真实归档结果显示，显著加速比也主要集中在 DSA、GDN、paged attention；
- `rmsnorm` 这类算子在官方库里已经较深度优化，继续深挖通常只有中低收益。

因此，本仓库当前只保留“高收益主战场”。
