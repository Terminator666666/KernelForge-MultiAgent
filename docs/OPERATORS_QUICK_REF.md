# 算子快速参考

当前主线只保留 3 类高收益算子。

---

## P0

### 1. DSA Sparse Attention
```bash
./scripts/start-campaign.sh dsa_sparse_attention 10
```
- 目标：DeepSeek 稀疏注意力主算子
- 特征：稀疏 gather、softmax、paged KV、shape dispatch
- 备注：当前最值得投入

### 2. GDN Prefill
```bash
./scripts/start-campaign.sh gdn_prefill 10
```
- 目标：Gated Delta Net 预填充
- 特征：变长序列、状态扫描、prefill 特化

### 3. DSA Top-k Indexer
```bash
./scripts/start-campaign.sh dsa_topk_indexer 10
```
- 目标：稀疏 top-k 索引器
- 特征：索引布局、访存、图捕获、低延迟调度

---

## 非主线

以下算子默认不进入主动闭环排期：

- `rmsnorm`
- `gemm`
- `rope`
- `sampling`
- `gqa_ragged`
- `gdn_decode`
- `paged_attention`

其中 `rmsnorm` 保留为流程样板，其他默认仅作兼容或历史参考。

---

## 相关文档

- 详细说明：[`docs/SUPPORTED_OPERATORS.md`](SUPPORTED_OPERATORS.md)
- 闭环流程：[`docs/CLOSED_LOOP.md`](CLOSED_LOOP.md)
- 总览：[`README.md`](../README.md)
