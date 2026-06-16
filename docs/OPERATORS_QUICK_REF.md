# 支持的算子类型快速参考

本文档快速列出 KernelForge-MultiAgent 支持的 10 种算子类型。

---

## 🔥 高优先级（5 个）- 推荐优先优化

### 1. GEMM (General Matrix Multiplication)
```bash
./scripts/start-campaign.sh gemm 10
```
- **操作**: C = A × B^T
- **应用**: 所有深度学习模型
- **收益**: ⭐⭐⭐⭐⭐ 占推理时间 60-80%
- **难度**: 高

### 2. RMSNorm (RMS Normalization)
```bash
./scripts/start-campaign.sh rmsnorm 10
```
- **操作**: RMS Layer Normalization  
- **应用**: Llama, Mistral, DeepSeek, Qwen
- **收益**: ⭐⭐⭐⭐⭐ 所有现代 LLM 标配
- **难度**: 低
- **示例**: ✅ 已有完整示例（推荐新手从这里开始）

### 3. GQA-Paged (Grouped Query Attention - Paged)
```bash
./scripts/start-campaign.sh gqa_paged 10
```
- **操作**: 分组查询注意力（Prefill + Decode）
- **应用**: Llama 2/3, Qwen, Mistral 系列
- **收益**: ⭐⭐⭐⭐⭐ 现代 LLM 的核心 Attention
- **难度**: 高

### 4. RoPE (Rotary Position Embedding)
```bash
./scripts/start-campaign.sh rope 10
```
- **操作**: 旋转位置编码
- **应用**: 几乎所有现代 LLM
- **收益**: ⭐⭐⭐⭐⭐ 位置编码标配
- **难度**: 中

### 5. Sampling (Token Sampling)
```bash
./scripts/start-campaign.sh sampling 10
```
- **操作**: Top-k, Top-p, Top-k+Top-p
- **应用**: 所有生成式模型
- **收益**: ⭐⭐⭐⭐⭐ 每个生成 token 都需要
- **难度**: 中

---

## ⚙️ 中优先级（3 个）- 特定架构

### 6. MoE (Mixture of Experts)
```bash
./scripts/start-campaign.sh moe 10
```
- **操作**: 专家混合路由和计算
- **应用**: Mixtral 8x7B/8x22B, Qwen 1.5/2.5 MoE, DeepSeek V2/V3
- **收益**: ⭐⭐⭐⭐ MoE 模型占推理 40-60%
- **难度**: 极高

### 7. MLA-Paged (Multi-head Latent Attention - Paged)
```bash
./scripts/start-campaign.sh mla_paged 10
```
- **操作**: 多头潜在注意力（Prefill + Decode）
- **应用**: DeepSeek V3 专用
- **收益**: ⭐⭐⭐ DeepSeek V3 的核心创新
- **难度**: 极高

### 8. GQA-Ragged (Grouped Query Attention - Ragged)
```bash
./scripts/start-campaign.sh gqa_ragged 10
```
- **操作**: 变长序列的 GQA
- **应用**: 批处理不同长度序列
- **收益**: ⭐⭐⭐ 减少 padding 开销
- **难度**: 高

---

## 🔬 低优先级（2 个）- 最新/实验性

### 9. DSA-Paged (DeepSeek Sparse Attention - Paged)
```bash
./scripts/start-campaign.sh dsa_paged 10
```
- **操作**: DeepSeek 稀疏注意力（Indexer + Sparse Attention）
- **应用**: DeepSeek V3.2 (2025)
- **收益**: ⭐⭐ 长上下文优化
- **难度**: 高

### 10. GDN (Gated Delta Net)
```bash
./scripts/start-campaign.sh gdn 10
```
- **操作**: 门控增量网络（Prefill + Decode）
- **应用**: Qwen3 Next (实验性)
- **收益**: ⭐⭐ 替代 Attention 的新架构
- **难度**: 高

---

## 🎯 推荐优化路线

### 路线 1: 通用 LLM（推荐新手）
```bash
1. rmsnorm       # ✅ 已有示例，容易上手
2. gemm          # 最高收益
3. rope          # 必需
4. gqa_paged     # 现代 Attention
5. sampling      # 生成必需
```
覆盖模型: Llama, Qwen, Mistral, Yi, Phi

### 路线 2: MoE 专项
```bash
1. rmsnorm
2. gemm
3. moe           # MoE 核心
4. rope
5. sampling
```
覆盖模型: Mixtral, Qwen MoE, DeepSeek V2/V3

### 路线 3: DeepSeek 全栈
```bash
1. rmsnorm
2. gemm
3. moe
4. mla_paged     # DeepSeek V3 核心
5. dsa_paged     # DeepSeek V3.2 新特性
```
覆盖模型: DeepSeek V2/V3/V3.2

---

## 📈 性能收益估算

| 算子 | 推理时间占比 | 优化潜力 | 影响面 |
|-----|------------|---------|--------|
| GEMM | 60-80% | 极高 ⭐⭐⭐⭐⭐ | 所有模型 |
| GQA-Paged | 10-20% | 极高 ⭐⭐⭐⭐⭐ | 现代 LLM |
| MoE | 40-60% (MoE 模型) | 极高 ⭐⭐⭐⭐⭐ | MoE 模型 |
| RMSNorm | 3-5% | 高 ⭐⭐⭐⭐ | 现代 LLM |
| RoPE | 2-3% | 中 ⭐⭐⭐ | 所有模型 |
| Sampling | 1-2% | 中 ⭐⭐⭐ | 生成模型 |

---

## 📚 更多信息

- **详细规范**: [`docs/SUPPORTED_OPERATORS.md`](SUPPORTED_OPERATORS.md)
- **闭环流程**: [`docs/CLOSED_LOOP.md`](CLOSED_LOOP.md)
- **快速开始**: [`README.md`](../README.md)

---

**推荐**: 从 **RMSNorm**（有示例）或 **GEMM**（最高收益）开始你的第一个 Campaign！
