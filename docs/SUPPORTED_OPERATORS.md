# 支持的算子类型

KernelForge-MultiAgent 现在支持 FlashInfer-Bench 标准的 **10 种算子类型**，与 FlashInfer-Bench 验收标准完全对齐。

---

## 算子类型列表

| 算子类型 | 英文名称 | 目录名 | FlashInfer-Bench 对应 |
|---------|---------|--------|---------------------|
| 1. 矩阵乘法 | GEMM | `gemm` | ✅ gemm |
| 2. 分组查询注意力（分页） | GQA Paged | `gqa_paged` | ✅ gqa-paged |
| 3. 分组查询注意力（变长） | GQA Ragged | `gqa_ragged` | ✅ gqa-ragged |
| 4. 多头潜在注意力（分页） | MLA Paged | `mla_paged` | ✅ mla-paged |
| 5. DeepSeek 稀疏注意力 | DSA Paged | `dsa_paged` | ✅ dsa-paged |
| 6. 专家混合 | MoE | `moe` | ✅ moe |
| 7. RMS 归一化 | RMSNorm | `rmsnorm` | ✅ rmsnorm |
| 8. 旋转位置编码 | RoPE | `rope` | ✅ rope |
| 9. Token 采样 | Sampling | `sampling` | ✅ sampling |
| 10. 门控增量网络 | GDN | `gdn` | ✅ gdn |

---

## 1. GEMM (General Matrix Multiplication)

### 基本信息
- **目录**: `kernels/operators/gemm/`, `prompts/gemm/`
- **用途**: 通用矩阵乘法，计算 C = A × B^T
- **应用场景**: 神经网络层计算、注意力机制、矩阵变换

### 变体
- **FP16 GEMM**: 16位浮点输入
- **FP8 GEMM**: 8位浮点输入，带缩放因子保持数值稳定性

### 维度（3个）
- `M`: 变量
- `N`, `K`: 常量

### 输入输出
**输入**:
- `A`: [M, K]
- `B`: [N, K]
- FP8 变体额外输入：
  - `A_scale`: [M]
  - `B_scale`: [N]

**输出**:
- `C`: [M, N]

---

## 2. GQA-Paged (Grouped Query Attention - Paged)

### 基本信息
- **目录**: `kernels/operators/gqa_paged/`, `prompts/gqa_paged/`
- **用途**: 分组查询注意力，分页内存布局
- **特点**: 多个 query heads 共享 key-value heads，高效 KV 缓存管理

### 变体
- **Prefill**: 预填充阶段，处理多个 token
- **Decode**: 解码阶段，单 token 生成

### Prefill 维度（8个）
- `total_q`, `num_pages`, `len_indptr`, `num_kv_indices`: 变量
- `num_qo_heads`, `num_kv_heads`, `head_dim`, `page_size`: 常量

### Prefill 输入输出
**输入**:
- `q`: query tensor [total_q, num_qo_heads, head_dim]
- `k_cache`, `v_cache`: 分页 KV 缓存 [num_pages, page_size, num_kv_heads, head_dim]
- `qo_indptr`, `kv_indptr`, `kv_indices`: 分页索引
- `sm_scale`: softmax 缩放（标量）

**输出**:
- `output`: 注意力输出 [total_q, num_qo_heads, head_dim]
- `lse`: log-sum-exp 值 [total_q, num_qo_heads]

---

## 3. GQA-Ragged (Grouped Query Attention - Ragged)

### 基本信息
- **目录**: `kernels/operators/gqa_ragged/`, `prompts/gqa_ragged/`
- **用途**: 变长序列的分组查询注意力
- **特点**: 使用 ragged tensor 布局，无需填充，提高内存效率

### 变体
- **Prefill**: 处理变长序列批次

### 维度（6个）
- `total_q`, `total_kv`, `len_indptr`: 变量
- `num_qo_heads`, `num_kv_heads`, `head_dim`: 常量

### 输入输出
**输入**:
- `q`: query tensor [total_q, num_qo_heads, head_dim]
- `k`, `v`: key-value tensors [total_kv, num_kv_heads, head_dim]
- `qo_indptr`, `kv_indptr`: 序列偏移量
- `sm_scale`: softmax 缩放（标量）

**输出**:
- `output`: 注意力输出 [total_q, num_qo_heads, head_dim]
- `lse`: log-sum-exp 值 [total_q, num_qo_heads]

---

## 4. MLA-Paged (Multi-head Latent Attention - Paged)

### 基本信息
- **目录**: `kernels/operators/mla_paged/`, `prompts/mla_paged/`
- **用途**: 多头潜在注意力，分页内存布局
- **特点**: 将 KV 表示分解为 CKV（压缩 KV）和 KPE（键位置编码），减少内存使用

### 变体
- **Prefill**: 预填充阶段
- **Decode**: 解码阶段

### 维度（8个）
- `total_q`, `num_pages`, `len_indptr`, `num_kv_indices`: 变量
- `num_qo_heads`, `head_dim_ckv`, `head_dim_kpe`, `page_size`: 常量

### 输入输出
**输入**:
- `q_nope`: 无位置编码的 query [total_q, num_qo_heads, head_dim_ckv]
- `q_pe`: query 位置编码 [total_q, num_qo_heads, head_dim_kpe]
- `ckv_cache`: 压缩 KV 缓存 [num_pages, page_size, head_dim_ckv]
- `kpe_cache`: 键位置编码缓存 [num_pages, page_size, head_dim_kpe]
- `qo_indptr`, `kv_indptr`, `kv_indices`: 分页索引
- `sm_scale`: softmax 缩放（标量）

**输出**:
- `output`: 注意力输出 [total_q, num_qo_heads, head_dim_ckv]
- `lse`: log-sum-exp 值 [total_q, num_qo_heads]

### 应用模型
- DeepSeek V3/R1
- Kimi K2

---

## 5. DSA-Paged (DeepSeek Sparse Attention - Paged)

### 基本信息
- **目录**: `kernels/operators/dsa_paged/`, `prompts/dsa_paged/`
- **用途**: 两阶段稀疏注意力机制
- **特点**: 计算复杂度从 O(n) 降至 O(k)，使用 FP8 量化

### 变体
- **Indexer**: 索引器，选择 top-K 相关 KV 缓存条目
- **Sparse Attention**: 稀疏注意力，仅在选中条目上执行 MLA 风格注意力

### Indexer 维度（9个）
- `batch_size`, `max_num_pages`, `num_pages`: 变量
- `num_index_heads`, `index_head_dim`, `page_size`, `topk`, `kv_cache_num_heads`, `head_dim_with_scale`: 常量

### Indexer 输入输出
**输入**:
- `q_index_fp8`: FP8 query [batch_size, num_index_heads, index_head_dim]
- `k_index_cache_fp8`: FP8 key 索引缓存 [num_pages, page_size, kv_cache_num_heads, head_dim_with_scale]
- `weights`: 学习的 head 权重 [batch_size, num_index_heads]
- `seq_lens`: 序列长度 [batch_size]
- `block_table`: 分页映射 [batch_size, max_num_pages]

**输出**:
- `topk_indices`: 选中的 token 索引 [batch_size, topk]（-1 表示填充）

### 应用模型
- DeepSeek V3.2

---

## 6. MoE (Mixture of Experts)

### 基本信息
- **目录**: `kernels/operators/moe/`, `prompts/moe/`
- **用途**: 专家混合层，将计算分配给多个专家子网络
- **特点**: 稀疏激活，gating 网络选择少数专家处理每个 token

### 维度（9个）
- `seq_len`: 变量
- `num_experts`, `num_local_experts`, `hidden_size`, `intermediate_size`, `gemm1_out_size`, `num_hidden_blocks`, `num_intermediate_blocks`, `num_gemm1_out_blocks`: 常量

### 输入输出
**输入**:
- `routing_logits`: 路由 logits [seq_len, num_experts]
- `routing_bias`: 路由偏置 [num_experts]
- `hidden_states`: 输入隐藏状态（FP8 量化）[seq_len, hidden_size]
- `hidden_states_scale`: 块级缩放因子 [num_hidden_blocks, seq_len]
- `gemm1_weights`: 第一个 GEMM 权重 [num_local_experts, gemm1_out_size, hidden_size]
- `gemm1_weights_scale`: 第一个 GEMM 缩放因子 [num_local_experts, num_gemm1_out_blocks, num_hidden_blocks]
- `gemm2_weights`: 第二个 GEMM 权重 [num_local_experts, hidden_size, intermediate_size]
- `gemm2_weights_scale`: 第二个 GEMM 缩放因子 [num_local_experts, num_hidden_blocks, num_intermediate_blocks]
- `local_expert_offset`: 本地专家在全局空间的偏移（标量）
- `routed_scaling_factor`: 路由权重缩放因子（标量）

**输出**:
- `output`: 最终 MoE 输出 [seq_len, hidden_size]

### 应用模型
- DeepSeek V3/V3.2
- Mixtral 8x7B/8x22B
- Qwen3 A3B/A22B 系列

---

## 7. RMSNorm (Root Mean Square Layer Normalization)

### 基本信息
- **目录**: `kernels/operators/rmsnorm/`, `prompts/rmsnorm/`
- **用途**: RMS 归一化技术
- **特点**: 通过输入元素的均方根归一化输入

### 变体
- **Standard RMSNorm**: 基本 RMS 归一化
- **Fused Add RMSNorm**: 单个融合操作中在归一化前添加残差连接

### 维度（2个）
- `batch_size`: 变量
- `hidden_size`: 常量

### 输入输出
**输入**:
- `hidden_states`: [batch_size, hidden_size]
- `weight`: [hidden_size]
- Fused Add RMSNorm 额外输入：
  - `residual`: [batch_size, hidden_size]

**输出**:
- `output`: [batch_size, hidden_size]

### 应用模型
- 几乎所有现代 LLM（Llama、Qwen、DeepSeek、Mistral 等）

---

## 8. RoPE (Rotary Position Embedding)

### 基本信息
- **目录**: `kernels/operators/rope/`, `prompts/rope/`
- **用途**: 旋转位置编码
- **特点**: 在注意力前对 query 和 key 应用旋转变换，直接编码位置信息

### 变体
- **Full RoPE**: `rotary_dim == head_size`
- **Partial RoPE**: `rotary_dim < head_size`

### 旋转样式
- **NeoX-style** (`is_neox=True`): 分割旋转维度的前后半部分
- **GPT-J interleaved** (`is_neox=False`): 旋转偶数/奇数索引

### 维度（6个）
- `num_tokens`, `num_qo_heads`, `num_kv_heads`, `max_seq_len`: 变量
- `head_size`, `rotary_dim`: 常量

### 输入输出
**输入**:
- `q`: [num_tokens, num_qo_heads, head_size]
- `k`: [num_tokens, num_kv_heads, head_size]
- `cos_sin_cache`: [max_seq_len, rotary_dim]（float32，前半 cos，后半 sin）
- `positions`: [num_tokens]（int64，cos_sin_cache 的索引）

**输出（原地修改）**:
- `q_out`: [num_tokens, num_qo_heads, head_size]
- `k_out`: [num_tokens, num_kv_heads, head_size]

### 应用模型
- Llama 系列
- Qwen 系列
- Mistral 系列

---

## 9. Sampling (Token Sampling)

### 基本信息
- **目录**: `kernels/operators/sampling/`, `prompts/sampling/`
- **用途**: 语言模型生成的 token 采样
- **特点**: 从概率分布中选择下一个 token

### 变体
- **Top-k Sampling**: 保留 k 个最高概率 token，重归一化后采样
- **Top-p Sampling**: 使用累积概率阈值过滤（核采样）
- **Top-k + Top-p Sampling**: 组合两种过滤方法

### 维度（2个）
- `batch_size`: 变量
- `vocab_size`: 常量

### 输入输出
**输入**:
- `probs`: softmax 后的概率分布 [batch_size, vocab_size]
- 采样特定参数：
  - `top_k`: 用于 top-k 采样 [batch_size]
  - `top_p`: 用于 top-p/核采样 [batch_size]

**输出**:
- `samples`: 采样的 token 索引 [batch_size]

### 应用模型
- 所有生成式 LLM

---

## 10. GDN (Gated Delta Net)

### 基本信息
- **目录**: `kernels/operators/gdn/`, `prompts/gdn/`
- **用途**: 门控增量网络线性注意力机制
- **特点**: 使用增量规则实现线性注意力，支持高效递归计算

### 变体
- **Prefill**: 变长序列的分块计算
- **Decode**: 单 token 生成，递归状态更新

### 增量规则更新
```
g = -exp(A_log) * softplus(a + dt_bias)   # 对数空间衰减门
beta = sigmoid(b)                          # 更新门
state = state * exp(g)                     # 对状态应用衰减
v_new = v - k @ state                      # 预测误差（增量规则）
v_new = v_new * beta                       # 应用更新门
state = state + k^T @ v_new                # 更新状态（外积）
output = scale * q @ state                 # 计算输出
```

### Prefill 维度（6个）
- `total_seq_len`, `num_seqs`: 变量
- `num_q_heads`, `num_k_heads`, `num_v_heads`, `head_size`: 常量

### Prefill 输入输出
**输入**:
- `q`: query tensor [total_seq_len, num_q_heads, head_size]
- `k`: key tensor [total_seq_len, num_k_heads, head_size]
- `v`: value tensor [total_seq_len, num_v_heads, head_size]
- `g`: 遗忘门（alpha）[total_seq_len, num_sab_heads]（float32，可选）
- `beta`: 更新门 [total_seq_len, num_sab_heads]（float32，可选）
- `cu_seqlens`: 累积序列长度 [num_seqs + 1]（int64）
- `initial_state`: 初始 KV 状态（可选）
- `scale`: softmax 缩放（标量，可选）

**输出**:
- `output`: 注意力输出 [total_seq_len, num_o_heads, head_size]
- `final_state`: 最终 KV 状态 [num_seqs, num_sab_heads, head_size, head_size]（float32）

### 配置
- **GQA**: `num_q_heads >= num_k_heads` 且 `num_q_heads % num_k_heads == 0`
- **GVA**: `num_v_heads >= num_q_heads` 且 `num_v_heads % num_q_heads == 0`

### 应用模型
- Qwen3 Next 80B A3B

---

## 目录结构

```
KernelForge-MultiAgent/
├── kernels/
│   └── operators/
│       ├── gemm/              # GEMM 算子
│       ├── gqa_paged/         # GQA-Paged 算子
│       ├── gqa_ragged/        # GQA-Ragged 算子
│       ├── mla_paged/         # MLA-Paged 算子
│       ├── dsa_paged/         # DSA-Paged 算子
│       ├── moe/               # MoE 算子
│       ├── rmsnorm/           # RMSNorm 算子
│       ├── rope/              # RoPE 算子
│       ├── sampling/          # Sampling 算子
│       └── gdn/               # GDN 算子
└── prompts/
    ├── gemm/                  # GEMM 优化提示词
    ├── gqa_paged/             # GQA-Paged 优化提示词
    ├── gqa_ragged/            # GQA-Ragged 优化提示词
    ├── mla_paged/             # MLA-Paged 优化提示词
    ├── dsa_paged/             # DSA-Paged 优化提示词
    ├── moe/                   # MoE 优化提示词
    ├── rmsnorm/               # RMSNorm 优化提示词
    ├── rope/                  # RoPE 优化提示词
    ├── sampling/              # Sampling 优化提示词
    └── gdn/                   # GDN 优化提示词
```

---

## FlashInfer-Bench 对应关系

| KernelForge 算子 | FlashInfer-Bench op_type | 文档链接 |
|-----------------|-------------------------|---------|
| `gemm` | `gemm` | [docs/op-types/gemm.mdx](https://bench.flashinfer.ai/docs/op-types/gemm) |
| `gqa_paged` | `gqa-paged` | [docs/op-types/gqa-paged.mdx](https://bench.flashinfer.ai/docs/op-types/gqa-paged) |
| `gqa_ragged` | `gqa-ragged` | [docs/op-types/gqa-ragged.mdx](https://bench.flashinfer.ai/docs/op-types/gqa-ragged) |
| `mla_paged` | `mla-paged` | [docs/op-types/mla-paged.mdx](https://bench.flashinfer.ai/docs/op-types/mla-paged) |
| `dsa_paged` | `dsa-paged` | [docs/op-types/dsa-paged.mdx](https://bench.flashinfer.ai/docs/op-types/dsa-paged) |
| `moe` | `moe` | [docs/op-types/moe.mdx](https://bench.flashinfer.ai/docs/op-types/moe) |
| `rmsnorm` | `rmsnorm` | [docs/op-types/rmsnorm.mdx](https://bench.flashinfer.ai/docs/op-types/rmsnorm) |
| `rope` | `rope` | [docs/op-types/rope.mdx](https://bench.flashinfer.ai/docs/op-types/rope) |
| `sampling` | `sampling` | [docs/op-types/sampling.mdx](https://bench.flashinfer.ai/docs/op-types/sampling) |
| `gdn` | `gdn` | [docs/op-types/gdn.mdx](https://bench.flashinfer.ai/docs/op-types/gdn) |

---

## 验收标准

所有生成的算子代码必须通过 FlashInfer-Bench 验证：

```bash
cd D:/Agent/flashinfer-bench-main/flashinfer-bench-main
flashinfer-bench run --local flashinfer-trace --op-type <op_type>
```

详细验证流程参见：[`docs/FLASHINFER_BENCH_VALIDATION.md`](FLASHINFER_BENCH_VALIDATION.md)

---

## 优化优先级

基于 FlashInfer-Bench 模型覆盖情况，推荐优化优先级：

### 高优先级（应用最广泛）
1. **RMSNorm** - 所有现代 LLM 必需
2. **GQA-Paged** - Llama 系列、Qwen 系列、Mistral 系列
3. **GEMM** - 基础线性层，所有模型必需
4. **RoPE** - 位置编码，大多数模型使用
5. **Sampling** - 所有生成式模型必需

### 中优先级（特定架构）
6. **MoE** - DeepSeek、Mixtral、Qwen3 MoE 变体
7. **MLA-Paged** - DeepSeek V3/R1、Kimi K2
8. **GQA-Ragged** - 变长序列批处理场景

### 低优先级（最新架构）
9. **DSA-Paged** - DeepSeek V3.2 专用
10. **GDN** - Qwen3 Next 80B 等新架构

---

**最后更新**: 2026-06-15  
**版本**: v2.0 - FlashInfer-Bench 对齐
