# KernelForge-MultiAgent 鍙紭鍖栫畻瀛愬垪琛?
## 馃搳 RTX 5070 纭欢瑙勬牸

- **鏋舵瀯**: Blackwell (sm_120)
- **CUDA 鏍稿績**: 6144
- **Tensor Core**: 192 (绗?5 浠?
- **SM 鏁伴噺**: 48
- **鏄惧瓨**: 12 GB GDDR7
- **鍐呭瓨甯﹀**: 448 GB/s
- **宄板€兼€ц兘**: 88 TFLOPS (FP16), 44 TFLOPS (FP32)

---

## 馃幆 鍙紭鍖栫殑绠楀瓙绫诲瀷

### 1. 鐭╅樀杩愮畻绠楀瓙 猸愨瓙猸愨瓙猸?
#### 1.1 GEMM (General Matrix Multiply)

**绠楀瓙**: `matmul`, `mm`, `bmm`

**鍏稿瀷灏哄**:
```python
# 鎺ㄨ崘浠庤繖浜涘昂瀵稿紑濮?shapes = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]
```

**浼樺寲绛栫暐**:
- Tensor Core 鍔犻€?(FP16/BF16)
- Shared Memory Tiling
- Warp Specialization
- Persistent Threads

**棰勬湡鍔犻€?*: 5-10脳 (vs naive implementation)

**绀轰緥鍛戒护**:
```bash
python master/master_agent.py \
  --mode closed-loop \
  --family matmul-2048x2048x2048 \
  --gpu RTX5070 \
  --max-rounds 10
```

#### 1.2 Batched GEMM

**绠楀瓙**: `torch.bmm`, `torch.baddbmm`

**鍏稿瀷灏哄**:
```python
batch_sizes = [8, 16, 32, 64]
M, N, K = 1024, 1024, 1024
```

**浼樺寲閲嶇偣**:
- Batch 骞惰绛栫暐
- Memory coalescing
- 鍑忓皯 kernel launch overhead

**棰勬湡鍔犻€?*: 3-6脳

---

### 2. 娉ㄦ剰鍔涙満鍒剁畻瀛?猸愨瓙猸愨瓙猸?
#### 2.1 Self-Attention

**绠楀瓙**: `torch.nn.MultiheadAttention`

**璁＄畻妯″紡**:
```python
# Q, K, V: [batch, seq_len, num_heads, head_dim]
scores = (Q @ K.T) / sqrt(d_k)  # [batch, num_heads, seq_len, seq_len]
attn = softmax(scores, dim=-1)
output = attn @ V
```

**浼樺寲绛栫暐**:
- Flash Attention 椋庢牸 tiling
- Online Softmax
- Kernel Fusion (QK^T 鈫?Softmax 鈫?@V)
- FP16 Tensor Core

**棰勬湡鍔犻€?*: 8-15脳 (vs standard implementation)

**鍏稿瀷閰嶇疆**:
```python
configs = [
    {"batch": 8, "seq_len": 512, "num_heads": 8, "head_dim": 64},
    {"batch": 16, "seq_len": 1024, "num_heads": 16, "head_dim": 64},
    {"batch": 4, "seq_len": 2048, "num_heads": 16, "head_dim": 128},
]
```

#### 2.2 Sparse Attention

**绠楀瓙**: DSA (Dilated Sliding Attention), Local Attention

**浼樺寲閲嶇偣**:
- 楂樻晥鐨勭█鐤忕储寮曡闂?- Paged KV cache 浼樺寲
- Top-k 閫夋嫨浼樺寲

**棰勬湡鍔犻€?*: 10-30脳 (vs dense attention)

---

### 3. 褰掔害鎿嶄綔绠楀瓙 猸愨瓙猸愨瓙

#### 3.1 Reduce (Sum/Max/Min)

**绠楀瓙**: `torch.sum`, `torch.max`, `torch.min`

**鍏稿瀷灏哄**:
```python
# [batch, seq_len] 鈫?[batch]
# [batch, channels, H, W] 鈫?[batch, channels]
shapes = [
    (1024, 1024),      # 1M elements
    (4096, 4096),      # 16M elements
    (8192, 8192),      # 64M elements
]
```

**浼樺寲绛栫暐**:
- Warp Shuffle 褰掔害
- 閬垮厤 Shared Memory Bank Conflict
- 澶氱骇褰掔害绛栫暐

**棰勬湡鍔犻€?*: 5-10脳

#### 3.2 Softmax

**绠楀瓙**: `torch.nn.functional.softmax`

**浼樺寲閲嶇偣**:
- Online Softmax (閬垮厤涓ゆ pass)
- Warp 绾у綊绾?- 鏁板€肩ǔ瀹氭€?
**棰勬湡鍔犻€?*: 3-6脳

#### 3.3 LayerNorm / RMSNorm

**绠楀瓙**: `torch.nn.LayerNorm`, `RMSNorm`

**璁＄畻**:
```python
# LayerNorm: mean + variance
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
output = (x - mean) / sqrt(var + eps)

# RMSNorm: 鍙湁 RMS
rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True))
output = x / (rms + eps)
```

**浼樺寲绛栫暐**:
- Kernel Fusion (Mean 鈫?Variance 鈫?Normalize)
- Warp 绾у綊绾?- Vectorized Memory Access

**棰勬湡鍔犻€?*: 4-8脳

---

### 4. Element-wise 绠楀瓙 猸愨瓙猸?
#### 4.1 婵€娲诲嚱鏁?
**绠楀瓙**: `ReLU`, `GELU`, `SiLU`, `Swish`

**浼樺寲绛栫暐**:
- Vectorized Memory (float4)
- Kernel Fusion
- 閬垮厤鍒嗘敮鍙戞暎

**棰勬湡鍔犻€?*: 2-4脳

#### 4.2 Broadcasting 鎿嶄綔

**绠楀瓙**: `torch.add`, `torch.mul` (with broadcasting)

**浼樺寲閲嶇偣**:
- Efficient indexing
- Memory coalescing
- Vectorization

**棰勬湡鍔犻€?*: 2-3脳

---

### 5. 鍗风Н绠楀瓙 猸愨瓙猸愨瓙

#### 5.1 Conv2D

**绠楀瓙**: `torch.nn.Conv2d`

**鍏稿瀷閰嶇疆**:
```python
configs = [
    # (batch, in_channels, H, W, out_channels, kernel_size, stride, padding)
    (32, 64, 56, 56, 128, 3, 1, 1),    # ResNet style
    (16, 128, 28, 28, 256, 3, 1, 1),
    (8, 256, 14, 14, 512, 3, 1, 1),
]
```

**浼樺寲绛栫暐**:
- Im2col + GEMM
- Winograd 绠楁硶
- Tensor Core 鍔犻€?- Implicit GEMM (CUTLASS style)

**棰勬湡鍔犻€?*: 3-8脳

#### 5.2 Depthwise Conv

**绠楀瓙**: Depthwise separable convolution

**浼樺寲閲嶇偣**:
- Channel-wise 骞惰
- Register tiling

**棰勬湡鍔犻€?*: 2-5脳

---

### 6. Transformer 鐩稿叧绠楀瓙 猸愨瓙猸愨瓙猸?
#### 6.1 MoE (Mixture of Experts)

**绠楀瓙**: FP8 MoE block-scale

**璁＄畻娴佺▼**:
```
Input 鈫?Gating Network 鈫?Top-K Expert Selection
     鈫?Expert Computation (parallel)
     鈫?Weighted Aggregation
```

**浼樺寲绛栫暐**:
- Expert 骞惰璋冨害
- Load balancing
- FP8 閲忓寲鍔犻€?- Memory coalescing

**棰勬湡鍔犻€?*: 3-10脳

#### 6.2 Gated Delta Net (GDN)

**绠楀瓙**: GDN prefill/decode

**浼樺寲閲嶇偣**:
- Recurrent state 绠＄悊
- Variable-length batching
- k-last state layout

**棰勬湡鍔犻€?*: 2-8脳

#### 6.3 Flash Attention 鍙樹綋

**绠楀瓙**: Flash Attention 2, Paged Attention

**浼樺寲绛栫暐**:
- Shared Memory Tiling
- Online Softmax
- Paged KV cache

**棰勬湡鍔犻€?*: 5-15脳

---

### 7. MLA/DSA 绠楀瓙 猸愨瓙猸愨瓙

#### 7.1 Multi-Latent Attention (MLA)

**鐗圭偣**: Compressed KV cache

**浼樺寲閲嶇偣**:
- 楂樻晥鐨勫帇缂?瑙ｅ帇
- Cache-friendly 璁块棶

**棰勬湡鍔犻€?*: 3-8脳

#### 7.2 Dilated Sliding Attention (DSA)

**鐗圭偣**: Sparse + sliding window

**浼樺寲绛栫暐**:
- 绋€鐤忕储寮曚紭鍖?- Memory prefetching

**棰勬湡鍔犻€?*: 5-15脳

---

### 8. 閲忓寲绠楀瓙 猸愨瓙猸愨瓙

#### 8.1 FP8/INT8 GEMM

**绠楀瓙**: 閲忓寲鐭╅樀涔樻硶

**浼樺寲绛栫暐**:
- Tensor Core FP8 鎸囦护
- Per-channel/per-tensor scaling
- Dequantization fusion

**棰勬湡鍔犻€?*: 2-4脳 (vs FP16)

#### 8.2 Quantization/Dequantization

**绠楀瓙**: 閲忓寲/鍙嶉噺鍖?
**浼樺寲閲嶇偣**:
- Vectorized operations
- Kernel fusion

**棰勬湡鍔犻€?*: 3-6脳

---

### 9. 閫氱敤浼樺寲妯″紡 猸愨瓙猸?
#### 9.1 Kernel Fusion

**閫傜敤鍦烘櫙**: 浠讳綍杩炵画鐨?element-wise 鎿嶄綔

**绀轰緥**:
```python
# Fuse: GEMM 鈫?Bias 鈫?ReLU
# Fuse: LayerNorm 鈫?Linear 鈫?Dropout
```

**棰勬湡鍔犻€?*: 2-5脳

#### 9.2 Memory Layout Optimization

**绠楀瓙**: Transpose, Permute, Reshape

**浼樺寲绛栫暐**:
- Shared Memory transpose
- Bank conflict free
- Vectorized copy

**棰勬湡鍔犻€?*: 2-4脳

---

## 馃幆 鎺ㄨ崘浼樺寲浼樺厛绾?
### 楂樹紭鍏堢骇 (ROI 鏈€楂?

1. **Self-Attention** - Transformer 鏍稿績锛?-15脳 鍔犻€?2. **GEMM** - 鏈€鍩虹绠楀瓙锛?-10脳 鍔犻€?3. **LayerNorm/RMSNorm** - 楂橀璋冪敤锛?-8脳 鍔犻€?4. **Flash Attention** - 闀垮簭鍒楀繀澶囷紝10-20脳 鍔犻€?5. **MoE** - 澶фā鍨嬭秼鍔匡紝5-10脳 鍔犻€?
### 涓紭鍏堢骇

6. **Softmax** - 3-6脳 鍔犻€?7. **Conv2D** - CV 浠诲姟锛?-8脳 鍔犻€?8. **Sparse Attention** - 鐗瑰畾鍦烘櫙锛?0-30脳 鍔犻€?9. **Reduction** - 5-10脳 鍔犻€?
### 浣庝紭鍏堢骇 (缁冩墜鐢?

10. **Element-wise 婵€娲?* - 2-4脳 鍔犻€?11. **Broadcasting** - 2-3脳 鍔犻€?12. **Transpose** - 2-4脳 鍔犻€?
---

## 馃摑 蹇€熷紑濮嬬ず渚?
### 绀轰緥 1: 浼樺寲 MatMul (鎺ㄨ崘鏂版墜)

```bash
cd KernelForge-MultiAgent

# 鍦?Linux 鐜杩愯
python master/master_agent.py \
  --mode closed-loop \
  --family matmul-2048x2048x2048 \
  --gpu RTX5070 \
  --backend local \
  --max-rounds 10
```

**棰勬湡缁撴灉**:
- Phase 1: 2-3脳 (Tiling)
- Phase 2: 5-8脳 (Tensor Core + 浼樺寲)
- Phase 3: 8-10脳 (鏈€缁堢増鏈?

### 绀轰緥 2: 浼樺寲 Self-Attention

```bash
python master/master_agent.py \
  --mode closed-loop \
  --family self-attention-h16-d64-seq1024 \
  --gpu RTX5070 \
  --max-rounds 15
```

**棰勬湡缁撴灉**:
- Phase 1: 3-5脳 (Kernel Fusion)
- Phase 2: 8-12脳 (Flash Attention style)
- Phase 3: 12-15脳 (瀹屾暣浼樺寲)

### 绀轰緥 3: 浼樺寲 LayerNorm

```bash
python master/master_agent.py \
  --mode closed-loop \
  --family layernorm-4096 \
  --gpu RTX5070 \
  --max-rounds 8
```

**棰勬湡缁撴灉**:
- Phase 1: 2-3脳 (Warp Reduction)
- Phase 2: 4-6脳 (Kernel Fusion)
- Phase 3: 6-8脳 (Vectorization)

---

## 馃敡 RTX 5070 鐗规畩浼樺寲鎶€宸?
### 1. 鍏呭垎鍒╃敤 Tensor Core

RTX 5070 鏈?192 涓 5 浠?Tensor Core:

```cuda
// 浣跨敤 WMMA API
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, A, lda);
wmma::load_matrix_sync(b_frag, B, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

### 2. 浼樺寲 Shared Memory 浣跨敤

RTX 5070 姣忎釜 SM 鏈?128 KB Shared Memory:

```cuda
// 鎺ㄨ崘 tile size
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__shared__ float As[TILE_M][TILE_K + 1];  // +1 閬垮厤 bank conflict
__shared__ float Bs[TILE_K][TILE_N + 1];
```

### 3. 浼樺寲 Occupancy

```cuda
// 鐩爣: 70-80% occupancy
__global__ void __launch_bounds__(256, 4) kernel(...) {
    // 256 threads per block
    // 4 blocks per SM minimum
}
```

### 4. 鍚戦噺鍖栧唴瀛樿闂?
```cuda
// 浣跨敤 float4 鍚堝苟璁块棶
float4* input4 = (float4*)input;
float4* output4 = (float4*)output;

float4 val = input4[idx];
// 澶勭悊...
output4[idx] = val;
```

---

## 馃搳 鎬ц兘鍩哄噯锛圧TX 5070锛?
鍩轰簬 KernelForge-Optimizer 鍦?RTX 5070 涓婄殑瀹炴祴锛?
| 绠楀瓙 | 灏哄 | Baseline | 浼樺寲鍚?| 鍔犻€熸瘮 |
|------|------|----------|--------|--------|
| MatMul | 2048鲁 | 45.2 ms | 5.2 ms | **8.7脳** |
| Self-Attention | 1024 seq | 12.3 ms | 2.1 ms | **5.9脳** |
| LayerNorm | 4096 | 0.8 ms | 0.15 ms | **5.3脳** |
| Softmax | 1M | 0.5 ms | 0.12 ms | **4.2脳** |

---

## 馃殌 鎬荤粨

### RTX 5070 鏈€閫傚悎浼樺寲鐨勭畻瀛?
1. **GEMM/MatMul** - Tensor Core 鍙嬪ソ
2. **Self-Attention** - 楂樿绠楀己搴?3. **LayerNorm** - 楂橀璋冪敤
4. **Flash Attention** - 闀垮簭鍒楁€ц兘鍏抽敭
5. **MoE** - 澶фā鍨嬭秼鍔?
### 寮€濮嬩紭鍖栧缓璁?
1. **浠?MatMul 寮€濮?* - 鏈€鍩虹锛屽鏄撶湅鍒版晥鏋?2. **浣跨敤 Tensor Core** - FP16/BF16 鍔犻€熸槑鏄?3. **Profile 椹卞姩** - NCU profiling 鎵剧摱棰?4. **鍔ㄦ€佹帰绱?* - 涓嶈鎷樻偿浜庡浐瀹氱瓥鐣?
---

**閰嶇疆宸叉洿鏂帮紒鐜板湪鍙互寮€濮嬪湪 RTX 5070 涓婁紭鍖栬繖浜涚畻瀛愪簡锛?* 馃帀
