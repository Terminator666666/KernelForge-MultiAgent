"""
简化的 Self-Attention CUDA Kernel
基于生成代码的优化思路，但使用标准 CUDA API

优化策略:
1. Tiling (分块计算)
2. Shared Memory 复用
3. Online Softmax
4. Warp-level 优化
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// 配置
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int WARP_SIZE = 32;

// Simplified Flash Attention Kernel
__global__ void flash_attention_simple_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // 线程块和线程 ID
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block = blockIdx.x;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 计算 Q 的起始位置
    int q_offset = ((batch_idx * heads + head_idx) * seq_len + q_block * BLOCK_M) * head_dim;
    int kv_offset = (batch_idx * heads + head_idx) * seq_len * head_dim;

    // Shared Memory
    __shared__ half smem_Q[BLOCK_M][BLOCK_K];
    __shared__ half smem_K[BLOCK_N][BLOCK_K];
    __shared__ half smem_V[BLOCK_N][BLOCK_K];
    __shared__ float smem_S[BLOCK_M][BLOCK_N];

    // 寄存器
    float acc_O[BLOCK_M / WARP_SIZE][BLOCK_K / WARP_SIZE] = {0.0f};
    float max_scores[BLOCK_M / WARP_SIZE];
    float sum_exp[BLOCK_M / WARP_SIZE];

    // 初始化
    for (int i = 0; i < BLOCK_M / WARP_SIZE; ++i) {
        max_scores[i] = -INFINITY;
        sum_exp[i] = 0.0f;
    }

    // 加载 Q (每个 warp 处理一部分)
    for (int m = warp_id; m < BLOCK_M; m += blockDim.x / WARP_SIZE) {
        for (int k = lane_id; k < head_dim && k < BLOCK_K; k += WARP_SIZE) {
            if (q_block * BLOCK_M + m < seq_len) {
                smem_Q[m][k] = Q[q_offset + m * head_dim + k];
            } else {
                smem_Q[m][k] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    // 遍历所有 K/V blocks
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        // 加载 K
        for (int n = warp_id; n < BLOCK_N; n += blockDim.x / WARP_SIZE) {
            for (int k = lane_id; k < head_dim && k < BLOCK_K; k += WARP_SIZE) {
                if (kv_block * BLOCK_N + n < seq_len) {
                    smem_K[n][k] = K[kv_offset + (kv_block * BLOCK_N + n) * head_dim + k];
                } else {
                    smem_K[n][k] = __float2half(0.0f);
                }
            }
        }

        // 加载 V
        for (int n = warp_id; n < BLOCK_N; n += blockDim.x / WARP_SIZE) {
            for (int k = lane_id; k < head_dim && k < BLOCK_K; k += WARP_SIZE) {
                if (kv_block * BLOCK_N + n < seq_len) {
                    smem_V[n][k] = V[kv_offset + (kv_block * BLOCK_N + n) * head_dim + k];
                } else {
                    smem_V[n][k] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // 计算 S = Q @ K^T
        for (int m = warp_id; m < BLOCK_M; m += blockDim.x / WARP_SIZE) {
            for (int n = lane_id; n < BLOCK_N; n += WARP_SIZE) {
                float sum = 0.0f;
                for (int k = 0; k < head_dim && k < BLOCK_K; ++k) {
                    sum += __half2float(smem_Q[m][k]) * __half2float(smem_K[n][k]);
                }
                smem_S[m][n] = sum * scale;
            }
        }
        __syncthreads();

        // Online Softmax + O 累加
        for (int m = warp_id; m < BLOCK_M; m += blockDim.x / WARP_SIZE) {
            // 找到当前行的最大值
            float row_max = -INFINITY;
            for (int n = 0; n < BLOCK_N; ++n) {
                row_max = fmaxf(row_max, smem_S[m][n]);
            }

            // 更新全局最大值和 exp sum
            float old_max = max_scores[m / (blockDim.x / WARP_SIZE)];
            float new_max = fmaxf(old_max, row_max);
            float exp_diff = expf(old_max - new_max);

            // 重新缩放之前的累加器
            for (int k = lane_id; k < head_dim && k < BLOCK_K; k += WARP_SIZE) {
                acc_O[m / (blockDim.x / WARP_SIZE)][k / WARP_SIZE] *= exp_diff;
            }

            // 计算新的 exp sum
            float new_sum = sum_exp[m / (blockDim.x / WARP_SIZE)] * exp_diff;

            // 累加当前块的贡献
            for (int n = 0; n < BLOCK_N; ++n) {
                float exp_val = expf(smem_S[m][n] - new_max);
                new_sum += exp_val;

                // O += exp_val * V[n, :]
                for (int k = lane_id; k < head_dim && k < BLOCK_K; k += WARP_SIZE) {
                    acc_O[m / (blockDim.x / WARP_SIZE)][k / WARP_SIZE] +=
                        exp_val * __half2float(smem_V[n][k]);
                }
            }

            max_scores[m / (blockDim.x / WARP_SIZE)] = new_max;
            sum_exp[m / (blockDim.x / WARP_SIZE)] = new_sum;
        }
        __syncthreads();
    }

    // 写回输出 (归一化)
    for (int m = warp_id; m < BLOCK_M; m += blockDim.x / WARP_SIZE) {
        float norm = 1.0f / sum_exp[m / (blockDim.x / WARP_SIZE)];
        for (int k = lane_id; k < head_dim && k < BLOCK_K; k += WARP_SIZE) {
            if (q_block * BLOCK_M + m < seq_len) {
                O[q_offset + m * head_dim + k] =
                    __float2half(acc_O[m / (blockDim.x / WARP_SIZE)][k / WARP_SIZE] * norm);
            }
        }
    }
}

torch::Tensor flash_attention_simple_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    auto batch = Q.size(0);
    auto heads = Q.size(1);
    auto seq_len = Q.size(2);
    auto head_dim = Q.size(3);

    auto O = torch::empty_like(Q);

    float scale = 1.0f / sqrtf(head_dim);

    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, heads, batch);
    dim3 block(128);

    flash_attention_simple_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        batch, heads, seq_len, head_dim, scale
    );

    return O;
}
"""

cpp_source = """
torch::Tensor flash_attention_simple_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);
"""

def compile_kernel():
    """编译简化的 Flash Attention kernel"""
    print("编译简化的 Flash Attention kernel...")

    module = load_inline(
        name='flash_attention_simple',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['flash_attention_simple_forward'],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '--gpu-architecture=sm_89'
        ],
        verbose=True
    )

    print("✓ 编译成功！")
    return module

if __name__ == "__main__":
    module = compile_kernel()
    print(f"Module functions: {dir(module)}")
