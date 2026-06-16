"""
纯 CUDA 实现 - 绕过 PyTorch C++ API 兼容性问题

根因：PyTorch 2.7 的 C++ API 在 C++14 模式下有类型不兼容问题
解决：使用纯 CUDA + ctypes，不依赖 torch::extension
"""

import torch
import numpy as np
import ctypes
import subprocess
import os
from pathlib import Path

# CUDA kernel 源代码
CUDA_SOURCE = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {

// Flash Attention Kernel - 纯 CUDA 实现
__global__ void flash_attention_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // 配置
    const int BLOCK_M = 32;
    const int BLOCK_N = 32;

    // 线程和块索引
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block = blockIdx.x;
    int tid = threadIdx.x;

    // 全局偏移
    int base_offset = (batch_idx * heads + head_idx) * seq_len * head_dim;
    int q_row_start = q_block * BLOCK_M;

    // Shared memory (动态分配)
    extern __shared__ char smem_buffer[];
    half* smem_Q = (half*)smem_buffer;
    half* smem_K = smem_Q + BLOCK_M * head_dim;
    half* smem_V = smem_K + BLOCK_N * head_dim;
    float* smem_S = (float*)(smem_V + BLOCK_N * head_dim);

    // 寄存器
    float acc_O[64];
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // 初始化
    for (int i = 0; i < head_dim; ++i) {
        acc_O[i] = 0.0f;
    }

    // 加载 Q block
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        int m = i / head_dim;
        int d = i % head_dim;
        int global_row = q_row_start + m;
        if (global_row < seq_len) {
            smem_Q[i] = Q[base_offset + global_row * head_dim + d];
        } else {
            smem_Q[i] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // 遍历 K/V blocks
    int num_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_blocks; ++kv_block) {
        int k_col_start = kv_block * BLOCK_N;

        // 加载 K
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int n = i / head_dim;
            int d = i % head_dim;
            int global_col = k_col_start + n;
            if (global_col < seq_len) {
                smem_K[i] = K[base_offset + global_col * head_dim + d];
            } else {
                smem_K[i] = __float2half(0.0f);
            }
        }

        // 加载 V
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int n = i / head_dim;
            int d = i % head_dim;
            int global_col = k_col_start + n;
            if (global_col < seq_len) {
                smem_V[i] = V[base_offset + global_col * head_dim + d];
            } else {
                smem_V[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // 计算 S = Q @ K^T
        for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int m = idx / BLOCK_N;
            int n = idx % BLOCK_N;

            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += __half2float(smem_Q[m * head_dim + d]) *
                         __half2float(smem_K[n * head_dim + d]);
            }
            smem_S[idx] = score * scale;
        }
        __syncthreads();

        // Online Softmax
        if (tid < BLOCK_M) {
            int m = tid;
            int global_row = q_row_start + m;

            if (global_row < seq_len) {
                // 找最大值
                float block_max = -INFINITY;
                for (int n = 0; n < BLOCK_N; ++n) {
                    block_max = fmaxf(block_max, smem_S[m * BLOCK_N + n]);
                }

                // 更新 max 和重新缩放
                float old_max = max_score;
                float new_max = fmaxf(old_max, block_max);
                float exp_diff = expf(old_max - new_max);

                for (int d = 0; d < head_dim; ++d) {
                    acc_O[d] *= exp_diff;
                }

                float new_sum = sum_exp * exp_diff;

                // 累加
                for (int n = 0; n < BLOCK_N; ++n) {
                    float exp_score = expf(smem_S[m * BLOCK_N + n] - new_max);
                    new_sum += exp_score;

                    for (int d = 0; d < head_dim; ++d) {
                        acc_O[d] += exp_score * __half2float(smem_V[n * head_dim + d]);
                    }
                }

                max_score = new_max;
                sum_exp = new_sum;
            }
        }
        __syncthreads();
    }

    // 写回输出
    if (tid < BLOCK_M) {
        int m = tid;
        int global_row = q_row_start + m;

        if (global_row < seq_len) {
            float norm = 1.0f / sum_exp;
            for (int d = 0; d < head_dim; ++d) {
                O[base_offset + global_row * head_dim + d] =
                    __float2half(acc_O[d] * norm);
            }
        }
    }
}

} // extern "C"
"""

def compile_pure_cuda():
    """编译纯 CUDA kernel"""

    print("="*80)
    print("  编译纯 CUDA Kernel")
    print("="*80)
    print()

    # 保存源文件
    cu_file = Path("flash_attention_pure.cu")
    with open(cu_file, 'w') as f:
        f.write(CUDA_SOURCE)

    print(f"✓ 源文件已保存: {cu_file}")

    # 编译为 .so
    so_file = Path("flash_attention_pure.so")

    compile_cmd = [
        "nvcc",
        "-shared",
        "-Xcompiler", "-fPIC",
        "-o", str(so_file),
        str(cu_file),
        "-arch=sm_89",
        "-O3",
        "--use_fast_math",
        "-lcudart"
    ]

    print(f"编译命令: {' '.join(compile_cmd)}")
    print()

    result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ 编译失败:")
        print(result.stderr)
        raise RuntimeError("NVCC compilation failed")

    print("✓ 编译成功!")
    print(f"✓ 输出文件: {so_file}")
    print()

    return so_file

def load_cuda_kernel(so_file):
    """加载编译好的 CUDA kernel"""

    lib = ctypes.CDLL(str(so_file))

    # 不需要设置参数类型，直接调用
    return lib.flash_attention_kernel

def flash_attention_cuda(Q, K, V, kernel_func):
    """调用纯 CUDA kernel"""

    batch, heads, seq_len, head_dim = Q.shape

    # 创建输出
    O = torch.empty_like(Q)

    # 配置
    BLOCK_M = 32
    scale = 1.0 / (head_dim ** 0.5)

    # Grid 和 Block
    grid_x = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_y = heads
    grid_z = batch
    block_size = 128

    # Shared memory 大小
    smem_size = (BLOCK_M + 32 * 2) * head_dim * 2 + BLOCK_M * 32 * 4

    # 调用 kernel
    kernel_func(
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_int(batch),
        ctypes.c_int(heads),
        ctypes.c_int(seq_len),
        ctypes.c_int(head_dim),
        ctypes.c_float(scale),
        # Launch configuration
        ctypes.c_int(grid_x), ctypes.c_int(grid_y), ctypes.c_int(grid_z),
        ctypes.c_int(block_size), ctypes.c_int(1), ctypes.c_int(1),
        ctypes.c_int(smem_size),
        ctypes.c_void_p(0)  # stream
    )

    torch.cuda.synchronize()

    return O

def benchmark_pure_cuda():
    """测试纯 CUDA kernel"""

    print("="*80)
    print("  测试纯 CUDA Kernel")
    print("="*80)
    print()

    # 编译
    so_file = compile_pure_cuda()

    # 加载
    print("加载 kernel...")
    kernel = load_cuda_kernel(so_file)
    print("✓ Kernel 已加载")
    print()

    # 配置
    batch = 2
    heads = 16
    seq_len = 1024
    head_dim = 64
    device = 'cuda'

    # 生成输入
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)

    print("配置:")
    print(f"  Batch: {batch}")
    print(f"  Heads: {heads}")
    print(f"  Sequence: {seq_len}")
    print(f"  Head dim: {head_dim}")
    print()

    # Warmup
    print("Warmup...")
    for _ in range(10):
        _ = flash_attention_cuda(Q, K, V, kernel)
    torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking...")
    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = flash_attention_cuda(Q, K, V, kernel)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)

    print()
    print("="*80)
    print("  性能结果")
    print("="*80)
    print(f"延迟:")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Std:    {np.std(times):.3f} ms")
    print()

    # 加载 baseline
    import json
    with open('benchmark_results/baseline_self_attention.json', 'r') as f:
        baseline = json.load(f)

    baseline_time = baseline['performance']['median_ms']
    speedup = baseline_time / np.median(times)

    print("="*80)
    print("  真实加速比")
    print("="*80)
    print(f"Baseline:  {baseline_time:.3f} ms")
    print(f"Optimized: {np.median(times):.3f} ms")
    print(f"加速比:    {speedup:.2f}×")
    print()

    if speedup > 1.0:
        print(f"🎉 成功！获得 {speedup:.2f}× 真实加速！")
    else:
        print(f"⚠️  性能: {speedup:.2f}× (baseline 已经很快)")

    # 保存结果
    result_dict = {
        'baseline_ms': baseline_time,
        'optimized_ms': float(np.median(times)),
        'speedup': float(speedup),
        'method': 'pure_cuda',
        'status': 'success'
    }

    with open('benchmark_results/pure_cuda_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    print()
    print("✓ 结果已保存: benchmark_results/pure_cuda_result.json")

if __name__ == "__main__":
    benchmark_pure_cuda()
