"""
持续解决所有问题 - 不放弃直到所有12个算子都获得真实加速比

当前问题清单:
1. GLIBCXX_3.4.32 not found - PyTorch JIT在conda环境中失败
2. ctypes无法正确调用CUDA kernel - 需要<<<grid, block>>>语法

解决方案:
方案A: 使用系统Python而非conda (彻底解决GLIBCXX)
方案B: 每个算子手工实现调用接口 (像Self-Attention一样)
方案C: 创建独立的CUDA程序，不依赖Python扩展

立即执行方案C: 创建独立CUDA可执行程序测试所有算子
"""

import subprocess
from pathlib import Path
import json
import re

def create_standalone_cuda_test():
    """创建独立的CUDA测试程序，不依赖Python扩展"""

    # 为每个算子创建独立的.cu文件和测试程序

    # Softmax独立测试
    softmax_cuda = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void softmax_kernel(
    const half* input,
    half* output,
    int total_rows,
    int row_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    const half* in_row = input + row * row_size;
    half* out_row = output + row * row_size;

    float max_val = -INFINITY;
    for (int i = 0; i < row_size; i++) {
        max_val = fmaxf(max_val, __half2float(in_row[i]));
    }

    float sum = 0.0f;
    for (int i = 0; i < row_size; i++) {
        float val = expf(__half2float(in_row[i]) - max_val);
        out_row[i] = __float2half(val);
        sum += val;
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < row_size; i++) {
        out_row[i] = __float2half(__half2float(out_row[i]) * inv_sum);
    }
}

int main() {
    // 配置
    int batch = 2, heads = 16, rows = 1024, cols = 1024;
    int total_rows = batch * heads * rows;
    size_t size = total_rows * cols * sizeof(half);

    // 分配内存
    half *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // 初始化随机数据
    half* h_input = (half*)malloc(size);
    for (int i = 0; i < total_rows * cols; i++) {
        h_input[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        softmax_kernel<<<(total_rows + 255) / 256, 256>>>(d_input, d_output, total_rows, cols);
    }
    cudaDeviceSynchronize();

    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    int iterations = 100;

    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        softmax_kernel<<<(total_rows + 255) / 256, 256>>>(d_input, d_output, total_rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }

    float avg_time = total_time / iterations;
    printf("Softmax optimized: %.3f ms\\n", avg_time);

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
"""

    # GELU独立测试
    gelu_cuda = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void gelu_kernel(
    const half* input,
    half* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = __half2float(input[idx]);
    const float sqrt_2_over_pi = 0.7978845608f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    float tanh_val = tanhf(inner);
    float result = 0.5f * x * (1.0f + tanh_val);

    output[idx] = __float2half(result);
}

int main() {
    int batch = 2, seq = 1024, hidden = 1024;
    int n = batch * seq * hidden;
    size_t size = n * sizeof(half);

    half *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    half* h_input = (half*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_input[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        gelu_kernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    // 测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int i = 0; i < 100; i++) {
        cudaEventRecord(start);
        gelu_kernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }

    printf("GELU optimized: %.3f ms\\n", total_time / 100);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
"""

    # RMSNorm独立测试
    rmsnorm_cuda = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void rmsnorm_kernel(
    const half* input,
    half* output,
    int batch_size,
    int hidden_dim,
    float eps
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= batch_size) return;

    const half* x = input + token_idx * hidden_dim;
    half* y = output + token_idx * hidden_dim;

    __shared__ float shared_sum[256];

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    shared_sum[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared_sum[0] / hidden_dim + eps);

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        y[i] = __float2half(__half2float(x[i]) * rms);
    }
}

int main() {
    int batch = 2, seq = 1024, hidden = 4096;
    int total_tokens = batch * seq;
    size_t size = total_tokens * hidden * sizeof(half);

    half *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    half* h_input = (half*)malloc(size);
    for (int i = 0; i < total_tokens * hidden; i++) {
        h_input[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        rmsnorm_kernel<<<total_tokens, 256>>>(d_input, d_output, total_tokens, hidden, 1e-6f);
    }
    cudaDeviceSynchronize();

    // 测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int i = 0; i < 100; i++) {
        cudaEventRecord(start);
        rmsnorm_kernel<<<total_tokens, 256>>>(d_input, d_output, total_tokens, hidden, 1e-6f);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }

    printf("RMSNorm optimized: %.3f ms\\n", total_time / 100);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
"""

    # LayerNorm独立测试
    layernorm_cuda = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void layernorm_kernel(
    const half* input,
    half* output,
    int batch_size,
    int hidden_dim,
    float eps
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= batch_size) return;

    const half* x = input + token_idx * hidden_dim;
    half* y = output + token_idx * hidden_dim;

    __shared__ float shared_data[256];

    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        sum += __half2float(x[i]);
    }

    shared_data[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    float mean = shared_data[0] / hidden_dim;

    float var = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = __half2float(x[i]) - mean;
        var += diff * diff;
    }

    shared_data[tid] = var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    float inv_std = rsqrtf(shared_data[0] / hidden_dim + eps);

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        y[i] = __float2half((__half2float(x[i]) - mean) * inv_std);
    }
}

int main() {
    int batch = 2, seq = 1024, hidden = 4096;
    int total_tokens = batch * seq;
    size_t size = total_tokens * hidden * sizeof(half);

    half *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    half* h_input = (half*)malloc(size);
    for (int i = 0; i < total_tokens * hidden; i++) {
        h_input[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        layernorm_kernel<<<total_tokens, 256>>>(d_input, d_output, total_tokens, hidden, 1e-5f);
    }
    cudaDeviceSynchronize();

    // 测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int i = 0; i < 100; i++) {
        cudaEventRecord(start);
        layernorm_kernel<<<total_tokens, 256>>>(d_input, d_output, total_tokens, hidden, 1e-5f);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }

    printf("LayerNorm optimized: %.3f ms\\n", total_time / 100);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
"""

    # 保存所有CUDA文件
    kernels = {
        'softmax': softmax_cuda,
        'gelu': gelu_cuda,
        'rmsnorm': rmsnorm_cuda,
        'layernorm': layernorm_cuda
    }

    results = []

    for name, code in kernels.items():
        print(f"\n{'='*80}")
        print(f"  测试: {name}")
        print(f"{'='*80}")

        # 保存代码
        cu_file = f"/tmp/{name}_standalone.cu"
        exe_file = f"/tmp/{name}_standalone"

        with open(cu_file, 'w') as f:
            f.write(code)

        # 编译
        print(f"  编译...", end=" ")
        compile_cmd = [
            "nvcc", cu_file, "-o", exe_file,
            "-arch=sm_89", "-O3", "--use_fast_math",
            "-lcudart"
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"失败")
            print(f"    {result.stderr[:200]}")
            continue

        print("成功")

        # 运行
        print(f"  运行...", end=" ")
        run_result = subprocess.run([exe_file], capture_output=True, text=True)

        if run_result.returncode != 0:
            print(f"失败")
            print(f"    {run_result.stderr[:200]}")
            continue

        print("成功")

        # 提取性能数据
        output = run_result.stdout
        match = re.search(r'(\d+\.\d+) ms', output)

        if match:
            optimized_ms = float(match.group(1))
            print(f"  ✓ 优化性能: {optimized_ms:.3f} ms")

            results.append({
                'operator': name,
                'optimized_ms': optimized_ms,
                'status': 'success'
            })
        else:
            print(f"  ✗ 无法提取性能数据")

    return results


if __name__ == "__main__":
    print("="*80)
    print("  使用独立CUDA程序测试所有算子")
    print("  避免Python扩展和GLIBCXX问题")
    print("="*80)

    results = create_standalone_cuda_test()

    print("\n" + "="*80)
    print(f"  完成: {len(results)}/4 算子成功")
    print("="*80)

    for r in results:
        print(f"  {r['operator']}: {r['optimized_ms']:.3f} ms")
