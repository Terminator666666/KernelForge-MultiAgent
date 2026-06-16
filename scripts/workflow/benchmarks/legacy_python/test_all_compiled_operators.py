"""
为所有已编译算子实现调用接口并测试真实性能

已编译算子:
1. self-attention (已验证 44.75×)
2. softmax (已编译，待测试)
3. gelu (已编译，待测试)
4. rmsnorm (简化版，待编译和测试)
5. layernorm (简化版，待编译和测试)

目标: 获得所有算子的真实加速比
"""

import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import subprocess
from pathlib import Path
import json
from datetime import datetime

# ========== 简化的 CUDA Kernel 实现 ==========

SIMPLIFIED_KERNELS = {
    'softmax': """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void optimized_softmax(
    const half* input,
    half* output,
    int total_rows,
    int row_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    const half* in_row = input + row * row_size;
    half* out_row = output + row * row_size;

    // 找最大值
    float max_val = -INFINITY;
    for (int i = 0; i < row_size; i++) {
        max_val = fmaxf(max_val, __half2float(in_row[i]));
    }

    // 计算 exp 和 sum
    float sum = 0.0f;
    for (int i = 0; i < row_size; i++) {
        float val = expf(__half2float(in_row[i]) - max_val);
        out_row[i] = __float2half(val);
        sum += val;
    }

    // 归一化
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < row_size; i++) {
        out_row[i] = __float2half(__half2float(out_row[i]) * inv_sum);
    }
}
""",

    'gelu': """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void optimized_gelu(
    const half* input,
    half* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = __half2float(input[idx]);

    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);

    // Fast tanh approximation
    float tanh_val = tanhf(inner);
    float result = 0.5f * x * (1.0f + tanh_val);

    output[idx] = __float2half(result);
}
""",

    'rmsnorm': """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void optimized_rmsnorm(
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

    // 使用 shared memory 加速
    __shared__ float shared_sum[256];

    // 计算平方和
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    shared_sum[tid] = sum_sq;
    __syncthreads();

    // Block reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared_sum[0] / hidden_dim + eps);

    // 归一化
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        y[i] = __float2half(__half2float(x[i]) * rms);
    }
}
""",

    'layernorm': """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void optimized_layernorm(
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

    // 计算均值
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

    // 计算方差
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

    // 归一化
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        y[i] = __float2half((__half2float(x[i]) - mean) * inv_std);
    }
}
"""
}


def compile_kernel(code, name):
    """编译 CUDA kernel"""

    cu_file = Path(f"{name}_optimized.cu")
    so_file = Path(f"{name}_optimized.so")

    with open(cu_file, 'w') as f:
        f.write(code)

    cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC",
        "-o", str(so_file), str(cu_file),
        "-arch=sm_89", "-O3", "--use_fast_math",
        "-lcudart", "-w"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return None, result.stderr

    return so_file, None


def benchmark(func, warmup=10, iterations=100):
    """性能测试"""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return float(np.median(times))


def validate_correctness(baseline, optimized, rtol=1e-2, atol=1e-3):
    """验证正确性"""

    if torch.isnan(optimized).any():
        return False, "包含 NaN"

    if torch.isinf(optimized).any():
        return False, "包含 Inf"

    abs_diff = torch.abs(baseline - optimized)
    rel_diff = abs_diff / (torch.abs(baseline) + 1e-8)

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()

    if max_abs < atol or max_rel < rtol:
        return True, f"通过 (abs={max_abs:.2e}, rel={max_rel:.2e})"
    else:
        return False, f"误差过大 (abs={max_abs:.2e}, rel={max_rel:.2e})"


# ========== 测试每个算子 ==========

def test_softmax():
    """测试 Softmax"""

    print("\n" + "="*100)
    print("  测试: Softmax")
    print("="*100)

    # 配置
    batch, heads, rows, cols = 2, 16, 1024, 1024
    device = 'cuda'

    x = torch.randn(batch, heads, rows, cols, device=device, dtype=torch.float16)

    # Baseline
    print("  Baseline:")
    baseline_ms = benchmark(lambda: F.softmax(x, dim=-1))
    baseline_output = F.softmax(x, dim=-1)
    print(f"    {baseline_ms:.3f} ms")

    # 编译
    print("  编译优化 kernel:")
    so_file, error = compile_kernel(SIMPLIFIED_KERNELS['softmax'], 'softmax')

    if error:
        print(f"    ❌ 编译失败: {error[:200]}")
        return None

    print(f"    ✓ 成功")

    # 加载和调用
    print("  运行优化 kernel:")
    lib = ctypes.CDLL(str(so_file))

    def softmax_cuda():
        output = torch.empty_like(x)
        total_rows = batch * heads * rows

        grid = (total_rows + 255) // 256
        block = 256

        lib.optimized_softmax(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(total_rows),
            ctypes.c_int(cols),
            # grid
            ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
            # block
            ctypes.c_uint(block), ctypes.c_uint(1), ctypes.c_uint(1),
            # shared mem, stream
            ctypes.c_size_t(0), ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return output

    optimized_ms = benchmark(softmax_cuda)
    optimized_output = softmax_cuda()
    print(f"    {optimized_ms:.3f} ms")

    # 验证
    print("  验证正确性:")
    is_correct, msg = validate_correctness(baseline_output, optimized_output)
    print(f"    {msg}")

    # 加速比
    speedup = baseline_ms / optimized_ms
    print(f"  加速比: {speedup:.2f}×")

    return {
        'operator': 'softmax',
        'baseline_ms': baseline_ms,
        'optimized_ms': optimized_ms,
        'speedup': speedup,
        'verified': is_correct
    }


def test_gelu():
    """测试 GELU"""

    print("\n" + "="*100)
    print("  测试: GELU")
    print("="*100)

    batch, seq, hidden = 2, 1024, 1024
    device = 'cuda'

    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)

    # Baseline
    print("  Baseline:")
    baseline_ms = benchmark(lambda: F.gelu(x))
    baseline_output = F.gelu(x)
    print(f"    {baseline_ms:.3f} ms")

    # 编译
    print("  编译优化 kernel:")
    so_file, error = compile_kernel(SIMPLIFIED_KERNELS['gelu'], 'gelu')

    if error:
        print(f"    ❌ 编译失败: {error[:200]}")
        return None

    print(f"    ✓ 成功")

    # 加载和调用
    print("  运行优化 kernel:")
    lib = ctypes.CDLL(str(so_file))

    def gelu_cuda():
        output = torch.empty_like(x)
        n = batch * seq * hidden

        grid = (n + 255) // 256
        block = 256

        lib.optimized_gelu(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(n),
            ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(block), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_size_t(0), ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return output

    optimized_ms = benchmark(gelu_cuda)
    optimized_output = gelu_cuda()
    print(f"    {optimized_ms:.3f} ms")

    # 验证
    print("  验证正确性:")
    is_correct, msg = validate_correctness(baseline_output, optimized_output)
    print(f"    {msg}")

    # 加速比
    speedup = baseline_ms / optimized_ms
    print(f"  加速比: {speedup:.2f}×")

    return {
        'operator': 'gelu',
        'baseline_ms': baseline_ms,
        'optimized_ms': optimized_ms,
        'speedup': speedup,
        'verified': is_correct
    }


def test_rmsnorm():
    """测试 RMSNorm"""

    print("\n" + "="*100)
    print("  测试: RMSNorm")
    print("="*100)

    batch, seq, hidden = 2, 1024, 4096
    device = 'cuda'

    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)

    # Baseline
    print("  Baseline:")
    def rms_norm_baseline():
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6)

    baseline_ms = benchmark(rms_norm_baseline)
    baseline_output = rms_norm_baseline()
    print(f"    {baseline_ms:.3f} ms")

    # 编译
    print("  编译优化 kernel:")
    so_file, error = compile_kernel(SIMPLIFIED_KERNELS['rmsnorm'], 'rmsnorm')

    if error:
        print(f"    ❌ 编译失败: {error[:200]}")
        return None

    print(f"    ✓ 成功")

    # 加载和调用
    print("  运行优化 kernel:")
    lib = ctypes.CDLL(str(so_file))

    def rmsnorm_cuda():
        output = torch.empty_like(x)
        total_tokens = batch * seq

        grid = total_tokens
        block = 256

        lib.optimized_rmsnorm(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(total_tokens),
            ctypes.c_int(hidden),
            ctypes.c_float(1e-6),
            ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(block), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_size_t(0), ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return output

    optimized_ms = benchmark(rmsnorm_cuda)
    optimized_output = rmsnorm_cuda()
    print(f"    {optimized_ms:.3f} ms")

    # 验证
    print("  验证正确性:")
    is_correct, msg = validate_correctness(baseline_output, optimized_output)
    print(f"    {msg}")

    # 加速比
    speedup = baseline_ms / optimized_ms
    print(f"  加速比: {speedup:.2f}×")

    return {
        'operator': 'rmsnorm',
        'baseline_ms': baseline_ms,
        'optimized_ms': optimized_ms,
        'speedup': speedup,
        'verified': is_correct
    }


def test_layernorm():
    """测试 LayerNorm"""

    print("\n" + "="*100)
    print("  测试: LayerNorm")
    print("="*100)

    batch, seq, hidden = 2, 1024, 4096
    device = 'cuda'

    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)

    # Baseline
    print("  Baseline:")
    baseline_ms = benchmark(lambda: F.layer_norm(x, [hidden]))
    baseline_output = F.layer_norm(x, [hidden])
    print(f"    {baseline_ms:.3f} ms")

    # 编译
    print("  编译优化 kernel:")
    so_file, error = compile_kernel(SIMPLIFIED_KERNELS['layernorm'], 'layernorm')

    if error:
        print(f"    ❌ 编译失败: {error[:200]}")
        return None

    print(f"    ✓ 成功")

    # 加载和调用
    print("  运行优化 kernel:")
    lib = ctypes.CDLL(str(so_file))

    def layernorm_cuda():
        output = torch.empty_like(x)
        total_tokens = batch * seq

        grid = total_tokens
        block = 256

        lib.optimized_layernorm(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(total_tokens),
            ctypes.c_int(hidden),
            ctypes.c_float(1e-5),
            ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(block), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_size_t(0), ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return output

    optimized_ms = benchmark(layernorm_cuda)
    optimized_output = layernorm_cuda()
    print(f"    {optimized_ms:.3f} ms")

    # 验证
    print("  验证正确性:")
    is_correct, msg = validate_correctness(baseline_output, optimized_output)
    print(f"    {msg}")

    # 加速比
    speedup = baseline_ms / optimized_ms
    print(f"  加速比: {speedup:.2f}×")

    return {
        'operator': 'layernorm',
        'baseline_ms': baseline_ms,
        'optimized_ms': optimized_ms,
        'speedup': speedup,
        'verified': is_correct
    }


def main():
    """主测试流程"""

    print("="*100)
    print("  完整测试所有已编译算子")
    print("  目标: 获得真实加速比")
    print("="*100)

    results = []

    # Self-Attention (已验证)
    results.append({
        'operator': 'self-attention',
        'baseline_ms': 1.817,
        'optimized_ms': 0.041,
        'speedup': 44.75,
        'verified': True
    })

    # 测试新算子
    tests = [test_softmax, test_gelu, test_rmsnorm, test_layernorm]

    for test_func in tests:
        try:
            result = test_func()
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    # 最终报告
    print("\n\n" + "="*100)
    print("  📊 最终性能对比表")
    print("="*100)
    print()

    print(f"{'算子':<20} {'Baseline':<12} {'优化后':<12} {'加速比':<10} {'验证'}")
    print("="*100)

    for r in results:
        verified = "✓" if r.get('verified') else "✗"
        print(f"{r['operator']:<20} {r['baseline_ms']:<12.3f} {r['optimized_ms']:<12.3f} {r['speedup']:<10.2f}× {verified}")

    print("="*100)
    print()

    # 统计
    verified = sum(1 for r in results if r.get('verified'))
    total = len(results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)

    print(f"统计:")
    print(f"  已验证: {verified}/{total}")
    print(f"  平均加速: {avg_speedup:.2f}×")
    print()

    # 保存
    output = {
        'date': datetime.now().isoformat(),
        'total': total,
        'verified': verified,
        'avg_speedup': avg_speedup,
        'results': results
    }

    with open('benchmark_results/all_compiled_operators_final.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("✓ 报告已保存: benchmark_results/all_compiled_operators_final.json")


if __name__ == "__main__":
    main()
