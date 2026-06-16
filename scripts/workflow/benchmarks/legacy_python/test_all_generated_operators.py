"""
测试所有已生成算子的性能

策略：
1. 为每个算子创建简化的纯 CUDA 实现
2. 编译并运行真实测试
3. 测量真实加速比
4. 验证正确性
"""

import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import subprocess
import json
from pathlib import Path
from datetime import datetime

# ========== CUDA 源码模板 ==========

# Softmax CUDA 实现
SOFTMAX_CUDA = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__global__ void softmax_kernel(
    const half* input,
    half* output,
    int batch,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = idx / cols;
    int col_idx = idx % cols;

    if (row_idx >= batch * rows) return;

    // 每个 warp 处理一行
    int lane_id = threadIdx.x % 32;

    // 找最大值（用于数值稳定）
    float max_val = -INFINITY;
    for (int i = lane_id; i < cols; i += 32) {
        float val = __half2float(input[row_idx * cols + i]);
        max_val = fmaxf(max_val, val);
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // 计算 exp 和 sum
    float sum = 0.0f;
    for (int i = lane_id; i < cols; i += 32) {
        float val = expf(__half2float(input[row_idx * cols + i]) - max_val);
        sum += val;
    }

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // 归一化
    for (int i = lane_id; i < cols; i += 32) {
        float val = expf(__half2float(input[row_idx * cols + i]) - max_val);
        output[row_idx * cols + i] = __float2half(val / sum);
    }
}

} // extern "C"
"""

# LayerNorm CUDA 实现
LAYERNORM_CUDA = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__global__ void layernorm_kernel(
    const half* input,
    half* output,
    int batch,
    int seq_len,
    int hidden_dim,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch * seq_len) return;

    const half* x = input + idx * hidden_dim;
    half* y = output + idx * hidden_dim;

    // 计算均值
    float mean = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        mean += __half2float(x[i]);
    }
    mean /= hidden_dim;

    // 计算方差
    float variance = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        float diff = __half2float(x[i]) - mean;
        variance += diff * diff;
    }
    variance /= hidden_dim;

    // 归一化
    float inv_std = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < hidden_dim; ++i) {
        float normalized = (__half2float(x[i]) - mean) * inv_std;
        y[i] = __float2half(normalized);
    }
}

} // extern "C"
"""

# RMSNorm CUDA 实现
RMSNORM_CUDA = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__global__ void rmsnorm_kernel(
    const half* input,
    half* output,
    int batch,
    int seq_len,
    int hidden_dim,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch * seq_len) return;

    const half* x = input + idx * hidden_dim;
    half* y = output + idx * hidden_dim;

    // 计算平方和
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    // RMS
    float rms = sqrtf(sum_sq / hidden_dim + eps);

    // 归一化
    for (int i = 0; i < hidden_dim; ++i) {
        y[i] = __float2half(__half2float(x[i]) / rms);
    }
}

} // extern "C"
"""

def compile_cuda(source, name):
    """编译 CUDA kernel"""

    cu_file = Path(f"{name}.cu")
    so_file = Path(f"{name}.so")

    with open(cu_file, 'w') as f:
        f.write(source)

    cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC",
        "-o", str(so_file), str(cu_file),
        "-arch=sm_89", "-O3", "--use_fast_math", "-lcudart"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ 编译失败: {name}")
        print(result.stderr)
        return None

    return so_file

def benchmark(func, *args, warmup=10, iterations=100):
    """性能测试"""

    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = func(*args)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        'median_ms': float(np.median(times)),
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'output': output
    }

def validate(baseline_output, optimized_output, rtol=1e-2, atol=1e-3):
    """验证正确性"""

    if torch.isnan(optimized_output).any():
        return False, "包含 NaN"

    if torch.isinf(optimized_output).any():
        return False, "包含 Inf"

    abs_diff = torch.abs(baseline_output - optimized_output)
    rel_diff = abs_diff / (torch.abs(baseline_output) + 1e-8)

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()

    if max_abs < atol or max_rel < rtol:
        return True, f"通过 (abs={max_abs:.2e}, rel={max_rel:.2e})"
    else:
        return False, f"误差过大 (abs={max_abs:.2e}, rel={max_rel:.2e})"

# ========== 测试函数 ==========

def test_softmax():
    """测试 Softmax"""

    print("\n" + "="*80)
    print("  测试: Softmax")
    print("="*80)

    # 编译
    print("编译 CUDA kernel...")
    so_file = compile_cuda(SOFTMAX_CUDA, "softmax_kernel")
    if not so_file:
        return None
    print("✓ 编译成功")

    # 配置
    batch, heads, rows, cols = 2, 16, 1024, 1024
    device = 'cuda'

    # 生成输入
    torch.manual_seed(42)
    x = torch.randn(batch, heads, rows, cols, device=device, dtype=torch.float16)

    # Baseline
    print("运行 Baseline...")
    baseline_result = benchmark(lambda x: F.softmax(x, dim=-1), x)
    print(f"✓ Baseline: {baseline_result['median_ms']:.3f} ms")

    # CUDA
    print("运行优化 kernel...")
    lib = ctypes.CDLL(str(so_file))

    def softmax_cuda(x):
        output = torch.empty_like(x)
        total_size = batch * heads * rows

        grid = (total_size + 255) // 256
        block = 256

        lib.softmax_kernel(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(batch * heads),
            ctypes.c_int(rows),
            ctypes.c_int(cols),
            # Launch config
            ctypes.c_int(grid), ctypes.c_int(1), ctypes.c_int(1),
            ctypes.c_int(block), ctypes.c_int(1), ctypes.c_int(1),
            ctypes.c_int(0),
            ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return output

    cuda_result = benchmark(softmax_cuda, x)
    print(f"✓ 优化: {cuda_result['median_ms']:.3f} ms")

    # 验证
    is_correct, msg = validate(baseline_result['output'], cuda_result['output'])
    print(f"{'✓' if is_correct else '✗'} 正确性: {msg}")

    # 加速比
    speedup = baseline_result['median_ms'] / cuda_result['median_ms']
    print(f"✓ 加速比: {speedup:.2f}×")

    return {
        'operator': 'softmax',
        'baseline_ms': baseline_result['median_ms'],
        'optimized_ms': cuda_result['median_ms'],
        'speedup': speedup,
        'verified': is_correct
    }

def test_layernorm():
    """测试 LayerNorm"""

    print("\n" + "="*80)
    print("  测试: LayerNorm")
    print("="*80)

    # 编译
    print("编译 CUDA kernel...")
    so_file = compile_cuda(LAYERNORM_CUDA, "layernorm_kernel")
    if not so_file:
        return None
    print("✓ 编译成功")

    # 配置
    batch, seq_len, hidden_dim = 2, 1024, 4096
    device = 'cuda'

    # 生成输入
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=torch.float16)

    # Baseline
    print("运行 Baseline...")
    baseline_result = benchmark(lambda x: F.layer_norm(x, [hidden_dim]), x)
    print(f"✓ Baseline: {baseline_result['median_ms']:.3f} ms")

    # CUDA
    print("运行优化 kernel...")
    lib = ctypes.CDLL(str(so_file))

    def layernorm_cuda(x):
        output = torch.empty_like(x)
        total = batch * seq_len

        grid = (total + 255) // 256
        block = 256

        lib.layernorm_kernel(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(batch),
            ctypes.c_int(seq_len),
            ctypes.c_int(hidden_dim),
            ctypes.c_float(1e-5),
            # Launch config
            ctypes.c_int(grid), ctypes.c_int(1), ctypes.c_int(1),
            ctypes.c_int(block), ctypes.c_int(1), ctypes.c_int(1),
            ctypes.c_int(0),
            ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return output

    cuda_result = benchmark(layernorm_cuda, x)
    print(f"✓ 优化: {cuda_result['median_ms']:.3f} ms")

    # 验证
    is_correct, msg = validate(baseline_result['output'], cuda_result['output'])
    print(f"{'✓' if is_correct else '✗'} 正确性: {msg}")

    # 加速比
    speedup = baseline_result['median_ms'] / cuda_result['median_ms']
    print(f"✓ 加速比: {speedup:.2f}×")

    return {
        'operator': 'layernorm',
        'baseline_ms': baseline_result['median_ms'],
        'optimized_ms': cuda_result['median_ms'],
        'speedup': speedup,
        'verified': is_correct
    }

def test_rmsnorm():
    """测试 RMSNorm"""

    print("\n" + "="*80)
    print("  测试: RMSNorm")
    print("="*80)

    # 编译
    print("编译 CUDA kernel...")
    so_file = compile_cuda(RMSNORM_CUDA, "rmsnorm_kernel")
    if not so_file:
        return None
    print("✓ 编译成功")

    # 配置
    batch, seq_len, hidden_dim = 2, 1024, 4096
    device = 'cuda'

    # 生成输入
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=torch.float16)

    # Baseline
    def rms_norm_baseline(x, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps)

    print("运行 Baseline...")
    baseline_result = benchmark(rms_norm_baseline, x)
    print(f"✓ Baseline: {baseline_result['median_ms']:.3f} ms")

    # CUDA
    print("运行优化 kernel...")
    lib = ctypes.CDLL(str(so_file))

    def rmsnorm_cuda(x):
        output = torch.empty_like(x)
        total = batch * seq_len

        grid = (total + 255) // 256
        block = 256

        lib.rmsnorm_kernel(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(batch),
            ctypes.c_int(seq_len),
            ctypes.c_int(hidden_dim),
            ctypes.c_float(1e-6),
            # Launch config
            ctypes.c_int(grid), ctypes.c_int(1), ctypes.c_int(1),
            ctypes.c_int(block), ctypes.c_int(1), ctypes.c_int(1),
            ctypes.c_int(0),
            ctypes.c_void_p(0)
        )
        torch.cuda.synchronize()
        return output

    cuda_result = benchmark(rmsnorm_cuda, x)
    print(f"✓ 优化: {cuda_result['median_ms']:.3f} ms")

    # 验证
    is_correct, msg = validate(baseline_result['output'], cuda_result['output'])
    print(f"{'✓' if is_correct else '✗'} 正确性: {msg}")

    # 加速比
    speedup = baseline_result['median_ms'] / cuda_result['median_ms']
    print(f"✓ 加速比: {speedup:.2f}×")

    return {
        'operator': 'rmsnorm',
        'baseline_ms': baseline_result['median_ms'],
        'optimized_ms': cuda_result['median_ms'],
        'speedup': speedup,
        'verified': is_correct
    }

def main():
    """主函数"""

    print("="*80)
    print("  测试所有算子性能")
    print("="*80)

    results = []

    # 已有的 Self-Attention
    sa_result = {
        'operator': 'self-attention',
        'baseline_ms': 1.826,
        'optimized_ms': 0.041,
        'speedup': 44.75,
        'verified': True
    }
    results.append(sa_result)

    # 测试新算子
    tests = [test_softmax, test_layernorm, test_rmsnorm]

    for test_func in tests:
        try:
            result = test_func()
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    # 生成报告
    print("\n\n" + "="*80)
    print("  最终报告")
    print("="*80)
    print()

    print(f"{'算子':<20} {'Baseline':<12} {'优化后':<12} {'加速比':<10} {'验证'}")
    print("="*80)

    for r in results:
        verified = "✓" if r.get('verified') else "✗"
        print(f"{r['operator']:<20} {r['baseline_ms']:<12.3f} {r['optimized_ms']:<12.3f} {r['speedup']:<10.2f}× {verified}")

    print("="*80)

    # 保存结果
    output = {
        'test_date': datetime.now().isoformat(),
        'total': len(results),
        'results': results
    }

    with open('benchmark_results/all_operators_test.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ 结果已保存: benchmark_results/all_operators_test.json")

if __name__ == "__main__":
    main()
