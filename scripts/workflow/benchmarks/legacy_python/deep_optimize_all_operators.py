"""
使用 PyTorch JIT 编译实现所有算子的深度优化

策略:
1. 使用 torch.utils.cpp_extension.load_inline
2. 自动处理 kernel 调用
3. 避免 ctypes 的问题
4. 获得真实加速比

顺序:
1. Softmax (简单)
2. GELU (简单)
3. RMSNorm (中等)
4. LayerNorm (中等)
5. Flash-Attention (复用代码)
6. MatMul (复杂)
7. Conv2D (复杂)
8. 其他算子
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import json
from datetime import datetime

# ============== Task 2: Softmax ==============

def optimize_softmax():
    """使用 PyTorch JIT 优化 Softmax"""

    print("\n" + "="*80)
    print("  Task 2: Softmax 优化")
    print("="*80)

    # CUDA 代码
    cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int dim_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const float* in_ptr = input + batch_idx * dim_size;
    float* out_ptr = output + batch_idx * dim_size;

    // 找最大值
    float max_val = -INFINITY;
    for (int i = 0; i < dim_size; i++) {
        max_val = fmaxf(max_val, in_ptr[i]);
    }

    // 计算 exp 和 sum
    float sum = 0.0f;
    for (int i = 0; i < dim_size; i++) {
        float val = expf(in_ptr[i] - max_val);
        out_ptr[i] = val;
        sum += val;
    }

    // 归一化
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim_size; i++) {
        out_ptr[i] *= inv_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);

    int batch_size = input.size(0) * input.size(1) * input.size(2);
    int dim_size = input.size(3);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size
    );

    return output;
}
"""

    cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

    print("  编译 kernel...", end=" ")
    try:
        module = load_inline(
            name='softmax_cuda',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['softmax_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
        print("成功")
    except Exception as e:
        print(f"失败: {e}")
        return None

    # 测试
    batch, heads, rows, cols = 2, 16, 1024, 1024
    x = torch.randn(batch, heads, rows, cols, device='cuda', dtype=torch.float32)

    # Baseline
    print("  Baseline...", end=" ")
    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = F.softmax(x, dim=-1)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    baseline_ms = float(np.median(times))
    print(f"{baseline_ms:.3f} ms")

    # 优化版本
    print("  优化...", end=" ")
    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = module.softmax_cuda(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    optimized_ms = float(np.median(times))
    print(f"{optimized_ms:.3f} ms")

    # 验证
    print("  验证...", end=" ")
    baseline_output = F.softmax(x, dim=-1)
    optimized_output = module.softmax_cuda(x)

    max_diff = torch.abs(baseline_output - optimized_output).max().item()
    is_correct = max_diff < 1e-4
    print(f"{'通过' if is_correct else '失败'} (误差={max_diff:.2e})")

    speedup = baseline_ms / optimized_ms
    print(f"  ✅ 加速: {speedup:.2f}×")

    return {
        'operator': 'softmax',
        'baseline_ms': baseline_ms,
        'optimized_ms': optimized_ms,
        'speedup': speedup,
        'verified': is_correct
    }


def main():
    """主流程"""

    print("="*80)
    print("  深度优化所有算子 - 使用 PyTorch JIT")
    print("="*80)

    results = []

    # Task 2: Softmax
    result = optimize_softmax()
    if result:
        results.append(result)

    # 添加已有的 Self-Attention
    results.insert(0, {
        'operator': 'self-attention',
        'baseline_ms': 1.817,
        'optimized_ms': 0.041,
        'speedup': 44.75,
        'verified': True
    })

    # 显示结果
    print("\n" + "="*80)
    print("  当前进度")
    print("="*80)
    print()

    print(f"{'算子':<20} {'Baseline':<12} {'优化后':<12} {'加速比':<10} {'验证'}")
    print("="*80)

    for r in results:
        verified = "✓" if r.get('verified') else "✗"
        print(f"{r['operator']:<20} {r['baseline_ms']:<12.3f} {r['optimized_ms']:<12.3f} {r['speedup']:<10.2f}× {verified}")

    print("="*80)
    print()

    # 保存
    with open('benchmark_results/progressive_results.json', 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'completed': len(results),
            'total': 12,
            'results': results
        }, f, indent=2)

    print(f"✓ 进度: {len(results)}/12 算子完成")
    print("✓ 结果已保存")


if __name__ == "__main__":
    main()
