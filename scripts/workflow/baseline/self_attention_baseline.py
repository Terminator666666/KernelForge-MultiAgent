#!/usr/bin/env python3
"""
Self-Attention Baseline - PyTorch 实现

用于对比生成的优化 CUDA kernel 的性能
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import argparse

def self_attention_baseline(Q, K, V, scale=None):
    """
    标准 Self-Attention 实现

    Args:
        Q: [batch, heads, seq_len, head_dim]
        K: [batch, heads, seq_len, head_dim]
        V: [batch, heads, seq_len, head_dim]
        scale: 缩放因子，默认为 1/sqrt(head_dim)

    Returns:
        output: [batch, heads, seq_len, head_dim]
    """
    if scale is None:
        scale = 1.0 / (Q.size(-1) ** 0.5)

    # Attention scores: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Output: Attention @ V
    output = torch.matmul(attn_weights, V)

    return output


def benchmark(func, *args, warmup=10, iterations=100, **kwargs):
    """
    性能测试函数

    Args:
        func: 要测试的函数
        *args: 函数参数
        warmup: 预热次数
        iterations: 测试次数
        **kwargs: 函数关键字参数

    Returns:
        dict: 包含性能统计和输出的字典
    """
    device = args[0].device

    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = func(*args, **kwargs)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)

    return {
        'median_ms': np.median(times),
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'output': output
    }


def main():
    parser = argparse.ArgumentParser(description='Self-Attention Baseline Benchmark')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--head-dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'float32'], help='Data type')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--save', type=str, default=None, help='Save results to file')

    args = parser.parse_args()

    # 配置
    batch = args.batch
    heads = args.heads
    seq_len = args.seq_len
    head_dim = args.head_dim
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32

    device = 'cuda'

    print("="*80)
    print("  Self-Attention Baseline Benchmark")
    print("="*80)
    print(f"Configuration:")
    print(f"  Batch size: {batch}")
    print(f"  Heads: {heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Data type: {args.dtype}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 生成随机输入
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    print("Running benchmark...")
    result = benchmark(
        self_attention_baseline, Q, K, V,
        warmup=args.warmup,
        iterations=args.iterations
    )

    print()
    print("="*80)
    print("  Results")
    print("="*80)
    print(f"Latency:")
    print(f"  Median: {result['median_ms']:.3f} ms")
    print(f"  Mean:   {result['mean_ms']:.3f} ms")
    print(f"  Std:    {result['std_ms']:.3f} ms")
    print(f"  Min:    {result['min_ms']:.3f} ms")
    print(f"  Max:    {result['max_ms']:.3f} ms")
    print()

    # 计算 FLOPS
    # Self-Attention FLOPs: 4 * batch * heads * seq_len^2 * head_dim
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    tflops = (flops / 1e12) / (result['median_ms'] / 1000)
    print(f"Throughput: {tflops:.2f} TFLOPS")
    print()

    # 保存结果
    if args.save:
        results_dict = {
            'config': {
                'batch': batch,
                'heads': heads,
                'seq_len': seq_len,
                'head_dim': head_dim,
                'dtype': args.dtype,
                'gpu': torch.cuda.get_device_name(0)
            },
            'performance': {
                'median_ms': float(result['median_ms']),
                'mean_ms': float(result['mean_ms']),
                'std_ms': float(result['std_ms']),
                'min_ms': float(result['min_ms']),
                'max_ms': float(result['max_ms']),
                'tflops': float(tflops)
            }
        }

        import json
        with open(args.save, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to: {args.save}")

        # 同时保存输入和输出用于验证
        output_file = args.save.replace('.json', '_data.pt')
        torch.save({
            'Q': Q.cpu(),
            'K': K.cpu(),
            'V': V.cpu(),
            'output': result['output'].cpu()
        }, output_file)
        print(f"Input/Output saved to: {output_file}")


if __name__ == "__main__":
    main()
