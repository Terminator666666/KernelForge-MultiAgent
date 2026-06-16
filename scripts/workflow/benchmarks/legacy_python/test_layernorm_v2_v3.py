"""
测试 LayerNorm Iteration 2 和 3
"""

import torch
import numpy as np
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import ctypes

def compile_kernel(cu_file, arch="sm_89"):
    """编译 CUDA kernel"""

    print(f"\n{'='*80}")
    print(f"  编译 {cu_file}")
    print(f"{'='*80}\n")

    so_file = cu_file.replace('.cu', '.so')

    compile_cmd = [
        "nvcc",
        "-shared",
        "-Xcompiler", "-fPIC",
        "-o", so_file,
        cu_file,
        f"-arch={arch}",
        "-O3",
        "--use_fast_math",
        "-lcudart"
    ]

    print(f"命令: {' '.join(compile_cmd)}\n")

    result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ 编译失败:")
        print(result.stderr)
        return None

    print(f"✓ 编译成功: {so_file}\n")
    return so_file

def test_kernel_version(so_file, func_name, version_name, batch_size=128, hidden_size=4096):
    """测试特定版本的 kernel"""

    print(f"\n{'='*80}")
    print(f"  测试 {version_name}")
    print(f"{'='*80}\n")

    # 加载库
    lib = ctypes.CDLL(so_file)

    try:
        launch_func = getattr(lib, func_name)
    except AttributeError:
        print(f"❌ 找不到函数: {func_name}")
        return None

    launch_func.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
    ]

    # 创建测试数据
    torch.manual_seed(42)
    device = 'cuda'

    input_data = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    gamma = torch.ones(hidden_size, device=device, dtype=torch.float32)
    beta = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    output_custom = torch.zeros_like(input_data)
    eps = 1e-5

    # PyTorch 参考结果
    layer_norm = torch.nn.LayerNorm(hidden_size, eps=eps).to(device)
    layer_norm.weight.data = gamma
    layer_norm.bias.data = beta
    output_ref = layer_norm(input_data)

    # 调用自定义 kernel
    try:
        launch_func(
            input_data.data_ptr(),
            output_custom.data_ptr(),
            gamma.data_ptr(),
            beta.data_ptr(),
            batch_size,
            hidden_size,
            eps,
            0
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"❌ Kernel 调用失败: {e}")
        return None

    # 正确性检查
    diff = torch.abs(output_custom - output_ref)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"正确性:")
    print(f"  最大误差: {max_diff:.6e}")
    print(f"  平均误差: {mean_diff:.6e}")

    threshold = 1e-4
    passed = max_diff < threshold

    if not passed:
        print(f"\n❌ 正确性测试失败")
        return None

    print(f"  ✓ 正确性通过\n")

    # 性能测试
    print(f"性能测试...")

    # Warmup
    for _ in range(100):
        launch_func(
            input_data.data_ptr(), output_custom.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, hidden_size, eps, 0
        )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        launch_func(
            input_data.data_ptr(), output_custom.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, hidden_size, eps, 0
        )
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)
    median_time = np.median(times)

    print(f"  Median: {median_time:.4f} ms\n")

    return {
        'median_ms': float(median_time),
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'max_diff': float(max_diff),
        'passed': passed
    }

def main():
    """主函数"""

    print("\n" + "="*80)
    print("  LayerNorm 多版本测试")
    print("  硬件: RTX 5070 (sm_89)")
    print("="*80)

    # 编译
    cu_file = "layernorm-4096_opt_v2.cu"
    so_file = compile_kernel(cu_file)
    if not so_file:
        return

    # 加载 baseline
    baseline_file = 'layernorm_optimization/baseline.json'
    if not os.path.exists(baseline_file):
        print("❌ 找不到 baseline")
        return

    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
        baseline_time = baseline['median_ms']

    print(f"\nBaseline (PyTorch): {baseline_time:.4f} ms")

    # 测试各版本
    versions = [
        ('launch_layernorm_optimized', 'V2 (float4 + __ldg)'),
        ('launch_layernorm_optimized_v3', 'V3 (float8)')
    ]

    results = {}

    for func_name, version_name in versions:
        result = test_kernel_version(so_file, func_name, version_name)

        if result:
            speedup = baseline_time / result['median_ms']
            print(f"{'='*80}")
            print(f"  {version_name} 结果")
            print(f"{'='*80}")
            print(f"延迟: {result['median_ms']:.4f} ms")
            print(f"加速比: {speedup:.2f}×")

            if speedup > 1.0:
                print(f"🎉 成功加速 {speedup:.2f}×!\n")
            else:
                print(f"⚠️  性能: {speedup:.2f}×\n")

            results[version_name] = {
                **result,
                'speedup': float(speedup),
                'baseline_ms': baseline_time
            }

    # 找到最佳版本
    if results:
        best_version = max(results.items(), key=lambda x: x[1]['speedup'])
        best_name, best_result = best_version

        print(f"\n{'='*80}")
        print(f"  最佳版本: {best_name}")
        print(f"{'='*80}")
        print(f"加速比: {best_result['speedup']:.2f}×")
        print(f"延迟: {best_result['median_ms']:.4f} ms")

        # 保存结果
        result_file = 'layernorm_optimization/iteration_2_3_results.json'
        with open(result_file, 'w') as f:
            json.dump({
                'baseline_ms': baseline_time,
                'versions': results,
                'best_version': best_name,
                'best_speedup': best_result['speedup'],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print(f"\n✓ 结果已保存: {result_file}")

if __name__ == "__main__":
    main()
