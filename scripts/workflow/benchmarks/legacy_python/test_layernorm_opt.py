"""
LayerNorm 优化版本测试
编译、运行、benchmark 和 NCU profiling
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

def test_correctness(so_file, batch_size=128, hidden_size=4096):
    """测试正确性"""

    print(f"\n{'='*80}")
    print(f"  正确性测试")
    print(f"{'='*80}\n")

    # 加载库
    lib = ctypes.CDLL(so_file)

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
    # 使用 CUDA runtime API 启动 kernel
    try:
        # 获取函数指针
        launch_func = lib.launch_layernorm_optimized

        # 设置参数类型
        launch_func.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # gamma
            ctypes.c_void_p,  # beta
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # hidden_size
            ctypes.c_float,   # eps
            ctypes.c_void_p   # stream
        ]

        # 调用
        launch_func(
            input_data.data_ptr(),
            output_custom.data_ptr(),
            gamma.data_ptr(),
            beta.data_ptr(),
            batch_size,
            hidden_size,
            eps,
            0  # default stream
        )

        torch.cuda.synchronize()

    except Exception as e:
        print(f"❌ Kernel 调用失败: {e}")
        return False

    # 比较结果
    diff = torch.abs(output_custom - output_ref)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"误差统计:")
    print(f"  最大误差: {max_diff:.6e}")
    print(f"  平均误差: {mean_diff:.6e}")

    # 判断是否通过
    threshold = 1e-4
    passed = max_diff < threshold

    if passed:
        print(f"\n✓ 正确性测试通过 (阈值: {threshold})")
    else:
        print(f"\n❌ 正确性测试失败 (最大误差 {max_diff:.6e} > 阈值 {threshold})")

    return passed

def benchmark_kernel(so_file, batch_size=128, hidden_size=4096, num_warmup=100, num_runs=1000):
    """Benchmark kernel 性能"""

    print(f"\n{'='*80}")
    print(f"  性能测试")
    print(f"{'='*80}\n")

    # 加载库
    lib = ctypes.CDLL(so_file)
    launch_func = lib.launch_layernorm_optimized

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
    output_data = torch.zeros_like(input_data)
    eps = 1e-5

    # Warmup
    print(f"Warmup ({num_warmup} 次)...")
    for _ in range(num_warmup):
        launch_func(
            input_data.data_ptr(), output_data.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, hidden_size, eps, 0
        )
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmark ({num_runs} 次)...")
    times = []

    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        launch_func(
            input_data.data_ptr(), output_data.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, hidden_size, eps, 0
        )
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)

    median_time = np.median(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n延迟:")
    print(f"  Median: {median_time:.4f} ms")
    print(f"  Mean:   {mean_time:.4f} ms")
    print(f"  Std:    {std_time:.4f} ms")

    return median_time, mean_time, std_time

def main():
    """主函数"""

    print("\n" + "="*80)
    print("  LayerNorm 优化版本测试")
    print("  硬件: RTX 5070 (sm_89)")
    print("="*80)

    # 配置
    cu_file = "layernorm-4096_opt.cu"
    batch_size = 128
    hidden_size = 4096

    # 1. 编译
    so_file = compile_kernel(cu_file)
    if not so_file:
        return

    # 2. 测试正确性
    passed = test_correctness(so_file, batch_size, hidden_size)
    if not passed:
        print("\n⚠️  正确性测试未通过，停止性能测试")
        return

    # 3. 性能测试
    median_time, mean_time, std_time = benchmark_kernel(so_file, batch_size, hidden_size)

    # 4. 对比 baseline
    baseline_file = 'layernorm_optimization/baseline.json'
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
            baseline_time = baseline['median_ms']

        speedup = baseline_time / median_time

        print(f"\n{'='*80}")
        print(f"  加速比分析")
        print(f"{'='*80}")
        print(f"\nBaseline (PyTorch): {baseline_time:.4f} ms")
        print(f"Optimized (Custom): {median_time:.4f} ms")
        print(f"加速比: {speedup:.2f}×")

        if speedup > 1.0:
            print(f"\n🎉 成功加速 {speedup:.2f}×!")
        else:
            print(f"\n⚠️  性能: {speedup:.2f}× (需要进一步优化)")

        # 保存结果
        result = {
            'kernel_file': cu_file,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
            'baseline_ms': baseline_time,
            'optimized_median_ms': float(median_time),
            'optimized_mean_ms': float(mean_time),
            'optimized_std_ms': float(std_time),
            'speedup': float(speedup),
            'correctness_passed': passed,
            'timestamp': datetime.now().isoformat()
        }

        result_file = 'layernorm_optimization/iteration_1_result.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ 结果已保存: {result_file}")

    else:
        print("\n⚠️  未找到 baseline 结果")

if __name__ == "__main__":
    main()
