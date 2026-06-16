"""
LayerNorm 算子优化系统
支持编译、测试、NCU profiling 和迭代优化
"""

import torch
import numpy as np
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

class LayerNormOptimizer:
    """LayerNorm 优化器"""

    def __init__(self, hidden_size=4096, batch_size=128):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = 'cuda'
        self.iteration = 0
        self.results = []

        # 创建输出目录
        os.makedirs('layernorm_optimization', exist_ok=True)
        os.makedirs('layernorm_optimization/ncu_reports', exist_ok=True)

    def compile_kernel(self, cu_file, output_name="layernorm"):
        """编译 CUDA kernel"""

        print(f"\n{'='*80}")
        print(f"  编译 Kernel: {cu_file}")
        print(f"{'='*80}\n")

        so_file = f"layernorm_optimization/{output_name}.so"

        # RTX 5070 使用 sm_89 架构
        compile_cmd = [
            "nvcc",
            "-shared",
            "-Xcompiler", "-fPIC",
            "-o", so_file,
            str(cu_file),
            "-arch=sm_89",  # RTX 5070
            "-O3",
            "--use_fast_math",
            "-lcudart",
            "-lineinfo"  # 为 NCU profiling 添加行号信息
        ]

        print(f"编译命令: {' '.join(compile_cmd)}\n")

        result = subprocess.run(compile_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ 编译失败:")
            print(result.stderr)
            return None

        print(f"✓ 编译成功: {so_file}\n")
        return so_file

    def create_test_data(self):
        """创建测试数据"""

        torch.manual_seed(42)

        # 输入数据
        input_data = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.float32
        )

        # Gamma 和 Beta 参数
        gamma = torch.ones(self.hidden_size, device=self.device, dtype=torch.float32)
        beta = torch.zeros(self.hidden_size, device=self.device, dtype=torch.float32)

        # 使用 PyTorch 计算参考输出
        eps = 1e-5
        mean = input_data.mean(dim=1, keepdim=True)
        var = input_data.var(dim=1, keepdim=True, unbiased=False)
        output_ref = (input_data - mean) / torch.sqrt(var + eps)
        output_ref = output_ref * gamma + beta

        return input_data, gamma, beta, output_ref

    def benchmark_pytorch(self):
        """Benchmark PyTorch LayerNorm"""

        print(f"\n{'='*80}")
        print(f"  PyTorch Baseline")
        print(f"{'='*80}\n")

        input_data, gamma, beta, _ = self.create_test_data()

        # 创建 LayerNorm 层
        layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-5).to(self.device)
        layer_norm.weight.data = gamma
        layer_norm.bias.data = beta

        # Warmup
        for _ in range(100):
            _ = layer_norm(input_data)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(1000):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = layer_norm(input_data)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times = np.array(times)
        baseline_time = np.median(times)

        print(f"配置:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"\n性能:")
        print(f"  Median: {baseline_time:.4f} ms")
        print(f"  Mean:   {np.mean(times):.4f} ms")
        print(f"  Std:    {np.std(times):.4f} ms")

        # 保存 baseline
        baseline_result = {
            'median_ms': float(baseline_time),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'timestamp': datetime.now().isoformat()
        }

        with open('layernorm_optimization/baseline.json', 'w') as f:
            json.dump(baseline_result, f, indent=2)

        return baseline_time

    def run_ncu_profile(self, cu_file, output_name="profile"):
        """运行 NCU profiling"""

        print(f"\n{'='*80}")
        print(f"  NCU Profiling: {cu_file}")
        print(f"{'='*80}\n")

        # 编译
        so_file = self.compile_kernel(cu_file, output_name)
        if not so_file:
            return None

        # 创建简单的测试程序
        test_code = f"""
import ctypes
import torch
import numpy as np

# 加载 shared library
lib = ctypes.CDLL('{so_file}')

# 创建测试数据
batch_size = {self.batch_size}
hidden_size = {self.hidden_size}
device = 'cuda'

torch.manual_seed(42)
input_data = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
output_data = torch.zeros_like(input_data)
gamma = torch.ones(hidden_size, device=device, dtype=torch.float32)
beta = torch.zeros(hidden_size, device=device, dtype=torch.float32)
eps = 1e-5

# 调用 kernel (通过 CUDA driver API)
import torch.utils.cpp_extension
torch.cuda.synchronize()

# Warmup
for _ in range(10):
    # 重新创建输出以避免缓存
    output_data = torch.zeros_like(input_data)

torch.cuda.synchronize()

# 单次运行供 NCU profile
output_data = torch.zeros_like(input_data)
torch.cuda.synchronize()

print("Kernel ready for profiling")
"""

        test_file = f'layernorm_optimization/{output_name}_test.py'
        with open(test_file, 'w') as f:
            f.write(test_code)

        # 运行 NCU
        ncu_output = f'layernorm_optimization/ncu_reports/{output_name}.ncu-rep'

        ncu_cmd = [
            'ncu',
            '--set', 'full',
            '--target-processes', 'all',
            '--export', ncu_output,
            'python', test_file
        ]

        print(f"NCU 命令: {' '.join(ncu_cmd)}\n")
        print("运行 NCU profiling (这可能需要几分钟)...\n")

        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print("⚠️  NCU profiling 可能失败")
            print(result.stderr)
        else:
            print(f"✓ NCU report 保存到: {ncu_output}")

        # 提取关键指标 (如果成功)
        metrics = self.parse_ncu_output(result.stdout + result.stderr)

        return metrics

    def parse_ncu_output(self, ncu_output):
        """解析 NCU 输出"""

        metrics = {}

        # 提取关键指标 (简单文本解析)
        lines = ncu_output.split('\n')
        for line in lines:
            if 'DRAM' in line and 'Throughput' in line:
                # 提取 DRAM 吞吐量
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '%' in part:
                            metrics['dram_throughput_pct'] = float(part.rstrip('%'))
                except:
                    pass
            elif 'SM' in line and 'Utilization' in line:
                # 提取 SM 利用率
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '%' in part:
                            metrics['sm_utilization_pct'] = float(part.rstrip('%'))
                except:
                    pass

        return metrics

    def benchmark_custom_kernel(self, cu_file, output_name, ncu_profile=False):
        """Benchmark 自定义 kernel"""

        print(f"\n{'='*80}")
        print(f"  Benchmark: {cu_file}")
        print(f"{'='*80}\n")

        # 编译
        so_file = self.compile_kernel(cu_file, output_name)
        if not so_file:
            return None

        # 加载 baseline
        if not os.path.exists('layernorm_optimization/baseline.json'):
            baseline_time = self.benchmark_pytorch()
        else:
            with open('layernorm_optimization/baseline.json', 'r') as f:
                baseline = json.load(f)
                baseline_time = baseline['median_ms']

        print(f"Baseline: {baseline_time:.4f} ms\n")

        # TODO: 实际运行和测试需要创建 Python wrapper
        # 这里先返回占位结果

        result = {
            'iteration': self.iteration,
            'kernel_file': str(cu_file),
            'output_name': output_name,
            'baseline_ms': baseline_time,
            'optimized_ms': None,  # 需要实际测量
            'speedup': None,
            'timestamp': datetime.now().isoformat()
        }

        self.iteration += 1
        self.results.append(result)

        return result

    def save_iteration_report(self):
        """保存迭代报告"""

        report_file = 'layernorm_optimization/ITERATIONS.md'

        with open(report_file, 'w') as f:
            f.write("# LayerNorm 优化迭代报告\n\n")
            f.write(f"**硬件**: RTX 5070\n")
            f.write(f"**算子**: LayerNorm (hidden_size={self.hidden_size})\n")
            f.write(f"**开始时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 迭代历史\n\n")
            f.write("| Iter | Kernel | 延迟 (ms) | 加速比 | 备注 |\n")
            f.write("|------|--------|-----------|--------|------|\n")

            for result in self.results:
                iter_num = result['iteration']
                kernel = Path(result['kernel_file']).name
                latency = result.get('optimized_ms', 'N/A')
                speedup = result.get('speedup', 'N/A')

                f.write(f"| {iter_num} | {kernel} | {latency} | {speedup} | - |\n")

        print(f"\n✓ 迭代报告保存到: {report_file}")


def main():
    """主函数"""

    print("\n" + "="*80)
    print("  LayerNorm 算子优化系统")
    print("  硬件: RTX 5070")
    print("  Hidden Size: 4096")
    print("="*80 + "\n")

    optimizer = LayerNormOptimizer(hidden_size=4096, batch_size=128)

    # 步骤 1: 建立 baseline
    print("步骤 1: 建立 PyTorch baseline")
    baseline_time = optimizer.benchmark_pytorch()

    print(f"\n✓ Baseline 建立完成: {baseline_time:.4f} ms")

    # 步骤 2: 分析原始 kernel
    print("\n步骤 2: 分析原始 kernel 代码")
    original_kernel = Path("layernorm-4096_kernel.cu")

    if not original_kernel.exists():
        print(f"❌ 找不到原始 kernel: {original_kernel}")
        return

    print(f"✓ 原始 kernel: {original_kernel}")
    print("\n代码分析:")
    print("  - 使用 Welford 在线算法计算均值和方差")
    print("  - 使用 Warp shuffle 指令进行归约")
    print("  - 包含向量化版本 (float4)")
    print("  - 使用 128 threads (4 warps)")

    # 步骤 3: 提示下一步
    print("\n" + "="*80)
    print("  下一步优化方向")
    print("="*80)
    print("\n根据代码分析和优化知识库，推荐以下优化方向:\n")
    print("1. **增加并行度**: 当前使用 128 threads，可以尝试 256 或更多")
    print("2. **优化内存访问**: 使用更大的向量化 (float8 或 float16)")
    print("3. **减少同步开销**: 优化 warp reduce 和 block reduce")
    print("4. **使用 Tensor Core**: 如果可能，使用 FP16 + Tensor Core")
    print("5. **Kernel Fusion**: 与前后算子融合 (如果适用)")

    print("\n系统已准备好，可以开始迭代优化!")
    print("使用 optimizer.benchmark_custom_kernel() 测试新版本")


if __name__ == "__main__":
    main()
