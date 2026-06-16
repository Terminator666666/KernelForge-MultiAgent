"""
Agent 优化工具包 - 为多智能体优化提供统一的编译、测试、NCU分析接口

功能：
1. 编译 CUDA kernel
2. 运行性能测试
3. 执行 NCU profiling
4. 解析性能数据
5. 生成优化报告
"""

import subprocess
import json
import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent

class OptimizationToolkit:
    """优化工具包"""

    def __init__(self, kernel_name, work_dir=None):
        """
        初始化工具包

        参数:
            kernel_name: kernel名称（例如：matmul-2048x2048x2048）
            work_dir: 工作目录
        """
        self.kernel_name = kernel_name
        self.work_dir = Path(work_dir) if work_dir else WORKFLOW_ROOT
        self.kernel_file = self._resolve_kernel_file(f"{kernel_name}_kernel.cu")
        self.opt_file = self._resolve_kernel_file(f"{kernel_name}_opt.cu")
        self.so_file = self.work_dir / f"{kernel_name}_opt.so"
        self.ncu_output = self.work_dir / f"{kernel_name}_ncu.txt"

    def _resolve_kernel_file(self, file_name):
        candidates = [
            REPO_ROOT / "kernels" / "generated" / file_name,
            self.work_dir / file_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def compile_kernel(self, source_file=None, output_file=None, arch="sm_89"):
        """
        编译 CUDA kernel

        参数:
            source_file: 源文件路径（默认使用 _opt.cu）
            output_file: 输出文件路径（默认使用 _opt.so）
            arch: GPU架构（RTX 5070 使用 sm_89）

        返回:
            (success, message)
        """
        if source_file is None:
            source_file = self.opt_file
        if output_file is None:
            output_file = self.so_file

        print(f"\n{'='*80}")
        print(f"  编译 Kernel: {source_file.name}")
        print(f"{'='*80}\n")

        # 编译命令
        compile_cmd = [
            "nvcc",
            "-shared",
            "-Xcompiler", "-fPIC",
            "-o", str(output_file),
            str(source_file),
            f"-arch={arch}",
            "-O3",
            "--use_fast_math",
            "-lcudart",
            "-lcublas"
        ]

        print(f"编译命令: {' '.join(compile_cmd)}\n")

        result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=self.work_dir)

        if result.returncode != 0:
            error_msg = f"编译失败:\n{result.stderr}"
            print(f"❌ {error_msg}")
            return False, error_msg

        print(f"✓ 编译成功: {output_file}")
        return True, str(output_file)

    def run_ncu_profile(self, executable=None, metrics=None):
        """
        运行 NCU profiling

        参数:
            executable: 可执行文件或测试脚本
            metrics: 要收集的指标列表

        返回:
            (success, ncu_data)
        """
        if metrics is None:
            # 默认收集的关键指标
            metrics = [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__time_duration.sum",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                "smsp__warp_issue_stalled_barrier_per_warp_active.pct"
            ]

        print(f"\n{'='*80}")
        print(f"  NCU Profiling: {self.kernel_name}")
        print(f"{'='*80}\n")

        # 构建 NCU 命令
        ncu_cmd = [
            "ncu",
            "--metrics", ",".join(metrics),
            "--target-processes", "all",
            "--export", str(self.ncu_output.with_suffix('')),
            "--force-overwrite"
        ]

        if executable:
            ncu_cmd.append(str(executable))

        print(f"NCU 命令: {' '.join(ncu_cmd)}\n")
        print("⏳ 开始 profiling（这可能需要几分钟）...\n")

        result = subprocess.run(ncu_cmd, capture_output=True, text=True, cwd=self.work_dir)

        if result.returncode != 0:
            error_msg = f"NCU profiling 失败:\n{result.stderr}"
            print(f"❌ {error_msg}")
            return False, error_msg

        print(f"✓ NCU profiling 完成")
        print(f"✓ 报告保存: {self.ncu_output}")

        # 解析 NCU 输出
        ncu_data = self._parse_ncu_output(result.stdout)

        return True, ncu_data

    def _parse_ncu_output(self, ncu_stdout):
        """解析 NCU 输出数据"""

        data = {
            "sm_efficiency": 0.0,
            "memory_throughput": 0.0,
            "kernel_time_us": 0.0,
            "warp_efficiency": 0.0,
            "bottleneck": "unknown"
        }

        lines = ncu_stdout.split('\n')

        for line in lines:
            # 解析 SM 效率
            if "sm__throughput" in line:
                try:
                    data["sm_efficiency"] = float(line.split()[-1].replace('%', ''))
                except:
                    pass

            # 解析内存吞吐
            if "dram__throughput" in line:
                try:
                    data["memory_throughput"] = float(line.split()[-1].replace('%', ''))
                except:
                    pass

            # 解析 kernel 时间
            if "gpu__time_duration" in line:
                try:
                    time_str = line.split()[-1]
                    if 'us' in time_str:
                        data["kernel_time_us"] = float(time_str.replace('us', ''))
                    elif 'ms' in time_str:
                        data["kernel_time_us"] = float(time_str.replace('ms', '')) * 1000
                except:
                    pass

            # 解析 warp 效率
            if "sm__warps_active" in line:
                try:
                    data["warp_efficiency"] = float(line.split()[-1].replace('%', ''))
                except:
                    pass

        # 判断瓶颈
        if data["sm_efficiency"] < 50:
            data["bottleneck"] = "compute"
        elif data["memory_throughput"] > 80:
            data["bottleneck"] = "memory"
        else:
            data["bottleneck"] = "balanced"

        return data

    def analyze_bottleneck(self, ncu_data):
        """
        分析性能瓶颈并给出优化建议

        参数:
            ncu_data: NCU profiling 数据

        返回:
            优化建议列表
        """
        suggestions = []

        bottleneck = ncu_data.get("bottleneck", "unknown")
        sm_eff = ncu_data.get("sm_efficiency", 0)
        mem_throughput = ncu_data.get("memory_throughput", 0)
        warp_eff = ncu_data.get("warp_efficiency", 0)

        print(f"\n{'='*80}")
        print(f"  性能瓶颈分析")
        print(f"{'='*80}\n")

        print(f"SM 效率: {sm_eff:.1f}%")
        print(f"内存吞吐: {mem_throughput:.1f}%")
        print(f"Warp 效率: {warp_eff:.1f}%")
        print(f"瓶颈类型: {bottleneck}\n")

        # 根据瓶颈给出建议
        if bottleneck == "compute":
            suggestions.extend([
                "使用 Tensor Core（针对 GEMM 类算子）",
                "增加指令级并行（ILP）",
                "使用 warp-level primitives",
                "优化循环展开",
                "减少分支和同步"
            ])
        elif bottleneck == "memory":
            suggestions.extend([
                "优化 shared memory 访问模式",
                "使用向量化访问（float4）",
                "增加数据重用（tiling）",
                "避免 bank conflict",
                "优化全局内存合并访问"
            ])
        else:
            suggestions.extend([
                "综合优化计算和访存",
                "调整 block size 和 grid size",
                "优化寄存器使用",
                "提高 occupancy"
            ])

        # Warp 效率低的额外建议
        if warp_eff < 60:
            suggestions.append("提高 warp occupancy - 调整 block size")

        print("优化建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

        return suggestions

    def save_optimization_report(self, iteration, ncu_data, suggestions, performance_gain=None):
        """
        保存优化报告

        参数:
            iteration: 迭代轮次
            ncu_data: NCU 数据
            suggestions: 优化建议
            performance_gain: 性能提升（相比上一轮）
        """
        report = {
            "kernel": self.kernel_name,
            "iteration": iteration,
            "timestamp": subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"]).decode().strip(),
            "ncu_metrics": ncu_data,
            "suggestions": suggestions,
            "performance_gain": performance_gain
        }

        report_file = self.work_dir / f"{self.kernel_name}_opt_report_iter{iteration}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ 优化报告已保存: {report_file}")

        return report_file

    def compare_with_baseline(self, opt_time_ms, baseline_file=None):
        """
        与 baseline 对比性能

        参数:
            opt_time_ms: 优化后的时间（毫秒）
            baseline_file: baseline 数据文件

        返回:
            加速比
        """
        # 尝试读取 baseline
        if baseline_file is None:
            baseline_file = self.work_dir / "benchmark_results" / f"{self.kernel_name}_baseline.json"

        if not baseline_file.exists():
            print(f"⚠️  Baseline 文件不存在: {baseline_file}")
            return None

        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)

        baseline_time = baseline_data.get("median_ms", 0)

        if baseline_time == 0:
            print("⚠️  Baseline 时间为 0，无法计算加速比")
            return None

        speedup = baseline_time / opt_time_ms

        print(f"\n{'='*80}")
        print(f"  性能对比")
        print(f"{'='*80}\n")
        print(f"Baseline:  {baseline_time:.3f} ms")
        print(f"Optimized: {opt_time_ms:.3f} ms")
        print(f"加速比:    {speedup:.2f}×\n")

        if speedup > 1.0:
            print(f"🎉 性能提升 {speedup:.2f}× ！")
        elif speedup > 0.95:
            print(f"⚠️  性能相近 ({speedup:.2f}×)")
        else:
            print(f"❌ 性能下降 ({speedup:.2f}×)")

        return speedup


def main():
    """测试工具包"""

    # 示例：MatMul 优化流程
    toolkit = OptimizationToolkit("matmul-2048x2048x2048")

    print("OptimizationToolkit 已初始化")
    print(f"Kernel: {toolkit.kernel_name}")
    print(f"工作目录: {toolkit.work_dir}")
    print(f"源文件: {toolkit.opt_file}")


if __name__ == "__main__":
    main()
