#!/usr/bin/env python3
"""
NCU数据提取和分析工具 - 正确解析.ncu-rep文件
"""

import subprocess
import json
import sys
from pathlib import Path

def extract_ncu_metrics(ncu_report_path):
    """
    从NCU报告中提取关键性能指标

    参数:
        ncu_report_path: .ncu-rep文件路径

    返回:
        包含性能指标的字典
    """

    # 使用NCU导出CSV格式
    cmd = [
        "sudo", "ncu",
        "--import", str(ncu_report_path),
        "--csv"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ NCU导出失败: {result.stderr}")
        return None

    # 解析CSV输出
    lines = result.stdout.strip().split('\n')

    metrics = {
        "dram_throughput": [],
        "sm_throughput": [],
        "warp_active": []
    }

    for line in lines:
        if "dram__throughput.avg.pct_of_peak_sustained_elapsed" in line:
            value = line.split(',')[-1].strip('"')
            metrics["dram_throughput"].append(float(value))
        elif "sm__throughput.avg.pct_of_peak_sustained_elapsed" in line:
            value = line.split(',')[-1].strip('"')
            metrics["sm_throughput"].append(float(value))
        elif "sm__warps_active.avg.pct_of_peak_sustained_active" in line:
            value = line.split(',')[-1].strip('"')
            metrics["warp_active"].append(float(value))

    # 计算平均值（取第一个kernel的数据）
    result_metrics = {
        "dram_throughput_pct": metrics["dram_throughput"][0] if metrics["dram_throughput"] else 0.0,
        "sm_throughput_pct": metrics["sm_throughput"][0] if metrics["sm_throughput"] else 0.0,
        "warp_active_pct": metrics["warp_active"][0] if metrics["warp_active"] else 0.0
    }

    # 判断瓶颈类型
    sm_eff = result_metrics["sm_throughput_pct"]
    mem_throughput = result_metrics["dram_throughput_pct"]

    if sm_eff > 60:
        result_metrics["bottleneck"] = "compute_bound"
        result_metrics["bottleneck_desc"] = "计算密集型 - SM利用率高"
    elif mem_throughput > 70:
        result_metrics["bottleneck"] = "memory_bound"
        result_metrics["bottleneck_desc"] = "内存受限 - 带宽瓶颈"
    else:
        result_metrics["bottleneck"] = "balanced"
        result_metrics["bottleneck_desc"] = "平衡 - 计算和访存均未饱和"

    return result_metrics

def generate_optimization_suggestions(metrics):
    """根据NCU指标生成优化建议"""

    suggestions = []

    bottleneck = metrics["bottleneck"]
    sm_eff = metrics["sm_throughput_pct"]
    mem_throughput = metrics["dram_throughput_pct"]
    warp_active = metrics["warp_active_pct"]

    if bottleneck == "compute_bound":
        suggestions.append(f"✅ SM吞吐已达到{sm_eff:.1f}% - 计算密集型特征明显")
        suggestions.append("🔧 继续提高计算效率:")
        suggestions.append("  - 增加指令级并行(ILP)")
        suggestions.append("  - 优化循环展开")
        suggestions.append("  - 减少分支divergence")

        if warp_active < 50:
            suggestions.append(f"⚠️ Warp活跃度仅{warp_active:.1f}% - 可提高occupancy")
            suggestions.append("  - 减少寄存器使用")
            suggestions.append("  - 优化shared memory配置")
            suggestions.append("  - 调整block size")

    elif bottleneck == "memory_bound":
        suggestions.append(f"⚠️ 内存吞吐{mem_throughput:.1f}% - 带宽受限")
        suggestions.append("🔧 优化访存模式:")
        suggestions.append("  - 增加数据重用(tiling)")
        suggestions.append("  - 使用shared memory缓存")
        suggestions.append("  - 向量化访存(float4)")
        suggestions.append("  - 消除bank conflict")

    else:
        suggestions.append("✅ 计算和访存较为平衡")
        suggestions.append("🔧 综合优化方向:")
        suggestions.append("  - 提高SM利用率")
        suggestions.append("  - 优化访存效率")
        suggestions.append("  - 调整并行粒度")

    if mem_throughput < 10:
        suggestions.append(f"✅ DRAM吞吐仅{mem_throughput:.1f}% - 访存压力小，适合继续增加计算")

    return suggestions

def main():
    if len(sys.argv) < 2:
        print("用法: python3 extract_ncu_metrics.py <ncu_report.ncu-rep>")
        sys.exit(1)

    ncu_report = Path(sys.argv[1])

    if not ncu_report.exists():
        print(f"❌ NCU报告不存在: {ncu_report}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  提取NCU性能指标")
    print(f"  报告: {ncu_report.name}")
    print(f"{'='*80}\n")

    # 提取指标
    metrics = extract_ncu_metrics(ncu_report)

    if not metrics:
        print("❌ 指标提取失败")
        sys.exit(1)

    # 打印指标
    print("📊 性能指标:")
    print(f"  SM吞吐:      {metrics['sm_throughput_pct']:.2f}%")
    print(f"  DRAM吞吐:    {metrics['dram_throughput_pct']:.2f}%")
    print(f"  Warp活跃度:  {metrics['warp_active_pct']:.2f}%")
    print(f"  瓶颈类型:    {metrics['bottleneck']} ({metrics['bottleneck_desc']})")
    print()

    # 生成优化建议
    suggestions = generate_optimization_suggestions(metrics)

    print("💡 优化建议:")
    for suggestion in suggestions:
        print(f"  {suggestion}")
    print()

    # 保存为JSON
    output_file = ncu_report.with_suffix('.metrics.json')

    output_data = {
        "ncu_report": str(ncu_report),
        "metrics": metrics,
        "suggestions": suggestions
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 指标已保存: {output_file}")
    print()

if __name__ == "__main__":
    main()
