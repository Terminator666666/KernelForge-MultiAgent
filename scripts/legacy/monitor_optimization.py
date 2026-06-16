#!/usr/bin/env python3
"""
实时监控优化进度脚本
每 30 秒更新一次进度
"""

import time
import json
from pathlib import Path
from datetime import datetime

REFERENCE_DIR = Path("reference")
OPERATORS = [
    "self-attention-h16-d64-seq1024",
    "flash-attention-seq2048",
    "fused-moe-fp8-experts8",
    "conv2d-resnet-style",
    "fp8-gemm-2048x2048x2048"
]

def clear_screen():
    import os
    os.system('clear' if os.name != 'nt' else 'cls')

def get_progress():
    """获取当前进度"""
    progress = []

    for op in OPERATORS:
        op_dir = REFERENCE_DIR / op

        if not op_dir.exists():
            progress.append({
                "name": op,
                "status": "pending",
                "files": 0,
                "speedup": "N/A"
            })
            continue

        code_files = list(op_dir.glob("kernel_iter*.cuda"))
        results_file = op_dir / "optimization_results.json"

        status = "running"
        speedup = "N/A"

        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                status = results.get('status', 'running')
                speedup = f"{results.get('best_speedup', 1.0):.2f}×"
            except:
                pass

        progress.append({
            "name": op,
            "status": status,
            "files": len(code_files),
            "speedup": speedup
        })

    return progress

def display_progress(iteration):
    """显示进度"""
    clear_screen()

    print("="*80)
    print("  🔄 优化进度实时监控")
    print(f"  更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  更新次数: {iteration}")
    print("="*80)
    print()

    progress = get_progress()

    # 统计
    completed = sum(1 for p in progress if p['status'] == 'completed')
    running = sum(1 for p in progress if p['status'] == 'running')
    pending = sum(1 for p in progress if p['status'] == 'pending')
    total_files = sum(p['files'] for p in progress)

    print(f"📊 总体进度:")
    print(f"  ✅ 已完成: {completed}/5")
    print(f"  🔄 进行中: {running}/5")
    print(f"  ⏳ 待处理: {pending}/5")
    print(f"  📁 总代码文件: {total_files} 个")
    print()

    # 进度条
    bar_length = 50
    filled = int(bar_length * completed / len(OPERATORS))
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"  [{bar}] {completed}/{len(OPERATORS)}")
    print()

    # 详细信息
    print("📋 算子详情:")
    print("-" * 80)
    print(f"{'序号':<4} {'算子名称':<45} {'状态':<10} {'文件':<6} {'加速比'}")
    print("-" * 80)

    for i, p in enumerate(progress, 1):
        name_short = p['name'] if len(p['name']) <= 43 else p['name'][:40] + "..."

        if p['status'] == 'completed':
            status_icon = "✅"
        elif p['status'] == 'running':
            status_icon = "🔄"
        else:
            status_icon = "⏳"

        print(f"{i:<4} {name_short:<45} {status_icon} {p['status']:<8} {p['files']:<6} {p['speedup']}")

    print("-" * 80)
    print()

    # 提示
    print("💡 提示:")
    print("  - 按 Ctrl+C 停止监控")
    print("  - 查看详细日志: tail -f logs/optimized_run.log")
    print("  - 查看进程状态: ps -p 6173")
    print()

    print("⏳ 下次更新: 30 秒后...")

def main():
    """主循环"""
    iteration = 0

    try:
        while True:
            iteration += 1
            display_progress(iteration)
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n⚠️  监控已停止")
        print("优化任务仍在后台运行")
        print("查看日志: tail -f logs/optimized_run.log")

if __name__ == "__main__":
    main()
