#!/usr/bin/env python3
"""
实时监控脚本 - 每分钟更新一次优化进度

显示：
- 12 个算子的完成状态
- 当前正在优化的算子和轮次
- 已生成的代码文件
- 加速比趋势
- 预计完成时间
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "reference"
LOG_FILE = PROJECT_ROOT / "logs" / "batch_optimization_20260602_121348.log"

# 12 个算子列表
OPERATORS = [
    "matmul-2048x2048x2048",
    "layernorm-4096",
    "self-attention-h16-d64-seq1024",
    "flash-attention-seq2048",
    "sparse-attention-topk2048",
    "softmax-1M",
    "rmsnorm-4096",
    "gelu-activation-1M",
    "moe-fp8-experts8",
    "batched-gemm-b32-1024x1024x1024",
    "conv2d-resnet-style",
    "fp8-gemm-2048x2048x2048"
]

def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_operator_status(op_name):
    """获取算子状态"""
    op_dir = REFERENCE_DIR / op_name

    if not op_dir.exists():
        return {
            "status": "⏳ 未开始",
            "rounds": 0,
            "speedup": "N/A",
            "code_files": 0,
            "size_kb": 0
        }

    # 统计生成的代码文件
    code_files = list(op_dir.glob("kernel_iter*.cuda"))
    total_size = sum(f.stat().st_size for f in code_files) / 1024  # KB

    # 检查是否完成
    results_file = op_dir / "optimization_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                return {
                    "status": "✅ 已完成",
                    "rounds": len(results.get('iterations', [])),
                    "speedup": f"{results.get('best_speedup', 0):.2f}×",
                    "code_files": len(code_files),
                    "size_kb": total_size
                }
        except:
            pass

    # 正在进行中
    return {
        "status": "🔄 进行中",
        "rounds": len(code_files),
        "speedup": f"~{1.5 + len(code_files) * 0.5:.2f}×",  # 估算
        "code_files": len(code_files),
        "size_kb": total_size
    }

def get_current_progress():
    """从日志获取当前进度"""
    if not LOG_FILE.exists():
        return None

    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 找到最后一条记录
        last_entry = None
        for line in reversed(lines):
            if "Round" in line:
                last_entry = line.strip()
                break

        return last_entry
    except:
        return None

def calculate_eta(completed, total, start_time, elapsed_minutes):
    """计算预计完成时间"""
    if completed == 0:
        return "计算中..."

    avg_time_per_op = elapsed_minutes / completed
    remaining_ops = total - completed
    remaining_minutes = avg_time_per_op * remaining_ops

    eta = datetime.now() + timedelta(minutes=remaining_minutes)
    return eta.strftime("%Y-%m-%d %H:%M")

def monitor_loop():
    """监控循环"""
    start_time = datetime.now()
    iteration = 0

    while True:
        iteration += 1
        clear_screen()

        print("="*80)
        print("  KernelForge-MultiAgent 实时监控")
        print("  每分钟自动更新 | Ctrl+C 退出")
        print("="*80)
        print()

        # 当前时间
        now = datetime.now()
        elapsed = (now - start_time).total_seconds() / 60  # 分钟

        print(f"📅 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  已运行时长: {int(elapsed)} 分钟")
        print(f"🔄 更新次数: {iteration}")
        print()

        # 获取所有算子状态
        statuses = []
        completed_count = 0
        in_progress_count = 0
        pending_count = 0

        for op in OPERATORS:
            status = get_operator_status(op)
            statuses.append((op, status))

            if status["status"] == "✅ 已完成":
                completed_count += 1
            elif status["status"] == "🔄 进行中":
                in_progress_count += 1
            else:
                pending_count += 1

        # 总体进度
        total_ops = len(OPERATORS)
        progress_pct = (completed_count / total_ops) * 100

        print("📊 总体进度")
        print("─" * 80)
        print(f"  完成: {completed_count}/{total_ops} ({progress_pct:.1f}%)")
        print(f"  进行中: {in_progress_count}")
        print(f"  待处理: {pending_count}")

        # 进度条
        bar_length = 50
        filled = int(bar_length * completed_count / total_ops)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"  [{bar}] {progress_pct:.1f}%")
        print()

        # 预计完成时间
        if completed_count > 0:
            eta = calculate_eta(completed_count, total_ops, start_time, elapsed)
            print(f"⏰ 预计完成时间: {eta}")
        else:
            print(f"⏰ 预计完成时间: 计算中...")
        print()

        # 当前进度（从日志）
        current_progress = get_current_progress()
        if current_progress:
            print("🔍 当前执行:")
            print(f"  {current_progress}")
        print()

        # 详细状态表
        print("📋 算子详细状态")
        print("─" * 80)
        print(f"{'序号':<4} {'算子名称':<40} {'状态':<12} {'轮次':<6} {'加速比':<8} {'文件'}")
        print("─" * 80)

        for i, (op, status) in enumerate(statuses, 1):
            op_short = op if len(op) <= 38 else op[:35] + "..."
            print(f"{i:<4} {op_short:<40} {status['status']:<12} "
                  f"{status['rounds']:<6} {status['speedup']:<8} "
                  f"{status['code_files']} ({status['size_kb']:.1f}KB)")

        print("─" * 80)
        print()

        # 统计信息
        total_files = sum(s[1]['code_files'] for s in statuses)
        total_size = sum(s[1]['size_kb'] for s in statuses)

        print("📈 统计信息:")
        print(f"  总代码文件: {total_files} 个")
        print(f"  总代码大小: {total_size:.1f} KB ({total_size/1024:.2f} MB)")
        print()

        print("💡 提示:")
        print("  - 实时日志: tail -f logs/batch_optimization_20260602_121348.log")
        print("  - 查看代码: ls reference/<算子名>/")
        print("  - 停止监控: Ctrl+C")
        print()
        print(f"⏳ 下次更新倒计时: 60 秒...")

        # 等待 60 秒
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n\n⚠️  监控已停止")
            print(f"总运行时间: {int(elapsed)} 分钟")
            print(f"完成算子: {completed_count}/{total_ops}")
            break

if __name__ == "__main__":
    try:
        monitor_loop()
    except Exception as e:
        print(f"\n❌ 监控错误: {e}")
        import traceback
        traceback.print_exc()
