"""
多智能体协调器 - 监控和汇总3个Agent的优化进度

功能：
1. 监控3个Agent的工作状态
2. 汇总优化结果
3. 生成对比报告
4. 可视化性能提升
"""

import json
import time
from pathlib import Path
from datetime import datetime
import subprocess

WORKFLOW_ROOT = Path(__file__).resolve().parent

class MultiAgentCoordinator:
    """多智能体协调器"""

    def __init__(self, work_dir=None):
        """初始化协调器"""
        self.work_dir = Path(work_dir) if work_dir else WORKFLOW_ROOT

        # 3个优化目标
        self.agents = {
            "LayerNorm": {
                "kernel": "layernorm-4096",
                "opt_file": "layernorm-4096_opt.cu",
                "status": "pending",
                "iterations": 0,
                "speedup": 0.0,
                "best_time_ms": None
            },
            "MatMul": {
                "kernel": "matmul-2048x2048x2048",
                "opt_file": "matmul-2048x2048x2048_opt.cu",
                "status": "pending",
                "iterations": 0,
                "speedup": 0.0,
                "best_time_ms": None
            },
            "FlashAttention": {
                "kernel": "flash-attention-seq2048",
                "opt_file": "flash-attention-seq2048_opt.cu",
                "status": "pending",
                "iterations": 0,
                "speedup": 0.0,
                "best_time_ms": None
            }
        }

    def check_agent_status(self, agent_name):
        """
        检查单个Agent的状态

        参数:
            agent_name: Agent名称

        返回:
            状态信息字典
        """
        agent = self.agents[agent_name]
        opt_file = self.work_dir / agent["opt_file"]

        # 检查优化文件是否生成
        if opt_file.exists():
            agent["status"] = "working"

            # 检查是否有优化报告
            report_files = list(self.work_dir.glob(f"{agent['kernel']}_opt_report_iter*.json"))
            agent["iterations"] = len(report_files)

            # 读取最新的报告
            if report_files:
                latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)

                agent["last_ncu"] = report_data.get("ncu_metrics", {})
                agent["last_update"] = report_data.get("timestamp", "unknown")

                if "performance_gain" in report_data and report_data["performance_gain"]:
                    agent["speedup"] = report_data["performance_gain"].get("speedup", 0.0)

        return agent

    def monitor_all_agents(self):
        """监控所有Agent的进度"""

        print(f"\n{'='*100}")
        print(f"  多智能体优化进度监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}\n")

        for agent_name in self.agents.keys():
            agent = self.check_agent_status(agent_name)

            print(f"【{agent_name} Agent】")
            print(f"  算子: {agent['kernel']}")
            print(f"  状态: {agent['status']}")
            print(f"  迭代轮次: {agent['iterations']}")

            if agent['speedup'] > 0:
                print(f"  当前加速比: {agent['speedup']:.2f}×")

            if "last_update" in agent:
                print(f"  最后更新: {agent['last_update']}")

            if "last_ncu" in agent:
                ncu = agent["last_ncu"]
                print(f"  NCU 指标:")
                print(f"    - SM 效率: {ncu.get('sm_efficiency', 0):.1f}%")
                print(f"    - 内存吞吐: {ncu.get('memory_throughput', 0):.1f}%")
                print(f"    - 瓶颈类型: {ncu.get('bottleneck', 'unknown')}")

            print()

    def generate_summary_report(self):
        """生成汇总报告"""

        print(f"\n{'='*100}")
        print(f"  多智能体优化汇总报告")
        print(f"{'='*100}\n")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "overall_stats": {
                "total_iterations": 0,
                "avg_speedup": 0.0,
                "best_agent": None,
                "best_speedup": 0.0
            }
        }

        total_speedup = 0
        agent_count = 0

        for agent_name in self.agents.keys():
            agent = self.check_agent_status(agent_name)

            summary["agents"][agent_name] = {
                "kernel": agent["kernel"],
                "status": agent["status"],
                "iterations": agent["iterations"],
                "speedup": agent["speedup"]
            }

            summary["overall_stats"]["total_iterations"] += agent["iterations"]

            if agent["speedup"] > 0:
                total_speedup += agent["speedup"]
                agent_count += 1

                if agent["speedup"] > summary["overall_stats"]["best_speedup"]:
                    summary["overall_stats"]["best_speedup"] = agent["speedup"]
                    summary["overall_stats"]["best_agent"] = agent_name

        if agent_count > 0:
            summary["overall_stats"]["avg_speedup"] = total_speedup / agent_count

        # 打印汇总
        print("整体统计:")
        print(f"  总迭代轮次: {summary['overall_stats']['total_iterations']}")
        print(f"  平均加速比: {summary['overall_stats']['avg_speedup']:.2f}×")

        if summary['overall_stats']['best_agent']:
            print(f"  最佳Agent: {summary['overall_stats']['best_agent']}")
            print(f"  最高加速比: {summary['overall_stats']['best_speedup']:.2f}×")

        print()

        # 各Agent详情
        print("各Agent详情:")
        for agent_name, data in summary["agents"].items():
            status_emoji = "✅" if data["speedup"] > 1.0 else "🔄" if data["status"] == "working" else "⏸️"
            print(f"  {status_emoji} {agent_name}: {data['iterations']}轮迭代, {data['speedup']:.2f}× 加速")

        # 保存报告
        report_file = self.work_dir / "multi_agent_summary.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 汇总报告已保存: {report_file}")

        return summary

    def wait_for_completion(self, check_interval=30, max_wait_time=3600):
        """
        等待所有Agent完成工作

        参数:
            check_interval: 检查间隔（秒）
            max_wait_time: 最大等待时间（秒）
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            self.monitor_all_agents()

            # 检查是否都完成
            all_done = all(
                agent["status"] == "completed"
                for agent in self.agents.values()
            )

            if all_done:
                print("✅ 所有Agent已完成工作！")
                break

            print(f"⏳ 等待中... (已等待 {int(time.time() - start_time)} 秒)")
            print(f"   下次检查: {check_interval} 秒后\n")

            time.sleep(check_interval)

        # 生成最终报告
        self.generate_summary_report()

    def compare_optimizations(self):
        """对比3个算子的优化效果"""

        print(f"\n{'='*100}")
        print(f"  优化效果横向对比")
        print(f"{'='*100}\n")

        results = []

        for agent_name in self.agents.keys():
            agent = self.check_agent_status(agent_name)

            if agent["speedup"] > 0:
                results.append({
                    "name": agent_name,
                    "kernel": agent["kernel"],
                    "speedup": agent["speedup"],
                    "iterations": agent["iterations"]
                })

        # 按加速比排序
        results.sort(key=lambda x: x["speedup"], reverse=True)

        print(f"{'排名':<6} {'Agent':<20} {'算子':<35} {'加速比':<12} {'迭代次数':<10}")
        print("-" * 100)

        for i, result in enumerate(results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{medal} #{i:<4} {result['name']:<20} {result['kernel']:<35} {result['speedup']:.2f}×{'':<8} {result['iterations']:<10}")

        print()


def main():
    """主函数 - 启动协调器"""

    coordinator = MultiAgentCoordinator()

    print("多智能体协调器已启动")
    print("监控3个优化Agent:")
    print("  1. LayerNorm Agent")
    print("  2. MatMul Agent")
    print("  3. FlashAttention Agent")
    print()

    # 持续监控
    try:
        coordinator.wait_for_completion(check_interval=60, max_wait_time=7200)  # 最多等待2小时

        # 最终对比
        coordinator.compare_optimizations()

    except KeyboardInterrupt:
        print("\n⚠️  监控被用户中断")
        print("生成当前进度报告...\n")
        coordinator.generate_summary_report()


if __name__ == "__main__":
    main()
