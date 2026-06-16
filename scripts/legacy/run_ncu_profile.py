"""
运行 NCU Profiling 分析瓶颈
"""

import subprocess
import os

def run_ncu_profile():
    """运行 NCU profiling"""

    print("\n" + "="*80)
    print("  NCU Profiling - LayerNorm 优化版本")
    print("="*80 + "\n")

    # 创建输出目录
    os.makedirs('layernorm_optimization/ncu_reports', exist_ok=True)

    # NCU 命令
    ncu_output = 'layernorm_optimization/ncu_reports/iteration_1'

    ncu_cmd = [
        'ncu',
        '--set', 'full',
        '--target-processes', 'all',
        '--export', ncu_output,
        '--force-overwrite',
        'python', 'test_layernorm_opt.py'
    ]

    print(f"NCU 命令: {' '.join(ncu_cmd)}\n")
    print("运行 NCU profiling (这可能需要几分钟)...\n")

    try:
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=600)

        print("NCU 输出:\n")
        print(result.stdout)

        if result.stderr:
            print("\nNCU 错误/警告:\n")
            print(result.stderr)

        if result.returncode == 0:
            print(f"\n✓ NCU report 保存到: {ncu_output}.ncu-rep")
            print(f"\n查看报告: ncu-ui {ncu_output}.ncu-rep")
        else:
            print(f"\n⚠️  NCU 可能遇到问题 (返回码: {result.returncode})")

        # 尝试生成文本摘要
        print("\n生成文本摘要...")
        summary_cmd = [
            'ncu',
            '--import', f'{ncu_output}.ncu-rep',
            '--page', 'details',
            '--csv'
        ]

        summary_result = subprocess.run(summary_cmd, capture_output=True, text=True, timeout=60)

        if summary_result.returncode == 0:
            summary_file = f'{ncu_output}_summary.csv'
            with open(summary_file, 'w') as f:
                f.write(summary_result.stdout)
            print(f"✓ 摘要保存到: {summary_file}")

    except subprocess.TimeoutExpired:
        print("\n⚠️  NCU profiling 超时")
    except Exception as e:
        print(f"\n❌ NCU profiling 失败: {e}")

if __name__ == "__main__":
    run_ncu_profile()
