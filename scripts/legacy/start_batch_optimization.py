#!/usr/bin/env python3
"""
快速启动脚本 - 批量优化 12 个算子

使用方法：
    python scripts/start_batch_optimization.py

注意：
1. 需要先在 config/api_config.json 中配置 API Key
2. 确保 NCU (Nsight Compute) 已安装
3. 在 Linux 环境中运行
"""

import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.batch_optimize_12 import main, SELECTED_OPERATORS


def show_menu():
    """显示交互式菜单"""
    print("="*70)
    print("  KernelForge-MultiAgent - 批量优化启动器")
    print("="*70)

    print("\n📊 已选择的 12 个算子:\n")

    for i, op in enumerate(SELECTED_OPERATORS, 1):
        priority_icon = "🌟" if op['priority'] == 1 else "⭐" if op['priority'] == 2 else "✨"
        print(f"{i:2d}. {priority_icon} {op['name']:<40} | {op['expected_speedup']:<10} | {op['description']}")

    print("\n" + "="*70)
    print("\n选项:")
    print("  1. 开始批量优化所有 12 个算子")
    print("  2. 选择特定算子优化")
    print("  3. 检查 API 配置")
    print("  4. 查看硬件信息")
    print("  0. 退出")

    choice = input("\n请选择 (0-4): ").strip()
    return choice


def check_api_config():
    """检查 API 配置"""
    config_path = project_root / "config" / "api_config.json"

    if not config_path.exists():
        print("\n⚠️  API 配置文件不存在!")
        print(f"   请创建: {config_path}")
        print("\n   示例内容:")
        print("""   {
     "api_key": "REPLACE_ME_LOCALLY",
     "base_url": "https://api.siliconflow.cn/v1",
     "model": "deepseek-ai/DeepSeek-V3"
   }""")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("\n✅ API 配置:")
    print(f"   Base URL: {config.get('base_url', 'N/A')}")
    print(f"   Model: {config.get('model', 'N/A')}")
    print(f"   API Key: {'已配置' if config.get('api_key') else '未配置'}")

    return bool(config.get('api_key'))


def show_hardware_info():
    """显示硬件信息"""
    print("\n🖥️  硬件配置:")
    print("   GPU: RTX 5070")
    print("   架构: Blackwell (sm_120)")
    print("   CUDA 核心: 6144")
    print("   Tensor Core: 192 (第 5 代)")
    print("   显存: 12 GB GDDR7")
    print("   内存带宽: 448 GB/s")


def select_operators():
    """选择要优化的算子"""
    print("\n请输入要优化的算子编号（用逗号分隔，如: 1,2,3）")
    print("或输入 'all' 优化所有算子")

    selection = input("\n选择: ").strip().lower()

    if selection == 'all':
        return list(range(len(SELECTED_OPERATORS)))

    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        indices = [i for i in indices if 0 <= i < len(SELECTED_OPERATORS)]
        return indices
    except:
        print("❌ 输入格式错误")
        return []


def main_menu():
    """主菜单"""
    while True:
        choice = show_menu()

        if choice == '0':
            print("\n👋 再见!")
            sys.exit(0)

        elif choice == '1':
            print("\n🚀 开始批量优化...")
            if check_api_config():
                main()
            else:
                print("\n❌ 请先配置 API")
            break

        elif choice == '2':
            indices = select_operators()
            if indices:
                # TODO: 实现选择性优化
                print(f"\n已选择 {len(indices)} 个算子")
                print("功能开发中...")
            input("\n按 Enter 继续...")

        elif choice == '3':
            check_api_config()
            input("\n按 Enter 继续...")

        elif choice == '4':
            show_hardware_info()
            input("\n按 Enter 继续...")

        else:
            print("\n❌ 无效选择")
            input("\n按 Enter 继续...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
