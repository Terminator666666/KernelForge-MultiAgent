#!/usr/bin/env python3
"""
直接启动批量优化 - 无需交互

直接运行 12 个算子的自动优化
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.batch_optimize_12 import main

if __name__ == "__main__":
    print("🚀 直接启动批量优化...")
    print("⚠️  这将优化 12 个算子，预计耗时 2-15 小时\n")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断优化")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
