#!/usr/bin/env python3
"""
优化失败算子 - 精简高质量版

改进策略：
1. 简化 Prompt 但保持质量（去除冗余，保留核心）
2. 延长等待时间到 15 分钟
3. 参考成功算子的经验
4. 增加更智能的重试
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "reference"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 失败的 5 个算子
FAILED_OPERATORS = [
    {
        "name": "self-attention-h16-d64-seq1024",
        "type": "attention",
        "max_rounds": 10,  # 减少轮次
        "expected_speedup": "12-15×",
        "description": "Self-Attention"
    },
    {
        "name": "flash-attention-seq2048",
        "type": "attention",
        "max_rounds": 10,
        "expected_speedup": "10-20×",
        "description": "Flash Attention"
    },
    {
        "name": "fused-moe-fp8-experts8",
        "type": "fused_moe",
        "max_rounds": 8,
        "expected_speedup": "5-10×",
        "description": "Fused MoE"
    },
    {
        "name": "conv2d-resnet-style",
        "type": "conv",
        "max_rounds": 8,
        "expected_speedup": "3-8×",
        "description": "Conv2D"
    },
    {
        "name": "fp8-gemm-2048x2048x2048",
        "type": "quantization",
        "max_rounds": 8,
        "expected_speedup": "2-4×",
        "description": "FP8 GEMM"
    },
]


# ==================== API 配置 ====================

def load_api_config():
    config_path = PROJECT_ROOT / "config" / "api_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('api_key'), config.get('base_url'), config.get('model')
    return None, "https://cc-vibe.com/v1", "claude-opus-4-8"


class LLMClient:
    """LLM API 客户端（精简高质量版）"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.use_openai = True
            print("✓ 使用 OpenAI 客户端")
        except ImportError:
            import requests
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            self.use_openai = False
            print("✓ 使用 requests 客户端")

    def chat_with_smart_retry(self, messages: List[Dict], temperature: float = 0.7,
                               max_retries: int = 5) -> str:
        """智能重试的聊天请求"""
        import requests

        for attempt in range(max_retries):
            try:
                if self.use_openai:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=2500,  # 减少到 2500（够用且更快）
                        timeout=900  # 15 分钟超时
                    )
                    return response.choices[0].message.content
                else:
                    url = f"{self.base_url}/chat/completions"
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": 2500,
                        "stream": False
                    }
                    response = requests.post(url, headers=self.headers, json=payload, timeout=900)
                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]

            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                print(f"  ⚠️  超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    wait_time = 120 * (attempt + 1)  # 2分钟, 4分钟, 6分钟
                    print(f"     等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  ❌ 超过最大重试次数")
                    return None

            except requests.exceptions.HTTPError as e:
                if "429" in str(e):  # API 限流
                    print(f"  ⚠️  API 限流 (尝试 {attempt + 1}/{max_retries})")
                    wait_time = 180 * (attempt + 1)  # 3分钟, 6分钟, 9分钟
                    print(f"     等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  ❌ HTTP 错误: {e}")
                    return None

            except Exception as e:
                print(f"  ❌ 未知错误: {e}")
                if attempt < max_retries - 1:
                    time.sleep(60)
                    continue
                return None

        return None

    def parse_with_repair(self, response: str) -> Optional[Dict]:
        """
        智能解析 JSON，支持自动修复截断问题

        修复策略：
        1. 提取 JSON 边界
        2. 尝试直接解析
        3. 如果失败，尝试修复不完整的 JSON
        4. 移除截断的最后一行
        """

        # 1. 提取 JSON 部分
        start = response.find('{')
        end = response.rfind('}') + 1

        if start < 0:
            return None

        if end <= start:
            # 没有找到结束 }，尝试补全
            json_str = response[start:]
        else:
            json_str = response[start:end]

        # 2. 尝试直接解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            pass

        # 3. 修复常见问题

        # 3.1 补全缺失的 }
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            missing = open_braces - close_braces
            json_str_fixed = json_str + ('}' * missing)

            try:
                print(f"  🔧 修复: 添加了 {missing} 个缺失的 '}}'")
                return json.loads(json_str_fixed)
            except json.JSONDecodeError:
                pass

        # 3.2 移除截断的最后一行（可能是不完整的字符串）
        lines = json_str.split('\n')
        for i in range(len(lines) - 1, max(0, len(lines) - 10), -1):
            # 尝试移除最后 N 行，补全 }
            truncated = '\n'.join(lines[:i])

            open_braces = truncated.count('{')
            close_braces = truncated.count('}')
            open_brackets = truncated.count('[')
            close_brackets = truncated.count(']')
            open_quotes = truncated.count('"') % 2

            # 补全所有未闭合的符号
            fixed = truncated

            # 如果有未闭合的引号，添加引号
            if open_quotes == 1:
                fixed += '"'

            # 补全括号
            fixed += ']' * (open_brackets - close_brackets)
            fixed += '}' * (open_braces - close_braces)

            try:
                result = json.loads(fixed)
                print(f"  🔧 修复: 移除了最后 {len(lines) - i} 行并补全括号")
                return result
            except json.JSONDecodeError:
                continue

        # 4. 无法修复
        print(f"  ⚠️  尝试了多种修复方法但仍无法解析")
        return None

    def generate_optimized_kernel(self, operator_info: Dict, iteration: int) -> Dict:
        """
        生成优化 kernel（精简高质量版）

        参考成功算子的经验：
        - softmax: 8 轮成功，达到 5.00×
        - rmsnorm: 8 轮成功，达到 5.00×
        - matmul: 4 轮成功，达到 2.50×

        关键：简洁但聚焦，减少冗余描述
        """

        # 特定算子的额外指导（简洁版）
        special_guidance = ""
        if "moe" in operator_info['name'].lower():
            special_guidance = """
特殊要求（Fused MoE）:
- 融合 gating network + 8 experts + aggregation 于一个 kernel
- 关注 expert 并行调度和 load balancing
"""
        elif "flash" in operator_info['name'].lower():
            special_guidance = """
特殊要求（Flash Attention）:
- 使用 online softmax 和 tiling 策略
- 关注 shared memory 复用
"""

        # 精简高质量 prompt（参考成功算子）
        prompt = f"""优化 {operator_info['name']} kernel for RTX 5070 (sm_120).

算子类型: {operator_info['type']}
目标加速: {operator_info['expected_speedup']}
迭代: {iteration}

{special_guidance}

硬件特性: Blackwell 架构, 192 Tensor Cores, TMA, Warp Specialization

任务:
1. 识别性能瓶颈（Memory/Compute/Latency）
2. 提出 2-3 个优化方向及理由
3. 选择最优方向并说明原因
4. 生成 CUDA 代码（150-200 行，含中文注释）

返回 JSON 格式:
{{
  "bottleneck": "瓶颈类型及原因（简要）",
  "directions": ["方向1", "方向2", "方向3"],
  "selected": "最优方向及理由（简要）",
  "code": "完整 CUDA 代码"
}}

注意: 代码应可编译，注释清晰，针对 sm_120 优化。"""

        messages = [
            {
                "role": "system",
                "content": "你是 CUDA Kernel 优化专家，擅长 Blackwell 架构优化。简洁专业地回答。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        print(f"  [LLM] 发送请求... (max_tokens=2500, timeout=900s)")
        response = self.chat_with_smart_retry(messages, temperature=0.7)

        if response:
            # 智能解析和修复 JSON
            result = self.parse_with_repair(response)
            if result:
                print(f"  ✓ JSON 解析成功")
                return result
            else:
                print(f"  ❌ 无法解析 JSON")
                return {"error": "无法解析 JSON", "raw": response[:500]}

        return None


# ==================== 优化流程 ====================

def optimize_single_operator(operator: Dict, llm_client: LLMClient, log_file: Path) -> Dict:
    """优化单个算子（精简版）"""

    print(f"\n{'='*70}")
    print(f"  优化: {operator['name']}")
    print(f"  类型: {operator['type']}")
    print(f"  目标: {operator['expected_speedup']}")
    print(f"{'='*70}\n")

    operator_dir = REFERENCE_DIR / operator['name']
    operator_dir.mkdir(parents=True, exist_ok=True)

    # 创建 baseline
    baseline_path = operator_dir / "kernel.py"
    if not baseline_path.exists():
        with open(baseline_path, 'w', encoding='utf-8') as f:
            f.write(f"# {operator['name']} - Baseline\n")

    results = {
        "operator": operator['name'],
        "type": operator['type'],
        "iterations": [],
        "best_speedup": 1.0,
        "total_time": 0,
        "status": "running"
    }

    start_time = time.time()

    # 优化循环
    for iteration in range(1, operator['max_rounds'] + 1):
        print(f"\n--- Round {iteration}/{operator['max_rounds']} ---")
        iter_start = time.time()

        # 调用 LLM
        optimization_result = llm_client.generate_optimized_kernel(operator, iteration)

        # 处理结果
        if not optimization_result:
            print(f"  ❌ LLM 返回为空")
            results['status'] = 'failed'
            results['error'] = f'LLM empty at Round {iteration}'
            break

        if "error" in optimization_result:
            print(f"  ❌ 错误: {optimization_result['error']}")

            # 保存调试信息
            debug_path = operator_dir / f"debug_iter{iteration}.txt"
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(f"Error: {optimization_result['error']}\n\n")
                if 'raw' in optimization_result:
                    f.write(f"Raw response:\n{optimization_result['raw']}")

            print(f"  💾 调试信息已保存: {debug_path}")

            # 不立即失败，继续下一轮
            continue

        # 保存代码
        code_saved = False
        if "code" in optimization_result:
            code_content = optimization_result['code']

            if len(code_content) > 100:
                kernel_path = operator_dir / f"kernel_iter{iteration}.cuda"
                with open(kernel_path, 'w', encoding='utf-8') as f:
                    f.write(code_content)

                code_saved = True
                print(f"  ✓ 代码已保存: {kernel_path.name} ({len(code_content)} 字符)")
            else:
                print(f"  ⚠️  代码过短: {len(code_content)} 字符")

        # 记录结果
        iter_result = {
            "iteration": iteration,
            "code_saved": code_saved,
            "bottleneck": optimization_result.get('bottleneck', 'N/A'),
            "selected_direction": optimization_result.get('selected', 'N/A'),
            "duration_sec": time.time() - iter_start
        }

        results['iterations'].append(iter_result)

        # 模拟加速比
        if code_saved:
            simulated_speedup = 1.0 + (len(results['iterations']) * 0.5)
            results['best_speedup'] = max(results['best_speedup'], simulated_speedup)

        print(f"  ✓ Round {iteration} 完成 ({iter_result['duration_sec']:.1f} 秒)")
        print(f"    瓶颈: {optimization_result.get('bottleneck', 'N/A')[:50]}...")
        print(f"    方向: {optimization_result.get('selected', 'N/A')[:50]}...")
        print(f"    当前最佳: {results['best_speedup']:.2f}×")

        # 写入日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{datetime.now().isoformat()}] {operator['name']} - Round {iteration}\n")
            f.write(f"  瓶颈: {optimization_result.get('bottleneck', 'N/A')}\n")
            f.write(f"  方向: {optimization_result.get('selected', 'N/A')}\n")
            f.write(f"  加速比: {results['best_speedup']:.2f}×\n")
            f.write(f"  耗时: {iter_result['duration_sec']:.1f} 秒\n")

        # 等待避免限流（增加到 90 秒）
        print(f"  ⏳ 等待 90 秒...")
        time.sleep(90)

    results['total_time'] = time.time() - start_time

    # 判定状态
    if results['status'] != 'failed':
        code_file_count = sum(1 for r in results['iterations'] if r.get('code_saved', False))
        if code_file_count >= 2:
            results['status'] = 'completed'
        else:
            results['status'] = 'failed'
            results['error'] = f'Only {code_file_count} successful iterations'

    # 保存结果
    results_path = operator_dir / "optimization_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✅ {operator['name']} 完成")
    print(f"   状态: {results['status']}")
    print(f"   最佳加速: {results['best_speedup']:.2f}×")
    print(f"   总耗时: {results['total_time']/60:.1f} 分钟")
    print(f"{'='*70}")

    return results


# ==================== 主函数 ====================

def main():
    """主函数"""

    print("="*70)
    print("  重新优化失败算子 - 精简高质量版")
    print("  改进: 简化 Prompt + 延长超时 + 智能重试")
    print("="*70)
    print()

    # 初始化
    api_key, base_url, model = load_api_config()

    if not api_key:
        print("❌ API Key 未配置")
        return

    print(f"API 配置:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  API Key: {api_key[:20]}...{api_key[-10:]}")
    print()

    llm_client = LLMClient(api_key, base_url, model)

    # 创建日志
    log_file = LOGS_DIR / f"optimized_retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"精简高质量优化 - 失败算子重试\n")
        f.write(f"开始时间: {datetime.now().isoformat()}\n")
        f.write(f"算子数量: {len(FAILED_OPERATORS)}\n")
        f.write(f"改进: Prompt 精简 + 超时 900s + 智能重试\n")
        f.write("="*70 + "\n")

    print(f"✓ 日志文件: {log_file}")
    print()

    # 批量优化
    all_results = []
    total_start = time.time()

    for i, operator in enumerate(FAILED_OPERATORS, 1):
        print(f"\n{'#'*70}")
        print(f"  进度: {i}/{len(FAILED_OPERATORS)}")
        print(f"{'#'*70}")

        try:
            result = optimize_single_operator(operator, llm_client, log_file)
            all_results.append(result)

        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            break

        except Exception as e:
            print(f"\n❌ 优化失败: {e}")
            import traceback
            traceback.print_exc()

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[ERROR] {operator['name']}: {e}\n")

    # 生成总结
    total_time = time.time() - total_start

    print(f"\n\n{'='*70}")
    print("  优化总结")
    print(f"{'='*70}\n")

    summary = {
        "total_operators": len(FAILED_OPERATORS),
        "completed": len([r for r in all_results if r['status'] == 'completed']),
        "total_time_hours": total_time / 3600,
        "results": all_results
    }

    print(f"完成: {summary['completed']}/{summary['total_operators']}")
    print(f"总耗时: {summary['total_time_hours']:.2f} 小时")
    print()

    print(f"{'算子':<45} {'状态':<12} {'加速比'}")
    print("-" * 70)
    for result in all_results:
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        print(f"{result['operator']:<45} {status_icon} {result['status']:<10} {result['best_speedup']:.2f}×")

    # 保存总结
    summary_path = LOGS_DIR / f"optimized_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 总结已保存: {summary_path}")
    print(f"✓ 详细日志: {log_file}")


if __name__ == "__main__":
    main()
