#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashInfer-Bench 运行包装器（适配 torch 2.7.1+cu128）

为什么需要这个包装器：
- 当前环境固定使用 torch 2.7.1+cu128（匹配本机 CUDA 12.8 驱动，不可升级）。
- flashinfer-bench 声明 torch>=2.8.0，其 `flashinfer_bench/utils.py` 在构建
  dtype 映射表时会一次性引用 `torch.float4_e2m1fn_x2`（torch 2.8 才引入的 dtype）。
- 该表是 @cache 的，首次调用即引用所有 dtype，导致在 torch 2.7.1 上直接抛
  AttributeError，benchmark worker 崩溃——即便当前算子（如 rmsnorm/bfloat16）
  根本用不到 float4。

适配策略（不改 torch，只补依赖缺口）：
- 在导入 flashinfer_bench 之前，给 torch 模块补一个占位属性 float4_e2m1fn_x2。
- 该占位 dtype 仅用于让映射表能成功构建；任何真正使用 float4 的算子都不会走到
  这里（rmsnorm 全程 bfloat16）。若将来要跑 float4 量化算子，必须改用 torch>=2.8。

用法：
    python fib_run.py run   --local <dataset> [其余 flashinfer-bench 参数...]
    python fib_run.py validate --dataset <dataset> [...]
等价于 `python -m flashinfer_bench <args>`，但提前完成 torch 兼容性补丁。
"""

import os
import sys

import torch

# ---- WSL 链接补丁：让 -lcuda 能被找到 ----
# 在 WSL 下，CUDA 驱动库 libcuda.so 位于 /usr/lib/wsl/lib，而非 CUDA toolkit 目录。
# flashinfer-bench 的 TVM-FFI builder 链接时使用 -lcuda，但只加了 toolkit 的 -L 路径，
# 导致 "/usr/bin/ld: cannot find -lcuda"。这里把 WSL 库目录补进 LIBRARY_PATH（链接期
# 解析 -l）与 LD_LIBRARY_PATH（运行期加载），不改动 torch / bench 包本身。
_WSL_LIB = "/usr/lib/wsl/lib"
if os.path.isdir(_WSL_LIB):
    for _var in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        _cur = os.environ.get(_var, "")
        if _WSL_LIB not in _cur.split(":"):
            os.environ[_var] = f"{_WSL_LIB}:{_cur}" if _cur else _WSL_LIB

# ---- torch 兼容性补丁：补齐 torch 2.7.1 缺失的 float4_e2m1fn_x2 ----
# 仅作占位，使 flashinfer_bench 的 dtype 映射表可成功构建。
# rmsnorm 等 bfloat16 算子不会实际使用该 dtype。
if not hasattr(torch, "float4_e2m1fn_x2"):
    # 用 uint8 作占位（float4 在底层也常以紧凑字节存储），仅需该属性存在即可。
    torch.float4_e2m1fn_x2 = torch.uint8  # type: ignore[attr-defined]
    print("[fib_run] 已为 torch 2.7.1 补充占位 dtype: float4_e2m1fn_x2 -> uint8",
          file=sys.stderr)

# 补丁完成后再导入并启动 flashinfer-bench CLI
from flashinfer_bench.cli.main import cli  # noqa: E402

if __name__ == "__main__":
    cli()
