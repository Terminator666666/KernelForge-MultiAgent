#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCU 采集驱动：在单进程内构建「我的 kernel」与「官方 baseline」，
对指定 workload 各执行一次，供 Nsight Compute (ncu) 采集真实硬件指标。

为什么这样写：
- ncu 需要 attach 一个真实进程并捕获其 CUDA kernel 启动。
- 我们复用 flashinfer-bench 的构建/输入生成组件，保证被 profile 的 kernel
  与正式验证时完全一致（同样的 TVM-FFI .so、同样的随机输入分布）。
- 脚本本身不做计时/不做循环，只各跑 1 次，让 ncu 用 -c / --launch-skip 精确截取。

用法（由 ncu 包裹调用）：
    ncu ... python ncu_driver.py --dataset <root> \
        --definition rmsnorm_h4096 \
        --solution kernelforge_rmsnorm_h4096_cuda_v1 \
        --baseline flashinfer_wrapper_2e27cd \
        --batch-size 16 \
        --which both|sol|base

--which 控制本次只跑哪一个实现，便于分两次采集、各自命名报告。
"""

import argparse
import os
import sys

import torch

# ---- WSL 链接补丁：让 TVM-FFI 链接期能找到 -lcuda（位于 /usr/lib/wsl/lib）----
_WSL_LIB = "/usr/lib/wsl/lib"
if os.path.isdir(_WSL_LIB):
    for _var in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        _cur = os.environ.get(_var, "")
        if _WSL_LIB not in _cur.split(":"):
            os.environ[_var] = f"{_WSL_LIB}:{_cur}" if _cur else _WSL_LIB

# ---- torch 兼容补丁：补齐 torch 2.7.1 缺失的 float4_e2m1fn_x2（仅占位）----
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8  # type: ignore[attr-defined]

from flashinfer_bench.bench.evaluators.utils import allocate_outputs  # noqa: E402
from flashinfer_bench.bench.utils import gen_inputs  # noqa: E402
from flashinfer_bench.compile import BuilderRegistry  # noqa: E402
from flashinfer_bench.data import TraceSet  # noqa: E402
from flashinfer_bench.data.workload import RandomInput, Workload  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--definition", required=True)
    p.add_argument("--solution", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--which", choices=["both", "sol", "base"], default="both")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=3,
                   help="profile 前的预热次数（让 JIT/缓存稳定，ncu 用 --launch-skip 跳过）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device

    ts = TraceSet.from_path(args.dataset)
    definition = ts.definitions[args.definition]

    # 构造一个指定 batch_size 的 workload（输入用随机，与官方一致）
    input_specs = {name: RandomInput() for name in definition.inputs.keys()}
    workload = Workload(
        axes={"batch_size": args.batch_size},
        inputs=input_specs,
        uuid=f"ncu-bs{args.batch_size}",
    )

    reg = BuilderRegistry.get_instance()

    inp = gen_inputs(definition, workload, device=device)

    sol_runnable = None
    base_runnable = None
    if args.which in ("both", "sol"):
        sol_runnable = reg.build(definition, ts.get_solution(args.solution))
    if args.which in ("both", "base") and args.baseline:
        base_runnable = reg.build(definition, ts.get_solution(args.baseline))

    def run_sol():
        out = allocate_outputs(definition, inp, device)
        with torch.no_grad():
            sol_runnable(*inp, *out)  # DPS
        torch.cuda.synchronize(device)

    def run_base():
        with torch.no_grad():
            base_runnable(*inp)  # value-returning
        torch.cuda.synchronize(device)

    # 预热（ncu 端用 --launch-skip 跳过这些启动，只截取最后一次）
    for _ in range(args.warmup):
        if sol_runnable is not None:
            run_sol()
        if base_runnable is not None:
            run_base()

    # 正式采集目标：每个实现各跑 1 次
    if sol_runnable is not None:
        run_sol()
    if base_runnable is not None:
        run_base()

    print(f"[ncu_driver] done which={args.which} batch_size={args.batch_size}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
