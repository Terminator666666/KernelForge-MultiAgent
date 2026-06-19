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
        --definition dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
        --solution kernelforge_dsa_sparse_attention_cuda_v1 \
        --baseline flashinfer_wrapper_5af199 \
        --workload-uuid 7ddaefc67273438a813360560a7931ea \
        --which both|sol|base

--which 控制本次只跑哪一个实现，便于分两次采集、各自命名报告。
优先使用 --workload-uuid 绑定真实 trace workload；--batch-size 仅保留给
旧的 batch_size 轴 definition 作为后备模式。
"""

import argparse
import os
import sys
from pathlib import Path

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
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors  # noqa: E402
from flashinfer_bench.compile import BuilderRegistry  # noqa: E402
from flashinfer_bench.data import TraceSet  # noqa: E402
from flashinfer_bench.data.workload import RandomInput, Workload  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--definition", required=True)
    p.add_argument("--solution", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--workload-uuid",
        default=None,
        help="直接指定 dataset 里的 workload uuid，优先用于真实 trace NCU 采样",
    )
    p.add_argument("--which", choices=["both", "sol", "base"], default="both")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=3,
                   help="profile 前的预热次数（让 JIT/缓存稳定，ncu 用 --launch-skip 跳过）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    dataset_root = Path(args.dataset).resolve()

    ts = TraceSet.from_path(str(dataset_root))
    definition = ts.definitions[args.definition]

    workload = None
    if args.workload_uuid:
        for wt in ts.workloads.get(args.definition, []):
            if wt.workload.uuid == args.workload_uuid:
                workload = wt.workload
                break
        if workload is None:
            raise SystemExit(
                f"错误：definition '{args.definition}' 中找不到 workload uuid '{args.workload_uuid}'"
            )
    else:
        if args.batch_size is None:
            raise SystemExit("错误：必须提供 --workload-uuid 或 --batch-size 其中之一。")
        # 后备模式：仅适用于 batch_size 为主轴的旧 definition
        input_specs = {name: RandomInput() for name in definition.inputs.keys()}
        workload = Workload(
            axes={"batch_size": args.batch_size},
            inputs=input_specs,
            uuid=f"ncu-bs{args.batch_size}",
        )

    reg = BuilderRegistry.get_instance()

    safe_tensors = load_safetensors(definition, workload, dataset_root)
    inp = gen_inputs(definition, workload, device=device, safe_tensors=safe_tensors)

    sol_runnable = None
    base_runnable = None
    if args.which in ("both", "sol"):
        sol_runnable = reg.build(definition, ts.get_solution(args.solution))
    if args.which in ("both", "base") and args.baseline:
        base_runnable = reg.build(definition, ts.get_solution(args.baseline))

    def run_sol():
        if sol_runnable.metadata.destination_passing_style:
            out = allocate_outputs(definition, inp, device)
            with torch.no_grad():
                sol_runnable(*inp, *out)
        else:
            with torch.no_grad():
                sol_runnable(*inp)
        torch.cuda.synchronize(device)

    def run_base():
        if base_runnable.metadata.destination_passing_style:
            out = allocate_outputs(definition, inp, device)
            with torch.no_grad():
                base_runnable(*inp, *out)
        else:
            with torch.no_grad():
                base_runnable(*inp)
        torch.cuda.synchronize(device)

    # 预热（ncu 端用 --launch-skip 跳过这些启动，只截取最后一次）
    for _ in range(args.warmup):
        if sol_runnable is not None:
            run_sol()
        if base_runnable is not None:
            run_base()

    # 正式采集目标：仅把被测执行段包进 profiler window，避免抓到前置随机数生成 kernel。
    cudart = torch.cuda.cudart()
    torch.cuda.synchronize(device)
    cudart.cudaProfilerStart()
    if sol_runnable is not None:
        run_sol()
    if base_runnable is not None:
        run_base()
    torch.cuda.synchronize(device)
    cudart.cudaProfilerStop()

    workload_tag = args.workload_uuid or f"batch_size={args.batch_size}"
    print(f"[ncu_driver] done which={args.which} workload={workload_tag}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
