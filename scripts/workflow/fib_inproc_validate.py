#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashInfer-Bench 单进程验证器（适配 WSL2 + torch 2.7.1+cu128）

为什么需要这个脚本：
- flashinfer-bench 官方的 `run` 命令使用 PersistentRunner / IsolatedRunner，
  二者都通过 multiprocessing(spawn) + CUDA IPC 在主进程与子进程之间共享 CUDA 张量。
- WSL2 不支持 CUDA IPC，子进程在 `rebuild_cuda_tensor -> _new_shared_cuda` 处报
  "CUDA error: invalid resource handle"，导致整个 benchmark 崩溃。
- 该问题与被测 kernel 无关，是运行环境（WSL）与框架进程模型不兼容。

本脚本的做法（结果与官方逻辑一致，仅去掉跨进程 IPC）：
- 直接调用 bench 自带的组件，全部在单进程内完成：
    BuilderRegistry.build_reference(definition)  -> 参考实现（纯 PyTorch，value-returning）
    BuilderRegistry.build(definition, solution)  -> 待测 solution（TVM-FFI / DPS）
    gen_inputs(...)            -> 与官方相同的随机输入生成
    compute_error_stats(...)   -> 与官方相同的误差/容差判定
    time_runnable(...)         -> 与官方相同的 CUPTI 计时
- 因此正确性判定（atol/rtol/matched_ratio）与官方 benchmark 完全相同。
- baseline 捕获阶段只记录 latency 与 workload 覆盖率；只有传入 --baseline 时，
  才输出相对 baseline 的 sol/base 加速比。

用法：
    python fib_inproc_validate.py --dataset <trace_set 根目录> \
        --definition rmsnorm_h4096 \
        --solution  kernelforge_rmsnorm_h4096_cuda_v1 \
        [--num-trials 1] [--warmup 3] [--iters 10] [--atol 1e-2] [--rtol 1e-2]

注意：本脚本只做验证与计时，不修改数据集，不写 trace。
"""

import argparse
import json
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

# 补丁完成后再导入 flashinfer-bench 组件
from flashinfer_bench.bench.config import ResolvedEvalConfig  # noqa: E402
from flashinfer_bench.bench.evaluators.utils import (  # noqa: E402
    allocate_outputs,
    normalize_result,
)
from flashinfer_bench.bench.timing import time_runnable  # noqa: E402
from flashinfer_bench.bench.utils import (  # noqa: E402
    compute_error_stats,
    gen_inputs,
    load_safetensors,
)
from flashinfer_bench.compile import BuilderRegistry  # noqa: E402
from flashinfer_bench.data import TraceSet  # noqa: E402


def collect_missing_safetensor_paths(workload, dataset_root: Path) -> list[str]:
    """收集当前 workload 缺失的 safetensors 文件路径。"""
    missing: list[str] = []
    for value in workload.inputs.values():
        if getattr(value, "type", None) != "safetensors":
            continue
        raw_path = str(getattr(value, "path", "")).strip()
        if not raw_path:
            continue
        path = dataset_root / raw_path.replace("./", "", 1)
        if not path.exists():
            missing.append(str(path))
    return missing


def load_workload_allowlist(path: Path) -> set[str]:
    """从 JSON/TXT 文件加载 workload uuid allowlist。"""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()

    if path.suffix.lower() in {".json", ".jsonl"}:
        payload = json.loads(text)
        if isinstance(payload, list):
            uuids = [
                item.strip()
                for item in payload
                if isinstance(item, str) and item.strip()
            ]
            return set(uuids)
        if isinstance(payload, dict):
            if isinstance(payload.get("workload_uuids"), list):
                uuids = [
                    item.strip()
                    for item in payload["workload_uuids"]
                    if isinstance(item, str) and item.strip()
                ]
                return set(uuids)
            if isinstance(payload.get("workloads"), list):
                uuids = []
                for item in payload["workloads"]:
                    if isinstance(item, str) and item.strip():
                        uuids.append(item.strip())
                    elif isinstance(item, dict):
                        uuid = str(item.get("uuid", "")).strip()
                        if uuid:
                            uuids.append(uuid)
                return set(uuids)
        raise ValueError(f"不支持的 allowlist JSON 结构: {path}")

    uuids = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        uuids.append(stripped.split()[0])
    return set(uuids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="TraceSet 根目录")
    p.add_argument("--definition", required=True, help="definition 名，如 rmsnorm_h4096")
    p.add_argument("--solution", required=True, help="待测 solution 名")
    p.add_argument("--baseline", default=None,
                   help="官方对照 solution 名（如 flashinfer baseline）。"
                        "指定后会额外计时该实现，并以它为锚点给出 sol/base 加速比。")
    p.add_argument(
        "--workload-uuid",
        default=None,
        help="只验证指定 workload uuid；用于首轮单 workload 预验证和 NCU 前快速探路",
    )
    p.add_argument(
        "--workload-allowlist-file",
        default=None,
        help="批量 workload allowlist 文件（JSON/TXT），只验证其中列出的真实 workload uuid",
    )
    p.add_argument("--num-trials", type=int, default=1, help="每个 workload 的随机试验次数")
    p.add_argument("--warmup", type=int, default=3, help="计时前的预热迭代数")
    p.add_argument("--iters", type=int, default=10, help="计时迭代数")
    p.add_argument("--atol", type=float, default=1e-2, help="绝对误差容差")
    p.add_argument("--rtol", type=float, default=1e-2, help="相对误差容差")
    p.add_argument("--device", default="cuda:0", help="运行设备")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    dataset_root = Path(args.dataset).resolve()

    if args.workload_uuid and args.workload_allowlist_file:
        print(
            "错误：--workload-uuid 与 --workload-allowlist-file 不能同时使用。",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print("错误：当前进程 torch.cuda 不可用，无法验证。", file=sys.stderr)
        return 2

    # ---- 加载数据集 ----
    ts = TraceSet.from_path(str(dataset_root))
    if args.definition not in ts.definitions:
        print(f"错误：数据集中找不到 definition '{args.definition}'", file=sys.stderr)
        return 2
    definition = ts.definitions[args.definition]

    solution = ts.get_solution(args.solution)
    if solution is None:
        print(f"错误：数据集中找不到 solution '{args.solution}'", file=sys.stderr)
        return 2

    workload_traces = ts.workloads.get(args.definition, [])
    if not workload_traces:
        print(f"错误：definition '{args.definition}' 没有 workload", file=sys.stderr)
        return 2

    allowlist_uuids: set[str] | None = None
    if args.workload_allowlist_file:
        allowlist_path = Path(args.workload_allowlist_file).resolve()
        if not allowlist_path.exists():
            print(f"错误：allowlist 文件不存在：{allowlist_path}", file=sys.stderr)
            return 2
        allowlist_uuids = load_workload_allowlist(allowlist_path)
        if not allowlist_uuids:
            print(f"错误：allowlist 文件为空：{allowlist_path}", file=sys.stderr)
            return 2
        dataset_uuids = {wt.workload.uuid for wt in workload_traces}
        missing_in_dataset = sorted(allowlist_uuids - dataset_uuids)
        if missing_in_dataset:
            print(
                f"错误：allowlist 中有 {len(missing_in_dataset)} 个 uuid 不在当前 definition 数据集中。",
                file=sys.stderr,
            )
            for uuid in missing_in_dataset[:10]:
                print(f"  - {uuid}", file=sys.stderr)
            return 2
        workload_traces = [wt for wt in workload_traces if wt.workload.uuid in allowlist_uuids]

    if args.workload_uuid:
        workload_traces = [wt for wt in workload_traces if wt.workload.uuid == args.workload_uuid]
        if not workload_traces:
            print(
                f"错误：definition '{args.definition}' 中找不到 workload uuid '{args.workload_uuid}'",
                file=sys.stderr,
            )
            return 2

    declared_workloads = list(workload_traces)
    eligible_workloads = []
    skipped_workloads: list[tuple[str, list[str]]] = []
    for wt in declared_workloads:
        missing_paths = collect_missing_safetensor_paths(wt.workload, dataset_root)
        if missing_paths:
            skipped_workloads.append((wt.workload.uuid, missing_paths))
            continue
        eligible_workloads.append(wt)

    if not eligible_workloads:
        print(
            f"错误：definition '{args.definition}' 没有任何可运行 workload；"
            "所有候选 workload 都缺少真实 safetensors 文件。",
            file=sys.stderr,
        )
        if skipped_workloads:
            print("缺失样例（最多前 5 个）:", file=sys.stderr)
            for uuid, paths in skipped_workloads[:5]:
                print(f"  - uuid={uuid}", file=sys.stderr)
                for path in paths[:3]:
                    print(f"      {path}", file=sys.stderr)
        return 2

    cfg = ResolvedEvalConfig(
        warmup_runs=args.warmup,
        iterations=args.iters,
        num_trials=args.num_trials,
        rtol=args.rtol,
        atol=args.atol,
    )

    reg = BuilderRegistry.get_instance()

    # ---- 构建参考实现与待测 solution ----
    ref_runnable = None
    if args.baseline:
        print(f"[build] 构建参考实现 (reference) ...", flush=True)
        ref_runnable = reg.build_reference(definition)

    print(f"[build] 构建 solution '{solution.name}' "
          f"(lang={solution.spec.language}, binding={solution.spec.binding}) ...", flush=True)
    sol_runnable = reg.build(definition, solution)
    is_dps = sol_runnable.metadata.destination_passing_style

    # ---- 可选：构建官方对照 baseline ----
    baseline_runnable = None
    baseline_is_dps = False
    if args.baseline:
        baseline_sol = ts.get_solution(args.baseline)
        if baseline_sol is None:
            print(f"错误：数据集中找不到 baseline solution '{args.baseline}'", file=sys.stderr)
            return 2
        print(f"[build] 构建 baseline '{baseline_sol.name}' "
              f"(lang={baseline_sol.spec.language}, binding={baseline_sol.spec.binding}) ...",
              flush=True)
        baseline_runnable = reg.build(definition, baseline_sol)
        baseline_is_dps = baseline_runnable.metadata.destination_passing_style

    print("=" * 72)
    print(f"definition : {definition.name}")
    print(f"solution   : {solution.name}  (DPS={is_dps})")
    print(f"workloads  : 声明 {len(declared_workloads)} 个 / 有效 {len(eligible_workloads)} 个 / "
          f"缺失跳过 {len(skipped_workloads)} 个   trials/iters = "
          f"{cfg.num_trials}/{cfg.iterations}   atol/rtol = {cfg.atol}/{cfg.rtol}")
    if args.workload_allowlist_file:
        print(f"allowlist  : {Path(args.workload_allowlist_file).resolve()}")
    if skipped_workloads:
        print("missing    : 仅对真实 safetensors 完整存在的 workload 计成绩；缺文件 workload 自动跳过")
    print("=" * 72)

    # 表头：baseline 捕获阶段只强调 latency；对比阶段才强调 sol/base
    if baseline_runnable is not None:
        header = (f"{'workload(axes)':<22}{'状态':<7}{'max_rel':>9}"
                  f"{'base(ms)':>10}{'sol(ms)':>9}{'sol/base':>10}")
    else:
        header = (f"{'workload(axes)':<22}{'状态':<7}{'max_rel':>9}{'sol(ms)':>9}")
    print(header)
    print("-" * len(header))

    total = 0
    passed = 0
    speedups_vs_base = []    # sol vs 官方 baseline
    sol_latencies = []
    base_latencies = []

    for wt in eligible_workloads:
        workload = wt.workload
        total += 1
        axes_str = ",".join(f"{k}={v}" for k, v in workload.axes.items())

        try:
            # 每个 trial 用新随机输入做正确性比对（与官方一致）
            max_abs = 0.0
            max_rel = 0.0
            incorrect = False
            inputs_cache = []

            for _ in range(cfg.num_trials):
                safe_tensors = load_safetensors(definition, workload, dataset_root)
                inp = gen_inputs(definition, workload, device=device, safe_tensors=safe_tensors)
                inputs_cache.append(inp)

                if ref_runnable is not None:
                    with torch.no_grad():
                        ref_res = ref_runnable(*inp)
                    torch.cuda.synchronize(device)
                    ref_out = normalize_result(definition, ref_res, device)
                else:
                    ref_runnable_local = reg.build_reference(definition)
                    with torch.no_grad():
                        ref_res = ref_runnable_local(*inp)
                    torch.cuda.synchronize(device)
                    ref_out = normalize_result(definition, ref_res, device)

                if is_dps:
                    out = allocate_outputs(definition, inp, device)
                    with torch.no_grad():
                        sol_runnable(*inp, *out)
                    torch.cuda.synchronize(device)
                    sol_out = out
                else:
                    with torch.no_grad():
                        sol_res = sol_runnable(*inp)
                    torch.cuda.synchronize(device)
                    sol_out = normalize_result(definition, sol_res, device)

                for s_t, r_t in zip(sol_out, ref_out):
                    if tuple(s_t.shape) != tuple(r_t.shape):
                        incorrect = True
                        break
                    if s_t.dtype != r_t.dtype:
                        incorrect = True
                        break
                    if torch.isinf(s_t).any().item() or torch.isnan(s_t).any().item():
                        incorrect = True
                        max_abs = max_rel = float("inf")
                        break
                    a, r, exceeds, _ratio = compute_error_stats(s_t, r_t, cfg)
                    max_abs = max(max_abs, a)
                    max_rel = max(max_rel, r)
                    if exceeds:
                        incorrect = True
                if incorrect:
                    break

            # ---- 计时（用第一组输入）----
            inp0 = inputs_cache[0]
            if is_dps:
                out0 = allocate_outputs(definition, inp0, device)
                sol_args = list(inp0) + out0
            else:
                sol_args = list(inp0)
            sol_ms = time_runnable(sol_runnable, sol_args, cfg.warmup_runs, cfg.iterations, device)

            # ---- 可选：官方 baseline 计时 ----
            base_ms = None
            if baseline_runnable is not None:
                if baseline_is_dps:
                    base_out0 = allocate_outputs(definition, inp0, device)
                    base_args = list(inp0) + base_out0
                else:
                    base_args = list(inp0)
                base_ms = time_runnable(
                    baseline_runnable, base_args, cfg.warmup_runs, cfg.iterations, device
                )

            status = "FAIL" if incorrect else "PASS"
            if not incorrect:
                passed += 1
                sol_latencies.append(sol_ms)

            if baseline_runnable is not None:
                sol_vs_base = base_ms / sol_ms if (base_ms and sol_ms > 0) else float("nan")
                if not incorrect and base_ms is not None:
                    speedups_vs_base.append(sol_vs_base)
                    base_latencies.append(base_ms)
                print(f"{axes_str:<22}{status:<7}{max_rel:>9.2e}"
                      f"{base_ms:>10.4f}{sol_ms:>9.4f}{sol_vs_base:>10.2f}")
            else:
                print(f"{axes_str:<22}{status:<7}{max_rel:>9.2e}{sol_ms:>9.4f}")

        except Exception as exc:  # noqa: BLE001
            print(f"{axes_str:<22}{'ERROR':<7}  {type(exc).__name__}: {exc}")

    print("-" * len(header))
    avg_sol_ms = sum(sol_latencies) / len(sol_latencies) if sol_latencies else float("nan")
    print(f"汇总: {passed}/{total} 通过正确性")
    print(f"  workload 覆盖率 = {total}/{len(declared_workloads)} "
          f"({total / len(declared_workloads):.1%})")
    if skipped_workloads:
        print(f"  缺失 safetensors 跳过 = {len(skipped_workloads)} 个 workload")
    print(f"  平均 latency (solution) = {avg_sol_ms:.4f} ms")
    if baseline_runnable is not None:
        if speedups_vs_base:
            avg_vs_base = sum(speedups_vs_base) / len(speedups_vs_base)
            avg_base_ms = sum(base_latencies) / len(base_latencies) if base_latencies else float("nan")
            verdict = "更快 ✅" if avg_vs_base >= 1.0 else "更慢 ❌"
            print(f"  平均 latency (baseline) = {avg_base_ms:.4f} ms")
            print(f"  平均加速比 (sol vs 官方 baseline '{args.baseline}') "
                  f"= {avg_vs_base:.3f}x  →  我的实现比官方 {verdict}")
        else:
            print(f"  官方 baseline '{args.baseline}' 无有效计时（可能运行失败）")

    return 0 if passed == total and total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
