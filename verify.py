#!/usr/bin/env python3
"""Repository verification entry point for KernelForge-MultiAgent.

By default this script checks the GitHub release layout and validates that the
tracked CUDA source package is present. Pass --cuda to compile and run the
standalone CUDA smoke test in scripts/workflow/benchmarks/cuda/verify_all_kernels.cu.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "scripts" / "workflow" / "benchmarks" / "cuda" / "verify_all_kernels.cu"
DEFAULT_BUILD_DIR = ROOT / "scripts" / "workflow" / "build" / "verify"

REQUIRED_PATHS = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "docs/reproduction.md",
    "docs/PROJECT_AUDIT.md",
    "docs/SUPPORTED_OPERATORS.md",
    "docs/FLASHINFER_BENCH_VALIDATION.md",
    "prompts/README.md",
    "kernels/README.md",
    "kernels/generated/all_operators.cu",
    # 当前主线聚焦的算子类型
    "kernels/operators/gemm/gemm_final.cu",
    "kernels/operators/gqa_paged/gqa_paged_final.cu",
    "kernels/operators/gqa_ragged/gqa_ragged_final.cu",
    "kernels/operators/mla_paged/mla_paged_final.cu",
    "kernels/operators/dsa_paged/dsa_paged_final.cu",
    "kernels/operators/moe/moe_final.cu",
    "kernels/operators/rmsnorm/rmsnorm_final.cu",
    "kernels/operators/rope/rope_final.cu",
    "kernels/operators/sampling/sampling_final.cu",
    "kernels/operators/gdn/gdn_final.cu",
    "skills/KernelWiki/SKILL.md",
    "skills/ncu-report-skill/SKILL.md",
    "scripts/workflow/agents/sub_agents.py",
    "scripts/workflow/agents/dynamic_exploration.py",
    "scripts/workflow/closed_loop_optimizer.py",
    "scripts/workflow/run_optimization_cycle.py",
    "scripts/workflow/benchmarks/cuda/verify_all_kernels.cu",
    "scripts/workflow/baseline/self_attention_baseline.py",
    "scripts/workflow/config/default_config.toml",
    "scripts/workflow/demo/end_to_end_demo.py",
    "scripts/workflow/master/master_agent.py",
]

FORBIDDEN_PATTERNS = [
    "*.o",
    "*.so",
    "*.a",
    "*.out",
    "*.exe",
    "*.ncu-rep",
    "*.nsys-rep",
    "*.log",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_layout() -> bool:
    ok = True
    for item in REQUIRED_PATHS:
        path = ROOT / item
        if path.exists():
            print(f"OK   {item}")
        else:
            print(f"MISS {item}")
            ok = False

    for pattern in FORBIDDEN_PATTERNS:
        matches = [
            p
            for p in ROOT.rglob(pattern)
            if ".git" not in p.parts and "build" not in p.parts
        ]
        if matches:
            ok = False
            for path in matches[:20]:
                print(f"ARTIFACT {rel(path)}")
            if len(matches) > 20:
                print(f"ARTIFACT ... {len(matches) - 20} more matches for {pattern}")

    return ok


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"Required tool not found on PATH: {name}")
    return tool


def build_cuda(args: argparse.Namespace) -> Path:
    nvcc = require_tool("nvcc")
    if sys.platform.startswith("win") and shutil.which("cl") is None:
        raise RuntimeError(
            "nvcc is available, but cl.exe is not on PATH. Open a Visual Studio "
            "Developer PowerShell or add MSVC build tools to PATH, then retry."
        )

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    build_dir = Path(args.build_dir).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / ("verify_all_kernels.exe" if sys.platform.startswith("win") else "verify_all_kernels")

    cmd = [
        nvcc,
        str(source),
        "-O3",
        "-lineinfo",
        f"-arch={args.arch}",
        "-o",
        str(output),
    ]
    print("RUN  " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return output


def run_cuda(args: argparse.Namespace) -> bool:
    binary = build_cuda(args)
    print(f"RUN  {binary}")
    completed = subprocess.run([str(binary)], cwd=ROOT)
    return completed.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Compile and run the CUDA smoke test.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="CUDA verification source file.")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR), help="Directory for compiled verification binaries.")
    parser.add_argument("--arch", default="sm_120", help="NVCC GPU architecture, for example sm_120 or sm_100.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout_ok = check_layout()
    if not layout_ok:
        return 1
    if args.cuda:
        try:
            return 0 if run_cuda(args) else 1
        except Exception as exc:
            print(f"CUDA verification failed before completion: {exc}", file=sys.stderr)
            return 2
    print("Layout verification passed. Use --cuda for the CUDA smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
