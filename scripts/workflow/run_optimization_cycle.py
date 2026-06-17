#!/usr/bin/env python3
"""One optimization-cycle harness generator.

This script prepares a CUDA benchmark harness for a named operator, compiles it
with the current optimized kernel, optionally profiles it with NCU, and writes a
JSON feedback file for the next agent round.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from operator_policy import canonicalize_family, list_primary_families  # type: ignore


OPERATOR_SOURCES = {
    # 当前主线只允许六类 family；底层先复用仓库现有实现目录
    "dsa_sparse_attention": REPO_ROOT / "kernels" / "operators" / "dsa_paged" / "dsa_paged_final.cu",
    "gdn_prefill": REPO_ROOT / "kernels" / "operators" / "gdn" / "gdn_final.cu",
    "gdn_decode": REPO_ROOT / "kernels" / "operators" / "gdn" / "gdn_final.cu",
    "dsa_topk_indexer": REPO_ROOT / "kernels" / "operators" / "dsa_paged" / "dsa_paged_final.cu",
    "paged_attention": REPO_ROOT / "kernels" / "operators" / "gqa_paged" / "gqa_paged_final.cu",
    "moe_fp8": REPO_ROOT / "kernels" / "operators" / "moe" / "moe_final.cu",
}


class OptimizationCycle:
    """Compile, benchmark, and profile one operator variant."""

    def __init__(self, kernel_name: str, iteration: int = 1, work_dir: Optional[Path] = None):
        self.kernel_name = kernel_name
        self.operator = self._operator_from_name(kernel_name)
        self.iteration = iteration
        self.work_dir = Path(work_dir).resolve() if work_dir else WORKFLOW_ROOT
        self.generated_dir = self.work_dir / "benchmarks" / "generated"
        self.test_file = self.generated_dir / f"test_{kernel_name}_opt.cu"
        executable_name = f"test_{kernel_name}_opt.exe" if os.name == "nt" else f"test_{kernel_name}_opt"
        self.executable = self.generated_dir / executable_name
        self.ncu_report = self.work_dir / f"{kernel_name}_ncu_report"
        self.feedback_file = self.work_dir / f"{kernel_name}_feedback_iter{iteration}.json"
        self.opt_file = self._resolve_kernel_file()

    def run(self) -> bool:
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.test_file.write_text(self._generate_test_harness(), encoding="utf-8")

        compile_result = self._compile()
        if compile_result.returncode != 0:
            self._write_feedback(
                {
                    "status": "compile_failed",
                    "stderr": compile_result.stderr[-4000:],
                    "stdout": compile_result.stdout[-4000:],
                }
            )
            return False

        benchmark_result = self._run_benchmark()
        if benchmark_result.returncode != 0:
            self._write_feedback(
                {
                    "status": "benchmark_failed",
                    "stderr": benchmark_result.stderr[-4000:],
                    "stdout": benchmark_result.stdout[-4000:],
                }
            )
            return False

        profile_result = self._run_ncu()
        feedback: Dict[str, Any] = {
            "status": "ok",
            "kernel": self.kernel_name,
            "operator": self.operator,
            "iteration": self.iteration,
            "kernel_file": str(self.opt_file),
            "test_file": str(self.test_file),
            "benchmark_stdout": benchmark_result.stdout,
            "profile_generated": profile_result is not None and profile_result.returncode == 0,
            "created_at": datetime.now().isoformat(),
        }
        if profile_result is not None and profile_result.returncode != 0:
            feedback["profile_warning"] = profile_result.stderr[-4000:]

        self._write_feedback(feedback)
        return True

    def _resolve_kernel_file(self) -> Path:
        candidates = [
            self.work_dir / f"{self.kernel_name}_opt.cu",
            self.work_dir / f"{self.operator}_opt.cu",
            REPO_ROOT / "kernels" / "generated" / f"{self.kernel_name}_opt.cu",
            REPO_ROOT / "kernels" / "generated" / f"{self.operator}_opt.cu",
            OPERATOR_SOURCES[self.operator],
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"No CUDA source found for {self.kernel_name}. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    def _compile(self) -> subprocess.CompletedProcess[str]:
        command = ["nvcc", "-O3", "-std=c++17", str(self.test_file), str(self.opt_file)]
        command.extend(str(path) for path in self._extra_compile_sources())
        command.extend(["-o", str(self.executable)])
        return subprocess.run(command, capture_output=True, text=True, cwd=self.work_dir)

    def _run_benchmark(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(self.executable)], capture_output=True, text=True, cwd=self.work_dir)

    def _run_ncu(self) -> Optional[subprocess.CompletedProcess[str]]:
        command = [
            "/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu",
            "--set",
            "full",
            "--target-processes",
            "all",
            "-o",
            str(self.ncu_report),
            str(self.executable),
        ]
        try:
            return subprocess.run(command, capture_output=True, text=True, cwd=self.work_dir)
        except FileNotFoundError:
            return None

    def _write_feedback(self, feedback: Dict[str, Any]) -> None:
        self.feedback_file.write_text(json.dumps(feedback, indent=2), encoding="utf-8")

    def _generate_test_harness(self) -> str:
        if self.operator == "paged_attention":
            return self._generate_gqa_paged_test()
        if self.operator in {"dsa_sparse_attention", "dsa_topk_indexer"}:
            return self._generate_dsa_paged_test()
        if self.operator == "moe_fp8":
            return self._generate_moe_test()
        if self.operator in {"gdn_prefill", "gdn_decode"}:
            return self._generate_gdn_test()
        raise ValueError(f"Unsupported operator: {self.operator}")

    @staticmethod
    def _operator_from_name(kernel_name: str) -> str:
        lowered = kernel_name.lower().replace("-", "_")
        if "dsa_sparse_attention" in lowered:
            return "dsa_sparse_attention"
        if "dsa_topk_indexer" in lowered:
            return "dsa_topk_indexer"
        if "gdn_prefill" in lowered:
            return "gdn_prefill"
        if "gdn_decode" in lowered:
            return "gdn_decode"
        if any(token in lowered for token in ("paged_attention", "gqa_paged_decode", "mla_paged_decode", "mla_paged_prefill")):
            return "paged_attention"
        if "moe_fp8" in lowered or ("moe" in lowered and "fp8" in lowered):
            return "moe_fp8"

        canonical = canonicalize_family(lowered)
        if canonical is not None:
            return canonical
        raise ValueError(
            f"Cannot infer operator from kernel name: {kernel_name}. "
            f"Allowed mainline families: {', '.join(list_primary_families())}"
        )

    @staticmethod
    def _generate_gemm_test() -> str:
        """生成 GEMM (C = A × B^T) 测试代码"""
        return r'''
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// GEMM kernel: C = A × B^T
void launch_gemm_optimized(
    const half* d_A, const half* d_B, half* d_C,
    int M, int N, int K, cudaStream_t stream
);

int main() {
    const int M = 512, N = 512, K = 512;
    half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, N * K * sizeof(half));  // B is [N, K] for B^T
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemset(d_A, 0, M * K * sizeof(half));
    cudaMemset(d_B, 0, N * K * sizeof(half));
    cudaMemset(d_C, 0, M * N * sizeof(half));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_gemm_optimized(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        launch_gemm_optimized(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("{\"avg_time_ms\": %.6f, \"operator\": \"gemm\", \"shape\": \"M=%d,N=%d,K=%d\"}\n",
                ms / 100.0f, M, N, K);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
'''

    @staticmethod
    def _generate_gqa_paged_test() -> str:
        """生成 GQA-Paged 测试代码（占位符）"""
        return r'''
#include <cuda_runtime.h>
#include <cstdio>

// TODO: Implement GQA-Paged test harness
int main() {
    std::printf("{\"status\": \"test_not_implemented\", \"operator\": \"gqa_paged\"}\n");
    return 0;
}
'''

    @staticmethod
    def _generate_gqa_ragged_test() -> str:
        """生成 GQA-Ragged 测试代码（占位符）"""
        return r'''
#include <cuda_runtime.h>
#include <cstdio>

// TODO: Implement GQA-Ragged test harness
int main() {
    std::printf("{\"status\": \"test_not_implemented\", \"operator\": \"gqa_ragged\"}\n");
    return 0;
}
'''

    @staticmethod
    def _generate_mla_paged_test() -> str:
        """生成 MLA-Paged 测试代码（占位符）"""
        return r'''
#include <cuda_runtime.h>
#include <cstdio>

// TODO: Implement MLA-Paged test harness
int main() {
    std::printf("{\"status\": \"test_not_implemented\", \"operator\": \"mla_paged\"}\n");
    return 0;
}
'''

    @staticmethod
    def _generate_dsa_paged_test() -> str:
        """生成 DSA-Paged 测试代码（占位符）"""
        return r'''
#include <cuda_runtime.h>
#include <cstdio>

// TODO: Implement DSA-Paged test harness
int main() {
    std::printf("{\"status\": \"test_not_implemented\", \"operator\": \"dsa_paged\"}\n");
    return 0;
}
'''

    @staticmethod
    def _generate_moe_test() -> str:
        """生成 MoE 测试代码（占位符）"""
        return r'''
#include <cuda_runtime.h>
#include <cstdio>

// TODO: Implement MoE test harness
int main() {
    std::printf("{\"status\": \"test_not_implemented\", \"operator\": \"moe\"}\n");
    return 0;
}
'''

    @staticmethod
    def _generate_rope_test() -> str:
        """生成 RoPE 测试代码（占位符）"""
        return r'''
#include <cuda_runtime.h>
#include <cstdio>

// TODO: Implement RoPE test harness
int main() {
    std::printf("{\"status\": \"test_not_implemented\", \"operator\": \"rope\"}\n");
    return 0;
}
'''

    @staticmethod
    def _generate_sampling_test() -> str:
        """生成 Sampling 测试代码（占位符）"""
        return r'''
#include <cuda_runtime.h>
#include <cstdio>

// TODO: Implement Sampling test harness
int main() {
    std::printf("{\"status\": \"test_not_implemented\", \"operator\": \"sampling\"}\n");
    return 0;
}
'''

    @staticmethod
    def _generate_gdn_test() -> str:
        """生成 GDN 测试代码（占位符）"""
        return r'''
#include <cuda_runtime.h>
#include <cstdio>

// TODO: Implement GDN test harness
int main() {
    std::printf("{\"status\": \"test_not_implemented\", \"operator\": \"gdn\"}\n");
    return 0;
}
'''
        return r'''
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void launch_matmul_optimized(
    const half* d_A, const half* d_B, half* d_C,
    int M, int N, int K, cudaStream_t stream
);

int main() {
    const int M = 512, N = 512, K = 512;
    half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemset(d_A, 0, M * K * sizeof(half));
    cudaMemset(d_B, 0, K * N * sizeof(half));
    cudaMemset(d_C, 0, M * N * sizeof(half));

    for (int i = 0; i < 10; ++i) {
        launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("{\"avg_time_ms\": %.6f}\n", ms / 100.0f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
'''

    @staticmethod
    def _generate_rmsnorm_test() -> str:
        """生成 RMSNorm 测试代码"""
        return r'''
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

void launch_rmsnorm_optimized(
    const half* input, half* output, const half* weight,
    int batch_size, int hidden_size, cudaStream_t stream
);

void launch_rmsnorm_true_naive_ex(
    const half* input, half* output, const half* weight,
    int batch_size, int hidden_size, float eps, cudaStream_t stream
);

static void launch_rmsnorm_naive(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    cudaStream_t stream
) {
    launch_rmsnorm_true_naive_ex(input, output, weight, batch_size, hidden_size, 1.0e-6f, stream);
}

static float benchmark_kernel(
    void (*kernel)(const half*, half*, const half*, int, int, cudaStream_t),
    const half* input,
    half* output,
    const half* weight,
    int batch,
    int hidden
) {
    for (int i = 0; i < 10; ++i) {
        kernel(input, output, weight, batch, hidden, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        kernel(input, output, weight, batch, hidden, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}

int main() {
    const int batches[] = {1, 16, 64, 256, 1024};
    const int num_batches = sizeof(batches) / sizeof(batches[0]);
    const int max_batch = 1024;
    const int hidden = 4096;
    half *d_input = nullptr, *d_output = nullptr, *d_output_naive = nullptr, *d_weight = nullptr;
    cudaMalloc(&d_input, max_batch * hidden * sizeof(half));
    cudaMalloc(&d_output, max_batch * hidden * sizeof(half));
    cudaMalloc(&d_output_naive, max_batch * hidden * sizeof(half));
    cudaMalloc(&d_weight, hidden * sizeof(half));
    cudaMemset(d_input, 0, max_batch * hidden * sizeof(half));
    cudaMemset(d_output, 0, max_batch * hidden * sizeof(half));
    cudaMemset(d_output_naive, 0, max_batch * hidden * sizeof(half));
    cudaMemset(d_weight, 0, hidden * sizeof(half));

    float log_speedup_sum = 0.0f;
    int valid_cases = 0;
    std::printf("{\"operator\":\"rmsnorm\",\"hidden\":%d,\"cases\":[", hidden);
    for (int idx = 0; idx < num_batches; ++idx) {
        const int batch = batches[idx];
        const float naive_ms = benchmark_kernel(
            launch_rmsnorm_naive,
            d_input,
            d_output_naive,
            d_weight,
            batch,
            hidden
        );
        const float opt_ms = benchmark_kernel(
            launch_rmsnorm_optimized,
            d_input,
            d_output,
            d_weight,
            batch,
            hidden
        );

        const float speedup = (opt_ms > 0.0f) ? (naive_ms / opt_ms) : 0.0f;
        if (naive_ms > 0.0f && opt_ms > 0.0f) {
            log_speedup_sum += logf(speedup);
            ++valid_cases;
        }

        if (idx != 0) {
            std::printf(",");
        }
        std::printf(
            "{\"batch\":%d,\"naive_ms\":%.6f,\"opt_ms\":%.6f,\"speedup\":%.6f}",
            batch,
            naive_ms,
            opt_ms,
            speedup
        );
    }
    const float geomean_speedup = valid_cases > 0 ? expf(log_speedup_sum / valid_cases) : 0.0f;
    std::printf("],\"geomean_speedup\":%.6f}\n", geomean_speedup);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_naive);
    cudaFree(d_weight);
    return 0;
}
'''

    def _extra_compile_sources(self) -> list[Path]:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one CUDA optimization feedback cycle")
    parser.add_argument(
        "kernel_name",
        help="Kernel 或主线 family 名（dsa_sparse_attention, gdn_prefill, gdn_decode, dsa_topk_indexer, paged_attention, moe_fp8）",
    )
    parser.add_argument("iteration", nargs="?", type=int, default=1)
    args = parser.parse_args()

    try:
        cycle = OptimizationCycle(args.kernel_name, args.iteration)
        return 0 if cycle.run() else 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
