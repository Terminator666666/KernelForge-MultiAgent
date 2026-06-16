# Reproduction

This repository is organized as a lightweight GitHub release for the
KernelForge-MultiAgent workflow. It keeps the prompts, skills, CUDA kernel
source package, validation programs, and project notes needed to reproduce the
optimization process.

## Environment

Use a CUDA-capable machine with:

- Python 3.10 to 3.13;
- NVIDIA CUDA Toolkit with `nvcc`;
- Nsight Compute for profiling;
- PyTorch and Triton if you run the Python or Triton-side experiments;
- Visual Studio Build Tools on Windows when compiling with `nvcc`.

Install Python dependencies with either `pip` or `uv`:

```bash
python -m pip install -r requirements.txt
```

On Windows, run CUDA compilation from a Visual Studio Developer PowerShell so
`cl.exe` is visible to `nvcc`.

## Verify the Release Layout

Run the repository verifier:

```bash
python verify.py
```

This checks that the GitHub release structure is present and that common binary
or profiler artifacts are not tracked.

## Run the CUDA Smoke Test

Compile and run the standalone CUDA verification program:

```bash
python verify.py --cuda --arch sm_120
```

Use `--arch sm_100` on B200, `--arch sm_90` on H100, or another architecture
matching your GPU. The smoke test source is:

```text
scripts/workflow/benchmarks/cuda/verify_all_kernels.cu
```

The verifier builds into `scripts/workflow/build/verify/`, which is ignored by Git.

## Run a Benchmark

The true-naive benchmark compares optimized kernels with deliberately simple
GPU baselines. Example for RMSNorm:

```bash
nvcc scripts/workflow/benchmarks/cuda/benchmark_true_naive.cu -O3 -lineinfo -arch=sm_120 -o scripts/workflow/build/benchmark_true_naive
scripts/workflow/build/benchmark_true_naive
```

Supported operators: `gemm`, `gqa_paged`, `gqa_ragged`, `mla_paged`, `dsa_paged`, 
`moe`, `rmsnorm`, `rope`, `sampling`, `gdn`.

Linux users can run the generated binary directly from `scripts/workflow/build/`.

## Run the Agent Loop

The closed-loop optimizer is kept under `scripts/workflow/` as a prototype
entry point. Example for RMSNorm:

```bash
python scripts/workflow/closed_loop_optimizer.py rmsnorm-h7168 3
```

Supported operator types: `gemm`, `gqa_paged`, `gqa_ragged`, `mla_paged`, `dsa_paged`, 
`moe`, `rmsnorm`, `rope`, `sampling`, `gdn`.

Use the phase prompts under `prompts/` to guide an agent session. Phase 1
establishes correctness, Phase 2 optimizes with profiler evidence, and Phase 3
validates and packages the final kernel.

## Skills

The workflow can consult two core skills under `skills/`:

- `skills/KernelWiki` — Blackwell/Hopper kernel optimization knowledge base
- `skills/ncu-report-skill` — Nsight Compute profiling and analysis

`KernelWiki` is intentionally large. If repository size is more important than
self-contained operation, publish it as a submodule instead of committing the
whole directory.

## Acceptance Criteria

**All generated operator code must pass FlashInfer-Bench validation to be considered successful.**

FlashInfer-Bench is the official benchmark suite from the FlashInfer project that validates:
- Correctness against reference implementations
- Performance on real-world workloads from the FlashInfer-Trace dataset
- Compliance with production kernel standards

### Running FlashInfer-Bench Validation

```bash
cd D:/Agent/flashinfer-bench-main/flashinfer-bench-main

# Install if needed
pip install flashinfer-bench

# Clone the FlashInfer-Trace dataset (with Git LFS pointer files only)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace

# Run validation against the dataset
flashinfer-bench run --local flashinfer-trace
```

The validation suite tests your kernel implementations against:
- Multiple operator types (attention, GEMM, normalization, MoE, etc.)
- Various workload configurations (batch sizes, sequence lengths, tensor shapes)
- Different data types (FP16, BF16, FP8, etc.)
- Real-world inference traces from production models

### Internal Development Workflow

Internal benchmarks and smoke tests are provided for rapid development iteration:

```bash
# Quick smoke test during development
python verify.py --cuda --arch sm_120

# Internal benchmark for performance comparison
nvcc scripts/workflow/benchmarks/cuda/benchmark_true_naive.cu -O3 -lineinfo -arch=sm_120 -o scripts/workflow/build/benchmark_true_naive
scripts/workflow/build/benchmark_true_naive
```

These internal tests help catch obvious errors early, but **they do not constitute acceptance**.

### False Speedup Controls

When developing kernels, reject any optimization that:

- fails correctness checks;
- uses non-deterministic inputs;
- omits warmup or synchronization;
- uses CPU timers instead of CUDA events;
- compares against a broken or unrealistic baseline;
- benchmarks a different kernel than the one being submitted.

If correctness fails, the candidate is rejected even when its measured latency is lower.

**Final acceptance requires passing FlashInfer-Bench validation with the official FlashInfer-Trace dataset.**

For detailed validation procedures, troubleshooting, and integration guidance, see [`FLASHINFER_BENCH_VALIDATION.md`](FLASHINFER_BENCH_VALIDATION.md).
