# Project File Audit

This audit records the GitHub release layout after reorganizing the project to
match the style of `mlsys2026-flashinfer-contest-main`.

## Top-Level Release Layout

| Path | Status | Notes |
| --- | --- | --- |
| `README.md` | Keep | Main repository entry point. |
| `verify.py` | Keep | Lightweight layout verifier and optional CUDA smoke-test entry point. |
| `pyproject.toml`, `requirements.txt` | Keep | Python dependency metadata. |
| `CLAUDE.md` | Keep | Agent-facing repository notes. |
| `docs/` | Keep | Reproduction notes, this audit, and legacy documentation. |
| `prompts/` | Keep | Phase prompts organized by operator. |
| `scripts/` | Keep | Release helper plus workflow/legacy subdirectories. |
| `skills/` | Keep | Local skills, including the requested `KernelWiki` and `ncu-report-skill`. |
| `kernels/` | Keep | Project-specific addition: consolidated CUDA operator source package. |

The reference FlashInfer repository has `docs/`, `prompts/`, `scripts/`, and
`skills/` as the main directories. This project keeps the same release shape
and adds `kernels/` because the user explicitly requested all generated
operator source to be published from one place.

## Publication Status

The repository is ready to upload as a public source snapshot of the agent
architecture after the owner chooses and adds a license. The active code path is
structured and documented, but the optimizer should be described as
prompt-driven and prototype-grade rather than fully autonomous production
automation.

No `uv.lock` or `.gitmodules` file is required for this layout. If `KernelWiki`
is too large for the intended GitHub repository, convert it to a submodule or a
separate release artifact before publishing.

## Workflow Code

Original engineering directories were moved under `scripts/workflow/` so they
do not clutter the GitHub root:

| Current Path | Purpose |
| --- | --- |
| `scripts/workflow/agents/` | Multi-agent abstractions. |
| `scripts/workflow/master/` | Campaign orchestration prototype. |
| `scripts/workflow/benchmarks/` | CUDA validation programs and legacy benchmark tests. |
| `scripts/workflow/config/` | Runtime/API configuration templates. |
| `scripts/workflow/baseline/` | PyTorch baseline experiment. |
| `scripts/workflow/demo/` | End-to-end multi-agent demo. |
| `scripts/workflow/*.py` | Active workflow helper scripts. |

Old path-specific shell/Python helpers were moved to `scripts/legacy/`. They
are preserved for historical context but are not the release entry points.

## Kernel Source Package

Maintained operator code is centralized under the current mainline-only package:

- `kernels/operators/dsa_paged/`
- `kernels/operators/gdn/`
- `kernels/generated/all_operators.cu`

The repository intentionally removed non-mainline operator source directories
from `kernels/operators/` and now keeps only the source buckets that map to the
active optimization families.

Compiled objects, `.so` files, Nsight Compute reports, logs, and generated
benchmark outputs are excluded by `.gitignore`.

## Useful Content

- `kernels/`: useful and should be committed.
- `prompts/`: useful for reproducing the agent workflow.
- `skills/KernelWiki`: useful but large; keep for a self-contained release, or
  convert to a submodule if repository size matters.
- `skills/ncu-report-skill`: useful for profiler-driven diagnosis.
- `docs/reproduction.md`: useful for GitHub readers.
- `scripts/workflow/benchmarks/cuda/verify_all_kernels.cu`: useful CUDA smoke
  test source.

## Legacy Or Optional Content

- `docs/legacy/`: old documents with historical value, not primary docs.
- `scripts/legacy/`: early scripts that rely on older path assumptions.
- `scripts/workflow/benchmarks/legacy_python/`: old Python benchmark experiments.

## Removed Or Ignored Content

The cleaned release excludes:

- compiled executables and libraries;
- `.o`, `.so`, `.a`, `.out`, `.exe`;
- `.ncu-rep`, `.nsys-rep`, generated profiler outputs;
- logs and `__pycache__`;
- benchmark result artifacts;
- local datasets and model checkpoints;
- real API keys or other secrets.

## False Speedup Policy

A reported speedup is valid only when:

- deterministic input initialization is used;
- optimized and baseline outputs match within tolerance;
- warmup and synchronization are included;
- timing uses CUDA events or another GPU-safe method;
- the baseline is a real CPU or true-naive GPU reference;
- the benchmark calls the actual final kernel implementation.

If correctness fails, the candidate is rejected even when timing looks faster.
