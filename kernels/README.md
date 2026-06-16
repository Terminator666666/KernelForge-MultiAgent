# Kernels

This directory contains the CUDA operator source code that should be kept in the repository.

## Layout

- `operators/softmax`: final and true-naive Softmax kernels.
- `operators/matmul`: final and true-naive FP16 MatMul kernels.
- `operators/layernorm`: final and true-naive LayerNorm kernels.
- `operators/rmsnorm`: final and true-naive RMSNorm kernels.
- `generated/all_operators.cu`: convenience include file for building the current operator set.
- `reference_archive`: reserved for historical search traces and iteration artifacts when they are intentionally published.

The old root-level `.cu` files were consolidated here where recoverable or moved to `benchmarks/cuda`.
Compiled objects, `.so` files, NCU reports, logs, and executable binaries are intentionally excluded.
