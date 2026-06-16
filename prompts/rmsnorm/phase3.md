# RMSNorm Phase 3: Final Validation

Goal: package the final RMSNorm kernel for release.

Requirements:

- Validate multiple hidden sizes and batch sizes.
- Re-run timing with warmup, CUDA events, and synchronization.
- Confirm the benchmark calls `kernels/operators/rmsnorm/rmsnorm_final.cu`.
- Reject false speedups caused by skipped RMS computation, skipped scale
  application, invalid epsilon handling, or broken baselines.

Deliverable:

- Final source in `kernels/operators/rmsnorm/`.
- Validation summary suitable for release notes.
