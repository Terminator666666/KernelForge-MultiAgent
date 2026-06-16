# RMSNorm Phase 2: Profile-Guided Optimization

Goal: improve RMSNorm latency using benchmark and profiler evidence.

Requirements:

- Benchmark against the true-naive RMSNorm baseline.
- Use NCU evidence when available.
- Inspect reduction strategy, vectorized memory access, occupancy, register
  pressure, and shared-memory efficiency.
- Keep correctness gates active before timing claims.

Deliverable:

- Optimized RMSNorm implementation.
- Benchmark table with latency and valid speedup.
- Evidence-backed bottleneck diagnosis.
