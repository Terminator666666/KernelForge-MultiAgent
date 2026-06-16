# RMSNorm 优化家族

本文件记录 `rmsnorm_h4096` 的真实闭环优化状态。当前已经有可验证的 Round 0 样板，
正确性通过，但相对官方 FlashInfer baseline 仍然偏慢，因此继续作为后续轮次的锚点。
当前仓库内已经补入 Round 1 候选实现，等待 Linux 环境下完成 FlashInfer-Bench 与
闭环 benchmark 验证后再决定是否升为新锚点。

---

## 家族信息

- **算子类型**: `rmsnorm`
- **definition**: `rmsnorm_h4096`
- **目标硬件**: RTX 5070 Laptop / sm_120（当前实测环境）
- **绑定方式**: `tvm-ffi`
- **官方 baseline**: `flashinfer_wrapper_2e27cd`

---

## 当前锚点变体

**变体 ID**: `round0-v1`  
**solution 名**: `kernelforge_rmsnorm_h4096_cuda_v1`  
**当前成绩**: `sol/base = 0.893x`  
**状态**: `REJECT，但保留为继续优化锚点`

**关键特性**:
- bfloat16 输入/输出，float32 累加
- `bfloat162` 向量化访存
- warp + shared memory 两级归约
- TVM-FFI DPS 接口，已接入 FlashInfer-Bench

**当前限制**:
- 小 batch 时并行度不足
- 访存事务不够紧凑，内存吞吐显著落后官方 baseline

---

## 当前候选变体

**变体 ID**: `round1-v2`  
**solution 名**: `kernelforge_rmsnorm_h4096_cuda_v2`  
**状态**: `PENDING（已实现，待验证）`

**代码位置**:
- `kernels/operators/rmsnorm/rmsnorm_h4096_tvmffi.cu`
- `kernels/operators/rmsnorm/rmsnorm_final.cu`

**本轮改动**:
- 增加小 batch 分裂并行路径：`partial_sum -> finalize_inv -> apply`
- 保留大 batch 单阶段快路径，避免额外 launch 开销
- 对通用工作流补齐 `launch_rmsnorm_optimized(batch, hidden, stream)` 接口
- 让闭环 harness 同时编译并对比 `rmsnorm_true_naive.cu`

**待验证点**:
- FlashInfer-Bench `rmsnorm_h4096` 正确性是否继续保持 `14/14`
- 小 batch（1/16/64）下 `sol/base` 是否相对 `round0-v1` 改善
- 三阶段路径的 launch 开销是否在 batch=128 附近仍然可接受

---

## 优化历史

### Round 0: 样板实现与基线验证
- **正确性**: `14/14` workload 通过
- **性能**: `sol/base = 0.893x`
- **决策**: `REJECT`
- **原因**: 正确性已达标，但未达到 `1.05x` 的验收线
- **锚点代码**: `reference/rmsnorm/variants/round0-v1/kernel.cu`

### Round 1: 深度闭环候选已落地，等待验证
- **候选 ID**: `round1-v2`
- **状态**: `PENDING`
- **实现位置**:
  - `kernels/operators/rmsnorm/rmsnorm_h4096_tvmffi.cu`
  - `kernels/operators/rmsnorm/rmsnorm_final.cu`
- **闭环增强**:
  - `scripts/workflow/run_optimization_cycle.py` 现可对比 true-naive 基线
  - benchmark 输出改为多 batch case + geomean speedup
- **说明**: 当前仅完成代码落地，尚未在 Linux 环境执行 benchmark / FIB / NCU

---

## 当前证据

### FlashInfer-Bench
- **definition**: `rmsnorm_h4096`
- **候选 solution**: `kernelforge_rmsnorm_h4096_cuda_v1`
- **官方 baseline**: `flashinfer_wrapper_2e27cd`

### NCU 结论
- batch=16 时，我的实现内存吞吐约 `5.70%`
- 同条件下官方 baseline 约 `35.99%`
- grid=16 仅覆盖约 `1/3 SM`

**结论**:
- 当前瓶颈不是数值正确性，而是小 batch 并行度与访存效率
- 下一轮应先解决 occupancy / memory transaction 问题，再谈更激进的融合

---

## 变体树

```text
round0-v1  (validated anchor, rejected for performance)
└── round1-v2  (implemented, pending validation)
```

---

## 下一轮方向

1. **先验证 round1-v2**
   - 在 Linux 环境中跑 FlashInfer-Bench 正确性与多 batch benchmark
2. **基于 NCU 判断 split 阈值**
   - 检查 `batch<=8/32/128` 的分裂策略是否合理
3. **必要时继续收紧访存**
   - 如果 apply 阶段仍是瓶颈，再考虑更细粒度 chunk 调度或输出阶段并行切分

---

## 相关文件

- `reference/rmsnorm/baseline.json`: 当前闭环状态源
- `reference/rmsnorm/solutions.jsonl`: 变体 DAG
- `reference/rmsnorm/TRAPS.md`: 陷阱库
- `reference/rmsnorm/variants/round0-v1/kernel.cu`: 当前锚点实现
- `kernels/operators/rmsnorm/rmsnorm_h4096_tvmffi.cu`: 当前 Round 1 候选实现

---

**最后更新**: 2026-06-16  
**维护者**: KernelForge-MultiAgent  
**状态**: Round 0 已验证，Round 1 待继续优化
