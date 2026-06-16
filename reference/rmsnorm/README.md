# RMSNorm 优化家族

本文件记录 `rmsnorm_h4096` 的真实闭环优化状态。当前已经有可验证的 Round 0 样板，
正确性通过，但相对官方 FlashInfer baseline 仍然偏慢，因此继续作为后续轮次的锚点。
当前已经完成 Round 1 与 Round 2 的真实闭环验证，二者都被 REJECT：
Round 1 的三段式 split 路径失败，Round 2 的单阶段寄存器预算路径也未追平官方 baseline，
因此继续保留 `round0-v1` 作为锚点。

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

## 最近候选变体

**变体 ID**: `round2-v1`  
**solution 名**: `kernelforge_rmsnorm_h4096_cuda_v3`  
**状态**: `REJECT（14/14 正确，但 sol/base = 0.710x）`

**代码位置**:
- `reference/rmsnorm/variants/round2-v1/kernel.cu`

**本轮改动**:
- 回退到单阶段 kernel，移除 Round 1 的三段式 split 路径
- 引入 `__launch_bounds__(256, 4)` 做寄存器预算实验
- 保留 `bfloat162` 向量化，并对输入/权重尝试不同的 cache hint
- 增加 2x ILP 的向量化循环步进

**真实结论**:
- FlashInfer-Bench `rmsnorm_h4096` 正确性继续保持 `14/14`
- 平均 `sol/base = 0.710x`，仍显著低于 `round0-v1`
- batch=16 的真实 NCU 证明单阶段寄存器预算仍无法解决结构性并行度不足，且 `long scoreboard stalls` 更重

---

## 优化历史

### Round 0: 样板实现与基线验证
- **正确性**: `14/14` workload 通过
- **性能**: `sol/base = 0.893x`
- **决策**: `REJECT`
- **原因**: 正确性已达标，但未达到 `1.05x` 的验收线
- **锚点代码**: `reference/rmsnorm/variants/round0-v1/kernel.cu`

### Round 1: 三段式 split 路径验证失败
- **候选 ID**: `round1-v2`
- **状态**: `REJECT`
- **实现位置**:
  - `kernels/operators/rmsnorm/rmsnorm_h4096_tvmffi.cu`
  - `kernels/operators/rmsnorm/rmsnorm_final.cu`
- **正确性**: `14/14`
- **性能**: `sol/base = 0.701x`
- **NCU 结论**:
  - `partial/apply` 仅 `64 blocks`，约 `0.296 waves/SM`
  - `finalize` 仅 `16 blocks`，约 `0.074 waves/SM`
  - `long/short scoreboard stalls` 明显高于官方 baseline
- **决策**: 放弃三段式 split 主方向，下一轮回到单阶段主路径

### Round 2: 单阶段寄存器预算路径仍失败
- **候选 ID**: `round2-v1`
- **状态**: `REJECT`
- **实现位置**:
  - `reference/rmsnorm/variants/round2-v1/kernel.cu`
- **正确性**: `14/14`
- **性能**: `sol/base = 0.710x`
- **NCU 结论**:
  - `single-pass` 仍只有 `16 blocks`，约 `0.074 waves/SM`
  - `registers/thread = 28`，但 `long scoreboard stalls = 16.49`
  - `memory throughput = 4.21%`，仍低于官方 baseline 的 `6.54%`
- **决策**: 仅靠寄存器预算和 cache hint 不足以收敛，下一轮必须直接解决小 batch 下的 CTA 数不足

---

## 当前证据

### FlashInfer-Bench
- **definition**: `rmsnorm_h4096`
- **候选 solution**: `kernelforge_rmsnorm_h4096_cuda_v1`
- **官方 baseline**: `flashinfer_wrapper_2e27cd`

### NCU 结论
- Round 1 split 路径：`partial/apply` 约 `0.296 waves/SM`，`finalize` 约 `0.074 waves/SM`
- Round 2 single-pass 路径：`16 blocks`，约 `0.074 waves/SM`，`long scoreboard = 16.49`
- 同条件下官方 baseline 的 `memory throughput` 仍高于当前候选

**结论**:
- 当前瓶颈不是数值正确性，而是小 batch 下的结构性并行度不足与严重访存等待
- 下一轮需要在不引入独立 finalize 小 grid 的前提下，显著增加每行可并行 CTA 数

---

## 变体树

```text
round0-v1  (validated anchor, rejected for performance)
├── round1-v2  (validated, rejected; split path regressed)
└── round2-v1  (validated, rejected; single-pass tuning regressed)
```

---

## 下一轮方向

1. **直接解决 batch=16 只有 16 CTA 的结构性问题**
   - 目标不是微调单 CTA，而是显式增加每行可并行工作数
2. **避免重走 Round 1 的 finalize 小 grid**
   - 不再接受独立 `finalize` 阶段这种极低 waves/SM 的结构
3. **考虑单 kernel 的多 CTA/row 收尾方案**
   - 保持 `bfloat162` 向量化，同时让并行度提升不再靠多次 launch

---

## 相关文件

- `reference/rmsnorm/baseline.json`: 当前闭环状态源
- `reference/rmsnorm/solutions.jsonl`: 变体 DAG
- `reference/rmsnorm/TRAPS.md`: 陷阱库
- `reference/rmsnorm/variants/round0-v1/kernel.cu`: 当前锚点实现
- `reference/rmsnorm/variants/round2-v1/kernel.cu`: Round 2 被拒候选

---

**最后更新**: 2026-06-16  
**维护者**: KernelForge-MultiAgent  
**状态**: Round 0 已验证；Round 1、Round 2 已 reject；准备进入 Round 3
