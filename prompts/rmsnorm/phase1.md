# RMSNorm Phase 1 Prompt

开发一个 CUDA 内核，在保持数值正确性的同时最小化延迟。目标机器是 NVIDIA B200，软件环境是 CUDA 13.2。

## 内核信息

- **算子类型**: `rmsnorm`
- **操作**: RMS Layer Normalization
- **FlashInfer-Bench 对应**: `rmsnorm`

### 输入输出规范

**标准 RMSNorm**:
- 输入: `hidden_states` [batch_size, hidden_size], `weight` [hidden_size]
- 输出: `output` [batch_size, hidden_size]

**Fused Add RMSNorm**:
- 输入: `hidden_states`, `residual`, `weight`
- 输出: `output` [batch_size, hidden_size]

## Phase 1 目标

**主要目标**: 产生第一个正确的 B200 RMSNorm 实现

1. ✅ 通过 FlashInfer-Bench 正确性检查
2. ✅ 清晰的设计和文档
3. ✅ 建立性能基线

## 工作流要求

### 1. 编写实现计划草稿
保存到 `docs/draft.md`，然后运行：
```bash
/humanize:gen-plan
```

### 2. 启动 RLCR 循环
```bash
/humanize:start-rlcr-loop
```

### 3. 记录和追踪
- `benchmark.csv`: 记录每个性能相关的 commit
- `solutions.jsonl`: 维护候选实现的 DAG
- `profile/`: NCU 性能分析记录

## 验证

```bash
cd D:/Agent/flashinfer-bench-main/flashinfer-bench-main
flashinfer-bench run --local flashinfer-trace --op-type rmsnorm
```

---

**最后更新**: 2026-06-15
