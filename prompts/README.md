# Prompt Release Notes

本目录存储用于重现 agent 优化工作流的提示词。

提示词按算子和阶段组织：

```text
prompts/
  gemm/
    phase1.md
    phase2.md
    phase3.md
  gqa_paged/
    phase1.md
    phase2.md
    phase3.md
  rmsnorm/
    phase1.md
    phase2.md
    phase3.md
  ...
```

---

## 通用工作流

每个阶段遵循相同的高层交互模式：

1. 从独立的任务实现工作区开始
2. 将本目录的阶段提示词粘贴到 agent 会话中
3. 要求 agent 调查仓库、工作负载元数据、性能分析证据、KernelWiki 和相关公开文档
4. 要求 agent 将其计划草稿写入 `docs/draft.md`
5. 运行 `/humanize:gen-plan` 将 `docs/draft.md` 转换为详细的实现计划
6. 运行 `/humanize:start-rlcr-loop` 启动实现和审查循环
7. 在活动实验工作区中记录性能 commits、候选关系、基准测试结果和 NCU 证据

---

## 阶段语义

### Phase 1: 研究和正确性
专注于研究并产生正确的 B200 内核。

**目标**：
- ✅ 通过 FlashInfer-Bench 正确性检查
- ✅ 理解算子的数学定义和计算流程
- ✅ 产生清晰的实现文档
- ✅ 建立性能基线

### Phase 2: 性能探索和优化
专注于性能分析引导的瓶颈分析和迭代性能优化。

**目标**：
- ✅ 系统识别所有可行的优化方向
- ✅ 使用 NCU 分析瓶颈
- ✅ 探索每个方向（最多 5 次迭代）
- ✅ 收集 before/after 基准测试
- ✅ 保持正确性

### Phase 3: 工作负载特化和打磨
专注于工作负载形状分析并为不同形状组特化最佳内核。

**目标**：
- ✅ 分析完整工作负载分布
- ✅ 为不同形状设计特化策略
- ✅ 在完整工作负载集上评估
- ✅ 最终打磨和文档完善

---

**最后更新**: 2026-06-15  
**版本**: v2.0 - 结构化工作流
