# 实现计划草稿模板

本文件是 Humanize 工作流的起点。在实现前，先在此编写高层实现计划，然后运行 `/humanize:gen-plan` 生成详细的可执行计划。

---

## 任务概述

**算子**: [算子名称，如 GEMM, RMSNorm, GQA-Paged]  
**目标**: [Phase 1/2/3 的具体目标]  
**当前阶段**: [Phase 1: 正确性 | Phase 2: 性能优化 | Phase 3: 工作负载特化]

---

## 算子定义

### 数学定义
```
[用数学公式描述算子的计算]
例如 RMSNorm:
output = weight * (hidden_states / sqrt(mean(hidden_states^2) + eps))
```

### 输入输出
- **输入**:
  - `输入1`: [形状] [数据类型] [描述]
  - `输入2`: [形状] [数据类型] [描述]
- **输出**:
  - `输出1`: [形状] [数据类型] [描述]

### 约束条件
- [列出所有约束，如维度关系、数据类型要求]

---

## 实现策略

### 1. 算法流程
```
[分步描述算法流程]
1. 步骤 1: ...
2. 步骤 2: ...
3. 步骤 3: ...
```

### 2. 内存布局
- **输入布局**: [描述输入张量的内存布局]
- **输出布局**: [描述输出张量的内存布局]
- **中间缓冲**: [是否需要 shared memory 或 registers]

### 3. 并行策略
- **线程块维度**: `(blockDim.x, blockDim.y, blockDim.z)`
- **网格维度**: `(gridDim.x, gridDim.y, gridDim.z)`
- **每个线程的工作**: [描述线程分工]

### 4. 优化方向 (Phase 2/3)
如果是 Phase 2 或 Phase 3，列出要探索的优化方向：

#### NCU 分析证据
```
[粘贴 NCU 关键指标]
- Memory bandwidth utilization: XX%
- Compute utilization: XX%
- Occupancy: XX%
- 主要瓶颈: [Memory-bound | Compute-bound | Latency-bound]
```

#### 候选优化方向
按优先级排序：

1. **优化方向 1**: [名称]
   - **预期收益**: [High | Medium | Low]
   - **实现风险**: [High | Medium | Low]
   - **NCU 证据**: [支持此方向的证据]
   - **实现步骤**:
     - 步骤 1
     - 步骤 2

2. **优化方向 2**: [名称]
   - ...

### 5. B200 特性利用
- [ ] **TMA** (Tensor Memory Accelerator): [如何使用]
- [ ] **TMEM** (Tensor Memory): [如何使用]
- [ ] **tcgen05**: [如何使用]
- [ ] **Warp Specialization**: [如何使用]
- [ ] **Persistent Scheduling**: [如何使用]

---

## 正确性验证

### 验证策略
1. [验证方法 1]
2. [验证方法 2]

### 代表性工作负载
- Small: [配置]
- Medium: [配置]
- Large: [配置]

### 容差标准
- FP16/BF16: `atol=1e-3, rtol=1e-2`
- FP8: `atol=1e-2, rtol=5e-2`

---

## 性能目标 (Phase 2/3)

### 基线性能
```
[当前基线的性能数据]
- Workload 1: X.XX ms
- Workload 2: X.XX ms
```

### 目标加速比
- **Phase 2 目标**: XXx 加速
- **Phase 3 目标**: XXx 加速

---

## 风险和挑战

1. **风险 1**: [描述]
   - **缓解措施**: [如何应对]

2. **风险 2**: [描述]
   - **缓解措施**: [如何应对]

---

## 实现清单

### Phase 1 清单
- [ ] 理解算子数学定义
- [ ] 设计内存布局
- [ ] 实现 naive 版本
- [ ] 通过正确性检查
- [ ] 建立性能基线
- [ ] 记录到 benchmark.csv

### Phase 2 清单
- [ ] NCU 性能分析
- [ ] 识别主要瓶颈
- [ ] 列出优化方向
- [ ] 探索优化方向 1 (最多 5 次迭代)
- [ ] 探索优化方向 2 (最多 5 次迭代)
- [ ] 记录 before/after 数据
- [ ] 更新 solutions.jsonl

### Phase 3 清单
- [ ] 分析工作负载分布
- [ ] 设计形状特化策略
- [ ] 实现特化版本
- [ ] 完整工作负载验证
- [ ] 最终性能评估
- [ ] 代码清理和文档

---

## 参考资源

### KernelWiki 查询关键词
- [关键词 1]
- [关键词 2]
- [关键词 3]

### 相关实现
- [参考实现 1]
- [参考实现 2]

### 技术文档
- [文档链接 1]
- [文档链接 2]

---

## 下一步

完成草稿后：
1. 保存此文件
2. 运行 `/humanize:gen-plan` 生成详细计划
3. 运行 `/humanize:start-rlcr-loop` 启动实现循环

---

**创建时间**: [日期]  
**最后更新**: [日期]  
**当前阶段**: [Phase X]
