## Humanize 工作流集成

KernelForge-MultiAgent 已集成 **Humanize 插件**，提供结构化的 plan-execute-verify 循环。

---

## 安装 Humanize

### 前置条件
- Claude Code 或 Codex
- Claude Plugin Marketplace 访问权限

### 安装步骤

```bash
# 1. 添加 PolyArch marketplace
/plugin marketplace add PolyArch/humanize

# 2. 安装 humanize 插件
/plugin install humanize@PolyArch

# 3. 验证安装
/plugin list
# 应该看到 humanize@PolyArch 在列表中
```

---

## Humanize 命令

### 1. `/humanize:gen-plan`
**用途**: 将实现计划草稿转换为详细的可执行计划

**输入**: `docs/draft.md` - 高层实现计划草稿  
**输出**: 详细的分步实现计划，包括：
- 分解的子任务
- 每个子任务的验收标准
- 依赖关系
- 风险评估
- 具体执行步骤

**使用场景**: 
- 完成 `docs/draft.md` 后，准备开始实现前
- 每个 Phase 开始时

**示例**:
```bash
# 1. 编写草稿
vim docs/draft.md

# 2. 在 Claude 对话中运行
/humanize:gen-plan

# Humanize 会读取 docs/draft.md 并生成详细计划
```

---

### 2. `/humanize:start-rlcr-loop`
**用途**: 启动 Research-Learn-Code-Review 循环

**RLCR 循环**:
- **Research**: 查找相关代码、文档、论文
- **Learn**: 理解实现细节和技术
- **Code**: 编写/修改代码
- **Review**: 验证正确性和性能

**特点**:
- 结构化的实现流程
- 自动追踪进度
- 每个子任务完成后自动 review
- 失败时自动回滚和重试

**使用场景**:
- 生成详细计划后
- 准备开始编码实现时

**示例**:
```bash
# 在生成计划后立即运行
/humanize:start-rlcr-loop

# Humanize 会：
# 1. 按计划执行每个子任务
# 2. 每个子任务结束后进行 review
# 3. 如果 review 通过，继续下一任务
# 4. 如果 review 失败，分析原因并重试
```

---

### 3. `/humanize:status`
**用途**: 查看当前 RLCR 循环的进度

**输出**:
- 当前执行的子任务
- 已完成的任务列表
- 剩余任务列表
- 当前循环状态（Research/Learn/Code/Review）

**使用场景**:
- 想了解当前进度
- RLCR 循环执行过程中

**示例**:
```bash
/humanize:status
```

---

## 工作流集成

### Phase 1: 正确性基线

```bash
# 1. 创建工作区
mkdir -p workspaces/rmsnorm-opt
cd workspaces/rmsnorm-opt
mkdir -p docs src profile

# 2. 复制草稿模板
cp ../../docs/draft_template.md docs/draft.md

# 3. 阅读 Phase 1 提示词
cat ../../prompts/rmsnorm/phase1.md

# 4. 编写实现计划草稿
vim docs/draft.md
# 填写：
# - 算子数学定义
# - 实现策略
# - 内存布局
# - CUDA 实现设计
# - 验证方法

# 5. 生成详细计划
/humanize:gen-plan

# 6. 开始实现循环
/humanize:start-rlcr-loop

# 7. 查看进度（可选）
/humanize:status

# 8. 验证结果
flashinfer-bench run --local $FIB_DATASET_PATH --op-type rmsnorm
```

### Phase 2: 性能优化

```bash
# 1. NCU 性能分析
mkdir -p profile/baseline
ncu --set full -o profile/baseline/report ./benchmark

# 2. 使用 ncu-report-skill 分析瓶颈
# 在 Claude 对话中使用 ncu-report-skill

# 3. 更新草稿，列出优化方向
vim docs/draft.md
# 添加：
# - NCU 分析证据
# - 识别的瓶颈
# - 候选优化方向（按优先级）
# - 每个方向的预期收益和风险

# 4. 生成优化计划
/humanize:gen-plan

# 5. 系统探索优化方向
/humanize:start-rlcr-loop
# 规则：每个方向最多 5 次迭代

# 6. 记录结果
echo "def456,$(date -Iseconds),rmsnorm,opt-v1,0.95,1.29x,Memory opt" >> benchmark.csv
```

### Phase 3: 工作负载特化

```bash
# 1. 分析工作负载分布
flashinfer-bench info --local $FIB_DATASET_PATH --op-type rmsnorm

# 2. 更新草稿，设计特化策略
vim docs/draft.md
# 添加：
# - 工作负载分布分析
# - 不同形状的瓶颈差异
# - 特化策略（dispatch 逻辑）
# - 特化路径

# 3. 生成特化计划
/humanize:gen-plan

# 4. 实现特化版本
/humanize:start-rlcr-loop

# 5. 完整验证
flashinfer-bench run --local $FIB_DATASET_PATH --op-type rmsnorm
```

---

## draft.md 编写指南

### 模板位置
`docs/draft_template.md` - 使用此模板开始

### 关键部分

#### 1. 任务概述
```markdown
**算子类型**: rmsnorm
**Phase**: Phase 1
**目标**: 产生正确的 B200 RMSNorm 实现
**预期加速比**: baseline
```

#### 2. 算子规范
```markdown
### 数学定义
output = weight * (hidden_states / sqrt(mean(hidden_states^2) + eps))

### 输入输出
- 输入: hidden_states [batch, hidden], weight [hidden]
- 输出: output [batch, hidden]
```

#### 3. 实现策略（Phase 1）
```markdown
### 算法流程
1. Warp-level reduction 计算平方和
2. 计算 RMS
3. 归一化
4. 缩放

### CUDA 实现设计
- Grid: (batch,)
- Block: (256,)
- Shared Memory: 用于 reduction
```

#### 4. 优化方向（Phase 2）
```markdown
### NCU 分析证据
- Memory bandwidth: 65%
- Compute utilization: 45%
- 主要瓶颈: Memory-bound

### 候选优化方向
1. **Warp-level reduction优化**
   - 预期收益: High (1.5x)
   - 实现风险: Low
   - 关键技术: __shfl_down_sync
```

#### 5. 工作负载特化（Phase 3）
```markdown
### 工作负载分布
- Small (batch<=8): 30%
- Medium (8<batch<=128): 50%
- Large (batch>128): 20%

### 特化策略
- Small: 单 warp 处理
- Large: 多 block 并行
```

---

## 与 KernelWiki 和 ncu-report-skill 协同

### 完整工作流

```
1. 阅读 Phase 提示词
   ↓
2. 使用 KernelWiki 研究相关实现
   - 查找 Blackwell RMSNorm 优化技术
   - 查找 warp-level primitives
   ↓
3. 编写 docs/draft.md
   - 填写算子规范
   - 设计实现策略
   ↓
4. /humanize:gen-plan
   - 生成详细可执行计划
   ↓
5. /humanize:start-rlcr-loop
   - 结构化实现
   ↓
6. NCU 性能分析（Phase 2）
   ↓
7. 使用 ncu-report-skill 分析瓶颈
   - 识别 Memory/Compute/Latency-bound
   - 获取优化建议
   ↓
8. 更新 draft.md 添加优化方向
   ↓
9. 重复步骤 4-5 进行优化
   ↓
10. FlashInfer-Bench 验证
```

---

## 最佳实践

### ✅ 应该做的

1. **详细的草稿**
   - 不要写得太简单
   - 包含足够的技术细节
   - 说明"为什么"而不只是"做什么"

2. **Phase 2 有证据支持**
   - 每个优化方向都引用 NCU 数据
   - 说明预期收益的依据
   - 记录探索结果（成功和失败）

3. **清晰的验收标准**
   - 每个子任务有明确的完成标准
   - 可测试、可验证

4. **记录所有尝试**
   - 更新 benchmark.csv
   - 更新 solutions.jsonl
   - 保存 NCU 报告

### ❌ 不应该做的

1. **草稿太简单**
   - 不要只写几句话
   - Humanize 需要足够的信息来生成详细计划

2. **跳过计划阶段**
   - 不要直接开始编码
   - 先 `/humanize:gen-plan`，再 `/humanize:start-rlcr-loop`

3. **忽略 RLCR 循环的 Review**
   - Review 失败说明有问题
   - 分析失败原因，不要强行继续

4. **优化无证据**
   - Phase 2 的每个优化方向必须有 NCU 证据
   - 不要"感觉"某个优化会有用

---

## 故障排除

### 问题 1: `/humanize:gen-plan` 生成的计划不够详细

**原因**: `docs/draft.md` 写得太简单

**解决方案**:
- 使用 `docs/draft_template.md` 模板
- 填写所有部分，不要留空
- 添加更多技术细节

### 问题 2: RLCR 循环卡住或反复失败

**原因**: 
- 子任务太大
- 验收标准不清晰
- 技术难度太高

**解决方案**:
- 将子任务分解得更小
- 明确验收标准
- 降低单个子任务的复杂度
- 必要时手动介入

### 问题 3: Review 总是失败

**原因**:
- 正确性检查未通过
- 性能目标未达到
- 代码质量问题

**解决方案**:
- 检查验收标准是否合理
- 查看具体的失败原因
- 调整目标或实现策略

---

## 参考资源

- **Humanize GitHub**: https://github.com/PolyArch/humanize
- **KernelWiki**: `skills/KernelWiki/`
- **ncu-report-skill**: `skills/ncu-report-skill/`
- **完整工作流**: `docs/STRUCTURED_WORKFLOW.md`
- **草稿模板**: `docs/draft_template.md`

---

**最后更新**: 2026-06-15  
**版本**: v1.0
