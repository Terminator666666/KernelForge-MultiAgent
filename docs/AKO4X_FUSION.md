# AKO4X 优秀思路融合总结

本文档记录从 AKO4X 项目中提取并融合到 KernelForge-MultiAgent 的核心优秀思路。

---

## 🎯 融合策略：闭环自动优化

**原则**: 提取 AKO4X 的闭环自动化核心思路，简化为单一运行模式。

**目标**: 专注于 **闭环自动优化**，移除多模式复杂性。

---

## ✅ 已融合的核心思路

### 1. Master-Sub 双智能体架构 ⭐⭐⭐⭐⭐

**来源**: AKO4X Master Agent + Sub Agent 分层

**价值**: 职责分离，Master 编排，Sub 执行

**融合方式**:
```
Master Agent (新增)
  - 位置: master/MASTER.md
  - 职责: Campaign 编排、决策、记忆维护
  - 工具: reference/ 归档、TRAPS.md、10-step loop

Sub Agent (保留现有)
  - 位置: workspaces/ 或 rounds/
  - 职责: 内核优化
  - 工具: Humanize RLCR、KernelWiki、ncu-report-skill
```

**优势**:
- ✅ 清晰的职责分离
- ✅ Master 提供战略规划
- ✅ Sub 专注执行优化
- ✅ 可扩展到多 Sub Agent 并行

---

### 2. Reference 家族归档系统 ⭐⭐⭐⭐⭐

**来源**: AKO4X `reference/<family>/` 结构

**价值**: 跨轮次持久化记忆

**融合方式**:
```
reference/<family>/
  ├── README.md        # 优化历史、当前最优、变体树
  ├── TRAPS.md         # 陷阱记录（核心创新！）
  ├── baseline.json    # 性能基准
  ├── solutions.jsonl  # 变体 DAG
  └── variants/        # 历史实现归档
      ├── baseline-v1/
      ├── opt-v1/
      └── opt-v2/
```

**对比 workspaces/**:
- workspaces/: 临时工作区，单次使用
- reference/: 持久化归档，跨轮次记忆

**优势**:
- ✅ 不重复失败的尝试
- ✅ 基于已有成果继续优化
- ✅ 完整的优化历史追溯

---

### 3. TRAPS.md 陷阱记忆 ⭐⭐⭐⭐⭐

**来源**: AKO4X `TRAPS.md` - Silent-bug patterns

**价值**: 从失败中学习，避免重复错误

**融合方式**:
```markdown
## 陷阱 1: 数值稳定性问题
**症状**: NaN, 溢出
**原因**: FP16 精度不足
**解决方案**: [具体代码]
**预防措施**: [检查清单]
```

**使用流程**:
1. 优化前 → 阅读 TRAPS.md
2. 遇到问题 → 检查是否已知陷阱
3. 优化后 → 记录新陷阱

**优势**:
- ✅ 具体的陷阱描述
- ✅ 可操作的解决方案
- ✅ 自动注入到 BRIEF.md

---

### 4. 10-Step Round Loop ⭐⭐⭐⭐

**来源**: AKO4X 闭环优化流程

**价值**: 结构化、可重复的优化循环

**融合方式**:
```
每轮 (Round):
  1. derive     → 生成子环境
  2. brief      → 注入上下文 (BRIEF.md)
  3. optimize   → Sub Agent 优化 (Humanize)
  4. benchmark  → 性能测试
  5. validate   → 正确性检查
  6. compare    → 与锚点对比
  7. decide     → Master 决策 (ACCEPT/REJECT)
  8. document   → 记录到 reference/
  9. lessons    → 提取教训 → TRAPS.md
  10. plan      → 规划下一轮
```

**与 Humanize 三阶段的关系**:
- Step 3 (optimize) = Phase 1/2/3 + Humanize RLCR
- 其他步骤是外层编排

**优势**:
- ✅ 标准化流程
- ✅ 每步有明确输入输出
- ✅ 自动化程度高

---

### 5. 证据驱动决策 ⭐⭐⭐⭐

**来源**: AKO4X 证据门控（evidence-gate）

**价值**: 客观、可重复的决策

**融合方式**:
```python
def decide(speedup, correctness, variance):
    if not correctness:
        return "REJECT", "Correctness failed"
    
    if variance > 0.05:
        return "REJECT", "High variance"
    
    if speedup >= anchor * 1.05:
        return "ACCEPT", "Significant improvement"
    else:
        return "REJECT", "Insufficient gain"
```

**决策依据**:
- ✅ 正确性（FlashInfer-Bench）
- ✅ 加速比（>= 1.05x）
- ✅ 稳定性（variance < 5%）

**优势**:
- ✅ 客观标准
- ✅ 可重复
- ✅ 避免主观偏见

---

## 🔄 保留的现有优势

### 1. Humanize RLCR 循环 ⭐⭐⭐⭐⭐
**保留原因**: 优秀的 plan-execute-verify 机制

**角色**: Sub Agent 的执行引擎

### 2. KernelWiki + ncu-report-skill ⭐⭐⭐⭐⭐
**保留原因**: 丰富的优化知识和性能分析

**角色**: Sub Agent 的知识库

### 3. FlashInfer-Bench 验收 ⭐⭐⭐⭐⭐
**保留原因**: 标准化、权威的验收标准

**角色**: 正确性和性能的最终裁判

---

## 🚫 未融合的 AKO4X 特性（及原因）

### 1. spawn.py 动态环境生成
**原因**: 
- 过于复杂的模板渲染系统
- 简化的脚本（start-campaign.sh, run-round.sh）已足够

**替代**: Bash 脚本创建轮次环境

### 2. Mode 1 Manual 和 Mode 3 Co-evolution
**原因**: 
- 降低概念复杂度
- 专注于闭环自动优化

**简化**: 只保留闭环模式

### 3. 复杂的配置系统
**原因**: 
- config.toml 过于繁琐
- 简单的 JSON 配置更直接

**替代**: campaign-<family>.json 简单配置

---

## 📊 融合效果对比

| 维度 | 融合前 | 融合后 |
|-----|-------|--------|
| **架构** | 单智能体 | ✅ Master-Sub 双层 |
| **记忆** | 单轮 CSV/JSON | ✅ 跨轮次家族归档 |
| **陷阱避免** | ❌ 无系统化 | ✅ TRAPS.md |
| **决策** | 主观判断 | ✅ 证据驱动 |
| **循环** | 三阶段 | ✅ 10-step loop |
| **工具** | Humanize | ✅ 保留 + 增强 |

---

## 🎯 三种运行模式

### Mode 1: Manual (手动单次)
**场景**: 快速实验、学习

**流程**:
```bash
# 使用 workspaces/
cd workspaces/<task>
vim docs/draft.md
/humanize:gen-plan
/humanize:start-rlcr-loop
```

**特点**:
- 单次优化
- 完全人工控制
- 适合探索

---

### Mode 2: Closed-loop (闭环自动)
**场景**: 生产优化、长期 Campaign

**流程**:
```bash
# Master Agent 编排
./scripts/start-campaign.sh rmsnorm 10

# 10 轮自动循环:
for round in 0..9:
    Master: derive + brief
    Sub: optimize (Humanize)
    Auto: benchmark + validate + compare
    Master: decide + document + lessons + plan
```

**特点**:
- 多轮自动迭代
- Master 编排
- reference/ 持久化记忆

---

### Mode 3: Co-evolution (协同进化)
**场景**: 高级优化（未来）

**流程**: Harness 和 Kernel 共同进化

**状态**: 🔜 未实现

---

## 📁 新增目录结构

```
KernelForge-MultiAgent/
├── master/               ✅ 新增
│   └── MASTER.md        # Master Agent 指南
├── reference/            ✅ 新增
│   └── <family>/
│       ├── README.md     # 优化历史
│       ├── TRAPS.md      # 陷阱记录
│       ├── baseline.json
│       ├── solutions.jsonl
│       └── variants/     # 变体归档
├── rounds/               ✅ 新增
│   └── round-<N>/<family>/  # 每轮工作区
├── docs/
│   ├── CLOSED_LOOP.md    ✅ 新增
│   └── ...
└── (其他保持不变)
```

---

## 🔧 使用示例

### Mode 1: 手动优化
```bash
# 1. 创建工作区
mkdir -p workspaces/rmsnorm-test
cd workspaces/rmsnorm-test

# 2. 使用 Humanize
vim docs/draft.md
/humanize:gen-plan
/humanize:start-rlcr-loop

# 3. 验证
flashinfer-bench run --op-type rmsnorm
```

### Mode 2: 闭环 Campaign
```bash
# 1. 启动 Campaign
./scripts/start-campaign.sh rmsnorm 5

# Master Agent 自动:
# - Round 0: 建立基线
# - Round 1: Warp reduction
# - Round 2: Shared memory
# - Round 3: TMA
# - Round 4: Register blocking

# 2. 查看进度
cat reference/rmsnorm/README.md
cat reference/rmsnorm/TRAPS.md

# 3. 查看当前最优
cat reference/rmsnorm/variants/<latest>/kernel.cu
```

---

## 📈 融合价值

### 1. 更强的记忆能力
- ✅ 跨轮次持久化
- ✅ 完整的优化历史
- ✅ 系统化的陷阱记录

### 2. 更好的自动化
- ✅ Master Agent 编排
- ✅ 10-step loop 标准化
- ✅ 证据驱动决策

### 3. 保留原有优势
- ✅ Humanize RLCR 循环
- ✅ KernelWiki 知识库
- ✅ FlashInfer-Bench 验收

### 4. 可扩展架构
- ✅ 双智能体架构
- ✅ 支持多种模式
- ✅ 未来可扩展 Co-evolution

---

## 🎓 关键创新

### 1. TRAPS.md 机制 🆕
**创新点**: 系统化的陷阱记录和预防

**价值**: 避免重复错误，积累优化经验

### 2. Reference 家族归档 🆕
**创新点**: 跨轮次持久化记忆

**价值**: 基于已有成果继续优化

### 3. Master-Sub 分层 🆕
**创新点**: 职责分离，编排与执行分离

**价值**: 更清晰的架构，更好的可扩展性

### 4. 10-Step Loop 🆕
**创新点**: 标准化的闭环流程

**价值**: 可重复、可自动化

---

## 📚 相关文档

- **AKO4X 融合**: 本文件
- **闭环流程**: `docs/CLOSED_LOOP.md`
- **Master Agent**: `master/MASTER.md`
- **家族归档**: `reference/<family>/README.md`
- **陷阱记录**: `reference/<family>/TRAPS.md`
- **整体工作流**: `docs/STRUCTURED_WORKFLOW.md`

---

**融合日期**: 2026-06-15  
**融合策略**: 精选核心思路，保留现有优势  
**融合比例**: AKO4X 30% + Humanize 70%  
**版本**: v1.0
