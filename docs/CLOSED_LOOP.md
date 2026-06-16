# 闭环优化流程

本文档描述 KernelForge-MultiAgent 的闭环优化流程，融合了 AKO4X 的 10-step round loop。

---

## 🔄 闭环优化概述

**闭环优化**是一个自动化、有记忆的迭代优化流程，每轮（Round）包含 10 个步骤，自动从上一轮的结果中学习并改进。

### 与传统工作流的区别

| 特性 | 传统工作流 | 闭环优化 |
|-----|----------|---------|
| **记忆** | ❌ 每次从头开始 | ✅ 跨轮次持久化 |
| **自动化** | ❌ 人工驱动 | ✅ 自动编排 |
| **陷阱避免** | ❌ 重复错误 | ✅ TRAPS.md 记录 |
| **变体管理** | ⚠️ 简单 DAG | ✅ 家族归档 |
| **决策依据** | ⚠️ 主观判断 | ✅ 证据驱动 |

---

## 📊 10-Step Round Loop

每轮优化包含以下 10 个步骤：

```
Round N:
  1. derive     → 生成子环境
  2. brief      → 注入上下文
  3. optimize   → 执行优化
  4. benchmark  → 性能测试
  5. validate   → 正确性检查
  6. compare    → 与锚点对比
  7. decide     → 接受/拒绝
  8. document   → 记录到归档
  9. lessons    → 提取教训
  10. plan      → 规划下一轮
```

---

## 详细步骤说明

### Step 1: derive (生成子环境)
**目的**: 为本轮创建独立的工作环境

**输入**:
- `reference/<family>/README.md` - 家族历史
- `reference/<family>/TRAPS.md` - 陷阱记录
- 上一轮的最优变体

**操作**:
```bash
# 创建轮次目录
mkdir -p rounds/round-<N>/<family>
cd rounds/round-<N>/<family>

# 复制模板
cp -r ../../../templates/workspace-template/* .

# 如果有锚点变体，复制其代码
if [ -f ../../../reference/<family>/variants/<anchor>/kernel.cu ]; then
    cp ../../../reference/<family>/variants/<anchor>/kernel.cu src/baseline.cu
fi
```

**输出**:
- 独立的工作目录
- 包含上一轮的最优实现（如果有）

---

### Step 2: brief (注入上下文)
**目的**: 向优化 agent 提供本轮的目标和约束

**注入内容**:
```markdown
# Round <N> Brief

## 当前状态
- 锚点变体: <anchor-id>
- 当前加速比: <X.Xx>
- 已探索方向: [方向1, 方向2, ...]

## 本轮目标
- 目标加速比: <X.Xx> (当前的 1.2x)
- 优先方向: <从 README.md 提取>
- 避免陷阱: <从 TRAPS.md 提取>

## 约束
- 最大迭代次数: 5 (per direction)
- NCU 分析: 必需
- 正确性: 必须通过 FlashInfer-Bench
```

**输出**:
- `rounds/round-<N>/<family>/BRIEF.md`

---

### Step 3: optimize (执行优化)
**目的**: 在子环境中进行内核优化

**执行**:
```bash
# 使用现有的 Humanize 工作流
cd rounds/round-<N>/<family>

# 1. 编写 draft
vim docs/draft.md

# 2. 生成计划
/humanize:gen-plan

# 3. 执行 RLCR 循环
/humanize:start-rlcr-loop
```

**约束**:
- 遵守 BRIEF.md 中的目标
- 参考 TRAPS.md 避免已知陷阱
- 每个优化方向最多 5 次迭代

**输出**:
- 优化后的内核: `src/<family>_opt.cu`
- 优化记录: `docs/optimization-log.md`

---

### Step 4: benchmark (性能测试)
**目的**: 测量优化后的性能

**操作**:
```bash
# 编译
nvcc -O3 -arch=sm_120 src/<family>_opt.cu -o bin/<family>_opt

# 运行基准测试（8 个代表性工作负载）
python ../../../scripts/run_benchmark.py \
    --kernel bin/<family>_opt \
    --workloads representative \
    --output benchmark-result.json

# 记录到 CSV
echo "<commit>,$(date -Iseconds),<family>,round-<N>,<latency>,<speedup>,Optimized" >> benchmark.csv
```

**输出**:
- `benchmark-result.json` - 详细性能数据
- `benchmark.csv` - 追加记录

---

### Step 5: validate (正确性检查)
**目的**: 确保优化没有破坏正确性

**操作**:
```bash
# FlashInfer-Bench 验证
flashinfer-bench run \
    --local $FIB_DATASET_PATH \
    --op-type <family> \
    --solution solution.json

# 检查结果
if [ $? -eq 0 ]; then
    echo "✅ Correctness PASSED"
else
    echo "❌ Correctness FAILED"
    exit 1
fi
```

**决策**:
- ❌ 失败 → 拒绝变体，记录到 TRAPS.md
- ✅ 通过 → 继续下一步

---

### Step 6: compare (与锚点对比)
**目的**: 判断是否有实质改进

**对比指标**:
```python
# 计算加速比
anchor_time = read_baseline_time(f"reference/{family}/baseline.json")
current_time = read_time("benchmark-result.json")
speedup = anchor_time / current_time

# 统计显著性检查
variance = read_variance("benchmark-result.json")
if variance > 0.05:  # 5% 变异系数
    print("⚠️ High variance, results unstable")
```

**判断标准**:
- 加速比 >= 1.05 → 显著改进
- 1.0 < 加速比 < 1.05 → 轻微改进
- 加速比 <= 1.0 → 无改进或退化

---

### Step 7: decide (接受/拒绝)
**目的**: 决定是否接受新变体作为锚点

**决策逻辑**:
```python
def decide(speedup, correctness, variance):
    if not correctness:
        return "REJECT", "Correctness failed"
    
    if variance > 0.05:
        return "REJECT", "High variance (unstable)"
    
    if speedup >= 1.05:
        return "ACCEPT", f"Significant improvement: {speedup:.2f}x"
    elif speedup >= 1.0:
        return "CONDITIONAL", f"Minor improvement: {speedup:.2f}x"
    else:
        return "REJECT", f"Regression: {speedup:.2f}x"
```

**输出**:
- `decision.json`:
  ```json
  {
    "round": N,
    "decision": "ACCEPT|REJECT|CONDITIONAL",
    "speedup": 1.25,
    "reason": "...",
    "timestamp": "2026-06-15T10:00:00"
  }
  ```

---

### Step 8: document (记录到归档)
**目的**: 持久化变体和经验到家族归档

**如果 ACCEPT**:
```bash
# 1. 创建变体目录
variant_id="opt-v$(date +%Y%m%d-%H%M%S)"
mkdir -p reference/<family>/variants/$variant_id

# 2. 复制文件
cp src/<family>_opt.cu reference/<family>/variants/$variant_id/kernel.cu
cp benchmark-result.json reference/<family>/variants/$variant_id/result.json
cp decision.json reference/<family>/variants/$variant_id/

# 3. 更新 solutions.jsonl
echo "{\"id\": \"$variant_id\", \"parent\": \"<anchor>\", \"speedup\": <X.X>, \"round\": <N>}" \
    >> reference/<family>/solutions.jsonl

# 4. 更新 README.md（当前最优变体）
```

**如果 REJECT**:
```bash
# 仍然记录，但标记为失败
mkdir -p reference/<family>/failed/$variant_id
# ... 复制相关文件
```

---

### Step 9: lessons (提取教训)
**目的**: 从本轮中学习，更新 TRAPS.md

**分析**:
```markdown
## 本轮发现

### 成功的优化
- 优化: Warp-level reduction
- 收益: 1.25x
- 关键: 使用 __shfl_down_sync
- 推荐: 未来优先尝试

### 失败的尝试
- 优化: TMA prefetch
- 问题: 反而降低了 3%
- 原因: 计算量太小，TMA overhead 明显
- 教训: RMSNorm 不适合 TMA
```

**更新 TRAPS.md**:
```bash
# 如果发现新陷阱
echo "## 陷阱 <N>: <标题>" >> reference/<family>/TRAPS.md
echo "<详细描述>" >> reference/<family>/TRAPS.md
```

---

### Step 10: plan (规划下一轮)
**目的**: 基于本轮结果规划下一轮

**策略**:
```python
if decision == "ACCEPT":
    # 成功 → 继续深入优化
    next_target = current_speedup * 1.2
    next_directions = identify_next_bottleneck()
else:
    # 失败 → 尝试不同方向
    next_target = current_speedup * 1.1
    next_directions = unexplored_directions()
```

**输出**:
- `reference/<family>/README.md` 更新（优化方向）
- `rounds/round-<N+1>/PLAN.md` 生成

---

## 🔁 轮次示例

### Round 0: 建立基线
```
1. derive     → 创建 rounds/round-0/rmsnorm/
2. brief      → 目标: 建立正确基线
3. optimize   → 实现 naive RMSNorm
4. benchmark  → 测得 1.23ms
5. validate   → ✅ 通过
6. compare    → 1.0x (自己是基线)
7. decide     → ACCEPT (作为锚点)
8. document   → 记录到 reference/rmsnorm/
9. lessons    → (无特殊发现)
10. plan      → Round 1: 优化 reduction
```

### Round 1: Warp-level Reduction
```
1. derive     → 基于 Round 0 的代码
2. brief      → 目标: 1.2x, 方向: warp reduction
3. optimize   → 实现 __shfl_down_sync
4. benchmark  → 测得 0.95ms
5. validate   → ✅ 通过
6. compare    → 1.29x vs Round 0
7. decide     → ACCEPT
8. document   → 记录新变体
9. lessons    → warp reduction 有效
10. plan      → Round 2: Shared memory 优化
```

---

## 🎯 使用闭环优化

### 启动闭环
```bash
cd D:/Agent/KernelForge-MultiAgent

# 选择算子家族
FAMILY="rmsnorm"

# 启动 Round 0
./scripts/start-round.sh $FAMILY 0
```

### 继续下一轮
```bash
# 自动继续（基于上一轮结果）
./scripts/continue-campaign.sh $FAMILY

# 或手动指定
./scripts/start-round.sh $FAMILY 1
```

### 查看状态
```bash
# 查看家族历史
cat reference/$FAMILY/README.md

# 查看变体树
python scripts/show-variant-tree.py $FAMILY

# 查看陷阱
cat reference/$FAMILY/TRAPS.md
```

---

## 📈 闭环优化的优势

### 1. 跨轮次记忆
- ✅ 不重复失败的尝试
- ✅ 基于已有成果继续优化
- ✅ 持久化的经验教训

### 2. 自动化编排
- ✅ 减少人工介入
- ✅ 标准化流程
- ✅ 可重复的结果

### 3. 证据驱动
- ✅ 每个决策有数据支持
- ✅ NCU 分析必需
- ✅ 统计显著性检查

### 4. 陷阱规避
- ✅ 从失败中学习
- ✅ 自动注入预防措施
- ✅ 减少重复错误

---

## 🔧 与 Humanize 集成

闭环优化与 Humanize 完美配合：

```
Step 3 (optimize) 使用 Humanize:
  ↓
编写 draft.md (注入 BRIEF + TRAPS)
  ↓
/humanize:gen-plan
  ↓
/humanize:start-rlcr-loop
  ↓
输出优化后的内核
```

**关键**：
- BRIEF.md 和 TRAPS.md 自动注入到 draft.md
- Humanize 在有约束的环境中工作
- 结果自动记录到家族归档

---

## 📚 相关文档

- `reference/<family>/README.md` - 家族归档
- `reference/<family>/TRAPS.md` - 陷阱记录
- `docs/STRUCTURED_WORKFLOW.md` - 整体工作流
- `docs/HUMANIZE_INTEGRATION.md` - Humanize 使用

---

**最后更新**: 2026-06-15  
**版本**: v1.0  
**灵感来源**: AKO4X 10-step round loop
