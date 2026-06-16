# KernelForge-MultiAgent 结构化优化工作流

本文档描述项目的结构化、有证据支持的优化工作流。

---

## 🎯 核心理念

**结构化、有证据支持的优化，而不是随机尝试**

本项目采用 **三阶段优化工作流**，结合：
- **Humanize** - 提供强大的 plan-execute-verify 循环
- **KernelWiki** - Blackwell/Hopper 优化知识库
- **ncu-report-skill** - Nsight Compute 性能分析

---

## 📊 工作流架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Phase 1: 正确性基线                      │
│  目标: 产生正确的 B200 实现，建立性能基准                    │
│                                                              │
│  输入: 算子定义、参考实现                                    │
│  输出: 正确的基线实现 + 性能数据                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 2: 性能探索优化                      │
│  目标: 系统探索优化方向，显著提升性能                        │
│                                                              │
│  输入: 基线实现 + NCU 分析                                   │
│  输出: 优化的实现 + 探索记录                                 │
│  规则: 每个方向最多 5 次迭代                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Phase 3: 工作负载特化                        │
│  目标: 针对不同形状特化，达到最优性能                        │
│                                                              │
│  输入: 优化实现 + 工作负载分布                               │
│  输出: 特化实现 + 完整评估                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 工具集成

### 1. Humanize 插件

**用途**: 提供结构化的计划-执行-审查循环

**使用流程**:
```bash
# 1. 编写实现计划草稿
vim docs/draft.md

# 2. 生成详细实现计划
/humanize:gen-plan

# 3. 启动 RLCR 循环 (Research-Learn-Code-Review)
/humanize:start-rlcr-loop

# 4. 查看进度
/humanize:status
```

**RLCR 循环**:
- **Research**: 查找相关代码和文档
- **Learn**: 理解实现细节和技术
- **Code**: 编写/修改代码
- **Review**: 验证正确性和性能

### 2. KernelWiki Skill

**用途**: 查找 Blackwell/Hopper 优化知识

**查询内容**:
- B200 硬件特性（TMA, TMEM, tcgen05）
- 算子特定优化技术
- CuTe DSL 示例
- 性能优化模式

**使用方式**: 在 Claude 对话中直接查询

### 3. ncu-report-skill

**用途**: 分析 Nsight Compute 性能报告

**分析内容**:
- 识别瓶颈（Memory/Compute/Latency-bound）
- 解释性能指标
- 建议优化方向

**使用方式**: 在 Claude 对话中分析 NCU 报告

---

## 📁 项目结构

```
KernelForge-MultiAgent/
├── kernels/operators/         # 最终内核实现
│   ├── gemm/
│   ├── rmsnorm/
│   └── ...
├── prompts/                   # 三阶段提示词
│   ├── gemm/
│   │   ├── phase1.md
│   │   ├── phase2.md
│   │   └── phase3.md
│   ├── rmsnorm/
│   └── README.md
├── workspaces/                # 独立任务工作区
│   ├── <task-name>/
│   │   ├── docs/
│   │   │   └── draft.md      # Humanize 草稿
│   │   ├── src/
│   │   ├── profile/          # NCU 报告
│   │   ├── benchmark.csv     # 性能记录
│   │   └── solutions.jsonl   # 候选 DAG
│   └── README.md
├── profile/                   # 全局性能分析
├── outputs/                   # 输出文件
├── runs/                      # 运行记录
├── skills/                    # 集成的 Skills
│   ├── KernelWiki/
│   └── ncu-report-skill/
└── docs/
    ├── STRUCTURED_WORKFLOW.md # 本文件
    ├── draft_template.md      # 草稿模板
    └── ...
```

---

## 🚀 完整优化流程

### 准备工作

#### 1. 安装 Humanize 插件
```bash
/plugin marketplace add PolyArch/humanize
/plugin install humanize@PolyArch
```

#### 2. 确保 Skills 已安装
```bash
# KernelWiki 和 ncu-report-skill 应该已经在 skills/ 目录
ls skills/KernelWiki/SKILL.md
ls skills/ncu-report-skill/SKILL.md
```

#### 3. 准备 FlashInfer-Trace 数据集
```bash
# 设置数据集路径
export FIB_DATASET_PATH="D:/Agent/flashinfer-bench-main/flashinfer-bench-main/data/flashinfer-trace"
```

---

### Phase 1: 建立正确性基线

#### 步骤 1: 创建工作区
```bash
mkdir -p workspaces/<op-name>-optimization
cd workspaces/<op-name>-optimization

# 初始化结构
mkdir -p docs src profile
touch docs/draft.md
echo "commit_hash,timestamp,kernel_name,workload_id,latency_ms,speedup,note" > benchmark.csv
echo '{"id": "init", "parent": null, "description": "Workspace initialized"}' > solutions.jsonl
```

#### 步骤 2: 阅读 Phase 1 提示词
```bash
cat ../../prompts/<op-name>/phase1.md
```

#### 步骤 3: 编写实现计划草稿
```bash
# 使用模板
cp ../../docs/draft_template.md docs/draft.md
vim docs/draft.md
```

**草稿应包括**:
- 算子的数学定义
- 内存布局和数据流
- 初步 CUDA 实现策略
- 正确性验证方法
- B200 特性利用点

#### 步骤 4: 生成详细计划
```bash
# 在 Claude 对话中
/humanize:gen-plan
```

Humanize 会将草稿转换为详细的可执行计划。

#### 步骤 5: 启动 RLCR 循环
```bash
/humanize:start-rlcr-loop
```

#### 步骤 6: 实现和验证
```bash
# 编写实现
vim src/<kernel>_baseline.cu

# 快速验证
cd ../..
python scripts/workflow/run_optimization_cycle.py <kernel>-baseline 1

# FlashInfer-Bench 验证
cd $FIB_DATASET_PATH/..
flashinfer-bench run --local flashinfer-trace --op-type <op_type> --limit 5
```

#### 步骤 7: 记录基线
```bash
cd workspaces/<op-name>-optimization

# 记录性能
echo "abc123,2026-06-15T10:00:00,<kernel>,baseline,1.23,1.0x,Phase 1 baseline" >> benchmark.csv

# 记录候选
echo '{"id": "baseline-v1", "parent": null, "description": "Phase 1 baseline", "speedup": 1.0, "correctness": true}' >> solutions.jsonl
```

#### Phase 1 成功标准
- ✅ 通过 FlashInfer-Bench 正确性检查
- ✅ 有清晰的实现文档
- ✅ 建立了性能基线

---

### Phase 2: 性能探索和优化

#### 步骤 1: NCU 性能分析
```bash
# 性能分析
mkdir -p profile/baseline
ncu --set full -o profile/baseline/report <executable>

# 使用 ncu-report-skill 分析
# 在 Claude 对话中使用 ncu-report-skill
```

#### 步骤 2: 识别优化方向
```bash
# 阅读 Phase 2 提示词
cat ../../prompts/<op-name>/phase2.md

# 更新草稿，列出优化方向
vim docs/draft.md
```

**列出优化方向** (按优先级):
1. 方向 1: [名称] - 预期收益 High, 风险 Low
2. 方向 2: [名称] - 预期收益 Medium, 风险 Medium
3. ...

#### 步骤 3: 系统探索

对每个优化方向：
```bash
# 1. 生成计划
/humanize:gen-plan

# 2. 启动循环 (最多 5 次迭代)
/humanize:start-rlcr-loop

# 3. NCU 分析
mkdir -p profile/<opt-name>
ncu --set full -o profile/<opt-name>/report <executable>

# 4. 记录结果
echo "def456,2026-06-15T11:00:00,<kernel>,opt-v1,0.95,1.29x,<opt-name>" >> benchmark.csv
echo '{"id": "opt-v1", "parent": "baseline-v1", "description": "<opt-name>", "speedup": 1.29, "correctness": true}' >> solutions.jsonl
```

**规则**:
- 每个方向最多 5 次迭代
- 5 次后无改进 → 记录证据，转向下一方向
- 收集 before/after 数据
- 保持正确性

#### Phase 2 成功标准
- ✅ 探索了所有主要优化方向
- ✅ 每个方向有 NCU 证据支持
- ✅ 性能显著提升
- ✅ 记录了探索过程

---

### Phase 3: 工作负载特化

#### 步骤 1: 分析工作负载分布
```bash
# 查看工作负载
flashinfer-bench info --local $FIB_DATASET_PATH --op-type <op_type>
```

#### 步骤 2: 设计特化策略
```bash
# 阅读 Phase 3 提示词
cat ../../prompts/<op-name>/phase3.md

# 更新草稿
vim docs/draft.md
```

**特化策略示例**:
- Tiny shapes: `batch_size <= 8`
- Medium shapes: `8 < batch_size <= 128`
- Large shapes: `batch_size > 128`

#### 步骤 3: 实现和评估
```bash
# 实现特化版本
/humanize:gen-plan
/humanize:start-rlcr-loop

# 完整工作负载验证
flashinfer-bench run --local $FIB_DATASET_PATH --op-type <op_type>
```

#### 步骤 4: 最终验收
```bash
# 复制到主仓库
cp src/<kernel>_final.cu ../../kernels/operators/<op_type>/

# 运行项目验证
cd ../..
python verify.py --cuda --arch sm_120
```

#### Phase 3 成功标准
- ✅ 完整工作负载集评估
- ✅ 特化策略有数据支持
- ✅ 代码质量高
- ✅ 通过所有验收

---

## 📊 记录系统

### benchmark.csv 格式
```csv
commit_hash,timestamp,kernel_name,workload_id,latency_ms,speedup,note
abc123,2026-06-15T10:00:00,gemm,baseline,1.23,1.0x,Phase 1 baseline
def456,2026-06-15T11:00:00,gemm,opt-v1,0.95,1.29x,Tiled GEMM
ghi789,2026-06-15T12:00:00,gemm,opt-v2,0.82,1.50x,+ Register blocking
```

### solutions.jsonl 格式
```jsonl
{"id": "baseline-v1", "parent": null, "description": "Naive implementation", "speedup": 1.0, "correctness": true}
{"id": "opt-v1", "parent": "baseline-v1", "description": "Tiled GEMM", "speedup": 1.29, "correctness": true}
{"id": "opt-v2", "parent": "opt-v1", "description": "+ Register blocking", "speedup": 1.50, "correctness": true}
{"id": "opt-v3-failed", "parent": "opt-v2", "description": "Warp specialization (failed)", "speedup": 0.95, "correctness": false}
```

---

## ✅ 最佳实践

### 应该做的
1. ✅ **始终从草稿开始** - 计划先行
2. ✅ **使用 Humanize** - 结构化执行
3. ✅ **记录所有尝试** - 包括失败
4. ✅ **保持 NCU 证据** - 数据驱动决策
5. ✅ **频繁验证** - 每次修改后验证
6. ✅ **独立工作区** - 避免污染主仓库

### 不应该做的
1. ❌ **不要跳过计划** - 盲目编码
2. ❌ **不要随机尝试** - 基于证据优化
3. ❌ **不要忽略失败** - 记录为什么失败
4. ❌ **不要省略验证** - 正确性第一
5. ❌ **不要混合任务** - 一个工作区一个任务

---

## 📈 成功案例参考

MLSys 2026 FlashInfer Contest 使用此工作流取得：
- **MoE Track**: 🥇 第 1 名
- **DSA Track**: 🥈 第 2 名  
- **GDN Track**: 🥉 第 3 名

**关键贡献**:
- **Humanize**: 最大贡献，提供结构化循环
- **KernelWiki**: 扩展优化知识
- **ncu-report-skill**: 细粒度性能分析

---

## 🔗 相关资源

- **Humanize**: https://github.com/PolyArch/humanize
- **KernelWiki**: `skills/KernelWiki/`
- **ncu-report-skill**: `skills/ncu-report-skill/`
- **FlashInfer-Bench**: https://github.com/flashinfer-ai/flashinfer-bench
- **参考项目**: https://github.com/mit-han-lab/mlsys2026-flashinfer-contest

---

**最后更新**: 2026-06-15  
**版本**: v1.0
