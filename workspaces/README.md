# KernelForge-MultiAgent 工作区管理

本目录用于存储独立的任务实现工作区。每个优化任务应在独立的工作区中进行，避免污染主仓库。

---

## 目录结构

```
workspaces/
├── <task-name-1>/          # 独立任务工作区
│   ├── docs/
│   │   └── draft.md        # Humanize 计划草稿
│   ├── src/
│   │   └── <kernel>.cu     # 内核实现
│   ├── profile/            # NCU 性能分析
│   ├── benchmark.csv       # 性能记录
│   ├── solutions.jsonl     # 候选实现 DAG
│   └── README.md
├── <task-name-2>/
│   └── ...
└── README.md               # 本文件
```

---

## 创建新工作区

### 方法 1: 从头开始

```bash
# 1. 创建工作区目录
mkdir -p workspaces/<task-name>
cd workspaces/<task-name>

# 2. 初始化结构
mkdir -p docs src profile
touch docs/draft.md
touch benchmark.csv
touch solutions.jsonl

# 3. 设置环境变量
export FIB_DATASET_PATH="$PROJECT_ROOT/data/flashinfer-trace"
```

### 方法 2: 从模板复制

```bash
# 复制工作区模板
cp -r workspaces/template workspaces/<task-name>
cd workspaces/<task-name>
```

---

## 工作区生命周期

### 1. 初始化
```bash
cd workspaces/<task-name>

# 创建初始记录文件
echo "commit_hash,timestamp,kernel_name,workload_id,latency_ms,speedup,note" > benchmark.csv
echo '{"id": "init", "parent": null, "description": "Workspace initialized", "speedup": 0}' > solutions.jsonl
```

### 2. Phase 1: 建立基线
```bash
# 编写计划草稿
vim docs/draft.md

# 生成详细计划
/humanize:gen-plan

# 启动 RLCR 循环
/humanize:start-rlcr-loop

# 记录基线结果
echo "abc123,2026-06-15T10:00:00,<kernel>,baseline,1.23,1.0x,Phase 1 baseline" >> benchmark.csv
```

### 3. Phase 2: 性能优化
```bash
# NCU 性能分析
mkdir -p profile/<optimization-name>
ncu --set full -o profile/<optimization-name>/report ./benchmark

# 使用 ncu-report-skill 分析
# 在 Claude 中使用 ncu-report-skill

# 记录优化结果
echo "def456,2026-06-15T11:00:00,<kernel>,opt-v1,0.95,1.29x,Memory optimization" >> benchmark.csv
```

### 4. Phase 3: 工作负载特化
```bash
# 完整工作负载评估
flashinfer-bench run --local $FIB_DATASET_PATH --op-type <op_type>

# 最终验收
cd $PROJECT_ROOT
python verify.py --cuda --arch sm_120
```

### 5. 清理和归档
```bash
# 将成功的实现移到主仓库
cp src/<kernel>_final.cu $PROJECT_ROOT/kernels/operators/<op_type>/

# 归档工作区
tar czf workspaces/<task-name>-archive.tar.gz workspaces/<task-name>
```

---

## 文件说明

### docs/draft.md
Humanize 计划草稿，包含：
- 实现目标
- 算法流程
- 优化策略
- 验证方法

### benchmark.csv
性能记录，CSV 格式：
```csv
commit_hash,timestamp,kernel_name,workload_id,latency_ms,speedup,note
abc123,2026-06-15T10:00:00,gemm,baseline,1.23,1.0x,Phase 1 baseline
def456,2026-06-15T11:00:00,gemm,opt-v1,0.95,1.29x,Tiled GEMM
ghi789,2026-06-15T12:00:00,gemm,opt-v2,0.82,1.50x,+ Register blocking
```

### solutions.jsonl
候选实现 DAG，每行一个 JSON 对象：
```jsonl
{"id": "baseline-v1", "parent": null, "description": "Naive implementation", "speedup": 1.0, "correctness": true}
{"id": "opt-v1", "parent": "baseline-v1", "description": "Tiled GEMM", "speedup": 1.29, "correctness": true}
{"id": "opt-v2", "parent": "opt-v1", "description": "+ Register blocking", "speedup": 1.50, "correctness": true}
{"id": "opt-v3-failed", "parent": "opt-v2", "description": "Warp specialization (failed)", "speedup": 0.95, "correctness": false}
```

### profile/
NCU 性能分析目录：
```
profile/
├── baseline/
│   ├── reports/
│   │   └── kernel_baseline.ncu-rep
│   └── analysis/
│       └── bottleneck_analysis.md
├── opt-v1/
│   ├── reports/
│   │   └── kernel_opt_v1.ncu-rep
│   └── analysis/
│       └── improvement_analysis.md
```

---

## 最佳实践

### ✅ 应该做的
1. **每个任务一个工作区** - 保持隔离
2. **记录所有尝试** - 包括失败的优化
3. **保持 NCU 证据** - 每个优化决策都有数据支持
4. **频繁提交** - 小步提交，便于回滚
5. **使用 Humanize** - 结构化的计划和执行
6. **验证正确性** - 每次修改后都要验证

### ❌ 不应该做的
1. **不要在主仓库中实现** - 使用独立工作区
2. **不要跳过草稿** - 计划先行
3. **不要盲目优化** - 基于 NCU 证据
4. **不要忽略失败** - 记录为什么失败
5. **不要省略验证** - 正确性第一
6. **不要混合任务** - 一个工作区一个任务

---

## 工作区示例

### 示例 1: GEMM 优化
```
workspaces/gemm-optimization/
├── docs/
│   ├── draft.md                    # Humanize 草稿
│   └── optimization-log.md         # 优化日志
├── src/
│   ├── gemm_naive.cu               # Naive 实现
│   ├── gemm_tiled.cu               # Tiled 版本
│   └── gemm_final.cu               # 最终版本
├── profile/
│   ├── naive/
│   ├── tiled/
│   └── final/
├── benchmark.csv                   # 性能追踪
├── solutions.jsonl                 # 候选 DAG
└── README.md                       # 工作区说明
```

### 示例 2: RMSNorm 优化
```
workspaces/rmsnorm-optimization/
├── docs/
│   └── draft.md
├── src/
│   ├── rmsnorm_baseline.cu
│   ├── rmsnorm_warp_reduction.cu
│   └── rmsnorm_final.cu
├── profile/
│   ├── baseline/
│   └── warp_reduction/
├── benchmark.csv
├── solutions.jsonl
└── README.md
```

---

## 工具集成

### Humanize
```bash
# 在工作区中使用 Humanize
cd workspaces/<task-name>
vim docs/draft.md

# 生成计划
/humanize:gen-plan

# 启动循环
/humanize:start-rlcr-loop

# 查看状态
/humanize:status
```

### KernelWiki
在 Claude 对话中直接查询相关优化技术。

### ncu-report-skill
在 Claude 对话中分析 profile/ 目录下的 NCU 报告。

---

## 清理策略

### 归档成功的工作区
```bash
# 复制最终实现到主仓库
cp workspaces/<task-name>/src/<kernel>_final.cu kernels/operators/<op_type>/

# 归档工作区
tar czf archives/<task-name>-$(date +%Y%m%d).tar.gz workspaces/<task-name>

# 可选：删除工作区
rm -rf workspaces/<task-name>
```

### 清理失败的工作区
```bash
# 保存失败日志
cp workspaces/<task-name>/docs/*.md docs/failed-attempts/<task-name>/

# 删除工作区
rm -rf workspaces/<task-name>
```

---

## 环境变量

推荐在 `~/.bashrc` 或 `~/.zshrc` 中设置：

```bash
# KernelForge-MultiAgent 项目根目录
export KFMA_ROOT="/path/to/KernelForge-MultiAgent"

# FlashInfer-Trace 数据集路径
export FIB_DATASET_PATH="$KFMA_ROOT/data/flashinfer-trace"

# 或者使用全局数据集
export FIB_DATASET_PATH="/path/to/flashinfer-trace"
```

---

**最后更新**: 2026-06-15  
**版本**: v1.0
