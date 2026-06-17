#!/usr/bin/env bash
# 运行单轮优化工作区初始化

# 说明：
# - 优先从 reference/<family>/baseline.json 读取锚点、数据集、definition、上一轮结论
# - family 必须属于当前主线六类算子，且所有新轮次都必须从官方 baseline 源码派生
# - 自动为本轮生成 round_config.json / BRIEF.md / draft.md
# - 自动生成 NCU / KernelWiki 证据模板，未补齐前不得进入最终决策
# - RTX 5070 / sm_120 的 NCU 证据只能使用 /usr/local/NVIDIA-Nsight-Compute-2025.2/ncu
# - 不执行 benchmark / validate，只准备好本轮工作区

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <family> <round_number>"
    echo "Example: $0 dsa_sparse_attention 0"
    exit 1
fi

FAMILY="$1"
ROUND="$2"

CANONICAL_FAMILY="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/operator_policy.py" canonical "$FAMILY" 2>/dev/null || true)"
if [ -z "$CANONICAL_FAMILY" ]; then
    echo "Error: unsupported family: $FAMILY"
    exit 1
fi
FAMILY="$CANONICAL_FAMILY"

REFERENCE_DIR="$PROJECT_ROOT/reference/$FAMILY"
ROUND_DIR="$PROJECT_ROOT/rounds/round-$ROUND/$FAMILY"
BASELINE_FILE="$REFERENCE_DIR/baseline.json"
ROUND_CONFIG="$ROUND_DIR/round_config.json"

if [ ! -d "$REFERENCE_DIR" ]; then
    echo "Error: Reference directory not found: $REFERENCE_DIR"
    echo "Please initialize the campaign first:"
    echo "  ./scripts/start-campaign.sh $FAMILY 10"
    exit 1
fi

echo "=========================================="
echo "Round $ROUND: $FAMILY"
echo "=========================================="

# baseline.json 必须先满足主线 family policy
if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/operator_policy.py" validate "$FAMILY" "$BASELINE_FILE" >/dev/null 2>&1; then
    echo "Error: $BASELINE_FILE does not satisfy the mainline operator policy."
    echo "Please fill or fix baseline.json before starting this round."
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/operator_policy.py" validate "$FAMILY" "$BASELINE_FILE" || true
    exit 1
fi

# 读取参考状态并合并 family policy 默认值
eval "$("$PYTHON_BIN" - "$FAMILY" "$ROUND" "$BASELINE_FILE" "$REFERENCE_DIR" "$ROUND_DIR" "$PROJECT_ROOT" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

family = sys.argv[1]
round_id = int(sys.argv[2])
baseline_file = Path(sys.argv[3])
reference_dir = Path(sys.argv[4])
round_dir = Path(sys.argv[5])
project_root = Path(sys.argv[6])

sys.path.insert(0, str(project_root / "scripts"))
from operator_policy import build_baseline_template  # type: ignore

defaults = build_baseline_template(family)
defaults.update(
    {
    "anchor": "",
    "anchor_kernel": "",
    "anchor_solution": "",
    "latest_sol_base": "",
    "latest_decision": "",
    "latest_reason": "",
    "next_direction": "",
    "next_technique": "",
    "ncu_summary": "",
    "author": "",
    "entry_symbol": "",
    "binding": "tvm-ffi",
    "baseline_dataset_group": defaults["baseline_dataset_group"],
    "official_baseline_source_json": "",
    "official_baseline_source_dir": "",
    "comparison_denominator": defaults["comparison_denominator"],
    "baseline_source_kind": defaults["baseline_source_kind"],
    "allowed_definition_prefixes": defaults["allowed_definition_prefixes"],
    "allowed_op_types": defaults["allowed_op_types"],
    "family_tier": defaults["family_tier"],
    "supported_targets": defaults["supported_targets"],
    "required_ncu_binary": defaults["required_ncu_binary"],
    "required_ncu_version": defaults["required_ncu_version"],
    }
)

data = {}
if baseline_file.exists():
    data = json.loads(baseline_file.read_text(encoding="utf-8"))

latest = data.get("latest_result") or {}

def pick(*values, default=""):
    for value in values:
        if value not in (None, ""):
            return value
    return default

dataset_path = pick(data.get("dataset_path"), defaults["dataset_path"])
definition = pick(data.get("definition"), defaults["definition"])
baseline_solution = pick(data.get("baseline_solution"), defaults["baseline_solution"])
anchor = pick(data.get("anchor"), defaults["anchor"])
anchor_kernel = pick(data.get("anchor_kernel"), defaults["anchor_kernel"])
if anchor_kernel and not os.path.isabs(anchor_kernel):
    anchor_kernel = str((baseline_file.parent / anchor_kernel).resolve())
anchor_solution = pick(data.get("anchor_solution"), data.get("solution_name"), defaults["anchor_solution"])
accept_threshold = pick(data.get("accept_threshold"), defaults["accept_threshold"])
latest_sol_base = pick(latest.get("avg_sol_vs_base"), data.get("best_sol_vs_base"), defaults["latest_sol_base"])
latest_decision = pick(latest.get("decision"), defaults["latest_decision"])
latest_reason = pick(latest.get("reason"), defaults["latest_reason"])
next_direction = pick(data.get("next_direction"), latest.get("next_direction"), defaults["next_direction"])
next_technique = pick(data.get("next_technique"), latest.get("next_technique"), defaults["next_technique"])
ncu_summary = pick(data.get("latest_ncu_summary"), latest.get("ncu_summary"), defaults["ncu_summary"])
solution_prefix = pick(data.get("solution_prefix"), defaults["solution_prefix"])
author = pick(data.get("author"), defaults["author"])
op_type = pick(data.get("op_type"), defaults["op_type"])
baseline_dataset_group = pick(data.get("baseline_dataset_group"), defaults["baseline_dataset_group"])
entry_symbol = pick(data.get("entry_symbol"), defaults["entry_symbol"])
binding = pick(data.get("binding"), defaults["binding"])
comparison_denominator = pick(data.get("comparison_denominator"), defaults["comparison_denominator"])
baseline_source_kind = pick(data.get("baseline_source_kind"), defaults["baseline_source_kind"])
required_ncu_binary = pick(data.get("required_ncu_binary"), defaults["required_ncu_binary"])
required_ncu_version = pick(data.get("required_ncu_version"), defaults["required_ncu_version"])
derive_from_official_baseline = bool(
    pick(data.get("derive_from_official_baseline"), defaults["derive_from_official_baseline"])
)
allowed_definition_prefixes = data.get("allowed_definition_prefixes") or defaults["allowed_definition_prefixes"]
allowed_op_types = data.get("allowed_op_types") or defaults["allowed_op_types"]
family_tier = pick(data.get("family_tier"), defaults["family_tier"])
supported_targets = data.get("supported_targets") or defaults["supported_targets"]

official_baseline_source_json = ""
if dataset_path and definition and baseline_solution:
    official_baseline_source_json = str(
        (
            Path(dataset_path)
            / "solutions"
            / "baseline"
            / baseline_dataset_group
            / definition
            / f"{baseline_solution}.json"
        ).resolve()
    )
official_baseline_source_dir = str((round_dir / "src" / "official_baseline").resolve())

candidate_variant = f"round{round_id}-v1"
candidate_solution = pick(
    data.get("candidate_solution"),
    f"{solution_prefix}{round_id + 1}" if solution_prefix else "",
)
candidate_kernel = str((round_dir / "src" / "kernel.cu").resolve())
baseline_copy = str((round_dir / "src" / "baseline.cu").resolve())
ncu_evidence_file = str((round_dir / "profile" / "ncu_evidence.json").resolve())
kernelwiki_evidence_file = str((round_dir / "docs" / "kernelwiki_evidence.json").resolve())
bootstrap = "1" if (round_id == 0 and latest_decision == "") else "0"

fields = {
    "DATASET_PATH": str(dataset_path),
    "DEFINITION": str(definition),
    "BASELINE_SOLUTION": str(baseline_solution),
    "ANCHOR": str(anchor),
    "ANCHOR_KERNEL": str(anchor_kernel),
    "ANCHOR_SOLUTION": str(anchor_solution),
    "ACCEPT_THRESHOLD": str(accept_threshold),
    "LATEST_SOL_BASE": str(latest_sol_base),
    "LATEST_DECISION": str(latest_decision),
    "LATEST_REASON": str(latest_reason),
    "NEXT_DIRECTION": str(next_direction),
    "NEXT_TECHNIQUE": str(next_technique),
    "NCU_SUMMARY": str(ncu_summary),
    "SOLUTION_PREFIX": str(solution_prefix),
    "AUTHOR": str(author),
    "OP_TYPE": str(op_type),
    "BASELINE_DATASET_GROUP": str(baseline_dataset_group),
    "ENTRY_SYMBOL": str(entry_symbol),
    "BINDING": str(binding),
    "COMPARISON_DENOMINATOR": str(comparison_denominator),
    "BASELINE_SOURCE_KIND": str(baseline_source_kind),
    "ALLOWED_DEFINITION_PREFIXES_JSON": json.dumps(allowed_definition_prefixes, ensure_ascii=False),
    "ALLOWED_OP_TYPES_JSON": json.dumps(allowed_op_types, ensure_ascii=False),
    "FAMILY_TIER": str(family_tier),
    "SUPPORTED_TARGETS_JSON": json.dumps(supported_targets, ensure_ascii=False),
    "REQUIRED_NCU_BINARY": str(required_ncu_binary),
    "REQUIRED_NCU_VERSION": str(required_ncu_version),
    "DERIVE_FROM_OFFICIAL_BASELINE": "1" if derive_from_official_baseline else "0",
    "OFFICIAL_BASELINE_SOURCE_JSON": str(official_baseline_source_json),
    "OFFICIAL_BASELINE_SOURCE_DIR": str(official_baseline_source_dir),
    "CANDIDATE_VARIANT": str(candidate_variant),
    "CANDIDATE_SOLUTION": str(candidate_solution),
    "CANDIDATE_KERNEL": candidate_kernel,
    "BASELINE_COPY": baseline_copy,
    "NCU_EVIDENCE_FILE": ncu_evidence_file,
    "KERNELWIKI_EVIDENCE_FILE": kernelwiki_evidence_file,
    "BOOTSTRAP_MODE": bootstrap,
}

for key, value in fields.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

# Step 1: derive (create round environment)
echo "Step 1/10: derive (creating round environment)"
mkdir -p "$ROUND_DIR"/{docs,src,profile}
mkdir -p "$OFFICIAL_BASELINE_SOURCE_DIR"

if [ "$DERIVE_FROM_OFFICIAL_BASELINE" = "1" ] && [ -n "$OFFICIAL_BASELINE_SOURCE_JSON" ] && [ -f "$OFFICIAL_BASELINE_SOURCE_JSON" ]; then
    "$PYTHON_BIN" - "$OFFICIAL_BASELINE_SOURCE_JSON" "$OFFICIAL_BASELINE_SOURCE_DIR" <<'PY'
import json
import sys
from pathlib import Path

src_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
payload = json.loads(src_json.read_text(encoding="utf-8"))
for source in payload.get("sources", []):
    rel_path = Path(source["path"])
    target = out_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source["content"], encoding="utf-8")
PY
    echo "  ✓ Extracted official baseline source: $OFFICIAL_BASELINE_SOURCE_JSON"
else
    echo "  ! Official baseline source json not found: $OFFICIAL_BASELINE_SOURCE_JSON"
fi

# 复制锚点内核，baseline.cu 作为参考，kernel.cu 作为本轮可直接编辑的工作副本
if [ -n "$ANCHOR_KERNEL" ] && [ -f "$ANCHOR_KERNEL" ]; then
    cp "$ANCHOR_KERNEL" "$BASELINE_COPY"
    cp "$ANCHOR_KERNEL" "$CANDIDATE_KERNEL"
    echo "  ✓ Copied anchor variant: $ANCHOR"
else
    echo "  ! No anchor kernel found; src/kernel.cu will need to be created manually"
fi

# Step 2: 写 round_config.json 与 BRIEF.md
echo "Step 2/10: brief (generating BRIEF.md)"

cat > "$ROUND_CONFIG" <<EOF
{
  "family": "$FAMILY",
  "round": $ROUND,
  "dataset_path": "$DATASET_PATH",
  "definition": "$DEFINITION",
  "baseline_solution": "$BASELINE_SOLUTION",
  "anchor_variant": "$ANCHOR",
  "anchor_solution": "$ANCHOR_SOLUTION",
  "anchor_kernel": "$ANCHOR_KERNEL",
  "candidate_variant": "$CANDIDATE_VARIANT",
  "candidate_solution": "$CANDIDATE_SOLUTION",
  "candidate_kernel": "$CANDIDATE_KERNEL",
  "ncu_evidence_file": "$NCU_EVIDENCE_FILE",
  "kernelwiki_evidence_file": "$KERNELWIKI_EVIDENCE_FILE",
  "requires_ncu_evidence": true,
  "requires_kernelwiki_evidence": true,
  "accept_threshold": $ACCEPT_THRESHOLD,
  "author": "$AUTHOR",
  "op_type": "$OP_TYPE",
  "baseline_dataset_group": "$BASELINE_DATASET_GROUP",
  "entry_symbol": "$ENTRY_SYMBOL",
  "binding": "$BINDING",
  "comparison_denominator": "$COMPARISON_DENOMINATOR",
  "baseline_source_kind": "$BASELINE_SOURCE_KIND",
  "allowed_definition_prefixes": $ALLOWED_DEFINITION_PREFIXES_JSON,
  "allowed_op_types": $ALLOWED_OP_TYPES_JSON,
  "family_tier": "$FAMILY_TIER",
  "supported_targets": $SUPPORTED_TARGETS_JSON,
  "required_ncu_binary": "$REQUIRED_NCU_BINARY",
  "required_ncu_version": "$REQUIRED_NCU_VERSION",
  "derive_from_official_baseline": $([ "$DERIVE_FROM_OFFICIAL_BASELINE" = "1" ] && echo "true" || echo "false"),
  "official_baseline_source_json": "$OFFICIAL_BASELINE_SOURCE_JSON",
  "official_baseline_source_dir": "$OFFICIAL_BASELINE_SOURCE_DIR"
}
EOF

if [ -f "$REFERENCE_DIR/TRAPS.md" ]; then
    TRAPS_SUMMARY="$(awk '
        /^### / {print "- " substr($0, 5); count++; if (count >= 6) exit}
    ' "$REFERENCE_DIR/TRAPS.md")"
fi

if [ -z "${TRAPS_SUMMARY:-}" ]; then
    TRAPS_SUMMARY="- 暂无已记录陷阱，请至少检查数值稳定性、共享内存冲突和小 batch 并行度"
fi

if [ "$BOOTSTRAP_MODE" = "1" ]; then
    GOAL_BLOCK=$(cat <<EOF
- **目标**: 先建立可验证基线
- **验收**: FlashInfer-Bench 正确性全部通过
- **性能口径**: 记录首次 sol/base，作为后续轮次的证据起点
EOF
)
else
    GOAL_BLOCK=$(cat <<EOF
- **目标速度线**: sol/base >= ${ACCEPT_THRESHOLD}x
- **当前锚点**: ${ANCHOR:-未建立} （solution: ${ANCHOR_SOLUTION:-未建立}）
- **上一轮结果**: ${LATEST_DECISION:-unknown}，sol/base = ${LATEST_SOL_BASE:-unknown}x
- **优化方向**: ${NEXT_DIRECTION:-请先补充 NCU 分析}
- **优先技术**: ${NEXT_TECHNIQUE:-请根据 NCU 结论选择}
EOF
)
fi

cat > "$ROUND_DIR/BRIEF.md" <<EOF
# Round $ROUND Brief - $FAMILY

## 当前状态
- Round: $ROUND
- Family: $FAMILY
- Anchor Variant: ${ANCHOR:-未建立}
- Anchor Kernel: ${ANCHOR_KERNEL:-未建立}
- Candidate Variant: $CANDIDATE_VARIANT
- Candidate Solution: ${CANDIDATE_SOLUTION:-未配置}

## 本轮目标
$GOAL_BLOCK

## 上一轮证据
- **决策原因**: ${LATEST_REASON:-暂无}
- **NCU 摘要**: ${NCU_SUMMARY:-暂无，执行 profile 后请补充到 reference/$FAMILY/baseline.json}
- **官方 baseline 派生规则**: 本轮必须从数据集里的官方 baseline 源码派生，参考源码位于：\`$OFFICIAL_BASELINE_SOURCE_DIR\`

## 本轮硬约束证据
- **真实 NCU 证据文件**: \`$NCU_EVIDENCE_FILE\`
- **KernelWiki 依据文件**: \`$KERNELWIKI_EVIDENCE_FILE\`
- **规则 1**: 必须同时填写 solution 与官方 baseline 的真实 NCU 报告
- **规则 2**: 必须记录本轮参考的 KernelWiki 页面，并说明其为何适用于本轮
- **规则 3**: NCU 只能使用 \`/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu\`；禁止使用 \`/usr/local/cuda/bin/ncu\`
- **规则 4**: 没有补齐这两个文件时，\`./scripts/evaluate-round.sh\` 会直接失败，不允许进入决策
- **规则 5**: 新轮次禁止从旧的自研 rejected 候选直接派生，必须先阅读并基于官方 baseline 源码展开改动
- **规则 6**: 本轮 family policy 固定为 \`$COMPARISON_DENOMINATOR\`，不得把 parent/历史 anchor 当最终分母

## 约束
- 必须使用真实 FlashInfer-Bench 数据集：\`$DATASET_PATH\`
- definition 固定为：\`$DEFINITION\`
- 官方 baseline 固定为：\`$BASELINE_SOLUTION\`
- 最终成绩分母固定为：\`$COMPARISON_DENOMINATOR\`
- 正确性不过即 REJECT
- 成绩只认 \`sol/base\`
- RTX 5070 / sm_120 属于 Blackwell，必须参考 \`skills/KernelWiki\`
- 对仅适用于 SM100/B200 的特性，必须在 KernelWiki 依据文件中明确标注“不适用于 sm_120”

## 需要规避的陷阱
$TRAPS_SUMMARY

## 资源
- 参考锚点代码：\`$BASELINE_COPY\`
- 官方 baseline 源码：\`$OFFICIAL_BASELINE_SOURCE_DIR\`
- 当前可编辑代码：\`$CANDIDATE_KERNEL\`
- 回写 solution 脚本：\`$ROUND_DIR/src/gen_solution.py\`
- 本轮配置：\`$ROUND_CONFIG\`
- NCU 证据模板：\`$NCU_EVIDENCE_FILE\`
- KernelWiki 证据模板：\`$KERNELWIKI_EVIDENCE_FILE\`
- 历史陷阱：\`$REFERENCE_DIR/TRAPS.md\`

## 推荐流程
1. 阅读本文件和 \`TRAPS.md\`
2. 先阅读 \`src/official_baseline/\` 下的官方 baseline 源码，再在 \`src/kernel.cu\` 上做派生改动
3. 查阅 \`skills/KernelWiki\`，把本轮参考页面写入 \`docs/kernelwiki_evidence.json\`
4. 把设计写入 \`docs/draft.md\`
5. 运行 \`python src/gen_solution.py\` 生成候选 solution
6. 分别对 candidate / baseline 做真实 NCU，并填写 \`profile/ncu_evidence.json\`
7. 回到仓库根目录执行 \`./scripts/evaluate-round.sh $FAMILY $ROUND\`
8. 如果需要新的优化方向，补做 NCU 并把结论写回 \`reference/$FAMILY/baseline.json\`

---

Generated: $(date -Iseconds)
EOF

echo "  ✓ BRIEF.md generated at: $ROUND_DIR/BRIEF.md"

if [ ! -f "$NCU_EVIDENCE_FILE" ]; then
cat > "$NCU_EVIDENCE_FILE" <<EOF
{
  "required": true,
  "status": "PENDING",
  "round": $ROUND,
  "family": "$FAMILY",
  "definition": "$DEFINITION",
  "device": "RTX 5070 Laptop / sm_120",
  "required_ncu_binary": "/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu",
  "required_ncu_version": "2025.2",
  "collected_at": "",
  "solution_profile": {
    "kernel_or_solution": "$CANDIDATE_SOLUTION",
    "report_path": "",
    "command": "",
    "batch_size": "",
    "workload": "",
    "key_metrics": [],
    "key_findings": []
  },
  "baseline_profile": {
    "kernel_or_solution": "$BASELINE_SOLUTION",
    "report_path": "",
    "command": "",
    "batch_size": "",
    "workload": "",
    "key_metrics": [],
    "key_findings": []
  },
  "comparison_summary": {
    "bottleneck": "",
    "decision_driver": "",
    "why_this_round": ""
  }
}
EOF
echo "  ✓ NCU evidence template created"
else
    echo "  ✓ NCU evidence template already exists, kept as-is"
fi

if [ ! -f "$KERNELWIKI_EVIDENCE_FILE" ]; then
cat > "$KERNELWIKI_EVIDENCE_FILE" <<EOF
{
  "required": true,
  "status": "PENDING",
  "round": $ROUND,
  "family": "$FAMILY",
  "architecture": "sm_120",
  "reviewed_at": "",
  "pages": [
    {
      "path": "skills/KernelWiki/wiki/patterns/low-sm-utilization.md",
      "id": "pattern-low-sm-utilization",
      "reason": "",
      "how_it_guides_this_round": "",
      "applicable_to_sm120": true
    },
    {
      "path": "skills/KernelWiki/wiki/patterns/memory-bound.md",
      "id": "pattern-memory-bound",
      "reason": "",
      "how_it_guides_this_round": "",
      "applicable_to_sm120": true
    },
    {
      "path": "skills/KernelWiki/wiki/techniques/vectorized-loads.md",
      "id": "technique-vectorized-loads",
      "reason": "",
      "how_it_guides_this_round": "",
      "applicable_to_sm120": true
    },
    {
      "path": "skills/KernelWiki/wiki/techniques/register-budgeting.md",
      "id": "technique-register-budgeting",
      "reason": "",
      "how_it_guides_this_round": "",
      "applicable_to_sm120": true
    }
  ],
  "decision_notes": ""
}
EOF
echo "  ✓ KernelWiki evidence template created"
else
    echo "  ✓ KernelWiki evidence template already exists, kept as-is"
fi

if [ -f "$PROJECT_ROOT/docs/draft_template.md" ]; then
    cp "$PROJECT_ROOT/docs/draft_template.md" "$ROUND_DIR/docs/draft.md"
    echo "  ✓ Draft template copied"
fi

if [ -n "$DATASET_PATH" ] && [ -n "$DEFINITION" ] && [ -n "$CANDIDATE_SOLUTION" ] && [ -n "$OP_TYPE" ]; then
    cat > "$ROUND_DIR/src/gen_solution.py" <<EOF
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将本轮候选 kernel.cu 写入 flashinfer-trace solution.json。
"""

import json
from pathlib import Path

KERNEL_SRC = Path(__file__).resolve().parent / "kernel.cu"
DATASET_ROOT = Path("$DATASET_PATH")
AUTHOR = "$AUTHOR"
OP_TYPE = "$OP_TYPE"
DEFINITION = "$DEFINITION"
SOLUTION_NAME = "$CANDIDATE_SOLUTION"
ENTRY_SYMBOL = "$ENTRY_SYMBOL"
BINDING = "$BINDING"
FAMILY = "$FAMILY"


def main() -> None:
    if not ENTRY_SYMBOL:
        raise SystemExit("ENTRY_SYMBOL 不能为空，请先在 reference/<family>/baseline.json 中补齐后再生成 solution。")

    cuda_source = KERNEL_SRC.read_text(encoding="utf-8")
    solution = {
        "name": SOLUTION_NAME,
        "definition": DEFINITION,
        "author": AUTHOR or "kernelforge",
        "description": f"KernelForge-MultiAgent 自动生成的 {FAMILY} 候选实现。",
        "spec": {
            "language": "cuda",
            "target_hardware": ["NVIDIA_Blackwell", "cuda"],
            "entry_point": f"kernel.cu::{ENTRY_SYMBOL}",
            "binding": BINDING,
            "destination_passing_style": True,
            "dependencies": []
        },
        "sources": [{"path": "kernel.cu", "content": cuda_source}]
    }

    out_dir = DATASET_ROOT / "solutions" / (AUTHOR or "kernelforge") / OP_TYPE / DEFINITION
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{SOLUTION_NAME}.json"
    out_path.write_text(json.dumps(solution, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 solution: {out_path}")


if __name__ == "__main__":
    main()
EOF
    chmod +x "$ROUND_DIR/src/gen_solution.py"
    echo "  ✓ Generic solution generator created"
fi

# 初始化跟踪文件
echo "commit_hash,timestamp,kernel_name,workload_id,latency_ms,speedup,note" > "$ROUND_DIR/benchmark.csv"
echo "{\"id\": \"$CANDIDATE_VARIANT\", \"parent\": \"${ANCHOR:-null}\", \"solution_name\": \"$CANDIDATE_SOLUTION\", \"description\": \"Round $ROUND candidate initialized\"}" > "$ROUND_DIR/solutions.jsonl"

echo ""
echo "✅ Round $ROUND environment ready"
echo ""
echo "=========================================="
echo "Next Steps (Sub Agent):"
echo "=========================================="
echo ""
echo "1. Read BRIEF.md:"
echo "   cat $ROUND_DIR/BRIEF.md"
echo ""
echo "2. Review TRAPS.md:"
echo "   cat $REFERENCE_DIR/TRAPS.md"
echo ""
echo "3. Edit draft.md:"
echo "   vim $ROUND_DIR/docs/draft.md"
echo ""
echo "4. Fill KernelWiki evidence:"
echo "   vim $KERNELWIKI_EVIDENCE_FILE"
echo ""
echo "5. Generate solution after editing kernel.cu:"
echo "   python $ROUND_DIR/src/gen_solution.py"
echo ""
echo "6. Run real NCU for candidate and baseline, then fill:"
echo "   vim $NCU_EVIDENCE_FILE"
echo ""
echo "7. Evaluate automatically:"
echo "   ./scripts/evaluate-round.sh $FAMILY $ROUND"
echo ""
echo "8. If rejected, update reference/$FAMILY/baseline.json with new NCU findings"
echo ""
echo "=========================================="
