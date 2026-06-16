#!/usr/bin/env bash
# 自动评估单轮优化结果

# 说明：
# - 读取 round_config.json / reference/<family>/baseline.json
# - 调用 fib_inproc_validate.py 做真实正确性与 sol/base 评估
# - 自动写入 decision.json，并把最新结果回写到 reference/<family>/baseline.json
# - 对 ACCEPT 的候选自动归档到 reference/<family>/variants/<candidate>/
# - 严格要求：没有本轮真实 NCU 证据和 KernelWiki 依据，不允许进入决策

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <family> <round_number>"
    echo "Example: $0 rmsnorm 0"
    exit 1
fi

FAMILY="$1"
ROUND="$2"

REFERENCE_DIR="$PROJECT_ROOT/reference/$FAMILY"
ROUND_DIR="$PROJECT_ROOT/rounds/round-$ROUND/$FAMILY"
BASELINE_FILE="$REFERENCE_DIR/baseline.json"
ROUND_CONFIG="$ROUND_DIR/round_config.json"
VALIDATE_LOG="$ROUND_DIR/validate.log"

echo "=========================================="
echo "Evaluating Round $ROUND: $FAMILY"
echo "=========================================="
echo ""

# Check if optimization is complete
if [ ! -d "$ROUND_DIR/src" ]; then
    echo "Error: Round directory not found: $ROUND_DIR"
    exit 1
fi

if [ ! -f "$ROUND_CONFIG" ]; then
    echo "Error: Round config not found: $ROUND_CONFIG"
    echo "Please run ./scripts/run-round.sh $FAMILY $ROUND first."
    exit 1
fi

eval "$("$PYTHON_BIN" - "$BASELINE_FILE" "$ROUND_CONFIG" <<'PY'
import json
import shlex
import sys
from pathlib import Path

baseline_file = Path(sys.argv[1])
round_config_file = Path(sys.argv[2])

baseline = json.loads(baseline_file.read_text(encoding="utf-8")) if baseline_file.exists() else {}
round_cfg = json.loads(round_config_file.read_text(encoding="utf-8"))

def pick(*values, default=""):
    for value in values:
        if value not in (None, ""):
            return value
    return default

fields = {
    "DATASET_PATH": pick(round_cfg.get("dataset_path"), baseline.get("dataset_path")),
    "DEFINITION": pick(round_cfg.get("definition"), baseline.get("definition")),
    "BASELINE_SOLUTION": pick(round_cfg.get("baseline_solution"), baseline.get("baseline_solution")),
    "CANDIDATE_SOLUTION": pick(round_cfg.get("candidate_solution"), baseline.get("candidate_solution"), baseline.get("solution_name")),
    "CANDIDATE_VARIANT": pick(round_cfg.get("candidate_variant"), f"round{round_cfg.get('round', 0)}-v1"),
    "CANDIDATE_KERNEL": pick(round_cfg.get("candidate_kernel")),
    "ANCHOR_VARIANT": pick(round_cfg.get("anchor_variant"), baseline.get("anchor")),
    "NCU_EVIDENCE_FILE": pick(round_cfg.get("ncu_evidence_file")),
    "KERNELWIKI_EVIDENCE_FILE": pick(round_cfg.get("kernelwiki_evidence_file")),
    "ACCEPT_THRESHOLD": str(pick(round_cfg.get("accept_threshold"), baseline.get("accept_threshold"), 1.05)),
}

for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

for required_name in DATASET_PATH DEFINITION BASELINE_SOLUTION CANDIDATE_SOLUTION; do
    if [ -z "${!required_name:-}" ]; then
        echo "Error: missing required config field: $required_name"
        exit 1
    fi
done

echo "Step 0/10: validating NCU + KernelWiki evidence"
eval "$("$PYTHON_BIN" - "$PROJECT_ROOT" "$ROUND_DIR" "$ROUND_CONFIG" <<'PY'
import json
import shlex
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
round_dir = Path(sys.argv[2])
round_cfg = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)

def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

ncu_file_raw = round_cfg.get("ncu_evidence_file", "")
kernelwiki_file_raw = round_cfg.get("kernelwiki_evidence_file", "")
if not ncu_file_raw:
    fail("Error: round_config.json 缺少 ncu_evidence_file")
if not kernelwiki_file_raw:
    fail("Error: round_config.json 缺少 kernelwiki_evidence_file")

ncu_file = resolve_path(ncu_file_raw)
kernelwiki_file = resolve_path(kernelwiki_file_raw)
if not ncu_file.exists():
    fail(f"Error: NCU evidence file not found: {ncu_file}")
if not kernelwiki_file.exists():
    fail(f"Error: KernelWiki evidence file not found: {kernelwiki_file}")

ncu = json.loads(ncu_file.read_text(encoding='utf-8'))
kernelwiki = json.loads(kernelwiki_file.read_text(encoding='utf-8'))

if round_cfg.get("derive_from_official_baseline") is True:
    baseline_src_json_raw = str(round_cfg.get("official_baseline_source_json", "")).strip()
    baseline_src_dir_raw = str(round_cfg.get("official_baseline_source_dir", "")).strip()
    if not baseline_src_json_raw:
        fail("Error: round_config.json 缺少 official_baseline_source_json")
    if not baseline_src_dir_raw:
        fail("Error: round_config.json 缺少 official_baseline_source_dir")
    baseline_src_json = resolve_path(baseline_src_json_raw)
    baseline_src_dir = resolve_path(baseline_src_dir_raw)
    if not baseline_src_json.exists():
        fail(f"Error: official baseline source json not found: {baseline_src_json}")
    if not baseline_src_dir.exists():
        fail(f"Error: official baseline source dir not found: {baseline_src_dir}")
    if not any(p.is_file() for p in baseline_src_dir.rglob("*")):
        fail(f"Error: official baseline source dir is empty: {baseline_src_dir}")

if ncu.get("status") != "COMPLETE":
    fail("Error: NCU evidence status 必须为 COMPLETE")
if kernelwiki.get("status") != "COMPLETE":
    fail("Error: KernelWiki evidence status 必须为 COMPLETE")

solution_profile = ncu.get("solution_profile") or {}
baseline_profile = ncu.get("baseline_profile") or {}
comparison_summary = ncu.get("comparison_summary") or {}
required_ncu_binary = str(
    ncu.get("required_ncu_binary", "/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu")
).strip()
required_ncu_version = str(ncu.get("required_ncu_version", "2025.2")).strip()

if required_ncu_binary != "/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu":
    fail("Error: required_ncu_binary 必须固定为 /usr/local/NVIDIA-Nsight-Compute-2025.2/ncu")
if required_ncu_version != "2025.2":
    fail("Error: required_ncu_version 必须固定为 2025.2")

for label, profile in (("solution", solution_profile), ("baseline", baseline_profile)):
    report_path_raw = str(profile.get("report_path", "")).strip()
    command = str(profile.get("command", "")).strip()
    if not report_path_raw:
        fail(f"Error: NCU evidence 缺少 {label} report_path")
    if not command:
        fail(f"Error: NCU evidence 缺少 {label} command")
    if required_ncu_binary not in command:
        fail(
            f"Error: {label} NCU command 必须使用 {required_ncu_binary}，"
            "禁止使用 /usr/local/cuda/bin/ncu"
        )
    report_path = Path(report_path_raw)
    if not report_path.is_absolute():
        report_path = (round_dir / report_path).resolve()
    if not report_path.exists():
        fail(f"Error: NCU report file not found for {label}: {report_path}")

bottleneck = str(comparison_summary.get("bottleneck", "")).strip()
decision_driver = str(comparison_summary.get("decision_driver", "")).strip()
why_this_round = str(comparison_summary.get("why_this_round", "")).strip()
if not bottleneck:
    fail("Error: NCU evidence comparison_summary.bottleneck 不能为空")
if not decision_driver:
    fail("Error: NCU evidence comparison_summary.decision_driver 不能为空")
if not why_this_round:
    fail("Error: NCU evidence comparison_summary.why_this_round 不能为空")

pages = kernelwiki.get("pages") or []
if not pages:
    fail("Error: KernelWiki evidence.pages 不能为空")

applicable_pages = 0
primary_page = ""
for idx, page in enumerate(pages):
    path_raw = str(page.get("path", "")).strip()
    page_id = str(page.get("id", "")).strip()
    reason = str(page.get("reason", "")).strip()
    guide = str(page.get("how_it_guides_this_round", "")).strip()
    if not path_raw or not page_id:
        fail(f"Error: KernelWiki evidence.pages[{idx}] 缺少 path 或 id")
    page_path = resolve_path(path_raw)
    if not page_path.exists():
        fail(f"Error: KernelWiki page not found: {page_path}")
    if not reason:
        fail(f"Error: KernelWiki evidence.pages[{idx}].reason 不能为空")
    if not guide:
        fail(f"Error: KernelWiki evidence.pages[{idx}].how_it_guides_this_round 不能为空")
    if page.get("applicable_to_sm120") is True:
        applicable_pages += 1
        if not primary_page:
            primary_page = path_raw

decision_notes = str(kernelwiki.get("decision_notes", "")).strip()
if not decision_notes:
    fail("Error: KernelWiki evidence.decision_notes 不能为空")
if applicable_pages <= 0:
    fail("Error: KernelWiki evidence 至少要有一个页面标注 applicable_to_sm120=true")

fields = {
    "NCU_EVIDENCE_FILE": str(ncu_file),
    "KERNELWIKI_EVIDENCE_FILE": str(kernelwiki_file),
    "NCU_BOTTLENECK": bottleneck,
    "NCU_DECISION_DRIVER": decision_driver,
    "NCU_REQUIRED_BINARY": required_ncu_binary,
    "KERNELWIKI_PAGE_COUNT": str(len(pages)),
    "KERNELWIKI_PRIMARY_PAGE": primary_page,
}

for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
echo "  ✓ NCU evidence    : $NCU_EVIDENCE_FILE"
echo "  ✓ NCU binary      : $NCU_REQUIRED_BINARY"
echo "  ✓ KernelWiki refs : $KERNELWIKI_EVIDENCE_FILE"
if [ "${DERIVE_FROM_OFFICIAL_BASELINE:-}" = "1" ]; then
  echo "  ✓ Baseline derive  : official baseline source extracted"
fi
echo "  ✓ NCU bottleneck  : $NCU_BOTTLENECK"
echo "  ✓ Decision driver : $NCU_DECISION_DRIVER"
echo ""

# Step 4-6: benchmark + validate + compare
echo "Step 4-6/10: benchmark + validate + compare"
echo "  dataset   : $DATASET_PATH"
echo "  definition: $DEFINITION"
echo "  solution  : $CANDIDATE_SOLUTION"
echo "  baseline  : $BASELINE_SOLUTION"
echo ""

set +e
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/workflow/fib_inproc_validate.py" \
    --dataset "$DATASET_PATH" \
    --definition "$DEFINITION" \
    --solution "$CANDIDATE_SOLUTION" \
    --baseline "$BASELINE_SOLUTION" \
    > "$VALIDATE_LOG" 2>&1
VALIDATE_EXIT=$?
set -e

echo "  ✓ Validation log saved to: $VALIDATE_LOG"

eval "$("$PYTHON_BIN" - "$VALIDATE_LOG" "$VALIDATE_EXIT" "$ACCEPT_THRESHOLD" <<'PY'
import re
import shlex
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
validate_exit = int(sys.argv[2])
accept_threshold = float(sys.argv[3])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

passed = total = 0
sol_ref = ""
sol_base = ""
summary_match = re.search(r"汇总:\s*(\d+)\s*/\s*(\d+)\s*通过正确性", text)
if summary_match:
    passed, total = summary_match.groups()
ref_match = re.search(r"平均加速比 \(sol vs 参考实现 PyTorch\)\s*=\s*([0-9.]+)x", text)
if ref_match:
    sol_ref = ref_match.group(1)
base_match = re.search(r"平均加速比 \(sol vs 官方 baseline .*?\)\s*=\s*([0-9.]+)x", text)
if base_match:
    sol_base = base_match.group(1)

correctness = "true" if validate_exit == 0 and total and passed == total else "false"
decision = "REJECT"
reason = "验证日志未提取到有效结果"

if correctness != "true":
    reason = f"正确性未全部通过（{passed}/{total}）"
elif sol_base:
    if float(sol_base) >= accept_threshold:
        decision = "ACCEPT"
        reason = f"sol/base = {sol_base}x，达到 {accept_threshold:.2f}x 验收线"
    else:
        reason = f"sol/base = {sol_base}x，低于 {accept_threshold:.2f}x 验收线"

fields = {
    "PASSED_WORKLOADS": str(passed),
    "TOTAL_WORKLOADS": str(total),
    "SOL_VS_REF": str(sol_ref),
    "SOL_VS_BASE": str(sol_base),
    "CORRECTNESS": correctness,
    "DECISION": decision,
    "REASON": reason,
}

for key, value in fields.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

echo ""
echo "Decision: $DECISION"
echo "Reason: $REASON"
echo ""

# Save decision
mkdir -p "$ROUND_DIR"
cat > "$ROUND_DIR/decision.json" <<EOF
{
  "round": $ROUND,
  "family": "$FAMILY",
  "decision": "$DECISION",
  "candidate_variant": "$CANDIDATE_VARIANT",
  "candidate_solution": "$CANDIDATE_SOLUTION",
  "anchor_variant": "$ANCHOR_VARIANT",
  "definition": "$DEFINITION",
  "dataset_path": "$DATASET_PATH",
  "baseline_solution": "$BASELINE_SOLUTION",
  "passed_workloads": ${PASSED_WORKLOADS:-0},
  "total_workloads": ${TOTAL_WORKLOADS:-0},
  "avg_sol_vs_ref": ${SOL_VS_REF:-null},
  "avg_sol_vs_base": ${SOL_VS_BASE:-null},
  "correctness": $CORRECTNESS,
  "reason": "$REASON",
  "ncu_evidence_file": "$NCU_EVIDENCE_FILE",
  "kernelwiki_evidence_file": "$KERNELWIKI_EVIDENCE_FILE",
  "ncu_bottleneck": "$NCU_BOTTLENECK",
  "ncu_decision_driver": "$NCU_DECISION_DRIVER",
  "kernelwiki_page_count": ${KERNELWIKI_PAGE_COUNT:-0},
  "kernelwiki_primary_page": "$KERNELWIKI_PRIMARY_PAGE",
  "validate_log": "$VALIDATE_LOG",
  "timestamp": "$(date -Iseconds)"
}
EOF

REFERENCE_VARIANT_DIR="$REFERENCE_DIR/variants/$CANDIDATE_VARIANT"
if [ "$DECISION" = "ACCEPT" ] && [ -f "$CANDIDATE_KERNEL" ]; then
    mkdir -p "$REFERENCE_VARIANT_DIR"
    cp "$CANDIDATE_KERNEL" "$REFERENCE_VARIANT_DIR/kernel.cu"
fi

# 回写 baseline.json 与 solutions.jsonl
"$PYTHON_BIN" - "$BASELINE_FILE" "$ROUND_CONFIG" "$ROUND_DIR/decision.json" "$REFERENCE_DIR/solutions.jsonl" <<'PY'
import json
from datetime import datetime
from pathlib import Path
import sys

baseline_file = Path(sys.argv[1])
round_config_file = Path(sys.argv[2])
decision_file = Path(sys.argv[3])
solutions_file = Path(sys.argv[4])

baseline = json.loads(baseline_file.read_text(encoding="utf-8")) if baseline_file.exists() else {}
round_cfg = json.loads(round_config_file.read_text(encoding="utf-8"))
decision = json.loads(decision_file.read_text(encoding="utf-8"))

baseline.setdefault("family", round_cfg.get("family"))
baseline.setdefault("dataset_path", round_cfg.get("dataset_path"))
baseline.setdefault("definition", round_cfg.get("definition"))
baseline.setdefault("baseline_solution", round_cfg.get("baseline_solution"))
baseline.setdefault("accept_threshold", round_cfg.get("accept_threshold", 1.05))
baseline.setdefault("author", round_cfg.get("author", ""))
baseline.setdefault("op_type", round_cfg.get("op_type", round_cfg.get("family")))
baseline.setdefault("entry_symbol", round_cfg.get("entry_symbol", ""))
baseline.setdefault("binding", round_cfg.get("binding", "tvm-ffi"))
baseline.setdefault("solution_prefix", "")

baseline["latest_result"] = {
    "round": decision["round"],
    "candidate_variant": decision["candidate_variant"],
    "solution_name": decision["candidate_solution"],
    "decision": decision["decision"],
    "passed_workloads": decision["passed_workloads"],
    "total_workloads": decision["total_workloads"],
    "avg_sol_vs_ref": decision["avg_sol_vs_ref"],
    "avg_sol_vs_base": decision["avg_sol_vs_base"],
    "ncu_summary": f"{decision.get('ncu_bottleneck', '')}; {decision.get('ncu_decision_driver', '')}".strip("; "),
    "kernelwiki_primary_page": decision.get("kernelwiki_primary_page", ""),
    "reason": decision["reason"],
    "evaluated_at": decision["timestamp"],
}

baseline["latest_ncu_summary"] = baseline["latest_result"]["ncu_summary"]

baseline["updated_at"] = datetime.now().isoformat()

if decision["decision"] == "ACCEPT":
    baseline["anchor"] = decision["candidate_variant"]
    baseline["anchor_kernel"] = f"variants/{decision['candidate_variant']}/kernel.cu"
    baseline["solution_name"] = decision["candidate_solution"]
    if decision["avg_sol_vs_base"] is not None:
        baseline["best_sol_vs_base"] = decision["avg_sol_vs_base"]

baseline_file.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")

entry = {
    "id": decision["candidate_variant"],
    "parent": decision.get("anchor_variant"),
    "solution_name": decision["candidate_solution"],
    "definition": decision["definition"],
    "correctness": decision["correctness"],
    "passed_workloads": decision["passed_workloads"],
    "total_workloads": decision["total_workloads"],
    "sol_vs_ref": decision["avg_sol_vs_ref"],
    "sol_vs_base": decision["avg_sol_vs_base"],
    "decision": decision["decision"],
    "description": f"{decision['reason']} | NCU: {decision.get('ncu_decision_driver', '')} | KernelWiki: {decision.get('kernelwiki_primary_page', '')}",
    "timestamp": decision["timestamp"],
}

rows = []
if solutions_file.exists():
    for raw in solutions_file.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))

replaced = False
for idx, row in enumerate(rows):
    if row.get("id") == entry["id"]:
        rows[idx] = entry
        replaced = True
        break
if not replaced:
    rows.append(entry)

solutions_file.parent.mkdir(parents=True, exist_ok=True)
solutions_file.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
    encoding="utf-8",
)
PY

echo "Evaluation complete!"
echo "Decision file : $ROUND_DIR/decision.json"
echo "Reference file: $BASELINE_FILE"
