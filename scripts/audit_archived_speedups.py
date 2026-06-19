#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计 reference/ 中已归档算子的加速比是否可由仓库内真实证据回溯。

审计规则：
1. 只对 reference/<family>/solutions.jsonl 中带有 decision 的条目做检查。
2. 如果存在对应的 rounds/round-*/<family>/decision.json，则要求：
   - candidate_variant 一致
   - avg_sol_vs_base 与 reference 记录一致（允许极小浮点误差）
   - validate.log 中能解析出相同的 sol/base 与 correctness 汇总
3. 如果不存在对应 decision.json，但同 family / 同 round 工作区存在：
   - 标记为 MISSING_ROUND_DECISION
4. 如果 round 工作区都不存在：
   - 标记为 MISSING_ROUND_DIR
5. baseline.json 的 latest_result / anchor 也会做一致性检查。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "reference"
ROUNDS_ROOT = REPO_ROOT / "rounds"
FLOAT_TOL = 1e-6


@dataclass
class EntryAudit:
    family: str
    variant: str
    decision: str
    status: str
    message: str
    reference_sol_vs_base: float | None
    evidence_sol_vs_base: float | None
    evidence_path: str | None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def parse_variant_round(variant: str) -> int | None:
    match = re.match(r"round(\d+)-v\d+", variant)
    if not match:
        return None
    return int(match.group(1))


def parse_validate_log(path: Path) -> tuple[tuple[int, int] | None, float | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    correctness_match = re.search(r"汇总:\s*(\d+)\s*/\s*(\d+)\s*通过正确性", text)
    speedup_match = re.search(r"平均加速比 \(sol vs 官方 baseline .*?\)\s*=\s*([0-9.]+)x", text)
    correctness = None
    speedup = None
    if correctness_match:
        correctness = (int(correctness_match.group(1)), int(correctness_match.group(2)))
    if speedup_match:
        speedup = float(speedup_match.group(1))
    return correctness, speedup


def float_equal(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)


def audit_entry(family: str, row: dict[str, Any]) -> EntryAudit:
    variant = str(row.get("id", "")).strip()
    decision = str(row.get("decision", "")).strip()
    reference_sol_vs_base = row.get("sol_vs_base")
    if isinstance(reference_sol_vs_base, str) and reference_sol_vs_base:
        reference_sol_vs_base = float(reference_sol_vs_base)

    round_id = parse_variant_round(variant)
    if round_id is None:
        return EntryAudit(
            family=family,
            variant=variant,
            decision=decision,
            status="SKIP",
            message="无法从 variant 解析 round 编号",
            reference_sol_vs_base=reference_sol_vs_base,
            evidence_sol_vs_base=None,
            evidence_path=None,
        )

    round_dir = ROUNDS_ROOT / f"round-{round_id}" / family
    decision_path = round_dir / "decision.json"
    validate_log_path = round_dir / "validate.log"

    if not round_dir.exists():
        return EntryAudit(
            family=family,
            variant=variant,
            decision=decision,
            status="MISSING_ROUND_DIR",
            message=f"缺少 round 工作区：{round_dir}",
            reference_sol_vs_base=reference_sol_vs_base,
            evidence_sol_vs_base=None,
            evidence_path=None,
        )

    if not decision_path.exists():
        return EntryAudit(
            family=family,
            variant=variant,
            decision=decision,
            status="MISSING_ROUND_DECISION",
            message=f"缺少原始 decision.json：{decision_path}",
            reference_sol_vs_base=reference_sol_vs_base,
            evidence_sol_vs_base=None,
            evidence_path=None,
        )

    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    if str(payload.get("candidate_variant", "")).strip() != variant:
        return EntryAudit(
            family=family,
            variant=variant,
            decision=decision,
            status="VARIANT_MISMATCH",
            message="reference 条目与 decision.json 的 candidate_variant 不一致",
            reference_sol_vs_base=reference_sol_vs_base,
            evidence_sol_vs_base=payload.get("avg_sol_vs_base"),
            evidence_path=str(decision_path),
        )

    evidence_sol_vs_base = payload.get("avg_sol_vs_base")
    if isinstance(evidence_sol_vs_base, str) and evidence_sol_vs_base:
        evidence_sol_vs_base = float(evidence_sol_vs_base)

    if not float_equal(reference_sol_vs_base, evidence_sol_vs_base):
        return EntryAudit(
            family=family,
            variant=variant,
            decision=decision,
            status="SPEEDUP_MISMATCH",
            message="reference.sol_vs_base 与 decision.avg_sol_vs_base 不一致",
            reference_sol_vs_base=reference_sol_vs_base,
            evidence_sol_vs_base=evidence_sol_vs_base,
            evidence_path=str(decision_path),
        )

    if not validate_log_path.exists():
        return EntryAudit(
            family=family,
            variant=variant,
            decision=decision,
            status="MISSING_VALIDATE_LOG",
            message=f"缺少 validate.log：{validate_log_path}",
            reference_sol_vs_base=reference_sol_vs_base,
            evidence_sol_vs_base=evidence_sol_vs_base,
            evidence_path=str(decision_path),
        )

    correctness, log_speedup = parse_validate_log(validate_log_path)
    if not float_equal(reference_sol_vs_base, log_speedup):
        return EntryAudit(
            family=family,
            variant=variant,
            decision=decision,
            status="LOG_SPEEDUP_MISMATCH",
            message="validate.log 解析出的 sol/base 与 reference 不一致",
            reference_sol_vs_base=reference_sol_vs_base,
            evidence_sol_vs_base=log_speedup,
            evidence_path=str(validate_log_path),
        )

    expected_pass = row.get("passed_workloads")
    expected_total = row.get("total_workloads")
    if correctness is not None and (expected_pass, expected_total) != correctness:
        return EntryAudit(
            family=family,
            variant=variant,
            decision=decision,
            status="CORRECTNESS_MISMATCH",
            message="validate.log 正确性汇总与 reference 不一致",
            reference_sol_vs_base=reference_sol_vs_base,
            evidence_sol_vs_base=log_speedup,
            evidence_path=str(validate_log_path),
        )

    return EntryAudit(
        family=family,
        variant=variant,
        decision=decision,
        status="OK",
        message="reference / decision / validate.log 三者一致",
        reference_sol_vs_base=reference_sol_vs_base,
        evidence_sol_vs_base=log_speedup,
        evidence_path=str(validate_log_path),
    )


def audit_baseline_latest(family: str, baseline_path: Path, rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    latest = payload.get("latest_result") or {}
    latest_variant = str(latest.get("candidate_variant", "")).strip()
    if latest_variant:
        match = next((row for row in rows if str(row.get("id", "")).strip() == latest_variant), None)
        if match is None:
            problems.append("baseline.latest_result 指向的 variant 不在 solutions.jsonl 中")
        else:
            latest_speedup = latest.get("avg_sol_vs_base")
            ref_speedup = match.get("sol_vs_base")
            if isinstance(latest_speedup, str) and latest_speedup:
                latest_speedup = float(latest_speedup)
            if isinstance(ref_speedup, str) and ref_speedup:
                ref_speedup = float(ref_speedup)
            if not float_equal(latest_speedup, ref_speedup):
                problems.append("baseline.latest_result.avg_sol_vs_base 与 solutions.jsonl 不一致")
    return problems


def main() -> int:
    report: dict[str, Any] = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "families": {},
        "summary": {
            "ok": 0,
            "issues": 0,
        },
    }

    for family_dir in sorted(p for p in REFERENCE_ROOT.iterdir() if p.is_dir()):
        family = family_dir.name
        solutions_path = family_dir / "solutions.jsonl"
        baseline_path = family_dir / "baseline.json"
        if not solutions_path.exists() or not baseline_path.exists():
            continue

        rows = load_jsonl(solutions_path)
        audited_entries: list[dict[str, Any]] = []
        for row in rows:
            if not row.get("decision"):
                continue
            result = audit_entry(family, row)
            audited_entries.append(result.__dict__)
            if result.status == "OK":
                report["summary"]["ok"] += 1
            elif result.status != "SKIP":
                report["summary"]["issues"] += 1

        baseline_problems = audit_baseline_latest(family, baseline_path, rows)
        if baseline_problems:
            report["summary"]["issues"] += len(baseline_problems)

        report["families"][family] = {
            "entries": audited_entries,
            "baseline_latest_problems": baseline_problems,
        }

    out_path = REPO_ROOT / "rounds" / "archive_speedup_audit.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
