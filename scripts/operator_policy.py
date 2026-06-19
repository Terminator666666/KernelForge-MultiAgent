#!/usr/bin/env python3
"""主线算子策略加载与校验工具。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "configs" / "operator_families.json"


def load_policy() -> Dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def canonicalize_family(name: str, policy: Optional[Dict[str, Any]] = None) -> Optional[str]:
    data = policy or load_policy()
    normalized = normalize_name(name)
    aliases = data.get("aliases", {})
    if normalized in aliases:
        return aliases[normalized]
    if normalized in data.get("families", {}):
        return normalized
    return None


def get_family_policy(name: str, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = policy or load_policy()
    canonical = canonicalize_family(name, data)
    if canonical is None:
        raise KeyError(f"unsupported family: {name}")
    return data["families"][canonical]


def list_primary_families(policy: Optional[Dict[str, Any]] = None) -> List[str]:
    data = policy or load_policy()
    return [
        family
        for family, spec in data.get("families", {}).items()
        if spec.get("tier") == "primary"
    ]


def build_baseline_template(name: str, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = policy or load_policy()
    canonical = canonicalize_family(name, data)
    if canonical is None:
        raise KeyError(f"unsupported family: {name}")

    rules = data["repository_rules"]
    spec = data["families"][canonical]
    template: Dict[str, Any] = {
        "family": canonical,
        "family_tier": spec["tier"],
        "dataset_path": rules["default_dataset_path"],
        "definition": spec.get("default_definition", ""),
        "baseline_solution": spec.get("default_baseline_solution", ""),
        "baseline_source_kind": spec["baseline_source_kind"],
        "comparison_denominator": spec["compare_against"],
        "accept_threshold": rules["default_accept_threshold"],
        "derive_from_official_baseline": bool(rules["derive_from_official_baseline"]),
        "op_type": spec.get("default_op_type", canonical),
        "baseline_dataset_group": spec.get("default_baseline_dataset_group", spec.get("default_op_type", canonical)),
        "solution_prefix": spec.get("default_solution_prefix", ""),
        "required_ncu_binary": rules["required_ncu_binary"],
        "required_ncu_version": rules["required_ncu_version"],
        "fast_iteration_workload_policy": rules.get("fast_iteration_workload_policy", "low_mid_high_3"),
        "fast_iteration_workload_count": rules.get("fast_iteration_workload_count", 3),
        "allowed_definition_prefixes": spec.get("allowed_definition_prefixes", []),
        "allowed_op_types": spec.get("allowed_op_types", []),
        "supported_targets": spec.get("targets", []),
        "status": "pending_baseline_capture"
    }
    notes = spec.get("notes")
    if notes:
        template["notes"] = notes
    return template


def validate_baseline_config(name: str, cfg: Dict[str, Any], policy: Optional[Dict[str, Any]] = None) -> List[str]:
    data = policy or load_policy()
    canonical = canonicalize_family(name, data)
    if canonical is None:
        return [f"不支持的 family: {name}"]

    rules = data["repository_rules"]
    spec = data["families"][canonical]
    errors: List[str] = []

    if spec.get("tier") != "primary":
        errors.append(f"{canonical} 不是当前主线三类算子之一")

    if str(cfg.get("family", canonical)) != canonical:
        errors.append(f"baseline.json 的 family 必须为 {canonical}")

    if cfg.get("derive_from_official_baseline") is not True:
        errors.append("derive_from_official_baseline 必须为 true")

    if cfg.get("comparison_denominator") != spec.get("compare_against"):
        errors.append(
            "comparison_denominator 必须与 family policy 一致："
            f"{spec.get('compare_against')}"
        )

    if cfg.get("baseline_source_kind") != spec.get("baseline_source_kind"):
        errors.append(
            "baseline_source_kind 必须与 family policy 一致："
            f"{spec.get('baseline_source_kind')}"
        )

    definition = str(cfg.get("definition", "")).strip()
    prefixes = spec.get("allowed_definition_prefixes", [])
    if not definition:
        errors.append("definition 不能为空")
    elif prefixes and not any(definition.startswith(prefix) for prefix in prefixes):
        errors.append(
            "definition 不符合 family policy 前缀约束："
            + ", ".join(prefixes)
        )

    op_type = str(cfg.get("op_type", "")).strip()
    allowed_op_types = spec.get("allowed_op_types", [])
    if not op_type:
        errors.append("op_type 不能为空")
    elif allowed_op_types and op_type not in allowed_op_types:
        errors.append(
            "op_type 不符合 family policy 约束："
            + ", ".join(allowed_op_types)
        )

    baseline_solution = str(cfg.get("baseline_solution", "")).strip()
    if not baseline_solution:
        errors.append("baseline_solution 不能为空；所有成绩必须对官方 baseline / expert baseline")

    baseline_dataset_group = str(
        cfg.get("baseline_dataset_group", spec.get("default_baseline_dataset_group", ""))
    ).strip()
    if not baseline_dataset_group:
        errors.append("baseline_dataset_group 不能为空；必须能定位到 flashinfer-trace/solutions/baseline/<group>/...")

    required_ncu_binary = str(cfg.get("required_ncu_binary", rules["required_ncu_binary"])).strip()
    required_ncu_version = str(cfg.get("required_ncu_version", rules["required_ncu_version"])).strip()
    fast_iteration_workload_policy = str(
        cfg.get("fast_iteration_workload_policy", rules.get("fast_iteration_workload_policy", "low_mid_high_3"))
    ).strip()
    fast_iteration_workload_count = int(
        cfg.get("fast_iteration_workload_count", rules.get("fast_iteration_workload_count", 3))
    )
    if required_ncu_binary != rules["required_ncu_binary"]:
        errors.append(f"required_ncu_binary 必须固定为 {rules['required_ncu_binary']}")
    if required_ncu_version != rules["required_ncu_version"]:
        errors.append(f"required_ncu_version 必须固定为 {rules['required_ncu_version']}")
    if fast_iteration_workload_policy != rules.get("fast_iteration_workload_policy", "low_mid_high_3"):
        errors.append(
            "fast_iteration_workload_policy 必须与仓库规则一致："
            f"{rules.get('fast_iteration_workload_policy', 'low_mid_high_3')}"
        )
    if fast_iteration_workload_count != int(rules.get("fast_iteration_workload_count", 3)):
        errors.append(
            "fast_iteration_workload_count 必须与仓库规则一致："
            f"{int(rules.get('fast_iteration_workload_count', 3))}"
        )

    return errors


def _cmd_canonical(args: argparse.Namespace) -> int:
    canonical = canonicalize_family(args.family)
    if canonical is None:
        print(f"unsupported family: {args.family}", file=sys.stderr)
        return 1
    print(canonical)
    return 0


def _cmd_template(args: argparse.Namespace) -> int:
    try:
        template = build_baseline_template(args.family)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(template, ensure_ascii=False, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"config not found: {args.config}", file=sys.stderr)
        return 1
    errors = validate_baseline_config(args.family, payload)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonical_parser = subparsers.add_parser("canonical", help="输出 canonical family 名")
    canonical_parser.add_argument("family")
    canonical_parser.set_defaults(func=_cmd_canonical)

    template_parser = subparsers.add_parser("template", help="输出 baseline.json 模板")
    template_parser.add_argument("family")
    template_parser.set_defaults(func=_cmd_template)

    validate_parser = subparsers.add_parser("validate", help="校验 baseline.json 是否符合 policy")
    validate_parser.add_argument("family")
    validate_parser.add_argument("config")
    validate_parser.set_defaults(func=_cmd_validate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
