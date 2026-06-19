#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为单个 definition 构建隔离的 dataset-mini。

目标：
1. 只复制当前 round 需要的 definition / workload / solution；
2. 复制 workload 依赖的 blob，避免 TraceSet.from_path() 扫全量数据集时被无关 schema 卡住；
3. 保持目录结构与 flashinfer-trace 一致，便于 fib_inproc_validate.py / ncu_driver.py 直接复用。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, help="原始 flashinfer-trace 根目录")
    parser.add_argument("--target-dataset", required=True, help="输出 dataset-mini 根目录")
    parser.add_argument("--definition", required=True, help="目标 definition 名")
    parser.add_argument(
        "--solution",
        action="append",
        default=[],
        help="需要复制进 dataset-mini 的 solution 名，可重复传入",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="如果目标目录已存在，则先删除后重建",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def find_unique_file(root: Path, pattern: str, predicate) -> Path:
    candidates = [p for p in root.rglob(pattern) if predicate(p)]
    if not candidates:
        raise FileNotFoundError(f"找不到匹配文件: pattern={pattern}")
    if len(candidates) > 1:
        raise RuntimeError(f"匹配文件不唯一: {[str(p) for p in candidates]}")
    return candidates[0]


def find_definition_file(source_root: Path, definition: str) -> Path:
    return find_unique_file(
        source_root / "definitions",
        f"{definition}.json",
        lambda p: p.stem == definition,
    )


def find_workload_file(source_root: Path, definition: str) -> Path:
    return find_unique_file(
        source_root / "workloads",
        f"{definition}.jsonl",
        lambda p: p.stem == definition,
    )


def find_solution_files(source_root: Path, solution_names: Iterable[str]) -> list[Path]:
    wanted = set(solution_names)
    found: dict[str, Path] = {}
    for path in sorted((source_root / "solutions").rglob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        name = payload.get("name")
        if name in wanted and name not in found:
            found[name] = path
    missing = wanted.difference(found)
    if missing:
        raise FileNotFoundError(f"找不到 solution 文件: {sorted(missing)}")
    return [found[name] for name in solution_names]


def collect_blob_paths(source_root: Path, workload_file: Path) -> list[Path]:
    blob_paths: list[Path] = []
    with workload_file.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            payload = json.loads(raw)
            inputs = payload.get("workload", {}).get("inputs", {})
            for spec in inputs.values():
                if spec.get("type") != "safetensors":
                    continue
                rel = spec.get("path", "")
                if not rel.startswith("./blob/"):
                    continue
                blob_paths.append(source_root / rel[2:])
    return sorted(set(blob_paths))


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_dataset).resolve()
    target_root = Path(args.target_dataset).resolve()

    if args.force_clean and target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    definition_file = find_definition_file(source_root, args.definition)
    workload_file = find_workload_file(source_root, args.definition)
    solution_files = find_solution_files(source_root, args.solution)
    blob_files = collect_blob_paths(source_root, workload_file)

    copy_file(definition_file, target_root / definition_file.relative_to(source_root))
    copy_file(workload_file, target_root / workload_file.relative_to(source_root))
    for solution_file in solution_files:
        copy_file(solution_file, target_root / solution_file.relative_to(source_root))
    for blob_file in blob_files:
        copy_file(blob_file, target_root / blob_file.relative_to(source_root))

    print(f"dataset-mini 已生成: {target_root}")
    print(f"  definition : {definition_file.relative_to(source_root)}")
    print(f"  workload   : {workload_file.relative_to(source_root)}")
    print(f"  solutions  : {len(solution_files)}")
    print(f"  blob files : {len(blob_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
