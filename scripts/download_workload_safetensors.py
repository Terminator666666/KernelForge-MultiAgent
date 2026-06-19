#!/usr/bin/env python3
"""
从 hf-mirror 的 flashinfer-trace tree 页面递归抓取真实 safetensors 文件。

设计目标：
1. 不再依赖本地 workloads/*.jsonl，直接从 tree 页面递归发现文件。
2. 每次运行前默认清理目标目录、断点续传残留和失败日志，保证本轮结果干净。
3. 默认支持按 definition / family / workload 子目录限定抓取范围。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_policy import canonicalize_family, get_family_policy, list_primary_families


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
WORKLOADS_ROOT_REL_PATH = "blob/workloads"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="flashinfer-trace 数据集根目录，例如 /mnt/d/Agent/flashinfer-trace",
    )
    parser.add_argument(
        "--repo-id",
        default="flashinfer-ai/flashinfer-trace",
        help="Hugging Face dataset repo id，默认 flashinfer-ai/flashinfer-trace",
    )
    parser.add_argument(
        "--base-url",
        default="https://hf-mirror.com",
        help="下载站点根地址，默认 https://hf-mirror.com",
    )
    parser.add_argument(
        "--definition",
        action="append",
        default=[],
        help="只处理指定 definition，可重复传入",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="按主线 family 选择 definition，可重复传入，例如 dsa_sparse_attention",
    )
    parser.add_argument(
        "--mainline-families",
        action="store_true",
        help="自动下载当前 README / CLAUDE 约束下的 3 个主线 family（每个 family 只取 default_definition）",
    )
    parser.add_argument(
        "--all-family-targets",
        action="store_true",
        help="按 family 下载其全部 supported_targets；默认只下载各 family 的 default_definition",
    )
    parser.add_argument(
        "--workload-subdir",
        action="append",
        default=[],
        help="只递归抓取指定 workloads 子目录，例如 dsa_paged，可重复传入",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="兼容保留参数；递归抓取模式下默认会重建目标目录并重新下载",
    )
    parser.add_argument(
        "--clean-target-dir",
        action="store_true",
        help="兼容保留参数；当前脚本默认就会在运行前清理目标目录",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="跳过默认清理逻辑，仅在你明确要复用已有下载结果时使用",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要下载的文件，不实际下载",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多处理多少个文件，便于先小范围探测网络",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="单次下载超时秒数，默认 20",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="失败后的重试次数，默认 1",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=1.0,
        help="重试等待秒数，默认 1",
    )
    return parser.parse_args()


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        prefix = handle.read(len(LFS_POINTER_PREFIX))
    return prefix == LFS_POINTER_PREFIX


class HrefCollector(HTMLParser):
    """提取页面中的 href 链接。"""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.hrefs.append(value)
                break


def definitions_from_families(
    families: list[str],
    include_mainline: bool,
    include_all_targets: bool,
) -> list[str]:
    requested_families = list(families)
    if include_mainline:
        requested_families.extend(list_primary_families())

    ordered_families: list[str] = []
    seen_families: set[str] = set()
    for family in requested_families:
        canonical = canonicalize_family(family)
        if canonical is None:
            raise KeyError(f"unsupported family: {family}")
        if canonical not in seen_families:
            seen_families.add(canonical)
            ordered_families.append(canonical)

    definitions: list[str] = []
    seen_definitions: set[str] = set()
    for family in ordered_families:
        policy = get_family_policy(family)
        if include_all_targets:
            candidate_definitions = [
                str(target.get("definition", "")).strip()
                for target in policy.get("targets", [])
            ]
        else:
            candidate_definitions = [str(policy.get("default_definition", "")).strip()]

        for definition in candidate_definitions:
            if definition and definition not in seen_definitions:
                seen_definitions.add(definition)
                definitions.append(definition)
    return definitions


def build_tree_url(base_url: str, repo_id: str, rel_path: str) -> str:
    repo_path = urllib.parse.quote(repo_id, safe="/")
    quoted_rel_path = "/".join(urllib.parse.quote(part) for part in rel_path.split("/"))
    return f"{base_url.rstrip('/')}/datasets/{repo_path}/tree/main/{quoted_rel_path}"


def definition_to_workload_subdir(definition: str) -> str:
    mapping = (
        ("dsa_sparse_attention_", "dsa_paged"),
        ("dsa_topk_indexer_", "dsa_paged"),
        ("gdn_prefill_", "gdn"),
        ("gdn_decode_", "gdn"),
        ("gqa_paged_", "gqa_paged"),
        ("mla_paged_", "mla_paged"),
    )
    for prefix, subdir in mapping:
        if definition.startswith(prefix):
            return subdir
    raise KeyError(f"无法从 definition 推断 workload 子目录: {definition}")


def build_start_tree_paths(
    wanted_definitions: set[str],
    workload_subdirs: list[str],
) -> list[str]:
    start_paths: list[str] = []
    seen: set[str] = set()

    for subdir in workload_subdirs:
        rel_path = f"{WORKLOADS_ROOT_REL_PATH}/{subdir.strip('/')}"
        if rel_path not in seen:
            seen.add(rel_path)
            start_paths.append(rel_path)

    for definition in sorted(wanted_definitions):
        subdir = definition_to_workload_subdir(definition)
        rel_path = f"{WORKLOADS_ROOT_REL_PATH}/{subdir}/{definition}"
        if rel_path not in seen:
            seen.add(rel_path)
            start_paths.append(rel_path)

    if not start_paths:
        start_paths.append(WORKLOADS_ROOT_REL_PATH)
    return start_paths


def should_keep_rel_path(rel_path: str, allowed_roots: list[str]) -> bool:
    normalized = rel_path.strip("/")
    for root in allowed_roots:
        root = root.strip("/")
        if normalized == root or normalized.startswith(root + "/"):
            return True
    return False


def fetch_text(url: str, timeout: float, retries: int, retry_wait: float) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "KernelForge-MultiAgent/flashinfer-trace-downloader"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "ignore")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(retry_wait)
            else:
                break
    assert last_error is not None
    raise last_error


def collect_page_hrefs(html_text: str) -> list[str]:
    parser = HrefCollector()
    parser.feed(html_text)
    return parser.hrefs


def extract_tree_rel_path(repo_id: str, href: str) -> str | None:
    tree_prefix = f"/datasets/{repo_id}/tree/main/"
    parsed = urllib.parse.urlsplit(urllib.parse.urljoin("https://local.invalid", href))
    if not parsed.path.startswith(tree_prefix):
        return None
    return urllib.parse.unquote(parsed.path[len(tree_prefix) :]).strip("/")


def extract_resolve_job(repo_id: str, href: str) -> tuple[str, str] | None:
    resolve_prefix = f"/datasets/{repo_id}/resolve/main/"
    absolute = urllib.parse.urljoin("https://local.invalid", href)
    parsed = urllib.parse.urlsplit(absolute)
    if not parsed.path.startswith(resolve_prefix):
        return None
    rel_path = urllib.parse.unquote(parsed.path[len(resolve_prefix) :]).strip("/")
    if not rel_path.endswith(".safetensors"):
        return None
    return rel_path, parsed.query


def definition_from_rel_path(rel_path: str) -> str:
    path = Path(rel_path)
    return path.parent.name


def crawl_download_jobs(
    dataset_root: Path,
    base_url: str,
    repo_id: str,
    start_tree_paths: list[str],
    timeout: float,
    retries: int,
    retry_wait: float,
    overwrite_existing: bool,
) -> list[tuple[Path, str, str, str]]:
    allowed_roots = [path.strip("/") for path in start_tree_paths]
    pending = list(start_tree_paths)
    visited_pages: set[str] = set()
    jobs: list[tuple[Path, str, str, str]] = []
    seen_files: set[str] = set()

    while pending:
        current_rel_path = pending.pop(0).strip("/")
        if current_rel_path in visited_pages:
            continue
        visited_pages.add(current_rel_path)

        page_url = build_tree_url(base_url, repo_id, current_rel_path)
        html_text = fetch_text(
            page_url,
            timeout=timeout,
            retries=retries,
            retry_wait=retry_wait,
        )
        hrefs = collect_page_hrefs(html_text)
        for href in hrefs:
            tree_rel_path = extract_tree_rel_path(repo_id, href)
            if tree_rel_path and should_keep_rel_path(tree_rel_path, allowed_roots):
                if tree_rel_path not in visited_pages:
                    pending.append(tree_rel_path)
                continue

            resolve_job = extract_resolve_job(repo_id, href)
            if not resolve_job:
                continue
            rel_path, query = resolve_job
            if not should_keep_rel_path(rel_path, allowed_roots):
                continue
            if rel_path in seen_files:
                continue
            local_path = dataset_root / rel_path
            if (
                not overwrite_existing
                and local_path.is_file()
                and not is_lfs_pointer(local_path)
            ):
                seen_files.add(rel_path)
                continue
            seen_files.add(rel_path)
            url = urllib.parse.urljoin(base_url.rstrip("/") + "/", href)
            if query and "?" not in url:
                url = f"{url}?{query}"
            jobs.append(
                (
                    local_path,
                    rel_path,
                    definition_from_rel_path(rel_path),
                    url,
                )
            )

    jobs.sort(key=lambda item: item[1])
    return jobs


def clean_target_dirs(dataset_root: Path, start_tree_paths: list[str]) -> None:
    for rel_path in start_tree_paths:
        target_dir = dataset_root / rel_path
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    failure_log = dataset_root / "download_failures.jsonl"
    if failure_log.exists():
        failure_log.unlink()


def download_one(
    url: str,
    target_path: Path,
    timeout: float,
    retries: int,
    retry_wait: float,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
            headers = {
                "User-Agent": "KernelForge-MultiAgent/flashinfer-trace-downloader",
            }
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                if resume_from > 0 and status == 200:
                    tmp_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        "远端未返回 Range 支持，已放弃断点续传并准备重试"
                    )

                mode = "ab" if resume_from > 0 and status == 206 else "wb"
                with tmp_path.open(mode) as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)

            if is_lfs_pointer(tmp_path):
                raise RuntimeError(f"下载结果仍然是 LFS pointer: {url}")

            tmp_path.replace(target_path)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(retry_wait)
            else:
                break

    assert last_error is not None
    raise last_error


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    selected_definitions = list(args.definition)
    if args.family or args.mainline_families:
        selected_definitions.extend(
            definitions_from_families(
                args.family,
                args.mainline_families,
                args.all_family_targets,
            )
        )
    wanted_definitions = set(selected_definitions)

    if wanted_definitions:
        print("目标 definitions:")
        for definition in sorted(wanted_definitions):
            print(f"  - {definition}")

    start_tree_paths = build_start_tree_paths(wanted_definitions, args.workload_subdir)
    print("递归抓取起点:")
    for rel_path in start_tree_paths:
        print(f"  - {rel_path}")

    should_clean = not args.skip_clean
    if should_clean and not args.dry_run:
        print("正在清理上一次下载残留...")
        clean_target_dirs(dataset_root, start_tree_paths)

    jobs = crawl_download_jobs(
        dataset_root=dataset_root,
        base_url=args.base_url,
        repo_id=args.repo_id,
        start_tree_paths=start_tree_paths,
        timeout=args.timeout,
        retries=args.retries,
        retry_wait=args.retry_wait,
        overwrite_existing=should_clean or args.force,
    )

    if args.max_files is not None:
        jobs = jobs[: max(0, args.max_files)]

    if not jobs:
        print("没有需要下载的 safetensors 文件。")
        return 0

    print(f"待处理文件数: {len(jobs)}")
    success_count = 0
    failure_count = 0
    failure_log = dataset_root / "download_failures.jsonl"

    for index, (local_path, rel_path, definition, url) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {definition}")
        print(f"  本地路径: {local_path}")
        print(f"  下载链接: {url}")
        if args.dry_run:
            continue
        try:
            download_one(
                url=url,
                target_path=local_path,
                timeout=args.timeout,
                retries=args.retries,
                retry_wait=args.retry_wait,
            )
            success_count += 1
            print("  结果: 下载成功")
        except urllib.error.HTTPError as exc:
            failure_count += 1
            with failure_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "definition": definition,
                            "path": rel_path,
                            "url": url,
                            "error": f"HTTP {exc.code}: {exc.reason}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"  结果: HTTP 错误 {exc.code}: {exc.reason}", file=sys.stderr)
            continue
        except Exception as exc:  # noqa: BLE001
            failure_count += 1
            with failure_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "definition": definition,
                            "path": rel_path,
                            "url": url,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"  结果: 下载失败: {exc}", file=sys.stderr)
            continue

    if args.dry_run:
        print("dry-run 完成，未实际下载。")
    else:
        print(f"下载完成: {success_count}/{len(jobs)}")
        if failure_count:
            print(f"失败任务: {failure_count}，详见 {failure_log}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
