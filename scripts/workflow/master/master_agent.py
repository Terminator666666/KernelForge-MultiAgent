#!/usr/bin/env python3
"""Master agent orchestration for KernelForge-MultiAgent.

The master agent owns campaign state, child workspace creation, prompt routing,
and archive bookkeeping. It keeps generated runtime state under
``scripts/workflow/trajectory`` so public source files stay clean.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CampaignConfig:
    """Configuration for one optimization campaign."""

    family: str
    gpu: str
    backend: str
    mode: int
    max_rounds: int
    dataset_path: str


@dataclass
class RoundResult:
    """Summary of one optimization round."""

    round_id: int
    variant_name: str
    parent_name: Optional[str]
    speedup: float
    passed_tests: bool
    kernel_changed: bool
    archived: bool
    phase2_proposals: Optional[List[Dict[str, Any]]] = None
    failure_reason: Optional[str] = None


@dataclass
class SubAgentTask:
    """Task assigned by the master to a sub-agent or external backend."""

    task_id: str
    agent_type: str
    phase: int
    input_data: Dict[str, Any]
    timeout: int = 18_000


class MasterAgent:
    """Coordinate multi-round kernel optimization campaigns."""

    def __init__(self, workflow_root: Path):
        self.workflow_root = Path(workflow_root).resolve()
        self.repo_root = self._find_repo_root(self.workflow_root)
        self.state_dir = self.workflow_root / "trajectory"
        self.reference_dir = self.state_dir / "reference"
        self.ledger_path = self.state_dir / "harness-ledger.md"

        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ledger_path.exists():
            self._init_ledger()

    def init_campaign(
        self,
        family: str,
        gpu: str = "B200",
        backend: str = "local",
        mode: int = 2,
    ) -> Dict[str, Any]:
        """Create or resume a campaign for an operator family."""

        family_dir = self.reference_dir / family
        baseline_path = family_dir / "baseline.json"

        if baseline_path.exists():
            baseline = self._read_json(baseline_path)
            environment = baseline.get("environment", {})
            if environment.get("gpu") != gpu or environment.get("backend") != backend:
                raise ValueError(
                    "Campaign environment mismatch: "
                    f"existing gpu={environment.get('gpu')} backend={environment.get('backend')}; "
                    f"requested gpu={gpu} backend={backend}"
                )
            if environment.get("mode") not in (None, mode):
                raise ValueError(
                    "Campaign mode mismatch: "
                    f"existing mode={environment.get('mode')}; requested mode={mode}"
                )
            return baseline

        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "variants").mkdir(exist_ok=True)
        (family_dir / "_failed").mkdir(exist_ok=True)

        baseline = {
            "source": "bootstrap",
            "environment": {
                "family": family,
                "gpu": gpu,
                "backend": backend,
                "mode": mode,
            },
            "workloads": {},
            "created_at": datetime.now().isoformat(),
        }
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        (family_dir / "README.md").write_text(
            f"# {family} Optimization History\n\nCurrent anchor: baseline (1.0x)\n",
            encoding="utf-8",
        )
        (family_dir / "TRAPS.md").write_text(
            f"# {family} Silent Failure Patterns\n\n",
            encoding="utf-8",
        )

        self._append_ledger(
            f"init-campaign family={family} gpu={gpu} backend={backend} mode={mode}: accepted"
        )
        return baseline

    def read_campaign_mode(self, family: str) -> int:
        baseline_path = self.reference_dir / family / "baseline.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Campaign does not exist: {family}")
        baseline = self._read_json(baseline_path)
        return int(baseline.get("environment", {}).get("mode", 2))

    def spawn_child(
        self,
        operator: str,
        parent_kernel_path: Optional[Path],
        name_label: str,
        family: str,
    ) -> Path:
        """Create an isolated child workspace for one optimization round."""

        baseline_path = self.reference_dir / family / "baseline.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Campaign is not initialized: {family}")

        baseline = self._read_json(baseline_path)
        environment = baseline.get("environment", {})
        child_dir = self.workflow_root / f"kfma-run-{name_label}"

        if child_dir.exists():
            raise ValueError(f"Child workspace already exists: {child_dir}")

        for relative in ("solution", "trajectory", ".ako", ".claude/skills"):
            (child_dir / relative).mkdir(parents=True, exist_ok=True)

        config = {
            "operator": operator,
            "family": family,
            "gpu": environment.get("gpu", "B200"),
            "backend": environment.get("backend", "local"),
            "parent": str(parent_kernel_path) if parent_kernel_path else None,
            "round_label": name_label,
        }
        self._write_toml(child_dir / "config.toml", config)

        kernel_dest = child_dir / "solution" / "kernel.py"
        if parent_kernel_path:
            shutil.copy2(parent_kernel_path, kernel_dest)
        else:
            kernel_dest.write_text(
                f"# Reference kernel placeholder\n# Operator: {operator}\n",
                encoding="utf-8",
            )

        (child_dir / "ITERATIONS.md").write_text(
            "# Iteration History\n\n"
            "| Iter | Strategy | Speedup | Passed tests | Notes |\n"
            "|------|----------|---------|--------------|-------|\n",
            encoding="utf-8",
        )

        self._copy_skills(child_dir / ".claude" / "skills")
        self._append_ledger(f"spawn-child family={family} round={name_label}: accepted")
        return child_dir

    def create_phase1_task(self, child_dir: Path, prompt: str, timeout: int = 18_000) -> Dict[str, Any]:
        """Persist a phase-1 task for a child agent or manual runner."""

        session_id = str(uuid.uuid4())
        ako_dir = child_dir / ".ako"
        ako_dir.mkdir(parents=True, exist_ok=True)

        (ako_dir / "session-id.txt").write_text(session_id, encoding="utf-8")
        (ako_dir / "phase1-prompt.txt").write_text(prompt, encoding="utf-8")

        task = SubAgentTask(
            task_id=session_id,
            agent_type="optimizer",
            phase=1,
            input_data={"prompt_path": str(ako_dir / "phase1-prompt.txt")},
            timeout=timeout,
        )
        (ako_dir / "phase1-task.json").write_text(
            json.dumps(asdict(task), indent=2),
            encoding="utf-8",
        )

        return {
            "session_id": session_id,
            "task_path": str(ako_dir / "phase1-task.json"),
            "prompt_path": str(ako_dir / "phase1-prompt.txt"),
            "kernel_path": str(child_dir / "solution" / "kernel.py"),
        }

    def create_retrospective_task(self, child_dir: Path, session_id: str) -> Dict[str, Any]:
        """Persist a phase-2 retrospective task for the same child workspace."""

        prompt = self._load_prompt_template("phase2_retrospective.md")
        if prompt is None:
            prompt = (
                "# Phase 2 Retrospective\n\n"
                "Review the previous attempt, identify unsupported claims, and propose a "
                "measurable next optimization step.\n"
            )

        ako_dir = child_dir / ".ako"
        prompt_path = ako_dir / "phase2-prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        task = SubAgentTask(
            task_id=session_id,
            agent_type="reviewer",
            phase=2,
            input_data={"prompt_path": str(prompt_path), "resume_session_id": session_id},
            timeout=3_600,
        )
        task_path = ako_dir / "phase2-task.json"
        task_path.write_text(json.dumps(asdict(task), indent=2), encoding="utf-8")

        return {
            "session_id": session_id,
            "task_path": str(task_path),
            "prompt_path": str(prompt_path),
        }

    def archive_variant(
        self,
        family: str,
        name: str,
        kernel_path: Path,
        parent: Optional[str],
        spawn_meta: Dict[str, Any],
        header_text: Optional[str],
        config_path: Optional[Path] = None,
        result_json_path: Optional[Path] = None,
        variance_json_path: Optional[Path] = None,
    ) -> Path:
        """Archive a validated kernel variant with provenance."""

        variant_dir = self.reference_dir / family / "variants" / name
        variant_dir.mkdir(parents=True, exist_ok=True)

        dest_kernel = variant_dir / "kernel.py"
        if header_text:
            content = Path(kernel_path).read_text(encoding="utf-8")
            dest_kernel.write_text(f"{header_text}\n\n{content}", encoding="utf-8")
        else:
            shutil.copy2(kernel_path, dest_kernel)

        for source, target_name in (
            (config_path, "config.toml"),
            (result_json_path, "results.json"),
            (variance_json_path, "variance.json"),
        ):
            if source and Path(source).exists():
                shutil.copy2(source, variant_dir / target_name)

        (variant_dir / "parent.txt").write_text(parent or "baseline", encoding="utf-8")
        (variant_dir / "spawn.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "parent": parent,
                    "spawn_meta": spawn_meta,
                    "archived_at": datetime.now().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self._append_ledger(f"archive-variant family={family} variant={name}: accepted")
        return variant_dir

    def archive_failed(
        self,
        child_dir: Path,
        round_id: str,
        family: str,
        exit_kind: str,
        last_action: str,
        last_stderr_tail: str,
    ) -> Path:
        """Archive failure evidence for a round that did not produce a valid variant."""

        failed_dir = self.reference_dir / family / "_failed" / round_id
        failed_dir.mkdir(parents=True, exist_ok=True)

        for source_name in (".ako/phase1-transcript.jsonl", "ITERATIONS.md"):
            source = child_dir / source_name
            if source.exists():
                shutil.copy2(source, failed_dir / Path(source_name).name)

        (failed_dir / "summary.md").write_text(
            f"# {round_id} Failure Summary\n\n"
            f"**Exit kind:** {exit_kind}\n\n"
            f"**Last action:** {last_action}\n\n"
            f"**stderr tail:**\n\n```text\n{last_stderr_tail}\n```\n",
            encoding="utf-8",
        )

        self._append_ledger(f"archive-failed family={family} round={round_id}: rejected")
        return failed_dir

    def run_campaign(self, config: CampaignConfig) -> Dict[str, Any]:
        """Prepare campaign state and create round task descriptors."""

        self.init_campaign(
            family=config.family,
            gpu=config.gpu,
            backend=config.backend,
            mode=config.mode,
        )

        created_rounds: List[Dict[str, Any]] = []
        for round_num in range(1, config.max_rounds + 1):
            round_label = f"{config.family}-r{round_num}"
            child_dir = self.spawn_child(
                operator=config.family,
                parent_kernel_path=None,
                name_label=round_label,
                family=config.family,
            )
            task = self.create_phase1_task(
                child_dir,
                self._default_phase1_prompt(config.family, config.dataset_path),
            )
            created_rounds.append({"round": round_num, "child_dir": str(child_dir), "task": task})
            break

        return {
            "family": config.family,
            "created_rounds": created_rounds,
            "state_dir": str(self.state_dir),
        }

    def _init_ledger(self) -> None:
        self.ledger_path.write_text(
            "# KernelForge-MultiAgent Audit Ledger\n\n"
            "Format: YYYY-MM-DD action: accepted|rejected reason\n\n",
            encoding="utf-8",
        )

    def _append_ledger(self, entry: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {entry}\n")

    def _copy_skills(self, destination: Path) -> None:
        source = self.repo_root / "skills"
        if not source.exists():
            return
        for skill_dir in source.iterdir():
            if skill_dir.is_dir():
                shutil.copytree(skill_dir, destination / skill_dir.name, dirs_exist_ok=True)

    def _load_prompt_template(self, name: str) -> Optional[str]:
        candidates = [
            self.repo_root / "prompts" / name,
            self.workflow_root / "prompts" / name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        return None

    @staticmethod
    def _default_phase1_prompt(family: str, dataset_path: str) -> str:
        return (
            f"# Phase 1 Optimization Task\n\n"
            f"Operator family: {family}\n"
            f"Dataset path: {dataset_path or 'not provided'}\n\n"
            "Produce a kernel change only when correctness and benchmark evidence can be collected.\n"
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_toml(path: Path, values: Dict[str, Any]) -> None:
        lines = []
        for key, value in values.items():
            if value is None:
                continue
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        for root in [start, *start.parents]:
            if (root / "README.md").exists() and (root / "skills").exists():
                return root
        return start


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelForge-MultiAgent master")
    parser.add_argument("--mode", choices=["manual", "closed-loop", "co-evolution"], default="closed-loop")
    parser.add_argument("--family", required=True)
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--dataset", default=os.getenv("AKO_DATASET_PATH", ""))
    args = parser.parse_args()

    mode_map = {"manual": 1, "closed-loop": 2, "co-evolution": 3}
    config = CampaignConfig(
        family=args.family,
        gpu=args.gpu,
        backend=args.backend,
        mode=mode_map[args.mode],
        max_rounds=args.max_rounds,
        dataset_path=args.dataset,
    )

    workflow_root = Path(__file__).resolve().parents[1]
    master = MasterAgent(workflow_root)
    summary = master.run_campaign(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
