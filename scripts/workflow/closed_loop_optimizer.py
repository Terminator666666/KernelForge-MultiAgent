#!/usr/bin/env python3
"""Multi-round closed-loop optimizer driver.

The driver runs ``run_optimization_cycle.py``, reads the generated feedback, and
writes the next-round prompt. It does not pretend to call an agent backend; the
prompt is an explicit artifact for a human or external agent runner.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WORKFLOW_ROOT = Path(__file__).resolve().parent


class ClosedLoopOptimizer:
    """Coordinate repeated benchmark/profile feedback cycles."""

    def __init__(self, kernel_name: str, max_iterations: int = 5, convergence_threshold: float = 0.05):
        self.kernel_name = kernel_name
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.work_dir = WORKFLOW_ROOT
        self.history: List[Dict[str, Any]] = []

    def run_multi_iteration(self) -> Dict[str, Any]:
        for iteration in range(1, self.max_iterations + 1):
            success, data = self.run_optimization_cycle(iteration)
            if not success:
                break

            if data:
                self.history.append(data)
                prompt = self.generate_agent_prompt(iteration + 1, data)
                if prompt:
                    self._write_prompt(iteration + 1, prompt)

            if self._has_converged():
                break

        return self._write_report()

    def run_optimization_cycle(self, iteration: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        command = [
            sys.executable,
            str(self.work_dir / "run_optimization_cycle.py"),
            self.kernel_name,
            str(iteration),
        ]
        result = subprocess.run(command, capture_output=True, text=True, cwd=self.work_dir)

        feedback_file = self.work_dir / f"{self.kernel_name}_feedback_iter{iteration}.json"
        feedback = self._read_json(feedback_file) if feedback_file.exists() else None

        if result.returncode != 0:
            return False, {
                "iteration": iteration,
                "status": "cycle_failed",
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
                "feedback": feedback,
            }

        return True, {
            "iteration": iteration,
            "status": "ok",
            "stdout": result.stdout,
            "feedback": feedback or {},
        }

    def generate_agent_prompt(self, iteration: int, previous_data: Dict[str, Any]) -> str:
        feedback = previous_data.get("feedback", {})
        benchmark_stdout = feedback.get("benchmark_stdout", "").strip()
        profile_note = "NCU report was generated." if feedback.get("profile_generated") else "NCU report was not generated."

        return f"""# Kernel Optimization Prompt

Kernel: {self.kernel_name}
Target iteration: {iteration}
Previous status: {feedback.get("status", previous_data.get("status", "unknown"))}

## Benchmark Output
```text
{benchmark_stdout or "No benchmark output captured."}
```

## Profiling
{profile_note}

## Task
1. Inspect the current optimized source for this kernel.
2. Propose one measurable change tied to the observed bottleneck.
3. Preserve correctness and avoid delegated speedups through banned libraries.
4. Save the next candidate as `{self.kernel_name}_opt.cu` or the operator-level `*_opt.cu`
   file under `scripts/workflow/` before running the next cycle.
"""

    def _write_prompt(self, iteration: int, prompt: str) -> Path:
        prompt_file = self.work_dir / f"{self.kernel_name}_agent_prompt_iter{iteration}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        return prompt_file

    def _has_converged(self) -> bool:
        if len(self.history) < 2:
            return False

        previous = self._avg_time_ms(self.history[-2])
        current = self._avg_time_ms(self.history[-1])
        if previous is None or current is None or previous <= 0.0:
            return False

        improvement = (previous - current) / previous
        return improvement < self.convergence_threshold

    def _write_report(self) -> Dict[str, Any]:
        report = {
            "kernel": self.kernel_name,
            "iterations": len(self.history),
            "history": self.history,
            "converged": self._has_converged(),
        }
        report_file = self.work_dir / f"{self.kernel_name}_multi_iteration_report.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _avg_time_ms(iteration_data: Dict[str, Any]) -> Optional[float]:
        feedback = iteration_data.get("feedback", {})
        stdout = str(feedback.get("benchmark_stdout", ""))
        try:
            parsed = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return None
        value = parsed.get("avg_time_ms")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-round CUDA optimization feedback")
    parser.add_argument("kernel_name", help="Kernel or operator name, such as softmax or matmul")
    parser.add_argument("max_iterations", nargs="?", type=int, default=5)
    args = parser.parse_args()

    optimizer = ClosedLoopOptimizer(args.kernel_name, args.max_iterations)
    report = optimizer.run_multi_iteration()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
