#!/usr/bin/env python3
"""Sub-agent interfaces for the KernelForge multi-agent workflow.

The module defines small, deterministic agent roles. It does not call an LLM by
itself; orchestration code can connect these roles to a real backend while
keeping evidence checks explicit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentResult:
    """Result returned by every agent phase."""

    success: bool
    agent_type: str
    phase: int
    output_data: Dict[str, Any]
    error_message: Optional[str] = None
    artifacts: List[Path] = field(default_factory=list)


class BaseAgent(ABC):
    """Common interface for all workflow agents."""

    def __init__(self, workspace: Path, config: Dict[str, Any]):
        self.workspace = Path(workspace).resolve()
        self.config = config
        self.agent_type = self.__class__.__name__

    @abstractmethod
    def execute_phase1(self, input_data: Dict[str, Any]) -> AgentResult:
        """Exploration and initial proposal phase."""

    @abstractmethod
    def execute_phase2(self, input_data: Dict[str, Any]) -> AgentResult:
        """Retrospective and refinement phase."""

    @abstractmethod
    def execute_phase3(self, input_data: Dict[str, Any]) -> AgentResult:
        """Final validation and packaging phase."""

    def load_skill(self, skill_name: str) -> Optional[str]:
        """Load a skill document from repo-level or child-workspace skill roots."""

        for root in self._candidate_skill_roots():
            skill_path = root / skill_name / "SKILL.md"
            if skill_path.exists():
                return skill_path.read_text(encoding="utf-8")
        return None

    def _candidate_skill_roots(self) -> List[Path]:
        roots: List[Path] = []
        search_roots = [self.workspace, *self.workspace.parents]

        for root in search_roots:
            for candidate in (root / "skills", root / ".claude" / "skills"):
                if candidate not in roots:
                    roots.append(candidate)

        return roots


class OptimizerAgent(BaseAgent):
    """Generate optimization candidates from measured bottlenecks."""

    def __init__(self, workspace: Path, config: Dict[str, Any]):
        super().__init__(workspace, config)
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> Dict[str, Dict[str, str]]:
        self.load_skill("strategy-library")
        return {
            "shared_memory_tiling": {
                "name": "Shared-memory tiling",
                "target": "data reuse and DRAM traffic",
            },
            "vectorized_memory": {
                "name": "Vectorized memory access",
                "target": "global memory transaction efficiency",
            },
            "warp_reduction": {
                "name": "Warp-level reduction",
                "target": "reduction latency and synchronization",
            },
            "occupancy_tuning": {
                "name": "Occupancy tuning",
                "target": "launch geometry and register pressure",
            },
            "kernel_fusion": {
                "name": "Kernel fusion",
                "target": "intermediate memory traffic",
            },
            "tensor_core": {
                "name": "Tensor Core path",
                "target": "matrix compute throughput",
            },
        }

    def execute_phase1(self, input_data: Dict[str, Any]) -> AgentResult:
        kernel_path = Path(input_data.get("kernel_path", ""))
        recommended = input_data.get("recommended_strategies", [])
        bottlenecks = input_data.get("bottlenecks", [])

        candidates = []
        for strategy_name in recommended:
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                continue
            candidates.append(
                {
                    "strategy": strategy_name,
                    "strategy_name": strategy["name"],
                    "target": strategy["target"],
                    "source_kernel": str(kernel_path) if str(kernel_path) else None,
                    "candidate_kernel": f"{kernel_path.stem}_{strategy_name}.cu"
                    if kernel_path.name
                    else f"candidate_{strategy_name}.cu",
                    "required_evidence": [
                        "correctness result",
                        "latency distribution",
                        "NCU metrics for the target bottleneck",
                    ],
                }
            )

        if not candidates:
            return AgentResult(
                success=False,
                agent_type=self.agent_type,
                phase=1,
                output_data={"bottlenecks": bottlenecks, "optimized_kernels": []},
                error_message="No known strategies were recommended.",
            )

        return AgentResult(
            success=True,
            agent_type=self.agent_type,
            phase=1,
            output_data={
                "bottlenecks": bottlenecks,
                "optimized_kernels": candidates,
            },
        )

    def execute_phase2(self, input_data: Dict[str, Any]) -> AgentResult:
        best_kernel = input_data.get("best_kernel")
        bottlenecks = input_data.get("detailed_bottlenecks", [])

        refinement_plan = [
            {
                "step": "isolate_primary_bottleneck",
                "input": bottlenecks,
                "output": "one hypothesis with a measurable target",
            },
            {
                "step": "generate_parameter_sweep",
                "input": best_kernel,
                "output": "bounded variants for block size, vector width, and unroll factor",
            },
            {
                "step": "profile_variants",
                "input": "candidate variants",
                "output": "correctness plus latency and NCU counters",
            },
        ]

        return AgentResult(
            success=True,
            agent_type=self.agent_type,
            phase=2,
            output_data={"refinement_plan": refinement_plan},
        )

    def execute_phase3(self, input_data: Dict[str, Any]) -> AgentResult:
        final_kernel = input_data.get("final_kernel")
        validation = input_data.get("validation_results", {})
        is_valid = bool(validation.get("passed_correctness"))

        return AgentResult(
            success=is_valid,
            agent_type=self.agent_type,
            phase=3,
            output_data={
                "packaged_solution": {
                    "kernel_path": final_kernel,
                    "validation": validation,
                }
            },
            error_message=None if is_valid else "Final kernel lacks correctness evidence.",
        )


class AnalyzerAgent(BaseAgent):
    """Analyze profiling data and recommend optimization directions."""

    def execute_phase1(self, input_data: Dict[str, Any]) -> AgentResult:
        metrics = input_data.get("profiling_metrics") or input_data.get("ncu_metrics") or {}
        operator = input_data.get("operator", "unknown")

        if not metrics:
            return AgentResult(
                success=False,
                agent_type=self.agent_type,
                phase=1,
                output_data={"operator": operator, "bottlenecks": [], "recommended_strategies": []},
                error_message="profiling_metrics or ncu_metrics are required.",
            )

        bottlenecks: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        dram_pct = self._metric(metrics, "dram_throughput_pct")
        sm_pct = self._metric(metrics, "sm_utilization_pct")
        l2_hit = self._metric(metrics, "l2_hit_rate")
        occupancy = self._metric(metrics, "occupancy_pct")

        if dram_pct is not None and dram_pct >= 70:
            bottlenecks.append(
                {
                    "type": "memory_bound",
                    "metric": "dram_throughput_pct",
                    "value": dram_pct,
                    "severity": "high" if dram_pct >= 85 else "medium",
                }
            )
            recommendations.extend(["vectorized_memory", "shared_memory_tiling"])

        if l2_hit is not None and l2_hit < 55:
            bottlenecks.append(
                {
                    "type": "poor_cache_locality",
                    "metric": "l2_hit_rate",
                    "value": l2_hit,
                    "severity": "medium",
                }
            )
            recommendations.append("shared_memory_tiling")

        if sm_pct is not None and sm_pct < 55:
            bottlenecks.append(
                {
                    "type": "compute_underutilized",
                    "metric": "sm_utilization_pct",
                    "value": sm_pct,
                    "severity": "medium",
                }
            )
            recommendations.append("occupancy_tuning")

        if occupancy is not None and occupancy < 50:
            recommendations.append("occupancy_tuning")

        if "matmul" in operator.lower():
            recommendations.append("tensor_core")
        if "norm" in operator.lower() or "softmax" in operator.lower():
            recommendations.append("warp_reduction")

        recommendations = list(dict.fromkeys(recommendations))

        return AgentResult(
            success=bool(bottlenecks or recommendations),
            agent_type=self.agent_type,
            phase=1,
            output_data={
                "operator": operator,
                "bottlenecks": bottlenecks,
                "recommended_strategies": recommendations,
            },
            error_message=None if bottlenecks or recommendations else "No actionable bottleneck found.",
        )

    def execute_phase2(self, input_data: Dict[str, Any]) -> AgentResult:
        feedback = input_data.get("feedback", {})
        detailed = {
            "changed_metric": feedback.get("changed_metric"),
            "regression": feedback.get("regression"),
            "next_measurement": feedback.get("next_measurement", "profile the modified kernel"),
        }
        return AgentResult(
            success=True,
            agent_type=self.agent_type,
            phase=2,
            output_data={"detailed_bottlenecks": [detailed]},
        )

    def execute_phase3(self, input_data: Dict[str, Any]) -> AgentResult:
        benchmark = input_data.get("benchmark_summary", {})
        speedup = benchmark.get("geomean_speedup")
        passed = bool(benchmark.get("passed_correctness"))

        return AgentResult(
            success=passed and speedup is not None,
            agent_type=self.agent_type,
            phase=3,
            output_data={"final_analysis": benchmark},
            error_message=None if passed and speedup is not None else "Final benchmark summary is incomplete.",
        )

    @staticmethod
    def _metric(metrics: Dict[str, Any], name: str) -> Optional[float]:
        value = metrics.get(name)
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None


class ProfilerAgent(BaseAgent):
    """Validate measured benchmark and profiling evidence."""

    def execute_phase1(self, input_data: Dict[str, Any]) -> AgentResult:
        results = input_data.get("benchmark_results", [])
        if not results:
            return AgentResult(
                success=False,
                agent_type=self.agent_type,
                phase=1,
                output_data={"benchmark_results": []},
                error_message="benchmark_results are required; this agent does not fabricate measurements.",
            )

        valid_results = [
            result
            for result in results
            if result.get("passed_correctness") and result.get("latency_ms") is not None
        ]

        return AgentResult(
            success=bool(valid_results),
            agent_type=self.agent_type,
            phase=1,
            output_data={"benchmark_results": valid_results},
            error_message=None if valid_results else "No benchmark result passed correctness.",
        )

    def execute_phase2(self, input_data: Dict[str, Any]) -> AgentResult:
        variance = input_data.get("variance_analysis", {})
        passed = variance.get("coefficient_of_variation_pct", 100.0) <= self.config.get(
            "max_cv_pct", 5.0
        )
        return AgentResult(
            success=passed,
            agent_type=self.agent_type,
            phase=2,
            output_data={"variance_analysis": variance},
            error_message=None if passed else "Benchmark variance exceeds configured threshold.",
        )

    def execute_phase3(self, input_data: Dict[str, Any]) -> AgentResult:
        full_benchmark = input_data.get("full_benchmark", {})
        passed = bool(full_benchmark.get("passed_correctness")) and bool(
            full_benchmark.get("sanitizer_passed", True)
        )
        return AgentResult(
            success=passed,
            agent_type=self.agent_type,
            phase=3,
            output_data={"full_benchmark": full_benchmark},
            error_message=None if passed else "Final profiling evidence failed validation.",
        )


class ReviewerAgent(BaseAgent):
    """Review kernel evidence and reject silent or delegated speedups."""

    BANNED_LIBRARY_TOKENS = (
        "flashinfer.",
        "deepgemm.",
        "cublas",
        "cudnn",
    )

    def execute_phase1(self, input_data: Dict[str, Any]) -> AgentResult:
        kernel_path = input_data.get("kernel_path")
        benchmark = input_data.get("benchmark_result", {})
        library_check = self._check_library_delegation(kernel_path)
        evidence_check = self._check_evidence(benchmark)

        approved = library_check["passed"] and evidence_check["passed"]
        return AgentResult(
            success=approved,
            agent_type=self.agent_type,
            phase=1,
            output_data={
                "library_check": library_check,
                "evidence_check": evidence_check,
                "approved": approved,
            },
            error_message=None if approved else "Review failed.",
        )

    def execute_phase2(self, input_data: Dict[str, Any]) -> AgentResult:
        before = input_data.get("before_metrics", {})
        after = input_data.get("after_metrics", {})
        changed = before != after
        passed = changed and bool(input_data.get("passed_correctness"))

        return AgentResult(
            success=passed,
            agent_type=self.agent_type,
            phase=2,
            output_data={
                "optimization_validity": {
                    "metrics_changed": changed,
                    "passed_correctness": bool(input_data.get("passed_correctness")),
                },
                "approved": passed,
            },
            error_message=None if passed else "No measurable valid improvement was provided.",
        )

    def execute_phase3(self, input_data: Dict[str, Any]) -> AgentResult:
        final_evidence = input_data.get("final_evidence", {})
        approved = bool(final_evidence.get("passed_correctness")) and bool(
            final_evidence.get("reproducible")
        )

        return AgentResult(
            success=approved,
            agent_type=self.agent_type,
            phase=3,
            output_data={
                "final_review": {
                    "approved_for_archive": approved,
                    "evidence": final_evidence,
                }
            },
            error_message=None if approved else "Final evidence is not sufficient for archive.",
        )

    def _check_library_delegation(self, kernel_path: Optional[str]) -> Dict[str, Any]:
        if not kernel_path:
            return {"passed": False, "banned_libs_found": [], "reason": "kernel_path is required"}

        path = Path(kernel_path)
        if not path.is_absolute():
            path = self.workspace / path

        if not path.exists():
            return {"passed": False, "banned_libs_found": [], "reason": f"not found: {path}"}

        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        found = [token for token in self.BANNED_LIBRARY_TOKENS if token in text]

        return {
            "passed": not found,
            "banned_libs_found": found,
            "reason": "ok" if not found else "banned library delegation found",
        }

    @staticmethod
    def _check_evidence(benchmark: Dict[str, Any]) -> Dict[str, Any]:
        passed_correctness = bool(benchmark.get("passed_correctness"))
        latency_present = benchmark.get("latency_ms") is not None
        baseline_present = benchmark.get("baseline_latency_ms") is not None
        speedup_present = benchmark.get("speedup") is not None
        passed = passed_correctness and latency_present and (baseline_present or speedup_present)

        return {
            "passed": passed,
            "passed_correctness": passed_correctness,
            "latency_present": latency_present,
            "baseline_or_speedup_present": baseline_present or speedup_present,
        }


class CoordinatorAgent(BaseAgent):
    """Coordinate agent execution for each workflow phase."""

    def __init__(self, workspace: Path, config: Dict[str, Any]):
        super().__init__(workspace, config)
        self.agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        self.agents[agent.agent_type] = agent

    def coordinate_phase(self, phase: int, input_data: Dict[str, Any]) -> Dict[str, AgentResult]:
        if phase == 1:
            return self._coordinate_phase1(input_data)

        results: Dict[str, AgentResult] = {}
        for agent_name in ("AnalyzerAgent", "OptimizerAgent", "ProfilerAgent", "ReviewerAgent"):
            agent = self.agents.get(agent_name)
            if not agent:
                continue
            method = getattr(agent, f"execute_phase{phase}")
            results[agent_name] = method(input_data)
        return results

    def _coordinate_phase1(self, input_data: Dict[str, Any]) -> Dict[str, AgentResult]:
        results: Dict[str, AgentResult] = {}

        analyzer = self.agents.get("AnalyzerAgent")
        if analyzer:
            analyzer_result = analyzer.execute_phase1(input_data)
            results["AnalyzerAgent"] = analyzer_result
            if analyzer_result.success:
                input_data = {
                    **input_data,
                    "bottlenecks": analyzer_result.output_data.get("bottlenecks", []),
                    "recommended_strategies": analyzer_result.output_data.get(
                        "recommended_strategies", []
                    ),
                }

        optimizer = self.agents.get("OptimizerAgent")
        if optimizer:
            optimizer_result = optimizer.execute_phase1(input_data)
            results["OptimizerAgent"] = optimizer_result
            if optimizer_result.success:
                input_data = {
                    **input_data,
                    "candidate_kernels": optimizer_result.output_data.get("optimized_kernels", []),
                }

        profiler = self.agents.get("ProfilerAgent")
        if profiler:
            profiler_result = profiler.execute_phase1(input_data)
            results["ProfilerAgent"] = profiler_result
            if profiler_result.success:
                best = max(
                    profiler_result.output_data.get("benchmark_results", []),
                    key=lambda item: item.get("speedup", 0.0),
                )
                input_data = {**input_data, "benchmark_result": best, "kernel_path": best.get("kernel")}

        reviewer = self.agents.get("ReviewerAgent")
        if reviewer and "benchmark_result" in input_data:
            results["ReviewerAgent"] = reviewer.execute_phase1(input_data)

        return results

    def execute_phase1(self, input_data: Dict[str, Any]) -> AgentResult:
        results = self.coordinate_phase(1, input_data)
        return self._aggregate_result(1, results)

    def execute_phase2(self, input_data: Dict[str, Any]) -> AgentResult:
        results = self.coordinate_phase(2, input_data)
        return self._aggregate_result(2, results)

    def execute_phase3(self, input_data: Dict[str, Any]) -> AgentResult:
        results = self.coordinate_phase(3, input_data)
        return self._aggregate_result(3, results)

    def _aggregate_result(self, phase: int, results: Dict[str, AgentResult]) -> AgentResult:
        success = bool(results) and all(result.success for result in results.values())
        return AgentResult(
            success=success,
            agent_type=self.agent_type,
            phase=phase,
            output_data={"agent_results": results},
            error_message=None if success else "One or more agents failed.",
        )


def create_agent(agent_type: str, workspace: Path, config: Dict[str, Any]) -> BaseAgent:
    """Factory for workflow agents."""

    agent_classes = {
        "optimizer": OptimizerAgent,
        "analyzer": AnalyzerAgent,
        "profiler": ProfilerAgent,
        "reviewer": ReviewerAgent,
        "coordinator": CoordinatorAgent,
    }
    agent_class = agent_classes.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent_class(workspace, config)


if __name__ == "__main__":
    workspace = Path.cwd()
    config: Dict[str, Any] = {"backend": "local", "gpu": "B200"}

    coordinator = create_agent("coordinator", workspace, config)
    assert isinstance(coordinator, CoordinatorAgent)

    for name in ("analyzer", "optimizer", "profiler", "reviewer"):
        coordinator.register_agent(create_agent(name, workspace, config))

    result = coordinator.execute_phase1(
        {
            "operator": "softmax",
            "kernel_path": "kernels/operators/softmax/softmax_final.cu",
            "profiling_metrics": {
                "dram_throughput_pct": 82.0,
                "sm_utilization_pct": 48.0,
                "l2_hit_rate": 42.0,
                "occupancy_pct": 44.0,
            },
            "benchmark_results": [],
        }
    )
    print(f"phase1 success={result.success}")
