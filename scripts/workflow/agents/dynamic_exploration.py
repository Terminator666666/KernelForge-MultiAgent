#!/usr/bin/env python3
"""Dynamic exploration engine for CUDA kernel optimization.

The engine turns profiling metrics into explicit optimization directions. It is
kept deterministic so that orchestration code can audit why a direction was
selected before handing it to an LLM or a human engineer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ProfilingData:
    """Subset of NCU metrics used by the exploration engine."""

    dram_throughput_pct: float
    sm_utilization_pct: float
    l2_hit_rate: float
    occupancy_pct: float
    warp_efficiency: float
    memory_workload_analysis: Dict[str, Any]
    compute_workload_analysis: Dict[str, Any]


@dataclass
class OptimizationDirection:
    """Candidate optimization direction generated from profiling evidence."""

    name: str
    rationale: str
    potential_speedup: str
    risk_level: str
    implementation_plan: str
    references: List[str]


class DynamicExplorationEngine:
    """Generate optimization directions from profile-derived bottlenecks."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict[str, Any]:
        for root in [self.workspace, *self.workspace.parents]:
            for skill_root in (root / "skills", root / ".claude" / "skills"):
                skill_path = skill_root / "optimization-knowledge" / "SKILL.md"
                if skill_path.exists():
                    return {"loaded": True, "path": str(skill_path)}
        return {"loaded": False}

    def analyze_bottleneck(self, profiling_data: ProfilingData) -> Dict[str, Any]:
        compute_intensity = self._estimate_compute_intensity(profiling_data)

        if profiling_data.dram_throughput_pct >= 70:
            primary = "memory_bound"
            root_cause = self._analyze_memory_bottleneck(profiling_data)
        elif profiling_data.sm_utilization_pct < 55:
            primary = "compute_underutilized"
            root_cause = self._analyze_compute_bottleneck(profiling_data)
        else:
            primary = "balanced"
            root_cause = "memory and compute metrics are both within expected ranges"

        theoretical_limit = self._calculate_theoretical_limit(profiling_data, compute_intensity)

        return {
            "primary_bottleneck": primary,
            "root_cause": root_cause,
            "compute_intensity": compute_intensity,
            "theoretical_limit": theoretical_limit,
            "gap_analysis": self._analyze_gap(profiling_data, theoretical_limit),
        }

    def generate_optimization_directions(
        self,
        bottleneck_analysis: Dict[str, Any],
        operator_info: Dict[str, Any],
    ) -> List[OptimizationDirection]:
        bottleneck = bottleneck_analysis.get("primary_bottleneck")
        root_cause = str(bottleneck_analysis.get("root_cause", ""))

        if bottleneck == "memory_bound":
            directions = self._explore_memory_optimizations(root_cause)
        elif bottleneck == "compute_underutilized":
            directions = self._explore_compute_optimizations(root_cause)
        else:
            directions = self._explore_balanced_optimizations()

        directions.extend(self._explore_operator_specific_directions(operator_info))
        return directions

    def generate_humanize_plan_prompt(
        self,
        directions: List[OptimizationDirection],
        selected_direction: OptimizationDirection,
        operator_info: Dict[str, Any],
    ) -> str:
        alternatives = "\n".join(
            f"- {direction.name}: {direction.rationale}"
            for direction in directions
            if direction != selected_direction
        )
        references = "\n".join(f"- {ref}" for ref in selected_direction.references)

        return f"""# Optimization Implementation Draft

## Selected Direction
{selected_direction.name}

## Rationale
{selected_direction.rationale}

## Expected Impact
- Potential speedup: {selected_direction.potential_speedup}
- Risk level: {selected_direction.risk_level}

## Implementation Plan
{selected_direction.implementation_plan}

## References
{references}

## Alternative Directions
{alternatives or "- None"}

## Operator Context
- Name: {operator_info.get("name", "unknown")}
- Definition: {operator_info.get("definition", "not provided")}
- Characteristics: {operator_info.get("characteristics", "not provided")}
"""

    def save_draft(self, draft_content: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(draft_content, encoding="utf-8")

    @staticmethod
    def _estimate_compute_intensity(profiling_data: ProfilingData) -> float:
        bytes_read = float(profiling_data.memory_workload_analysis.get("bytes_read", 0.0))
        bytes_written = float(profiling_data.memory_workload_analysis.get("bytes_written", 0.0))
        flops = float(profiling_data.compute_workload_analysis.get("flops", 0.0))
        traffic = bytes_read + bytes_written
        if traffic <= 0.0 or flops <= 0.0:
            return 0.0
        return flops / traffic

    @staticmethod
    def _analyze_memory_bottleneck(profiling_data: ProfilingData) -> str:
        reasons: List[str] = []

        if profiling_data.l2_hit_rate < 50:
            reasons.append("low L2 hit rate suggests weak data locality")
        if profiling_data.dram_throughput_pct > 85:
            reasons.append("DRAM bandwidth is close to saturation")
        if profiling_data.memory_workload_analysis.get("uncoalesced_accesses", 0) > 0:
            reasons.append("uncoalesced memory accesses were reported")

        return "; ".join(reasons) if reasons else "high DRAM throughput"

    @staticmethod
    def _analyze_compute_bottleneck(profiling_data: ProfilingData) -> str:
        reasons: List[str] = []

        if profiling_data.occupancy_pct < 50:
            reasons.append("low occupancy limits active warps")
        if profiling_data.warp_efficiency < 80:
            reasons.append("low warp efficiency suggests divergence or serialization")
        if profiling_data.sm_utilization_pct < 35:
            reasons.append("very low SM utilization suggests launch or scheduling overhead")

        return "; ".join(reasons) if reasons else "compute units are underutilized"

    @staticmethod
    def _calculate_theoretical_limit(
        profiling_data: ProfilingData,
        compute_intensity: float,
    ) -> Dict[str, Any]:
        return {
            "compute_intensity_flops_per_byte": compute_intensity,
            "observed_dram_throughput_pct": profiling_data.dram_throughput_pct,
            "observed_sm_utilization_pct": profiling_data.sm_utilization_pct,
        }

    @staticmethod
    def _analyze_gap(
        profiling_data: ProfilingData,
        theoretical_limit: Dict[str, Any],
    ) -> str:
        if profiling_data.dram_throughput_pct >= 85:
            return "reduce memory traffic or improve locality before adding compute work"
        if profiling_data.sm_utilization_pct < 50:
            return "increase useful work per launch and improve occupancy"
        if theoretical_limit["compute_intensity_flops_per_byte"] == 0.0:
            return "collect traffic and FLOP counters for a roofline estimate"
        return "profile focused variants to identify the next limiting metric"

    @staticmethod
    def _explore_memory_optimizations(root_cause: str) -> List[OptimizationDirection]:
        directions: List[OptimizationDirection] = []

        if "locality" in root_cause or "bandwidth" in root_cause:
            directions.append(
                OptimizationDirection(
                    name="Shared-memory tiling",
                    rationale=(
                        "The profile points to memory pressure. Tiling can reuse data "
                        "inside a block and reduce DRAM transactions."
                    ),
                    potential_speedup="1.5x-4x when reuse is available",
                    risk_level="medium",
                    implementation_plan=(
                        "1. Identify reusable tensor dimensions.\n"
                        "2. Choose a tile size within shared-memory limits.\n"
                        "3. Add cooperative loads and boundary handling.\n"
                        "4. Validate correctness and profile bank conflicts."
                    ),
                    references=["CUDA shared memory guide", "CUTLASS tiling patterns"],
                )
            )

        if "uncoalesced" in root_cause:
            directions.append(
                OptimizationDirection(
                    name="Memory access reordering",
                    rationale="Reported uncoalesced accesses can waste memory transactions.",
                    potential_speedup="1.2x-3x",
                    risk_level="low",
                    implementation_plan=(
                        "1. Map thread lanes to contiguous data.\n"
                        "2. Use vectorized loads where alignment allows.\n"
                        "3. Keep a scalar tail path for boundaries.\n"
                        "4. Verify transaction efficiency with NCU."
                    ),
                    references=["CUDA memory coalescing rules"],
                )
            )

        if not directions:
            directions.append(
                OptimizationDirection(
                    name="Memory traffic audit",
                    rationale="The bottleneck is memory-bound but the root cause is not specific yet.",
                    potential_speedup="unknown until counters are collected",
                    risk_level="low",
                    implementation_plan=(
                        "1. Collect sector counts, requested bytes, and cache hit rates.\n"
                        "2. Compare reads and writes against the theoretical minimum.\n"
                        "3. Pick the largest avoidable traffic source."
                    ),
                    references=["NCU memory workload analysis"],
                )
            )

        return directions

    @staticmethod
    def _explore_compute_optimizations(root_cause: str) -> List[OptimizationDirection]:
        directions: List[OptimizationDirection] = []

        if "occupancy" in root_cause:
            directions.append(
                OptimizationDirection(
                    name="Occupancy-aware launch tuning",
                    rationale="Low occupancy indicates too few active warps or excessive resource use.",
                    potential_speedup="1.2x-2.5x",
                    risk_level="low",
                    implementation_plan=(
                        "1. Sweep block size and register pressure.\n"
                        "2. Check achieved occupancy and eligible warps.\n"
                        "3. Keep only variants that preserve correctness."
                    ),
                    references=["CUDA occupancy calculator"],
                )
            )

        if "warp efficiency" in root_cause:
            directions.append(
                OptimizationDirection(
                    name="Warp-level control-flow cleanup",
                    rationale="Low warp efficiency usually comes from divergence or serialized work.",
                    potential_speedup="1.1x-2x",
                    risk_level="medium",
                    implementation_plan=(
                        "1. Locate divergent branches.\n"
                        "2. Replace per-thread branches with warp primitives where possible.\n"
                        "3. Profile branch efficiency and instruction count."
                    ),
                    references=["CUDA warp-level primitives"],
                )
            )

        if not directions:
            directions.append(
                OptimizationDirection(
                    name="Launch overhead and instruction audit",
                    rationale="Compute utilization is low but the limiting counter is not isolated.",
                    potential_speedup="unknown until counters are collected",
                    risk_level="low",
                    implementation_plan=(
                        "1. Measure launch count and kernel duration.\n"
                        "2. Check instruction throughput and eligible warps.\n"
                        "3. Decide between fusion, occupancy tuning, or algorithm changes."
                    ),
                    references=["NCU compute workload analysis"],
                )
            )

        return directions

    @staticmethod
    def _explore_balanced_optimizations() -> List[OptimizationDirection]:
        return [
            OptimizationDirection(
                name="Variant sweep around the current best kernel",
                rationale="No single bottleneck dominates, so small parameter sweeps are safer.",
                potential_speedup="1.05x-1.5x",
                risk_level="low",
                implementation_plan=(
                    "1. Sweep one parameter at a time.\n"
                    "2. Track latency variance and key NCU metrics.\n"
                    "3. Keep only variants with reproducible improvement."
                ),
                references=["benchmark variance analysis"],
            )
        ]

    @staticmethod
    def _explore_operator_specific_directions(
        operator_info: Dict[str, Any],
    ) -> List[OptimizationDirection]:
        name = str(operator_info.get("name", "")).lower()
        directions: List[OptimizationDirection] = []

        if "matmul" in name:
            directions.append(
                OptimizationDirection(
                    name="Tensor Core path validation",
                    rationale="Matrix multiplication can often benefit from Tensor Core instructions.",
                    potential_speedup="2x-8x when dimensions and datatypes align",
                    risk_level="medium",
                    implementation_plan=(
                        "1. Confirm datatype and layout support.\n"
                        "2. Implement a WMMA or CUTLASS-style tile path.\n"
                        "3. Compare against a non-delegated baseline."
                    ),
                    references=["NVIDIA WMMA programming guide"],
                )
            )

        if "softmax" in name or "norm" in name:
            directions.append(
                OptimizationDirection(
                    name="Warp-level reduction path",
                    rationale="Softmax and norm operators are often reduction-heavy.",
                    potential_speedup="1.3x-3x",
                    risk_level="medium",
                    implementation_plan=(
                        "1. Keep reductions inside a warp when rows are small.\n"
                        "2. Use block reductions for larger rows.\n"
                        "3. Validate numerical tolerance across edge cases."
                    ),
                    references=["CUDA cooperative groups", "warp shuffle reductions"],
                )
            )

        return directions


def demo_dynamic_exploration() -> None:
    workspace = Path.cwd()
    engine = DynamicExplorationEngine(workspace)
    profiling = ProfilingData(
        dram_throughput_pct=85.0,
        sm_utilization_pct=45.0,
        l2_hit_rate=35.0,
        occupancy_pct=40.0,
        warp_efficiency=75.0,
        memory_workload_analysis={
            "uncoalesced_accesses": 25,
            "bytes_read": 1_000_000,
            "bytes_written": 250_000,
        },
        compute_workload_analysis={"flops": 12_000_000},
    )
    analysis = engine.analyze_bottleneck(profiling)
    directions = engine.generate_optimization_directions(
        analysis,
        {"name": "softmax", "definition": "row-wise softmax"},
    )
    print(f"primary_bottleneck={analysis['primary_bottleneck']}")
    print(f"directions={len(directions)}")


if __name__ == "__main__":
    demo_dynamic_exploration()
