"""Statistics and report generation for the GPU LER benchmark.

Helper module for ``benchmark_ler_gpu.py`` (Wilson intervals, binomial stats, CSV/
JSON/Markdown writers). The general-purpose reproducible harness with hot/cold split
and tail-latency percentiles lives in ``qector_decoder_v3.benchmarking``; the
PyMatching circuit-level head-to-head is ``scripts/competitive_stim_ler.py``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple


def wilson_score_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    denominator = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denominator
    width = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return max(0.0, center - width), min(1.0, center + width)


def binomial_std(k: int, n: int) -> float:
    """Return the standard deviation of a binomial proportion."""
    if n == 0:
        return 0.0
    probability = k / n
    return math.sqrt(probability * (1.0 - probability) / n)


@dataclass
class LERResult:
    """One logical-error-rate benchmark result."""

    decoder: str
    code_distance: int
    error_model: str
    error_rate: float
    batch_size: int
    n_shots: int
    n_logical_errors: int
    ler: float
    ler_std: float
    ci_lower: float
    ci_upper: float
    elapsed_time_s: float
    throughput: float
    gpu_failures: int = 0
    gpu_degraded: bool = False
    gpu_recoveries: int = 0


@dataclass
class BenchmarkReport:
    """Complete CUDA/OpenCL/CPU LER benchmark report."""

    timestamp: str
    python_version: str
    qector_version: str
    cpu: str
    opencl_available: bool
    cuda_available: bool
    results: List[LERResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def print_markdown_table(results: List[LERResult], model: str) -> None:
    """Print one error-model result table."""
    print(f"\n### {model.upper()}\n")
    print(
        "| d | p | Batch | Decodeur | Shots | LER | std | IC 95% | "
        "Temps | Throughput | GPU fail | Degrade |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for result in results:
        if result.error_model.startswith(model):
            print(_result_row(result))


def print_markdown_summary(report: BenchmarkReport) -> None:
    """Print the complete Markdown summary."""
    print("\n---")
    print("# QECTOR v3 - Benchmark LER GPU vs CPU")
    print(f"\n**Date** : {report.timestamp}")
    print(f"**CPU** : {report.cpu}")
    print(f"**Python** : {report.python_version}")
    print(f"**QECTOR** : {report.qector_version}")
    print(f"**GPU OpenCL disponible** : {'oui' if report.opencl_available else 'non'}")
    print(f"**GPU CUDA natif disponible** : {'oui' if report.cuda_available else 'non'}")
    print(f"**Total runs** : {len(report.results)}")
    for model in sorted({result.error_model for result in report.results}):
        print_markdown_table(report.results, model)


def save_csv(results: List[LERResult], path: Path) -> None:
    """Save benchmark results as CSV."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "distance",
                "error_model",
                "error_rate",
                "decoder",
                "batch_size",
                "shots",
                "logical_errors",
                "ler",
                "ler_std",
                "ci_95_lower",
                "ci_95_upper",
                "elapsed_s",
                "throughput",
                "gpu_failures",
                "gpu_degraded",
                "gpu_recoveries",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.code_distance,
                    result.error_model,
                    result.error_rate,
                    result.decoder,
                    result.batch_size,
                    result.n_shots,
                    result.n_logical_errors,
                    result.ler,
                    result.ler_std,
                    result.ci_lower,
                    result.ci_upper,
                    result.elapsed_time_s,
                    result.throughput,
                    result.gpu_failures,
                    result.gpu_degraded,
                    result.gpu_recoveries,
                ]
            )


def save_json(report: BenchmarkReport, path: Path) -> None:
    """Save the complete report as JSON."""
    path.write_text(report.to_json(), encoding="utf-8")


def generate_markdown_report(report: BenchmarkReport, path: Path) -> None:
    """Save the complete report as Markdown."""
    lines = [
        "# QECTOR v3 - Benchmark LER GPU vs CPU",
        "",
        f"**Date** : {report.timestamp}",
        f"**CPU** : {report.cpu}",
        f"**Python** : {report.python_version}",
        f"**QECTOR** : {report.qector_version}",
        f"**GPU OpenCL disponible** : {'oui' if report.opencl_available else 'non'}",
        f"**GPU CUDA natif disponible** : {'oui' if report.cuda_available else 'non'}",
        f"**Total runs** : {len(report.results)}",
        "",
        "## Objectif",
        "",
        "Comparer le LER et la performance entre CUDA natif, OpenCL et CPU.",
        "",
        "## Resultats",
        "",
    ]
    for model in sorted({result.error_model for result in report.results}):
        lines.extend(
            [
                f"### {model.upper()}",
                "",
                "| d | p | Batch | Decodeur | Shots | LER | std | IC 95% | "
                "Temps | Throughput | GPU fail | Degrade |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        lines.extend(
            _result_row(result)
            for result in report.results
            if result.error_model.startswith(model)
        )
        lines.append("")
    lines.extend(
        [
            "## Validation",
            "",
            "CUDA, OpenCL et CPU doivent produire un LER identique pour la meme graine.",
            "Les deux chemins GPU utilisent un fallback CPU et un mode degrade apres 3 echecs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _result_row(result: LERResult) -> str:
    return (
        f"| {result.code_distance} | {result.error_rate} | {result.batch_size} | "
        f"{result.decoder} | {result.n_shots} | {result.ler:.6f} | "
        f"{result.ler_std:.6f} | [{result.ci_lower:.6f}, {result.ci_upper:.6f}] | "
        f"{result.elapsed_time_s:.2f}s | {result.throughput:.0f} | "
        f"{result.gpu_failures} | {'oui' if result.gpu_degraded else 'non'} |"
    )
