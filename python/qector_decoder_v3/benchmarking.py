"""
qector_decoder_v3.benchmarking — reproducible, statistically-honest benchmarks.

A benchmark is only credible if someone else can reproduce it and if it reports
tail latency, not just an average.  This harness captures everything needed for
that:

    * **Environment**: CPU model & cores, RAM, OS, Python / NumPy / Rust versions,
      package versions, and whether CUDA / OpenCL are present.
    * **Methodology**: explicit seed, warmup count, timed iteration count.
    * **Statistics**: mean, median, std, min, max, p50/p90/p95/p99 and a 95%
      bootstrap-free normal confidence interval on the mean.
    * **Hot vs cold path**: construction time (cold) is measured separately from
      decode-only time (hot).
    * **Memory**: peak Python allocation via ``tracemalloc`` (and RSS via
      ``psutil`` when installed).
    * **Output**: structured ``dict`` plus ``to_json`` / ``to_csv`` writers and a
      ``GitHub Actions``-friendly artifact layout.

Example
-------
>>> from qector_decoder_v3 import codes
>>> from qector_decoder_v3.benchmarking import benchmark_decoder, capture_environment
>>> code = codes.rotated_surface_code(5)
>>> r = benchmark_decoder("blossom", code, n_trials=2000, warmup=200, seed=1)
>>> r["latency_us"]["p99"]
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "capture_environment",
    "git_commit",
    "percentiles",
    "time_iterations",
    "benchmark_decoder",
    "BenchmarkReport",
    "write_json",
    "write_csv",
]


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------
def capture_environment() -> Dict[str, Any]:
    """Capture hardware/software details needed to reproduce a benchmark."""
    env: Dict[str, Any] = {
        "timestamp_unix": None,  # caller stamps this (Date.now is non-deterministic in some runtimes)
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        env["numpy_version"] = np.__version__
    except Exception:  # pragma: no cover
        env["numpy_version"] = None

    env["rust_version"] = _safe_cmd(["rustc", "--version"])
    env["cargo_version"] = _safe_cmd(["cargo", "--version"])
    env["git_commit"] = git_commit()

    # package versions
    for pkg in ("qector_decoder_v3", "pymatching", "stim", "scipy"):
        env[f"{pkg}_version"] = _pkg_version(pkg)

    # memory
    env.update(_memory_info())

    # accelerators
    try:
        import qector_decoder_v3 as qd

        env["cuda_available"] = bool(qd.cuda_is_available())
        env["opencl_available"] = bool(qd.opencl_is_available())
    except Exception:  # pragma: no cover
        env["cuda_available"] = None
        env["opencl_available"] = None

    return env


def git_commit() -> str:
    """Return the current git commit hash, or ``"unknown"`` if unavailable.

    Looks for the repository the package lives in so the value is meaningful even
    when the script's CWD is elsewhere. Used to stamp every artifact with the
    exact build it came from (credibility requirement: no ``"Git commit: unknown"``).
    """
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = _safe_cmd(["git", "-C", repo, "rev-parse", "HEAD"])
    return out if out else "unknown"


def _safe_cmd(cmd: Sequence[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        try:
            return md.version(name.replace("_", "-"))
        except md.PackageNotFoundError:
            return md.version(name)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None


def _memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"ram_total_gb": None, "ram_available_gb": None}
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
        info["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def percentiles(samples: Sequence[float], ps=(50, 90, 95, 99)) -> Dict[str, float]:
    """Percentiles (linear interpolation) keyed as ``pXX``."""
    arr = np.asarray(samples, dtype=np.float64)
    out: Dict[str, float] = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def summarize(samples_seconds: Sequence[float]) -> Dict[str, float]:
    """Full latency summary (in microseconds) from per-iteration seconds."""
    us = np.asarray(samples_seconds, dtype=np.float64) * 1e6
    n = us.size
    mean = float(us.mean())
    std = float(us.std(ddof=1)) if n > 1 else 0.0
    # 95% normal CI on the mean.
    half = 1.959963985 * std / (n**0.5) if n > 1 else 0.0
    summary = {
        "n": int(n),
        "mean": mean,
        "median": float(np.median(us)),
        "std": std,
        "min": float(us.min()),
        "max": float(us.max()),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }
    summary.update(percentiles(us.tolist()))
    return summary


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------
def time_iterations(
    fn: Callable[[], Any], n_trials: int, warmup: int = 0
) -> List[float]:
    """Time ``fn`` per-call ``n_trials`` times after ``warmup`` untimed calls."""
    for _ in range(max(0, warmup)):
        fn()
    samples: List[float] = []
    perf = time.perf_counter
    for _ in range(n_trials):
        t0 = perf()
        fn()
        samples.append(perf() - t0)
    return samples


# ---------------------------------------------------------------------------
# Decoder benchmark
# ---------------------------------------------------------------------------
def _build_decoder(kind: str, code):
    from . import (
        BlossomDecoder,
        BPOSDDecoder,
        CPUBatchDecoder,
        FastUnionFindDecoder,
        SparseBlossomDecoder,
        UnionFindDecoder,
    )

    c2q, nq = code.check_to_qubits, code.n_qubits
    builders = {
        "union_find": lambda: UnionFindDecoder(c2q, nq),
        "fast_union_find": lambda: FastUnionFindDecoder(c2q, nq),
        "blossom": lambda: BlossomDecoder(c2q, nq),
        "sparse_blossom": lambda: SparseBlossomDecoder(c2q, nq),
        "cpu_batch": lambda: CPUBatchDecoder(c2q, nq),
        "bp_osd": lambda: BPOSDDecoder(c2q, nq, 0.05),
    }
    if kind not in builders:
        raise ValueError(f"unknown decoder kind {kind!r}; choose from {list(builders)}")
    return builders[kind]


def benchmark_decoder(
    kind: str,
    code,
    n_trials: int = 1000,
    warmup: int = 100,
    p: float = 0.08,
    seed: int = 1234,
    measure_memory: bool = True,
) -> Dict[str, Any]:
    """Benchmark a decoder on a code with full hot/cold and tail-latency stats.

    Returns a structured ``dict`` (also wrappable in :class:`BenchmarkReport`).
    The **cold path** times decoder construction; the **hot path** times
    decode-only on pre-built syndromes.  Syndromes are reachable (generated from
    real errors) and fixed by ``seed`` for reproducibility.
    """
    rng = np.random.default_rng(seed)
    H = code.parity_check_matrix()
    n_qubits = code.n_qubits

    # Reachable syndromes (errors -> H e).
    errors = (rng.random((max(n_trials, 1), n_qubits)) < p).astype(np.uint8)
    syndromes = (errors @ H.T) & 1
    syndromes = syndromes.astype(np.uint8)

    builder = _build_decoder(kind, code)

    # Cold path: construction time (median of a few builds).
    cold_samples = time_iterations(
        lambda: builder(), n_trials=min(20, max(3, n_trials // 50)), warmup=1
    )
    decoder = builder()

    # Optional correctness gate (cheap, on a handful of syndromes).
    valid = True
    for i in range(min(64, len(syndromes))):
        corr = np.asarray(decoder.decode(syndromes[i]), dtype=np.uint8).reshape(-1)
        if not np.array_equal((H @ corr) & 1, syndromes[i]):
            valid = False
            break

    # Hot path: decode-only.
    idx = {"i": 0}
    m = len(syndromes)

    def one_decode():
        i = idx["i"]
        idx["i"] = (i + 1) % m
        decoder.decode(syndromes[i])

    peak_kib = None
    if measure_memory:
        tracemalloc.start()
    hot_samples = time_iterations(one_decode, n_trials=n_trials, warmup=warmup)
    if measure_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_kib = round(peak / 1024, 2)

    report = {
        "decoder": kind,
        "code": code.name,
        "n_qubits": int(code.n_qubits),
        "n_checks": int(code.n_checks),
        "distance": code.distance,
        "physical_error_rate": p,
        "seed": seed,
        "n_trials": n_trials,
        "warmup": warmup,
        "syndrome_faithful": valid,
        "cold_path_us": summarize(cold_samples),
        "latency_us": summarize(hot_samples),
        "throughput_decodes_per_s": (1.0 / statistics.median(hot_samples))
        if hot_samples
        else None,
        "peak_python_alloc_kib": peak_kib,
    }
    return report


# ---------------------------------------------------------------------------
# Report container + writers
# ---------------------------------------------------------------------------
class BenchmarkReport:
    """A set of decoder benchmarks plus the environment they ran in."""

    def __init__(
        self, results: List[Dict[str, Any]], environment: Optional[dict] = None
    ):
        self.results = results
        self.environment = environment or capture_environment()

    def to_dict(self) -> Dict[str, Any]:
        return {"environment": self.environment, "results": self.results}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def to_csv(self) -> str:
        return _results_to_csv(self.results)

    def save(self, json_path: str, csv_path: Optional[str] = None) -> None:
        write_json(json_path, self.to_dict())
        if csv_path:
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(self.to_csv())


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def write_csv(path: str, results: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(_results_to_csv(results))


def _results_to_csv(results: List[Dict[str, Any]]) -> str:
    cols = [
        "decoder",
        "code",
        "n_qubits",
        "n_checks",
        "distance",
        "physical_error_rate",
        "n_trials",
        "syndrome_faithful",
        "lat_mean_us",
        "lat_median_us",
        "lat_p50_us",
        "lat_p90_us",
        "lat_p95_us",
        "lat_p99_us",
        "lat_std_us",
        "throughput_per_s",
        "cold_median_us",
        "peak_python_alloc_kib",
    ]
    lines = [",".join(cols)]
    for r in results:
        lat = r.get("latency_us", {})
        cold = r.get("cold_path_us", {})
        row = [
            r.get("decoder", ""),
            r.get("code", ""),
            r.get("n_qubits", ""),
            r.get("n_checks", ""),
            r.get("distance", ""),
            r.get("physical_error_rate", ""),
            r.get("n_trials", ""),
            r.get("syndrome_faithful", ""),
            _fmt(lat.get("mean")),
            _fmt(lat.get("median")),
            _fmt(lat.get("p50")),
            _fmt(lat.get("p90")),
            _fmt(lat.get("p95")),
            _fmt(lat.get("p99")),
            _fmt(lat.get("std")),
            _fmt(r.get("throughput_decodes_per_s")),
            _fmt(cold.get("median")),
            _fmt(r.get("peak_python_alloc_kib")),
        ]
        lines.append(",".join(str(x) for x in row))
    return "\n".join(lines) + "\n"


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
