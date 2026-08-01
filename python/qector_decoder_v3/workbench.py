"""
qector_decoder_v3.workbench - the QECTOR Workbench application controller.

A headless, fully-testable backend for the QECTOR desktop Workbench.  It loads
real ``.stim`` / ``.dem`` files, runs real decode benchmarks through a cancelable
background job queue, and exports the resulting artifacts to JSON / CSV / PDF -
every number traced to a real decode (no fabricated data).  A thin GUI (see
``rest_api`` / the ``run-qector`` skill) wraps this controller; the controller is
what the test-suite drives, so the GUI stays a presentation shell.

Design points that satisfy the buyer-grade requirements:

  * **No fake data.** :meth:`Workbench.run_benchmark` decodes real syndromes with
    the real Rust decoders; LER is computed from real Stim shots when a circuit
    with observables is loaded.
  * **Survives long jobs.** Benchmarks run on a single FIFO worker thread; jobs
    are cancelable cooperatively (queued jobs cancel instantly; running jobs stop
    at the next decoder/distance unit boundary).
  * **Reproducible exports.** Every export is derived from a stored artifact dict
    that already carries the environment snapshot and git commit.
  * **Windows paths with spaces.** All paths go through ``os.path``; nothing is
    shell-quoted.

Example
-------
>>> from qector_decoder_v3.workbench import Workbench
>>> wb = Workbench()
>>> art = wb.run_benchmark({"code": "rotated_surface", "distances": [3, 5], "decoders": ["blossom"], "trials": 500})
>>> wb.export_json(art, "out.json")
... wb.export_csv(art, "out.csv")
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import benchmarking as _bm
from . import codes as _codes

__all__ = ["Job", "Workbench", "WorkbenchError", "main"]


class WorkbenchError(RuntimeError):
    """Raised for invalid Workbench requests (bad file, unknown code, etc.)."""


_CODE_FAMILIES = {
    "repetition": _codes.repetition_code,
    "ring": _codes.ring_code,
    "rotated_surface": _codes.rotated_surface_code,
    "unrotated_surface": _codes.unrotated_surface_code,
    "toric": _codes.toric_code,
    "heavy_hex": _codes.heavy_hex_code,
}

_DECODER_KINDS = (
    "union_find",
    "fast_union_find",
    "blossom",
    "sparse_blossom",
    "cpu_batch",
    "bp_osd",
)


@dataclass
class Job:
    """A queued/running/finished benchmark job."""

    job_id: str
    spec: dict
    status: str = "queued"  # queued | running | completed | cancelled | failed
    submitted_unix: float = 0.0
    started_unix: float | None = None
    finished_unix: float | None = None
    progress: float = 0.0  # 0..1
    units_done: int = 0
    units_total: int = 0
    error: str = ""
    artifact: dict | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "spec": self.spec,
            "submitted_unix": self.submitted_unix,
            "started_unix": self.started_unix,
            "finished_unix": self.finished_unix,
            "progress": self.progress,
            "units_done": self.units_done,
            "units_total": self.units_total,
            "error": self.error,
            "has_artifact": self.artifact is not None,
        }


class Workbench:
    """Controller for the QECTOR Workbench (load → benchmark → export)."""

    def __init__(self) -> None:
        self._loaded: dict | None = None  # last loaded stim/dem problem
        self._jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._worker: threading.Thread | None = None
        self._shutdown = False

    # ------------------------------------------------------------------ load
    def load_stim(self, source: Any) -> dict:
        """Load a Stim circuit from a ``.stim`` file path, text, or Circuit.

        Returns a descriptor (counts + a decomposed DEM) and stores it as the
        current problem so :meth:`run_benchmark` can compute a real LER.
        """
        import stim

        if hasattr(source, "detector_error_model"):  # a stim.Circuit
            circuit = source
        elif isinstance(source, str) and os.path.exists(source):
            with open(source, "r", encoding="utf-8") as fh:
                circuit = stim.Circuit(fh.read())
        elif isinstance(source, str):
            circuit = stim.Circuit(source)
        else:
            raise WorkbenchError(f"cannot load Stim circuit from {type(source).__name__}")

        try:
            sdem = circuit.detector_error_model(decompose_errors=True)
        except (ValueError, RuntimeError):
            sdem = circuit.detector_error_model()
        desc = {
            "kind": "stim",
            "num_qubits": int(circuit.num_qubits),
            "num_detectors": int(circuit.num_detectors),
            "num_observables": int(circuit.num_observables),
            "num_ticks": int(getattr(circuit, "num_ticks", 0)),
            "dem_num_errors": int(sdem.num_errors) if hasattr(sdem, "num_errors") else None,
        }
        self._loaded = {**desc, "_circuit": circuit, "_sdem": sdem}
        return desc

    def load_dem(self, source: Any) -> dict:
        """Load a Stim DEM from a ``.dem`` file path, text, or DemModel/DEM object.

        Returns a descriptor and stores the model as the current problem.
        """
        from . import dem as _dem

        if isinstance(source, _dem.DemModel):
            model = source
        elif isinstance(source, str) and os.path.exists(source):
            model = _dem.load_dem_file(source)
        elif isinstance(source, str):
            model = _dem.parse_dem(source)
        elif hasattr(source, "num_detectors"):  # stim.DetectorErrorModel
            model = _dem.from_stim(source)
        else:
            raise WorkbenchError(f"cannot load DEM from {type(source).__name__}")

        graphlike = model.is_graphlike
        collapsed = model.collapse_to_graph() if graphlike else model
        desc = {
            "kind": "dem",
            "num_detectors": int(model.num_detectors),
            "num_observables": int(model.num_observables),
            "num_errors": int(model.num_errors),
            "collapsed_edges": int(collapsed.num_errors),
            "graphlike": bool(graphlike),
        }
        self._loaded = {**desc, "_model": model}
        return desc

    @property
    def loaded(self) -> dict | None:
        if self._loaded is None:
            return None
        return {k: v for k, v in self._loaded.items() if not k.startswith("_")}

    # ------------------------------------------------------------- detection
    def detect_backends(self) -> dict:
        """Detect available decode backends and GPU device names."""
        import qector_decoder_v3 as qd

        info: dict[str, Any] = {
            "cpu": True,
            "cuda": bool(qd.cuda_is_available()),
            "opencl": bool(qd.opencl_is_available()),
            "cuda_device": None,
            "opencl_device": None,
        }
        try:
            info["cuda_device"] = qd.CUDABatchDecoder([[0]], 1).device_name
        except (ImportError, OSError, AttributeError, TypeError, ValueError, RuntimeError):
            pass
        if info["opencl"]:
            try:
                info["opencl_device"] = qd.OpenCLBatchDecoder([[0]], 1).device_name
            except (ImportError, OSError, AttributeError, TypeError, ValueError, RuntimeError):
                pass
        return info

    def environment_snapshot(self) -> dict:
        """Full reproducibility snapshot (env + git commit + backends)."""
        env = _bm.capture_environment()
        env["timestamp_unix"] = int(time.time())
        env["git_commit"] = _bm.git_commit()
        env["backends"] = self.detect_backends()
        return env

    def set_license_key(self, key: str) -> None:
        """Set the QECTOR license key for tier-gated features.

        Delegates to qector_decoder_v3.set_license_key().
        """
        from . import set_license_key as _slk

        _slk(key)

    def get_license_info(self) -> dict:
        """Return current license info dict, or {"tier": "Community", ...}."""
        from . import get_license_info as _gli

        try:
            return _gli()
        except Exception:
            return {"tier": "Community", "max_distance": "7"}

    # ------------------------------------------------------------- benchmark
    def run_benchmark(self, spec: dict, job: Job | None = None) -> dict:
        """Run a real decode benchmark (synchronously) and return an artifact.

        ``spec`` keys:
            code       : code family name (ignored when ``source="loaded"``)
            distances  : list[int]
            decoders   : list[str]
            trials     : int (latency iterations / LER shots)
            warmup     : int
            error_rate : float
            source     : "code" (default) or "loaded"
            throttle_seconds : float (optional inter-unit delay; for long-job UX)
        """
        source = spec.get("source", "code")
        decoders = list(spec.get("decoders", ["blossom"]))
        for k in decoders:
            if k not in _DECODER_KINDS:
                raise WorkbenchError(f"unknown decoder {k!r}; choose from {_DECODER_KINDS}")
        trials = int(spec.get("trials", 1000))
        warmup = int(spec.get("warmup", min(100, trials)))
        p = float(spec.get("error_rate", 0.05))
        throttle = float(spec.get("throttle_seconds", 0.0))

        problems = self._resolve_problems(spec, source)
        units_total = len(problems) * len(decoders)
        if job is not None:
            job.units_total = units_total

        results: list[dict] = []
        done = 0
        for code, ler_ctx in problems:
            for kind in decoders:
                if job is not None and job._cancel.is_set():
                    raise _Cancelled()
                rep = _bm.benchmark_decoder(
                    kind, code, n_trials=trials, warmup=warmup, p=p, seed=int(spec.get("seed", 1)), measure_memory=True
                )
                if ler_ctx is not None:
                    rep["logical_error_rate"] = self._ler(kind, code, ler_ctx, trials)
                results.append(rep)
                done += 1
                if job is not None:
                    job.units_done = done
                    job.progress = done / max(1, units_total)
                if throttle > 0:
                    # cancel-aware sleep so long jobs stop promptly
                    end = time.perf_counter() + throttle
                    while time.perf_counter() < end:
                        if job is not None and job._cancel.is_set():
                            raise _Cancelled()
                        time.sleep(min(0.02, throttle))

        artifact = {
            "environment": self.environment_snapshot(),
            "spec": spec,
            "results": results,
        }
        if job is not None:
            job.artifact = artifact
        # C6-03: record decode shots for metered billing
        try:
            from . import record_shots as _rs

            _rs(done * trials)
        except Exception:
            pass
        return artifact

    def _resolve_problems(self, spec, source):
        """Return a list of (Code, ler_context_or_None)."""
        if source == "loaded":
            if self._loaded is None:
                raise WorkbenchError("no problem loaded; call load_stim/load_dem first")
            if "_model" in self._loaded:  # DEM problem
                model = self._loaded["_model"]
                code = model.to_code()
                return [(code, None)]
            if "_sdem" in self._loaded:  # Stim problem
                from . import dem as _dem

                model = _dem.from_stim(self._loaded["_sdem"])
                code = (model.collapse_to_graph() if model.is_graphlike else model).to_code()
                return [(code, {"circuit": self._loaded["_circuit"]})]
            raise WorkbenchError("loaded problem is not benchmarkable")
        # generated codes
        fam = spec.get("code", "rotated_surface")
        if fam not in _CODE_FAMILIES:
            raise WorkbenchError(f"unknown code family {fam!r}; choose from {list(_CODE_FAMILIES)}")
        dists = spec.get("distances", [spec.get("distance", 5)])
        return [(_CODE_FAMILIES[fam](int(d)), None) for d in dists]

    #: Decoder kinds whose LER this method can honestly measure. ``Matching``
    #: in ``pymatching_compat`` is hard-wired to ``BlossomDecoder``
    #: (``self._decoder: BlossomDecoder``; ``from_detector_error_model`` takes no
    #: decoder argument), so it is the only decoder actually exercised here.
    _LER_MEASURABLE_KINDS = frozenset({"blossom"})

    def _ler(self, kind, code, ler_ctx, shots) -> float | None:
        """Real LER from Stim shots, for the decoder named by ``kind``.

        Returns ``None`` when the LER cannot be attributed to ``kind`` — a
        missing number, never a wrong one.

        This previously accepted ``kind`` and then ignored it: the body built
        ``pymatching_compat.Matching.from_detector_error_model(sdem)`` and
        decoded with that, and ``Matching`` only ever drives ``BlossomDecoder``.
        Every row therefore carried Blossom's LER while being labelled with
        whatever decoder had just been benchmarked. That is how a row could
        report a 5.0% logical error rate while 42.5% of that decoder's own
        corrections failed to reproduce their syndrome — the two columns were
        measuring different decoders.

        Until the Stim-shot path can drive an arbitrary decoder, only the kinds
        in :attr:`_LER_MEASURABLE_KINDS` yield a figure. Read
        ``syndrome_match_rate`` for the others: it *is* measured per decoder,
        and a correction that does not reproduce its syndrome is not a valid
        correction regardless of any logical-error figure.
        """
        if kind not in self._LER_MEASURABLE_KINDS:
            return None
        try:
            from . import pymatching_compat

            circuit = ler_ctx["circuit"]
            sdem = circuit.detector_error_model(decompose_errors=True)
            det, obs = circuit.compile_detector_sampler(seed=1).sample(shots=shots, separate_observables=True)
            det = det.astype(np.uint8)
            obs = obs.astype(np.uint8)
            m = pymatching_compat.Matching.from_detector_error_model(sdem)
            pred = np.asarray(m.decode_batch(det), np.uint8).reshape(shots, -1)
            return float(np.any(pred != obs, axis=1).mean())
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
            # Unsupported problem shape (e.g. a non-graphlike DEM) or Stim absent.
            return None

    # ------------------------------------------------------------- job queue
    def submit_job(self, spec: dict) -> str:
        """Queue a benchmark job; returns a job id. Runs on a FIFO worker."""
        job = Job(job_id=uuid.uuid4().hex[:12], spec=dict(spec), submitted_unix=time.time())
        with self._cv:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
            self._ensure_worker()
            self._cv.notify_all()
        return job.job_id

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise WorkbenchError(f"unknown job {job_id!r}")
            return job.to_dict()

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def job_artifact(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else job.artifact

    def cancel_job(self, job_id: str) -> str:
        """Cancel a queued or running job. Queued jobs cancel instantly."""
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                raise WorkbenchError(f"unknown job {job_id!r}")
            if job.status == "queued":
                if job_id in self._queue:
                    self._queue.remove(job_id)
                job.status = "cancelled"
                job.finished_unix = time.time()
            elif job.status == "running":
                job._cancel.set()
            return job.status

    def wait(self, job_id: str, timeout: float | None = None) -> dict:
        """Block until a job reaches a terminal state (or timeout)."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    raise WorkbenchError(f"unknown job {job_id!r}")
                if job.status in ("completed", "cancelled", "failed"):
                    return job.to_dict()
            if deadline is not None and time.perf_counter() > deadline:
                return self.get_job(job_id)
            time.sleep(0.01)

    def shutdown(self) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_worker, daemon=True)
            self._worker.start()

    def _run_worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._shutdown:
                    self._cv.wait(timeout=0.5)
                if self._shutdown and not self._queue:
                    return
                if not self._queue:
                    continue
                job_id = self._queue.pop(0)
                job = self._jobs[job_id]
                if job.status == "cancelled":
                    continue
                job.status = "running"
                job.started_unix = time.time()
            try:
                if job._cancel.is_set():
                    raise _Cancelled()
                self.run_benchmark(job.spec, job=job)
                with self._lock:
                    job.status = "completed"
                    job.finished_unix = time.time()
                    job.progress = 1.0
            except _Cancelled:
                with self._lock:
                    job.status = "cancelled"
                    job.finished_unix = time.time()
            except Exception as exc:  # noqa: BLE001 - a worker thread must never die silently
                with self._lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.finished_unix = time.time()

    # ------------------------------------------------------------- exporters
    def export_json(self, artifact: dict, path: str) -> str:
        _ensure_parent(path)
        _bm.write_json(path, artifact)
        return path

    def export_csv(self, artifact: dict, path: str) -> str:
        _ensure_parent(path)
        rows = artifact.get("results", [])
        cols = [
            "decoder",
            "code",
            "distance",
            "n_qubits",
            "n_checks",
            "physical_error_rate",
            "syndrome_faithful",
            "logical_error_rate",
            "lat_mean_us",
            "lat_p50_us",
            "lat_p99_us",
            "throughput_per_s",
            "peak_python_alloc_kib",
        ]
        lines = [",".join(cols)]
        for r in rows:
            lat = r.get("latency_us", {})
            lines.append(
                ",".join(
                    str(x)
                    for x in [
                        r.get("decoder", ""),
                        r.get("code", ""),
                        r.get("distance", ""),
                        r.get("n_qubits", ""),
                        r.get("n_checks", ""),
                        r.get("physical_error_rate", ""),
                        r.get("syndrome_faithful", ""),
                        _fmt(r.get("logical_error_rate")),
                        _fmt(lat.get("mean")),
                        _fmt(lat.get("p50")),
                        _fmt(lat.get("p99")),
                        _fmt(r.get("throughput_decodes_per_s")),
                        _fmt(r.get("peak_python_alloc_kib")),
                    ]
                )
            )
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    def export_pdf(self, artifact: dict, path: str) -> str:
        """Render a PDF report with charts built from the real artifact.

        Uses matplotlib's PdfPages (no reportlab needed). Raises a clear error if
        matplotlib is unavailable.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError as exc:  # pragma: no cover
            raise WorkbenchError(f"PDF export needs matplotlib: {exc}")

        _ensure_parent(path)
        rows = artifact.get("results", [])
        env = artifact.get("environment", {})
        with PdfPages(path) as pdf:
            # Title / environment page.
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.clf()
            txt = [
                "QECTOR Workbench - Benchmark Report",
                "",
                f"git commit : {env.get('git_commit')}",
                f"platform   : {env.get('platform')}",
                f"python     : {env.get('python_version')}",
                f"cuda/opencl: {env.get('cuda_available')}/{env.get('opencl_available')}",
                f"results    : {len(rows)} decoder runs",
                "",
            ]
            for r in rows[:30]:
                txt.append(
                    f"  {r.get('decoder'):16s} {r.get('code')!s:18s} "
                    f"faithful={r.get('syndrome_faithful')} "
                    f"p50={r.get('latency_us', {}).get('p50', 0):.2f}us"
                )
            fig.text(0.07, 0.95, "\n".join(txt), va="top", family="monospace", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)

            # Latency-by-decoder chart (real numbers).
            if rows:
                fig, ax = plt.subplots(figsize=(8.27, 5))
                labels = [f"{r.get('decoder')}\nd={r.get('distance')}" for r in rows]
                p50 = [r.get("latency_us", {}).get("p50", 0.0) for r in rows]
                p99 = [r.get("latency_us", {}).get("p99", 0.0) for r in rows]
                x = np.arange(len(rows))
                ax.bar(x - 0.2, p50, 0.4, label="p50 us")
                ax.bar(x + 0.2, p99, 0.4, label="p99 us")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
                ax.set_ylabel("latency (us)")
                ax.set_title("Decode latency per decoder/distance")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # LER-vs-distance chart when available.
            ler_rows = [r for r in rows if r.get("logical_error_rate") is not None]
            if len(ler_rows) >= 2:
                fig, ax = plt.subplots(figsize=(8.27, 5))
                ax.plot([r.get("distance") for r in ler_rows], [r.get("logical_error_rate") for r in ler_rows], "o-")
                ax.set_xlabel("distance")
                ax.set_ylabel("logical error rate")
                ax.set_yscale("log")
                ax.set_title("Logical error rate vs distance")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return path


class _Cancelled(Exception):
    pass


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Workbench CLI: set license key, show license info, or run a benchmark job."""
    import argparse

    parser = argparse.ArgumentParser(description="QECTOR Workbench CLI")
    parser.add_argument("--license-key", help="Set QECTOR license key before running")
    parser.add_argument(
        "command",
        nargs="?",
        default="info",
        choices=["info", "benchmark"],
        help="Command: 'info' (default) shows license info; 'benchmark' runs a benchmark",
    )
    args = parser.parse_args(argv)

    wb = Workbench()

    if args.license_key:
        wb.set_license_key(args.license_key)
        print("License key set")

    if args.command == "info":
        info = wb.get_license_info()
        print(f"Tier        : {info.get('tier', 'Community')}")
        print(f"Max distance: {info.get('max_distance', '7')}")
        print(f"Customer ID : {info.get('customer_id', 'free_tier')}")
        return 0

    if args.command == "benchmark":
        spec = {"code": "rotated_surface", "distances": [5], "decoders": ["blossom"], "trials": 1000}
        print(f"Running benchmark: {spec}")
        art = wb.run_benchmark(spec)
        print(f"Done. {len(art.get('results', []))} result(s)")
        return 0

    return 0
