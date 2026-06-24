"""
qector_decoder_v3.backend — Automatic backend selection.

QECTOR ships several execution paths with different fixed costs and crossover
points: a single-thread CPU decoder (lowest latency for tiny batches), a Rayon
data-parallel CPU decoder (best for medium batches), and CUDA / OpenCL GPU batch
decoders (best for large batches, when a device is present and healthy).

:class:`AutoDecoder` picks the right one per call from the batch size, the
available hardware, and — optionally — a one-off **runtime calibration** that
measures the real crossover on this machine.  It degrades gracefully: if a GPU
path raises, it falls back to CPU and records why.  Everything is overridable.

Example
-------
>>> from qector_decoder_v3 import codes
>>> from qector_decoder_v3.backend import AutoDecoder
>>> code = codes.rotated_surface_code(7)
>>> dec = AutoDecoder(code.check_to_qubits, code.n_qubits)
>>> dec.calibrate()                      # optional: tune the GPU threshold
>>> out = dec.batch_decode(syndromes)    # picks CPU / Rayon / GPU automatically
>>> dec.diagnostics()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import (
    BatchDecoder,
    CPUBatchDecoder,
    CUDABatchDecoder,
    FastUnionFindDecoder,
    OpenCLBatchDecoder,
    cuda_is_available,
    opencl_is_available,
)

__all__ = ["Backend", "BackendConfig", "AutoDecoder"]


class Backend:
    """Backend identifiers."""

    CPU_SINGLE = "cpu_single"
    CPU_RAYON = "cpu_rayon"
    CUDA = "cuda"
    OPENCL = "opencl"
    ALL = (CPU_SINGLE, CPU_RAYON, CUDA, OPENCL)


@dataclass
class BackendConfig:
    """Tunable thresholds and policy for :class:`AutoDecoder`.

    Attributes
    ----------
    rayon_threshold : int
        Batches at or above this size use the Rayon CPU path instead of the
        single-thread decoder.
    gpu_threshold : int
        Batches at or above this size use a GPU path (if available).  Set by
        :meth:`AutoDecoder.calibrate`.
    allow_gpu : bool
        Master switch for GPU usage.
    prefer : str
        Preferred GPU backend when both are present: ``"cuda"`` or ``"opencl"``.
    force : str | None
        Force a specific backend for every call (one of :data:`Backend.ALL`),
        bypassing automatic selection.
    """

    rayon_threshold: int = 8
    gpu_threshold: int = 4096
    allow_gpu: bool = True
    prefer: str = Backend.CUDA
    force: Optional[str] = None


@dataclass
class _Diag:
    last_backend: str = ""
    last_reason: str = ""
    calls: int = 0
    gpu_failures: int = 0
    gpu_fallbacks: int = 0
    calibrated: bool = False
    calibration: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class AutoDecoder:
    """Workload-aware decoder that routes to CPU / Rayon / CUDA / OpenCL.

    Same surface as the other decoders: ``decode`` for one syndrome and
    ``batch_decode`` for a 2-D array.  Decoders for each backend are created
    lazily on first use so unavailable hardware costs nothing.
    """

    def __init__(self, check_to_qubits, n_qubits=None, config: Optional[BackendConfig] = None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        self._c2q = [[int(q) for q in check] for check in check_to_qubits]
        self._nq = None if n_qubits is None else int(n_qubits)
        self.config = config or BackendConfig()
        self._diag = _Diag()

        # Lazy decoder slots.
        self._cpu_single: Optional[FastUnionFindDecoder] = None
        self._cpu_rayon: Optional[BatchDecoder] = None
        self._cpu_batch: Optional[CPUBatchDecoder] = None
        self._cuda: Optional[CUDABatchDecoder] = None
        self._opencl: Optional[OpenCLBatchDecoder] = None

        self._cuda_ok = bool(self.config.allow_gpu and cuda_is_available())
        self._opencl_ok = bool(self.config.allow_gpu and opencl_is_available())

    # -- availability ------------------------------------------------------
    def available_backends(self) -> List[str]:
        avail = [Backend.CPU_SINGLE, Backend.CPU_RAYON]
        if self._cuda_ok:
            avail.append(Backend.CUDA)
        if self._opencl_ok:
            avail.append(Backend.OPENCL)
        return avail

    # -- lazy builders -----------------------------------------------------
    def _get_cpu_single(self) -> FastUnionFindDecoder:
        if self._cpu_single is None:
            self._cpu_single = FastUnionFindDecoder(self._c2q, self._nq)
        return self._cpu_single

    def _get_cpu_rayon(self) -> BatchDecoder:
        if self._cpu_rayon is None:
            self._cpu_rayon = BatchDecoder(self._c2q, self._nq)
        return self._cpu_rayon

    def _get_cuda(self) -> Optional[CUDABatchDecoder]:
        if not self._cuda_ok:
            return None
        if self._cuda is None:
            try:
                self._cuda = CUDABatchDecoder(self._c2q, self._nq)
            except Exception as exc:  # pragma: no cover - hardware dependent
                self._cuda_ok = False
                self._diag.warnings.append(f"CUDA init failed: {exc}")
                return None
        return self._cuda

    def _get_opencl(self) -> Optional[OpenCLBatchDecoder]:
        if not self._opencl_ok:
            return None
        if self._opencl is None:
            try:
                self._opencl = OpenCLBatchDecoder(self._c2q, self._nq)
            except Exception as exc:  # pragma: no cover - hardware dependent
                self._opencl_ok = False
                self._diag.warnings.append(f"OpenCL init failed: {exc}")
                return None
        return self._opencl

    # -- selection ---------------------------------------------------------
    def select(self, batch_size: int) -> str:
        """Return the backend that *would* run for a given batch size."""
        if self.config.force is not None:
            return self.config.force
        if (
            self.config.allow_gpu
            and batch_size >= self.config.gpu_threshold
            and (self._cuda_ok or self._opencl_ok)
        ):
            if self.config.prefer == Backend.OPENCL and self._opencl_ok:
                return Backend.OPENCL
            if self._cuda_ok:
                return Backend.CUDA
            return Backend.OPENCL
        if batch_size >= self.config.rayon_threshold:
            return Backend.CPU_RAYON
        return Backend.CPU_SINGLE

    # -- decoding ----------------------------------------------------------
    def decode(self, syndrome) -> np.ndarray:
        """Decode a single syndrome (always the single-thread CPU path)."""
        s = _as_u8_1d(syndrome)
        self._diag.calls += 1
        self._diag.last_backend = Backend.CPU_SINGLE
        self._diag.last_reason = "single syndrome"
        return self._get_cpu_single().decode(s)

    def batch_decode(self, syndromes) -> np.ndarray:
        """Decode a 2-D batch, routing to the best available backend."""
        syn = _as_u8_2d(syndromes)
        n = syn.shape[0]
        self._diag.calls += 1
        chosen = self.select(n)

        if chosen in (Backend.CUDA, Backend.OPENCL):
            out = self._run_gpu(chosen, syn)
            if out is not None:
                return out
            chosen = Backend.CPU_RAYON  # fell back

        if chosen == Backend.CPU_RAYON:
            self._diag.last_backend = Backend.CPU_RAYON
            self._diag.last_reason = f"batch={n} >= rayon_threshold"
            return np.asarray(self._get_cpu_rayon().parallel_batch_decode(syn))

        self._diag.last_backend = Backend.CPU_SINGLE
        self._diag.last_reason = f"batch={n} below thresholds"
        single = self._get_cpu_single()
        return np.stack([np.asarray(single.decode(syn[i])) for i in range(n)])

    def _run_gpu(self, which: str, syn: np.ndarray) -> Optional[np.ndarray]:
        dec = self._get_cuda() if which == Backend.CUDA else self._get_opencl()
        if dec is None:
            self._diag.gpu_fallbacks += 1
            self._diag.warnings.append(f"{which} unavailable; falling back to CPU")
            return None
        try:
            out = np.asarray(dec.batch_decode(syn))
            if getattr(dec, "is_degraded", False):
                self._diag.warnings.append(f"{which} reported degraded mode")
            self._diag.last_backend = which
            self._diag.last_reason = f"batch={syn.shape[0]} >= gpu_threshold"
            return out
        except Exception as exc:  # pragma: no cover - hardware dependent
            self._diag.gpu_failures += 1
            self._diag.gpu_fallbacks += 1
            self._diag.warnings.append(f"{which} decode failed ({exc}); CPU fallback")
            return None

    # -- calibration -------------------------------------------------------
    def calibrate(self, sizes=(64, 256, 1024, 4096, 16384), repeats: int = 3, seed: int = 0):
        """Measure the CPU/GPU crossover on this machine and set ``gpu_threshold``.

        Times the Rayon CPU path against the fastest available GPU path on random
        batches and sets ``config.gpu_threshold`` to the smallest size where the
        GPU is faster.  Emits a performance warning when the GPU never wins.
        """
        rng = np.random.default_rng(seed)
        n_checks = len(self._c2q)
        timings = {"cpu": {}, "gpu": {}}
        gpu_name = None
        gpu_dec = None
        if self._cuda_ok:
            gpu_dec, gpu_name = self._get_cuda(), Backend.CUDA
        elif self._opencl_ok:
            gpu_dec, gpu_name = self._get_opencl(), Backend.OPENCL

        cpu = self._get_cpu_rayon()
        crossover = None
        for n in sizes:
            syn = (rng.random((n, n_checks)) < 0.08).astype(np.uint8)
            cpu_t = _best_time(lambda: cpu.parallel_batch_decode(syn), repeats)
            timings["cpu"][n] = cpu_t
            if gpu_dec is not None:
                try:
                    gpu_t = _best_time(lambda: gpu_dec.batch_decode(syn), repeats)
                    timings["gpu"][n] = gpu_t
                    if crossover is None and gpu_t < cpu_t:
                        crossover = n
                except Exception as exc:  # pragma: no cover
                    self._diag.warnings.append(f"calibration {gpu_name} failed: {exc}")
                    gpu_dec = None

        if gpu_dec is None:
            self._diag.warnings.append("no GPU available for calibration; GPU disabled")
            self.config.gpu_threshold = 1 << 62
        elif crossover is None:
            self._diag.warnings.append(
                "GPU never beat CPU in calibration; keeping CPU for all sizes"
            )
            self.config.gpu_threshold = 1 << 62
        else:
            self.config.gpu_threshold = int(crossover)

        self._diag.calibrated = True
        self._diag.calibration = {
            "gpu_backend": gpu_name,
            "crossover": crossover,
            "gpu_threshold": self.config.gpu_threshold,
            "timings": timings,
        }
        return self._diag.calibration

    # -- diagnostics -------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "available_backends": self.available_backends(),
            "config": {
                "rayon_threshold": self.config.rayon_threshold,
                "gpu_threshold": self.config.gpu_threshold,
                "allow_gpu": self.config.allow_gpu,
                "prefer": self.config.prefer,
                "force": self.config.force,
            },
            "last_backend": self._diag.last_backend,
            "last_reason": self._diag.last_reason,
            "calls": self._diag.calls,
            "gpu_failures": self._diag.gpu_failures,
            "gpu_fallbacks": self._diag.gpu_fallbacks,
            "calibrated": self._diag.calibrated,
            "calibration": self._diag.calibration,
            "warnings": list(self._diag.warnings),
        }

    @property
    def last_backend(self) -> str:
        return self._diag.last_backend

    @property
    def n_qubits(self) -> int:
        return self._get_cpu_single().n_qubits

    @property
    def n_checks(self) -> int:
        return len(self._c2q)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_u8_1d(syndrome) -> np.ndarray:
    s = syndrome if isinstance(syndrome, np.ndarray) else np.asarray(syndrome, dtype=np.uint8)
    if s.dtype != np.uint8:
        s = s.astype(np.uint8)
    return s.reshape(-1)


def _as_u8_2d(syndromes) -> np.ndarray:
    s = syndromes if isinstance(syndromes, np.ndarray) else np.asarray(syndromes, dtype=np.uint8)
    if s.dtype != np.uint8:
        s = s.astype(np.uint8)
    if s.ndim != 2:
        raise ValueError(f"syndromes must be 2D, got shape {s.shape}")
    # Force C-contiguity: the Rust/GPU batch decoders read the buffer row-major, so
    # a Fortran-ordered or non-contiguous batch would otherwise decode incorrectly.
    return np.ascontiguousarray(s, dtype=np.uint8)


def _best_time(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best
