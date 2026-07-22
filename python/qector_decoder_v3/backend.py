"""
qector_decoder_v3.backend — Workload-Aware AutoDecoder with Self-Auto-Debug & Multi-Tier Fallback.

Provides automatic hardware routing, real-time performance calibration, and a 
7-tier fault-tolerant self-debugging fallback engine for quantum error correction decoding.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
import os
from typing import List, Optional, Union, Dict, Any, cast

import numpy as np

from . import (
    BatchDecoder,
    CPUBatchDecoder,
    CUDABatchDecoder,
    FastUnionFindDecoder,
    UnionFindDecoder,
    BlossomDecoder,
    SparseBlossomDecoder,
    LookupTableDecoder,
    OpenCLBatchDecoder,
    cuda_is_available,
    opencl_is_available,
)

logger = logging.getLogger("qector_decoder_v3.backend")

__all__ = ["Backend", "BackendConfig", "AutoDecoder"]


class Backend:
    """Backend identifiers & Fallback Tiers."""

    CPU_SINGLE = "cpu_single"
    CPU_RAYON = "cpu_rayon"
    CPU_BATCH = "cpu_batch"
    CUDA = "cuda"
    OPENCL = "opencl"
    BLOSSOM = "blossom"
    SPARSE_BLOSSOM = "sparse_blossom"
    LOOKUP_TABLE = "lookup_table"

    ALL = (
        CPU_SINGLE,
        CPU_RAYON,
        CPU_BATCH,
        CUDA,
        OPENCL,
        BLOSSOM,
        SPARSE_BLOSSOM,
        LOOKUP_TABLE,
    )

    # Multi-tier fallback hierarchy (highest performance to highest compatibility)
    TIERS = [CUDA, OPENCL, CPU_RAYON, CPU_BATCH, CPU_SINGLE, BLOSSOM, LOOKUP_TABLE]


@dataclass
class BackendConfig:
    """Tunable thresholds and policy for :class:`AutoDecoder`.

    Attributes
    ----------
    rayon_threshold : int
        Batches at or above this size use the Rayon CPU path instead of the
        single-thread decoder.
    gpu_threshold : int
        Batches at or above this size use a GPU path (if available).
    allow_gpu : bool
        Master switch for GPU usage.
    prefer : str
        Preferred GPU backend when both are present: ``"cuda"`` or ``"opencl"``.
    force : str | None
        Force a specific backend for every call (one of :data:`Backend.ALL`),
        bypassing automatic selection.
    enable_auto_debug : bool
        Enable robust multi-tier self-debugging and automatic exception recovery.
    """

    rayon_threshold: int = 8
    gpu_threshold: int = 4096
    allow_gpu: bool = True
    prefer: str = Backend.CUDA
    force: Optional[str] = None
    enable_auto_debug: bool = True


@dataclass
class _Diag:
    last_backend: str = ""
    last_reason: str = ""
    calls: int = 0
    gpu_failures: int = 0
    gpu_fallbacks: int = 0
    cpu_fallbacks: int = 0
    calibrated: bool = False
    calibration: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    debug_log: List[Dict[str, Any]] = field(default_factory=list)
    backend_health: Dict[str, bool] = field(default_factory=dict)


class AutoDecoder:
    """Workload-aware, self-debugging decoder that routes to CUDA / OpenCL / Rayon / CPU / Fallback.

    Features automatic hardware discovery, adaptive runtime performance calibration,
    and a robust multi-tier self-debugging fallback engine that traps hardware or solver
    errors and seamlessly recovers execution without failing downstream callers.
    """

    def __init__(self, check_to_qubits, n_qubits=None, config: Optional[BackendConfig] = None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        self._c2q = [[int(q) for q in check] for check in check_to_qubits]
        self._nq = None if n_qubits is None else int(n_qubits)
        self.config = config or BackendConfig()
        self._diag = _Diag()

        # Lazy decoder instances
        self._decoders: Dict[str, Any] = {}

        self._cuda_ok = bool(self.config.allow_gpu and cuda_is_available())
        opencl_auto = os.environ.get("QECTOR_ENABLE_OPENCL_AUTO", "").lower() in {"1", "true", "yes", "on"}
        self._opencl_ok = bool(self.config.allow_gpu and opencl_auto and opencl_is_available())
        
        if self.config.allow_gpu and not opencl_auto and opencl_is_available():
            self._diag.warnings.append("OpenCL auto-routing disabled; set QECTOR_ENABLE_OPENCL_AUTO=1 to enable it")

        # Initialize health status
        for b in Backend.ALL:
            self._diag.backend_health[b] = True

    # -- availability ------------------------------------------------------
    def available_backends(self) -> List[str]:
        avail = [Backend.CPU_SINGLE, Backend.CPU_RAYON, Backend.CPU_BATCH]
        if self._cuda_ok and self._diag.backend_health.get(Backend.CUDA, True):
            avail.append(Backend.CUDA)
        if self._opencl_ok and self._diag.backend_health.get(Backend.OPENCL, True):
            avail.append(Backend.OPENCL)
        avail.extend([Backend.BLOSSOM, Backend.SPARSE_BLOSSOM, Backend.LOOKUP_TABLE])
        return avail

    @staticmethod
    def auto_select(checks, n_qubits, batch_size=1):
        """Smart backend selection based on code properties and batch size.

        Args:
            checks: ``check_to_qubits`` list.
            n_qubits: Number of data qubits.
            batch_size: Number of syndromes to decode at once.

        Returns:
            String backend name: ``"lookup_table"``, ``"cpu_batch"``,
            ``"union_find"``, or ``"blossom"``.
        """
        n_det = len(checks)
        if n_det <= 16 and batch_size == 1:
            return "lookup_table"
        if batch_size >= 64:
            return "cpu_batch"
        if n_det <= 8:
            return "union_find"
        return "blossom"

    # -- lazy builders -----------------------------------------------------
    def _get_decoder(self, backend: str) -> Any:
        if backend in self._decoders:
            return self._decoders[backend]

        dec = None
        if backend == Backend.CPU_SINGLE:
            dec = FastUnionFindDecoder(self._c2q, self._nq)
        elif backend == Backend.CPU_RAYON:
            dec = BatchDecoder(self._c2q, self._nq)
        elif backend == Backend.CPU_BATCH:
            dec = CPUBatchDecoder(self._c2q, self._nq)
        elif backend == Backend.CUDA:
            if self._cuda_ok:
                try:
                    dec = CUDABatchDecoder(self._c2q, self._nq)
                except Exception as exc:
                    self._cuda_ok = False
                    self._diag.backend_health[Backend.CUDA] = False
                    self._diag.warnings.append(f"CUDA init failed: {exc}")
        elif backend == Backend.OPENCL:
            if self._opencl_ok:
                try:
                    dec = OpenCLBatchDecoder(self._c2q, self._nq)
                except Exception as exc:
                    self._opencl_ok = False
                    self._diag.backend_health[Backend.OPENCL] = False
                    self._diag.warnings.append(f"OpenCL init failed: {exc}")
        elif backend == Backend.BLOSSOM:
            dec = BlossomDecoder(self._c2q, self._nq)
        elif backend == Backend.SPARSE_BLOSSOM:
            dec = SparseBlossomDecoder(self._c2q, self._nq)
        elif backend == Backend.LOOKUP_TABLE:
            try:
                dec = LookupTableDecoder(self._c2q, self._nq)
            except Exception:
                dec = FastUnionFindDecoder(self._c2q, self._nq)

        if dec is not None:
            self._decoders[backend] = dec
        return dec

    def _get_cpu_single(self) -> FastUnionFindDecoder:
        return self._get_decoder(Backend.CPU_SINGLE)

    def _get_cpu_rayon(self) -> BatchDecoder:
        return self._get_decoder(Backend.CPU_RAYON)

    def _get_cuda(self) -> Optional[CUDABatchDecoder]:
        return self._get_decoder(Backend.CUDA)

    def _get_opencl(self) -> Optional[OpenCLBatchDecoder]:
        return self._get_decoder(Backend.OPENCL)

    # -- selection ---------------------------------------------------------
    def select(self, batch_size: int) -> str:
        """Return the backend that *would* run for a given batch size."""
        if self.config.force is not None:
            return self.config.force

        if self.config.allow_gpu and batch_size >= self.config.gpu_threshold:
            if self.config.prefer == Backend.OPENCL and self._opencl_ok and self._diag.backend_health.get(Backend.OPENCL, True):
                return Backend.OPENCL
            if self._cuda_ok and self._diag.backend_health.get(Backend.CUDA, True):
                return Backend.CUDA
            if self._opencl_ok and self._diag.backend_health.get(Backend.OPENCL, True):
                return Backend.OPENCL

        if batch_size >= self.config.rayon_threshold and self._diag.backend_health.get(Backend.CPU_RAYON, True):
            return Backend.CPU_RAYON

        return Backend.CPU_SINGLE

    # -- robust self auto debugging decoding ------------------------------
    def decode(self, syndrome) -> np.ndarray:
        """Decode a single syndrome with auto-debug error recovery."""
        s = _as_u8_1d(syndrome)
        self._diag.calls += 1

        if not self.config.enable_auto_debug:
            self._diag.last_backend = Backend.CPU_SINGLE
            self._diag.last_reason = "single syndrome"
            return cast(np.ndarray, self._get_cpu_single().decode(s))

        # Self auto-debug fallback chain for single decode
        primary = Backend.CPU_SINGLE
        tiers = [primary, Backend.BLOSSOM, Backend.SPARSE_BLOSSOM, Backend.LOOKUP_TABLE]

        for b in tiers:
            try:
                dec = self._get_decoder(b)
                if dec is None:
                    continue
                out = np.asarray(dec.decode(s), dtype=np.uint8)
                self._diag.last_backend = b
                self._diag.last_reason = f"single syndrome (tier={b})"
                return out
            except Exception as exc:
                self._record_auto_debug_failure(b, exc, "single_decode")

        # Ultimate pure python fallback
        return self._python_fallback_decode(s)

    def batch_decode(self, syndromes) -> np.ndarray:
        """Decode a 2-D batch, routing to the best available backend with multi-tier fallback."""
        syn = _as_u8_2d(syndromes)
        n = syn.shape[0]
        self._diag.calls += 1
        chosen = self.select(n)

        if not self.config.enable_auto_debug:
            return self._batch_decode_direct(chosen, syn)

        # Robust multi-tier self-debugging execution chain
        fallback_chain = [chosen]
        for t in [Backend.CUDA, Backend.OPENCL, Backend.CPU_RAYON, Backend.CPU_BATCH, Backend.CPU_SINGLE, Backend.BLOSSOM]:
            if t not in fallback_chain:
                fallback_chain.append(t)

        for b in fallback_chain:
            if not self._diag.backend_health.get(b, True) and b in (Backend.CUDA, Backend.OPENCL):
                continue

            try:
                out = self._exec_backend_batch(b, syn)
                if out is not None:
                    self._diag.last_backend = b
                    self._diag.last_reason = f"batch={n} routing={b}"
                    return out
            except Exception as exc:
                self._record_auto_debug_failure(b, exc, f"batch_decode(n={n})")
                if b in (Backend.CUDA, Backend.OPENCL):
                    self._diag.gpu_failures += 1
                    self._diag.gpu_fallbacks += 1
                else:
                    self._diag.cpu_fallbacks += 1

        # Ultimate fallback across rows
        single = self._get_cpu_single()
        return np.stack([np.asarray(single.decode(syn[i]), dtype=np.uint8) for i in range(n)])

    def _batch_decode_direct(self, chosen: str, syn: np.ndarray) -> np.ndarray:
        if chosen in (Backend.CUDA, Backend.OPENCL):
            out = self._run_gpu(chosen, syn)
            if out is not None:
                return out
            chosen = Backend.CPU_RAYON

        if chosen == Backend.CPU_RAYON:
            self._diag.last_backend = Backend.CPU_RAYON
            self._diag.last_reason = f"batch={syn.shape[0]} >= rayon_threshold"
            return np.asarray(self._get_cpu_rayon().parallel_batch_decode(syn))

        self._diag.last_backend = Backend.CPU_SINGLE
        self._diag.last_reason = f"batch={syn.shape[0]} below thresholds"
        single = self._get_cpu_single()
        return np.stack([np.asarray(single.decode(syn[i])) for i in range(syn.shape[0])])

    def _exec_backend_batch(self, b: str, syn: np.ndarray) -> Optional[np.ndarray]:
        dec = self._get_decoder(b)
        if dec is None:
            return None

        if hasattr(dec, "parallel_batch_decode"):
            return np.asarray(dec.parallel_batch_decode(syn), dtype=np.uint8)
        elif hasattr(dec, "batch_decode"):
            return np.asarray(dec.batch_decode(syn), dtype=np.uint8)
        elif hasattr(dec, "decode"):
            return np.stack([np.asarray(dec.decode(syn[i]), dtype=np.uint8) for i in range(syn.shape[0])])
        return None

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
        except Exception as exc:
            self._diag.gpu_failures += 1
            self._diag.gpu_fallbacks += 1
            self._diag.warnings.append(f"{which} decode failed ({exc}); CPU fallback")
            return None

    def _record_auto_debug_failure(self, backend: str, exc: Exception, context: str) -> None:
        self._diag.backend_health[backend] = False
        msg = f"AutoDebug caught error on {backend} [{context}]: {exc}"
        self._diag.warnings.append(msg)
        self._diag.debug_log.append({
            "timestamp": time.time(),
            "backend": backend,
            "context": context,
            "error": str(exc),
        })
        logger.warning(msg)

    def _python_fallback_decode(self, syndrome: np.ndarray) -> np.ndarray:
        nq = int(self._get_cpu_single().n_qubits)
        out = np.zeros(nq, dtype=np.uint8)
        for i, check in enumerate(self._c2q):
            if i < len(syndrome) and syndrome[i]:
                for q in check:
                    if q < nq:
                        out[q] ^= 1
        return out

    def reset_backend_health(self) -> None:
        """Resets all backend health states back to active."""
        for b in Backend.ALL:
            self._diag.backend_health[b] = True
        self._cuda_ok = bool(self.config.allow_gpu and cuda_is_available())
        opencl_auto = os.environ.get("QECTOR_ENABLE_OPENCL_AUTO", "").lower() in {"1", "true", "yes", "on"}
        self._opencl_ok = bool(self.config.allow_gpu and opencl_auto and opencl_is_available())

    # -- calibration -------------------------------------------------------
    def calibrate(self, sizes=(64, 256, 1024, 4096, 16384), repeats: int = 3, seed: int = 0):
        """Measure CPU/GPU crossover and tune parameters automatically."""
        rng = np.random.default_rng(seed)
        n_checks = len(self._c2q)
        timings: dict[str, dict[int, float]] = {"cpu": {}, "gpu": {}}
        gpu_name = None
        gpu_dec: Optional[Union[CUDABatchDecoder, OpenCLBatchDecoder]] = None
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
                except Exception as exc:
                    self._diag.warnings.append(f"calibration {gpu_name} failed: {exc}")
                    gpu_dec = None

        if gpu_dec is None:
            self._diag.warnings.append("no GPU available for calibration; GPU disabled")
            self.config.gpu_threshold = 1 << 62
        elif crossover is None:
            self._diag.warnings.append("GPU never beat CPU in calibration; keeping CPU for all sizes")
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
                "enable_auto_debug": self.config.enable_auto_debug,
            },
            "last_backend": self._diag.last_backend,
            "last_reason": self._diag.last_reason,
            "calls": self._diag.calls,
            "gpu_failures": self._diag.gpu_failures,
            "gpu_fallbacks": self._diag.gpu_fallbacks,
            "cpu_fallbacks": self._diag.cpu_fallbacks,
            "calibrated": self._diag.calibrated,
            "calibration": self._diag.calibration,
            "backend_health": dict(self._diag.backend_health),
            "warnings": list(self._diag.warnings),
            "debug_log": list(self._diag.debug_log),
        }

    @property
    def last_backend(self) -> str:
        return self._diag.last_backend

    @property
    def n_qubits(self) -> int:
        return int(self._get_cpu_single().n_qubits)

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
    return np.ascontiguousarray(s, dtype=np.uint8)


def _best_time(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best
