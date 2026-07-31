"""
qector_decoder_v3.gpu_backend - the CuPy GPU backend abstraction.

This is the single, policy-aware foundation that every other GPU-aware module in
the package builds on. It answers three questions uniformly:

* *Is a GPU usable here?* - :func:`has_cupy`, :func:`has_cuda_rust`,
  :func:`gpu_available`.
* *Which array library should I compute with right now?* -
  :func:`get_array_module` / :func:`xp` (NumPy-or-CuPy, honouring both a global
  preference and the location of the data you pass in).
* *How do I move data across the host/device boundary, and how much am I moving?*
  - :func:`to_device`, :func:`to_host` / :func:`asnumpy`, :func:`is_on_gpu`, and
  the module-level :data:`TELEMETRY` counters.

CuPy is **optional**. If it is not installed, or no usable CUDA device is present,
every function degrades to NumPy on the host with no exceptions raised - so code
written against this module runs unchanged on a CPU-only box. When CuPy *is*
present and a device is usable (e.g. cupy 14.x on this machine's GTX 1660 Ti),
the GPU path is taken by default.

The Rust ``CUDABatchDecoder`` is a separate, build-time CUDA path; this module
does not depend on it but reports its availability via :func:`has_cuda_rust` so
callers can reason about both GPU paths from one place.

Example
-------
>>> from qector_decoder_v3 import gpu_backend as gb
>>> xp = gb.get_array_module(prefer_gpu=True)  # cupy if usable, else numpy
>>> dev = gb.to_device([1, 0, 1, 1])  # host -> device (counted)
>>> gb.is_on_gpu(dev)  # True iff really on the GPU
>>> host = gb.to_host(dev)  # device -> host (counted)
>>> gb.get_backend().summary()  # one-glance capability report
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import numpy as np

__all__ = [
    "TELEMETRY",
    "GpuBackend",
    "asnumpy",
    "get_array_module",
    "get_backend",
    "get_prefer_gpu",
    "gpu_available",
    "has_cuda_rust",
    "has_cupy",
    "is_on_gpu",
    "note_fallback",
    "note_gpu_call",
    "reset_telemetry",
    "set_prefer_gpu",
    "to_device",
    "to_host",
    "xp",
]


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
#: Global preference for using the GPU when one is available. Toggle with
#: :func:`set_prefer_gpu`. Detection (:func:`gpu_available`) is unaffected by
#: this flag - it reports hardware, not policy.
_PREFER_GPU: bool = True

#: Cached :class:`GpuBackend` capability snapshot (built lazily, hardware is
#: assumed stable for the life of the process). Reset is not normally needed.
_BACKEND: GpuBackend | None = None

#: Cross-module transfer/usage counters. Other modules read and mutate this in
#: place; it is reset (never rebound) so external references stay valid.
#:
#: * ``h2d``       - host -> device array transfers performed by :func:`to_device`.
#: * ``d2h``       - device -> host transfers performed by :func:`to_host`.
#: * ``gpu_calls`` - GPU compute operations, recorded by callers via
#:   :func:`note_gpu_call`.
#: * ``fallbacks`` - times a requested GPU action degraded to the CPU/host path.
TELEMETRY: dict[str, int] = {"h2d": 0, "d2h": 0, "gpu_calls": 0, "fallbacks": 0}


# ---------------------------------------------------------------------------
# CuPy / device probing (cached - hardware does not change mid-process)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _cupy() -> ModuleType | None:
    """Return the imported ``cupy`` module, or ``None`` if it is unavailable."""
    try:
        import cupy  # type: ignore[import-not-found]

        return cast(ModuleType, cupy)
    except (ImportError, RuntimeError):  # cupy absent, or its own CUDA stack broken
        return None


@functools.lru_cache(maxsize=1)
def _probe() -> tuple[bool, str | None, int | None]:
    """Probe device 0 once. Returns ``(usable, device_name, total_mem_bytes)``.

    "Usable" means CuPy imported, a CUDA device is enumerated, and a tiny
    allocation on it succeeds - so a present-but-broken driver reports unusable
    rather than crashing later inside a hot loop.
    """
    cp = _cupy()
    if cp is None:
        return (False, None, None)
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return (False, None, None)
        dev = cp.cuda.Device(0)
        dev.use()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name") if isinstance(props, dict) else None
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        total = int(cp.cuda.runtime.memGetInfo()[1])
        # Confirm we can actually allocate on the device before declaring success.
        _scratch = cp.zeros(8, dtype=cp.uint8)
        del _scratch
        return (True, name, total)
    except (RuntimeError, OSError, ValueError, AttributeError):  # pragma: no cover
        # A probe must degrade to "unusable", never crash the caller — these
        # cover driver/runtime failures from cupy (CUDARuntimeError is a
        # RuntimeError), missing device libs (OSError), and API drift.
        return (False, None, None)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def has_cupy() -> bool:
    """Return ``True`` if the ``cupy`` module can be imported."""
    return _cupy() is not None


def has_cuda_rust() -> bool:
    """Return ``True`` iff the Rust ``CUDABatchDecoder`` reports a usable device.

    This is the *build-time* CUDA path baked into the compiled core, distinct
    from the CuPy runtime path. ``False`` when the crate was built without the
    ``cuda`` feature or no driver device is present.
    """
    try:
        from . import CUDABatchDecoder
    except (ImportError, AttributeError):  # pragma: no cover - import wiring
        return False
    try:
        return bool(CUDABatchDecoder.is_available())
    except (OSError, AttributeError, RuntimeError):  # pragma: no cover - hardware dependent
        return False


def gpu_available() -> bool:
    """Return ``True`` iff CuPy is importable **and** a CUDA device is usable.

    Independent of the :func:`set_prefer_gpu` policy flag - it reports hardware
    capability only.
    """
    return _probe()[0]


# ---------------------------------------------------------------------------
# Array-module dispatch
# ---------------------------------------------------------------------------
def _resolve_module(prefer_gpu: bool) -> ModuleType:
    """Pick CuPy vs NumPy from policy alone (no data inspection)."""
    if prefer_gpu and _PREFER_GPU and gpu_available():
        cp = _cupy()
        if cp is not None:
            return cp
    return np


def get_array_module(*arrays: Any, prefer_gpu: bool = True) -> ModuleType:
    """Return the array module (``cupy`` or ``numpy``) to compute with.

    Policy-aware analogue of ``cupy.get_array_module``:

    * If **any** input array already lives on the GPU, ``cupy`` is returned
      regardless of ``prefer_gpu`` - you cannot operate on device arrays with
      NumPy, so the data location wins.
    * Otherwise ``cupy`` is returned only when ``prefer_gpu`` is true, the global
      preference (:func:`set_prefer_gpu`) is on, and a device is usable; else
      ``numpy``.

    Parameters
    ----------
    *arrays:
        Zero or more array-likes whose device residency steers the choice.
    prefer_gpu:
        Per-call request for the GPU. Defaults to ``True``.
    """
    cp = _cupy()
    if cp is not None:
        for a in arrays:
            if is_on_gpu(a):
                return cp
    return _resolve_module(prefer_gpu)


def xp(prefer_gpu: bool = True) -> ModuleType:
    """Alias for :func:`get_array_module` with no array arguments.

    Idiomatic shorthand for ``xp = gpu_backend.xp(); a = xp.zeros(...)``.
    """
    return get_array_module(prefer_gpu=prefer_gpu)


# ---------------------------------------------------------------------------
# Host/device transfer
# ---------------------------------------------------------------------------
def is_on_gpu(a: Any) -> bool:
    """Return ``True`` iff ``a`` is a CuPy device array."""
    cp = _cupy()
    if cp is None:
        return False
    return isinstance(a, cp.ndarray)


def to_device(a: Any) -> Any:
    """Move ``a`` onto the GPU when policy and hardware allow; else keep on host.

    * Already-on-GPU input is returned unchanged.
    * A host transfer is performed (and counted in ``TELEMETRY['h2d']``) only
      when a device is usable and the global preference is on.
    * When the GPU is unavailable or disabled, a host ``numpy`` array is returned
      and ``TELEMETRY['fallbacks']`` is incremented; a device array asked to stay
      on host in this mode is brought back via :func:`to_host`.
    """
    cp = _cupy()
    if gpu_available() and _PREFER_GPU and cp is not None:
        if is_on_gpu(a):
            return a
        dev = cp.asarray(a)
        TELEMETRY["h2d"] += 1
        return dev
    # Host / fallback path.
    if is_on_gpu(a):
        return to_host(a)
    TELEMETRY["fallbacks"] += 1
    return np.asarray(a)


def to_host(a: Any) -> np.ndarray:
    """Return ``a`` as a host NumPy array, copying off the GPU if necessary.

    A device -> host copy is counted in ``TELEMETRY['d2h']``.
    """
    if is_on_gpu(a):
        cp = _cupy()
        assert cp is not None  # is_on_gpu implies cupy imported
        host = cp.asnumpy(a)
        TELEMETRY["d2h"] += 1
        return cast(np.ndarray, host)
    return np.asarray(a)


def asnumpy(a: Any) -> np.ndarray:
    """Alias for :func:`to_host` - mirrors ``cupy.asnumpy``."""
    return to_host(a)


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------
def note_gpu_call(n: int = 1) -> None:
    """Record ``n`` GPU compute operations in ``TELEMETRY['gpu_calls']``."""
    TELEMETRY["gpu_calls"] += int(n)


def note_fallback(n: int = 1) -> None:
    """Record ``n`` GPU->CPU fallbacks in ``TELEMETRY['fallbacks']``."""
    TELEMETRY["fallbacks"] += int(n)


def reset_telemetry() -> None:
    """Zero every :data:`TELEMETRY` counter in place (references stay valid)."""
    for key in TELEMETRY:
        TELEMETRY[key] = 0


# ---------------------------------------------------------------------------
# Capability snapshot
# ---------------------------------------------------------------------------
@dataclass
class GpuBackend:
    """Immutable-ish snapshot of this machine's GPU capability + active policy.

    Attributes
    ----------
    cupy_available:
        CuPy is importable.
    cuda_rust_available:
        The Rust ``CUDABatchDecoder`` build-time CUDA path is usable.
    device_name:
        CUDA device 0 name, or ``None`` if no usable device.
    total_mem_bytes:
        Total global memory of device 0 in bytes, or ``None``.
    prefer_gpu:
        Current global GPU preference (see :func:`set_prefer_gpu`).
    """

    cupy_available: bool
    cuda_rust_available: bool
    device_name: str | None
    total_mem_bytes: int | None
    prefer_gpu: bool = True

    def module(self) -> ModuleType:
        """Return the array module this backend would compute with right now."""
        if self.prefer_gpu and self.cupy_available and gpu_available():
            cp = _cupy()
            if cp is not None:
                return cp
        return np

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly capability report for logs/diagnostics."""
        return {
            "cupy_available": self.cupy_available,
            "cuda_rust_available": self.cuda_rust_available,
            "gpu_available": gpu_available(),
            "device_name": self.device_name,
            "total_mem_bytes": self.total_mem_bytes,
            "prefer_gpu": self.prefer_gpu,
            "active_module": self.module().__name__,
        }


def get_backend() -> GpuBackend:
    """Return the cached :class:`GpuBackend` capability snapshot (built once)."""
    global _BACKEND
    if _BACKEND is None:
        _usable, name, total = _probe()
        _BACKEND = GpuBackend(
            cupy_available=has_cupy(),
            cuda_rust_available=has_cuda_rust(),
            device_name=name,
            total_mem_bytes=total,
            prefer_gpu=_PREFER_GPU,
        )
    return _BACKEND


def set_prefer_gpu(flag: bool) -> None:
    """Set the global GPU preference and sync the cached backend snapshot.

    Setting ``False`` forces every policy-driven path (:func:`get_array_module`,
    :func:`xp`, :func:`to_device`, :meth:`GpuBackend.module`) onto NumPy/host,
    even when a device is available - useful for benchmarking or reproducibility.
    """
    global _PREFER_GPU
    _PREFER_GPU = bool(flag)
    if _BACKEND is not None:
        _BACKEND.prefer_gpu = _PREFER_GPU


def get_prefer_gpu() -> bool:
    """Return the current global GPU preference flag."""
    return _PREFER_GPU
