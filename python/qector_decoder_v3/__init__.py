"""
QECTOR Decoder v3 — Source-available Rust/Python QEC decoders with reproducible, artifact-hashed benchmark evidence.
Rust core + PyO3 bindings. Zero-copy NumPy. GIL-free decode.
"""

# pyright: reportMissingImports=false
# The compiled Rust extension provides this submodule at runtime in the installed package.
import importlib as _importlib


def _load_native_module():
    return _importlib.import_module(".qector_decoder_v3", __name__)  # pyright: ignore[reportMissingImports]


_native_module = _load_native_module()

_RustUnionFindDecoder = _native_module.UnionFindDecoder
_RustFastUnionFindDecoder = _native_module.FastUnionFindDecoder
_RustBlossomDecoder = _native_module.BlossomDecoder
_RustSlidingWindowDecoder = _native_module.SlidingWindowDecoder
_RustStreamingDecoder = _native_module.StreamingDecoder
_RustBatchDecoder = _native_module.BatchDecoder
_RustCPUBatchDecoder = _native_module.CPUBatchDecoder
_RustOpenCLBatchDecoder = _native_module.OpenCLBatchDecoder
_RustBenchmarkSuite = _native_module.BenchmarkSuite
_RustLookupTableDecoder = _native_module.LookupTableDecoder
_RustBPOSDDecoder = _native_module.BPOSDDecoder
_RustNeuralPredecoder = _native_module.NeuralPredecoder
_RustDetectorGraph = _native_module.DetectorGraph
_RustGNNPredecoder = _native_module.GNNPredecoder
_RustGNNTrainer = _native_module.GNNTrainer
try:
    _RustLERBenchmark = _native_module.LERBenchmark
except (AttributeError, ImportError):
    _RustLERBenchmark = None  # type: ignore[assignment]
_RustSparseBlossomDecoder = _native_module.SparseBlossomDecoder
_RustHybridDecoder = _native_module.HybridDecoder
py_check_to_edges = _native_module.py_check_to_edges
py_generate_surface_code_checks = _native_module.py_generate_surface_code_checks
py_generate_toy_code_checks = _native_module.py_generate_toy_code_checks
py_generate_ring_code_checks = _native_module.py_generate_ring_code_checks
py_generate_repetition_code_checks = _native_module.py_generate_repetition_code_checks
try:
    run_mcp_server = _native_module.run_mcp_server
except AttributeError:
    def run_mcp_server(*args, **kwargs):
        raise RuntimeError("run_mcp_server requires the 'grpc' feature (maturin develop --features full)")


try:
    _RustCUDABatchDecoder = _native_module.CUDABatchDecoder
    cuda_is_available = _native_module.cuda_is_available
except (AttributeError, ImportError):
    _RustCUDABatchDecoder = None  # type: ignore[assignment]

    def cuda_is_available():
        return False


try:
    _RustOpenCLBatchDecoder = _native_module.OpenCLBatchDecoder
    _rust_opencl_is_available = _native_module.opencl_is_available
except (AttributeError, ImportError):
    _RustOpenCLBatchDecoder = None

    def _rust_opencl_is_available():
        return False


try:
    run_grpc_server = _native_module.run_grpc_server
except (AttributeError, ImportError):
    run_grpc_server = None

try:
    start_metrics_server = _native_module.start_metrics_server
except (AttributeError, ImportError):
    start_metrics_server = None  # type: ignore[assignment]

try:
    import importlib.util as _importlib_util

    if _importlib_util.find_spec("torch") is not None:
        import torch.nn.functional as F  # noqa: F401
    else:
        F = None  # type: ignore[assignment]
except Exception:
    F = None  # type: ignore[assignment]

import numpy as _np

# os / sys / subprocess are imported locally inside the functions that need them
# to avoid polluting the public module namespace (see end of file)

# The authoritative runtime version is whatever the compiled Rust core reports.
# ``__fallback_version__`` is the Python-packaging version (pyproject/Cargo) and is
# used ONLY when the compiled module does not expose ``__version__`` (e.g. an old
# wheel). We never overwrite a real compiled ``__version__`` with it — doing so
# would falsely claim a version the loaded binary is not — so after a version bump
# ``__version__`` keeps reporting the *built* value until the Rust wheel is rebuilt.
__fallback_version__ = "0.6.4"

try:
    __version__ = _native_module.__version__
except (AttributeError, ImportError):
    __version__ = __fallback_version__

from typing import Optional

_OPENCL_HEALTH_CACHE: Optional[bool] = None


def _validate_check_to_qubits(check_to_qubits, n_qubits=None, *, reject_hyperedges=False):
    """Validate and normalize check_to_qubits for all decoders.

    Args:
        check_to_qubits: List of lists of qubit indices.
        n_qubits: Optional total qubit count.
        reject_hyperedges: If True, raise ValueError when any qubit appears in
            more than 2 checks (Union-Find / FastUnionFind limitation).

    Returns:
        Tuple of (normalized_c2q, resolved_n_qubits).

    Raises:
        ValueError: On empty input, negative indices, out-of-range qubits,
            duplicate qubits in a check, or hyperedges when rejected.
        TypeError: On non-integer qubit indices.

    Note:
        Empty checks (detectors with no associated mechanism) are allowed.
        They become boundary detectors that never fire from any single error.
    """
    if not check_to_qubits:
        raise ValueError("check_to_qubits must be non-empty")

    normalized = []
    max_q = -1
    for i, check in enumerate(check_to_qubits):
        if not check:
            normalized.append([])
            continue
        cleaned, check_max_q = _clean_single_check(check, i)
        normalized.append(cleaned)
        if check_max_q > max_q:
            max_q = check_max_q

    inferred_nq = max_q + 1 if max_q >= 0 else 0
    if n_qubits is not None:
        nq = int(n_qubits)
        if nq <= 0:
            raise ValueError(f"n_qubits must be positive, got {nq}")
        if max_q >= nq:
            raise ValueError(f"Qubit index {max_q} >= n_qubits {nq}")
    else:
        nq = inferred_nq

    if reject_hyperedges:
        _raise_if_hyperedges(normalized)

    return normalized, nq


def _clean_single_check(check, index):
    """Validate and normalize one check's qubit list for `_validate_check_to_qubits`.

    Returns ``(cleaned_qubits, local_max_qubit)``. Raises the same ValueError /
    TypeError that the inline validation used to raise for invalid entries.

    Note:
        Empty checks are allowed (returns empty list, local_max_q = -1).
        The caller ``_validate_check_to_qubits`` already handles the empty-checks
        pass-through case, so this function is kept simple for backward compat.
    """
    if not check:
        return [], -1
    cleaned = []
    seen = set()
    local_max_q = -1
    for q in check:
        if not isinstance(q, (int, bool, _np.integer, getattr(_np, "bool_", bool))):
            raise TypeError(f"Qubit index must be integer, got {type(q).__name__} in check {index}")
        qi = int(q)
        if qi < 0:
            raise ValueError(f"Negative qubit index {qi} in check {index}")
        if qi in seen:
            raise ValueError(f"Duplicate qubit {qi} in check {index}")
        seen.add(qi)
        cleaned.append(qi)
        if qi > local_max_q:
            local_max_q = qi
    return cleaned, local_max_q


def _raise_if_hyperedges(normalized):
    """Raise ValueError if any qubit participates in more than 2 checks."""
    qubit_degree = {}
    for check in normalized:
        for q in check:
            qubit_degree[q] = qubit_degree.get(q, 0) + 1
    for q, deg in qubit_degree.items():
        if deg > 2:
            raise ValueError(
                f"UnionFindDecoder / FastUnionFindDecoder only support stabilizer codes "
                f"where every qubit participates in at most 2 checks.\n"
                f"Qubit {q} participates in {deg} checks (hyperedge).\n"
                f"Use BlossomDecoder, SparseBlossomDecoder, or BPOSDDecoder instead.\n"
                f"(Codes with weight>2 checks, e.g. from generate_surface_code_checks(), "
                f"can still be graphlike if each qubit touches <=2 checks; this code is not)"
            )


def _opencl_raw_available() -> bool:
    try:
        return bool(_rust_opencl_is_available())
    except Exception:
        return False


def _opencl_health_check() -> bool:
    """Return True only when OpenCL can construct and decode a tiny batch.

    Some OpenCL driver stacks report a device but crash the process during
    context/kernel setup. Probe that risky path in a child process so production
    code can fall back to CPU instead of taking down the caller.
    """
    import os as _os_local
    import subprocess as _subprocess_local
    import sys as _sys_local

    global _OPENCL_HEALTH_CACHE
    if _os_local.environ.get("QECTOR_DISABLE_OPENCL", "").lower() in {"1", "true", "yes", "on"}:
        return False
    if _os_local.environ.get("QECTOR_OPENCL_PROBE_CHILD") == "1":
        return _opencl_raw_available()
    if _OPENCL_HEALTH_CACHE is not None:
        return _OPENCL_HEALTH_CACHE
    if not _opencl_raw_available():
        _OPENCL_HEALTH_CACHE = False
        return False

    code = (
        "import os\n"
        "os.environ['QECTOR_OPENCL_PROBE_CHILD']='1'\n"
        "import numpy as np\n"
        "import qector_decoder_v3 as qd\n"
        "dec = qd.OpenCLBatchDecoder([[0, 1]], 2)\n"
        "out = _np.asarray(dec.batch_decode(_np.array([[1]], dtype=_np.uint8)), dtype=_np.uint8)\n"
        "assert out.shape == (1, 2)\n"
    )
    try:
        proc = _subprocess_local.run(
            [_sys_local.executable, "-c", code],
            stdout=_subprocess_local.DEVNULL,
            stderr=_subprocess_local.DEVNULL,
            timeout=float(_os_local.environ.get("QECTOR_OPENCL_PROBE_TIMEOUT", "10")),
        )
        _OPENCL_HEALTH_CACHE = proc.returncode == 0
    except Exception:
        _OPENCL_HEALTH_CACHE = False
    return _OPENCL_HEALTH_CACHE


def opencl_is_available():
    return _opencl_health_check()


class UnionFindDecoder:
    """Production-ready Union-Find quantum error correction decoder.

    Rust core with PyO3 bindings. Zero-copy NumPy interop.
    GIL is released during decode for true parallelism.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        self._inner = _RustUnionFindDecoder(c2q, nq)

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode(syndromes)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class FastUnionFindDecoder:
    """SIMD-accelerated Union-Find with zero-allocation hot path and AVX2 runtime dispatch.
    Consistently faster than UnionFindDecoder on surface and repetition codes (1.1M shots/s).
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        self._inner = _RustFastUnionFindDecoder(c2q, nq)

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode(syndromes)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class BlossomDecoder:
    """Minimum-Weight Perfect Matching (MWPM) decoder via Edmonds' Blossom algorithm.

    Supports weighted edges for higher decoding accuracy on realistic codes.
    """

    def __init__(self, check_to_qubits, n_qubits=None, edge_weights=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustBlossomDecoder(c2q, nq, edge_weights)

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode(syndromes)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks

    @property
    def edges(self):
        return self._inner.edges


class SlidingWindowDecoder:
    """Sliding-window decoder with exponential decay weighting.

    Maintains a window of the last W rounds. Each round's syndrome is weighted
    by ``decay_factor ** age`` so that more recent rounds contribute more.
    The weighted cumulative syndrome is thresholded at 0.5 and decoded with
    the standard Union-Find decoder.
    """

    def __init__(self, check_to_qubits, n_qubits=None, window_size=10, decay_factor=0.8):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustSlidingWindowDecoder(c2q, nq, window_size, decay_factor)

    def update(self, round_syndrome):
        if not isinstance(round_syndrome, _np.ndarray):
            round_syndrome = _np.array(round_syndrome, dtype=_np.uint8)
        if round_syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {round_syndrome.dtype}")
        return self._inner.update(round_syndrome)

    def flush(self):
        self._inner.flush()

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks

    @property
    def window_size(self):
        return self._inner.window_size

    @property
    def decay_factor(self):
        return self._inner.decay_factor

    @property
    def current_round(self):
        return self._inner.current_round


class StreamingDecoder:
    """Streaming decoder that accumulates syndromes over multiple rounds.

    Rust core with circular history buffer and OR accumulation.
    """

    def __init__(self, check_to_qubits, n_qubits=None, history_size=10):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustStreamingDecoder(c2q, nq, history_size)

    def update(self, round_syndrome):
        if not isinstance(round_syndrome, _np.ndarray):
            round_syndrome = _np.array(round_syndrome, dtype=_np.uint8)
        if round_syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {round_syndrome.dtype}")
        return self._inner.update(round_syndrome)

    def flush(self):
        self._inner.flush()

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class BatchDecoder:
    """Parallel batch decoder using Rayon (Rust data parallelism).

    Distributes batch decoding across all CPU cores.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=False)
        self._inner = _RustBatchDecoder(c2q, nq)

    def parallel_batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.parallel_batch_decode(syndromes)

    def batch_decode(self, syndromes):
        """Alias for ``parallel_batch_decode`` for API consistency with the
        other batch decoders."""
        return self.parallel_batch_decode(syndromes)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class CPUBatchDecoder:
    """SIMD-friendly CPU batch decoder with pooled buffers and SoA transposition."""

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=False)
        self._inner = _RustCPUBatchDecoder(c2q, nq)

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode_simd(syndromes)

    def batch_decode_par(self, syndromes):
        """Rayon-parallel batch decode (best for large batches)."""
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode_par(syndromes)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class OpenCLBatchDecoder:
    """GPU-accelerated OpenCL batch decoder.

    Uses NVIDIA/AMD/Intel GPU via OpenCL for parallel batch decoding.
    Falls back to CPU UnionFind for small batches (< 8) or after repeated GPU failures.
    Automatically recovers from degraded mode after periodic GPU health checks.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        import os as _os_local

        if _os_local.environ.get("QECTOR_OPENCL_PROBE_CHILD") != "1" and not _opencl_health_check():
            raise RuntimeError(
                "OpenCL backend is unavailable or failed its health check; "
                "use CPUBatchDecoder/AutoDecoder fallback or set QECTOR_DISABLE_OPENCL=1"
            )
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=False)
        self._inner = _RustOpenCLBatchDecoder(c2q, nq)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode(syndromes)

    def reset(self):
        """Reset all GPU counters and exit degraded mode.

        Forces the decoder to retry GPU on the next call.
        Useful after driver update, GPU maintenance, or manual intervention.
        """
        self._inner.reset()

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks

    @property
    def consecutive_failures(self):
        """Number of consecutive GPU failures since last success."""
        return self._inner.consecutive_failures

    @property
    def total_failures(self):
        """Total number of GPU failures since decoder creation."""
        return self._inner.total_failures

    @property
    def is_degraded(self):
        """True if decoder is in CPU-only mode after repeated GPU failures."""
        return self._inner.is_degraded

    @property
    def gpu_recoveries(self):
        """Number of times the GPU recovered after being in degraded mode."""
        return self._inner.gpu_recoveries

    @property
    def device_name(self):
        """Human-readable OpenCL GPU device name (e.g. 'NVIDIA GeForce RTX 3080')."""
        return self._inner.device_name

    @staticmethod
    def is_available():
        """Return True if OpenCL is available and passes a safe decode probe."""
        return _opencl_health_check()


class CUDABatchDecoder:
    """GPU-accelerated native CUDA batch decoder.

    Uses a compiled CUDA kernel loaded through the CUDA Driver API. Falls back
    to CPU UnionFind for tiny batches or after repeated CUDA failures.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        if _RustCUDABatchDecoder is None:
            raise RuntimeError("qector-decoder-v3 was built without the 'cuda' feature")
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=False)
        self._inner = _RustCUDABatchDecoder(c2q, nq)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode(syndromes)

    def reset(self):
        self._inner.reset()

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks

    @property
    def device_name(self):
        return self._inner.device_name

    @property
    def compute_capability(self):
        return self._inner.compute_capability

    @property
    def consecutive_failures(self):
        return self._inner.consecutive_failures

    @property
    def total_failures(self):
        return self._inner.total_failures

    @property
    def is_degraded(self):
        return self._inner.is_degraded

    @property
    def gpu_recoveries(self):
        return self._inner.gpu_recoveries

    @staticmethod
    def is_available():
        """Return True if a CUDA driver device is available in this build."""
        if _RustCUDABatchDecoder is None:
            return False
        return _RustCUDABatchDecoder.is_available()


class SparseBlossomDecoder:
    """Region-growing Sparse Blossom decoder with RadixHeap.

    Supports dynamic weight overrides from GNN Pre-Decoder for enriched decoding.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustSparseBlossomDecoder(c2q, nq)

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def decode_with_weights(self, syndrome, weights):
        """Decode with per-qubit dynamic weight overrides.

        Honesty note: ``SparseBlossomDecoder`` is a region-growing
        **minimum-cardinality** matcher. The ``weights`` are applied only as an
        approximate rescale of the BFS region-growth distances; this does **not**
        perform an exact weighted MWPM (the correction is syndrome-valid but not
        weight-optimal), and empirically the override has little to no effect on
        the logical error rate. For true weighted minimum-weight perfect matching
        use :class:`BlossomDecoder` (exact Edmonds) or the ``pymatching_compat``
        layer. A one-time warning is emitted on first weighted use.

        Args:
            syndrome: _np.ndarray of shape (n_checks,) with dtype uint8.
            weights: List of (qubit_id, weight) tuples.

        Returns:
            _np.ndarray of shape (n_qubits,) with correction.
        """
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        if not isinstance(weights, list):
            weights = list(weights)
        return self._inner.decode_with_weights(syndrome, weights)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode(syndromes)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class BPOSDDecoder:
    """Belief Propagation + Ordered Statistics Decoding.

    Min-sum BP with OSD stage for improved LER on complex codes.
    """

    def __init__(self, check_to_qubits, n_qubits=None, error_rate=0.1):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustBPOSDDecoder(c2q, nq, error_rate)

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def decode_timed(self, syndrome, max_latency_ms=10.0):
        """Decode with a wall-clock deadline in milliseconds.

        Args:
            syndrome: np.ndarray of shape (n_checks,) with dtype uint8.
            max_latency_ms: Max allowed BP time in ms (default: 10.0).
                Falls back to hard-decision if BP times out.

        Returns:
            np.ndarray of shape (n_qubits,) with correction bits.
        """
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode_timed(syndrome, max_latency_ms)

    def bp_decode(self, syndrome, max_iterations=20):
        """Run Belief Propagation and return log-likelihood ratios (LLRs) for each qubit.

        Args:
            syndrome: _np.ndarray of shape (n_checks,) with dtype uint8.
            max_iterations: Number of BP iterations (default: 20).

        Returns:
            _np.ndarray of shape (n_qubits,) with LLR values.
            Positive LLR -> more likely 0, Negative LLR -> more likely 1.
        """
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.bp_decode(syndrome, max_iterations)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class NeuralPredecoder:
    """Lightweight MLP pre-decoder with Xavier initialization and SGD training."""

    def __init__(self, n_input, n_output, n_hidden1=None, n_hidden2=None):
        self._inner = _RustNeuralPredecoder(n_input, n_output, n_hidden1, n_hidden2)

    def train(self, syndromes, corrections, n_epochs, learning_rate=0.01):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if not isinstance(corrections, _np.ndarray):
            corrections = _np.array(corrections, dtype=_np.uint8)
        self._inner.train(syndromes, corrections, n_epochs, learning_rate)

    def predict(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        return self._inner.predict(syndrome)

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        return self._inner.decode(syndrome)

    @property
    def n_input(self):
        return self._inner.n_input

    @property
    def n_output(self):
        return self._inner.n_output

    @property
    def n_hidden1(self):
        return self._inner.n_hidden1

    @property
    def n_hidden2(self):
        return self._inner.n_hidden2


class GNNPredecoder:
    """Graph Neural Network Pre-Decoder for dynamic edge weight prediction.

    MPNN 3 layers + MLP readout. Predicts adjusted edge weights for SparseBlossom.

    **v2.0** : Full backpropagation through MPNN layers (P0). All layers are trainable.

    Dimensions must match the DetectorGraph:
    - node_feat_dim = 10 (NodeFeatures::DIM)
    - edge_feat_dim = 8 (EdgeFeatures::DIM)
    """

    NODE_FEAT_DIM = 10
    EDGE_FEAT_DIM = 8

    def __init__(self, node_feat_dim=None, edge_feat_dim=None, hidden_size=16, n_layers=2):
        """Create a GNNPredecoder.

        If node_feat_dim and edge_feat_dim are not provided, uses the standard
        dimensions matching DetectorGraph (10 and 8).
        """
        nfd = node_feat_dim if node_feat_dim is not None else self.NODE_FEAT_DIM
        efd = edge_feat_dim if edge_feat_dim is not None else self.EDGE_FEAT_DIM
        self._inner = _RustGNNPredecoder(nfd, efd, hidden_size, n_layers)

        # Load PyTorch implementation if torch is installed.
        try:
            from .torch_predecoder import TorchGNNPredecoder
            self._torch_model = TorchGNNPredecoder(nfd, efd, hidden_size, n_layers)
            self._torch_model.double()  # Cast to float64 to match Rust f64.
            self.use_torch = True
        except ImportError:
            self.use_torch = False

    def _sync_to_rust(self):
        if self.use_torch:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self._torch_model.save_weights(tmp_path)
                self._inner.load_weights(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def _sync_to_torch(self):
        if self.use_torch:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self._inner.save_weights(tmp_path)
                self._torch_model.load_weights(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    @classmethod
    def new_standard(cls, hidden_size=16, n_layers=2):
        """Create a GNNPredecoder with standard dimensions matching DetectorGraph."""
        return cls(cls.NODE_FEAT_DIM, cls.EDGE_FEAT_DIM, hidden_size, n_layers)

    @property
    def learning_rate(self):
        return self._inner.learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        self._inner.learning_rate = lr

    @property
    def l2_lambda(self):
        return self._inner.l2_lambda

    @l2_lambda.setter
    def l2_lambda(self, val):
        self._inner.l2_lambda = val

    def save_weights(self, path):
        """Save model weights in SafeTensors format."""
        if self.use_torch:
            self._torch_model.save_weights(str(path))
            self._sync_to_rust()
        else:
            self._inner.save_weights(str(path))

    def load_weights(self, path):
        """Load model weights from a SafeTensors file."""
        if self.use_torch:
            self._torch_model.load_weights(str(path))
            self._sync_to_rust()
        else:
            self._inner.load_weights(str(path))

    def forward(self, graph):
        """Predict adjusted edge weights for a DetectorGraph."""
        if self.use_torch:
            import torch
            self._torch_model.eval()
            with torch.no_grad():
                node_feats = torch.tensor(graph.node_features, dtype=torch.float64)
                edge_feats = torch.tensor(graph.edge_features, dtype=torch.float64)
                edge_src = torch.tensor(graph.edge_src, dtype=torch.long)
                edge_dst = torch.tensor(graph.edge_dst, dtype=torch.long)
                out = self._torch_model(node_feats, edge_feats, edge_src, edge_dst)
                return out.cpu().numpy().tolist()
        else:
            return self._inner.forward(graph._inner if isinstance(graph, DetectorGraph) else graph)

    def train(self, graphs, targets, n_epochs):
        """Train the GNN on a list of graphs and target edge weights."""
        if self.use_torch:
            if F is None:
                raise ImportError("PyTorch is required to train GNNPredecoder")
            import torch
            import torch.optim as optim
            optimizer = optim.SGD(self._torch_model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)
            self._torch_model.train()
            
            for epoch in range(n_epochs):
                for graph, target in zip(graphs, targets):
                    node_feats = torch.tensor(graph.node_features, dtype=torch.float64)
                    edge_feats = torch.tensor(graph.edge_features, dtype=torch.float64)
                    edge_src = torch.tensor(graph.edge_src, dtype=torch.long)
                    edge_dst = torch.tensor(graph.edge_dst, dtype=torch.long)
                    target_t = torch.tensor(target, dtype=torch.float64)
                    
                    optimizer.zero_grad()
                    preds = self._torch_model(node_feats, edge_feats, edge_src, edge_dst)
                    loss = F.mse_loss(preds, target_t)
                    loss.backward()
                    optimizer.step()
            self._sync_to_rust()
        else:
            inner_graphs = [g._inner if isinstance(g, DetectorGraph) else g for g in graphs]
            self._inner.train(inner_graphs, targets, n_epochs)

    def predict_with_node_probs(self, graph):
        """Predict edge weights and node error probabilities."""
        if self.use_torch:
            import torch
            self._torch_model.eval()
            with torch.no_grad():
                node_feats = torch.tensor(graph.node_features, dtype=torch.float64)
                edge_feats = torch.tensor(graph.edge_features, dtype=torch.float64)
                edge_src = torch.tensor(graph.edge_src, dtype=torch.long)
                edge_dst = torch.tensor(graph.edge_dst, dtype=torch.long)
                
                weights = self._torch_model(node_feats, edge_feats, edge_src, edge_dst)
                
                N = node_feats.size(0)
                node_probs = torch.zeros(N, dtype=torch.float64)
                node_counts = torch.zeros(N, dtype=torch.float64)
                
                node_probs.scatter_add_(0, edge_src, weights)
                node_probs.scatter_add_(0, edge_dst, weights)
                
                node_counts.scatter_add_(0, edge_src, torch.ones_like(edge_src, dtype=torch.float64))
                node_counts.scatter_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=torch.float64))
                
                mask = node_counts > 0
                node_probs[mask] /= node_counts[mask]
                node_probs = 1.0 / (1.0 + (-node_probs + 1.0).exp())
                
                return weights.cpu().numpy().tolist(), node_probs.cpu().numpy().tolist()
        else:
            return self._inner.predict_with_node_probs(graph._inner if isinstance(graph, DetectorGraph) else graph)



class DetectorGraph:
    """Detector graph used by the GNN and hybrid decoder paths."""

    NODE_FEAT_DIM = 10
    EDGE_FEAT_DIM = 8

    def __init__(
        self,
        check_to_qubits,
        syndrome,
        check_positions=None,
        check_types=None,
        base_weights=None,
        n_qubits=None,
    ):
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        syn = [int(bit) for bit in syndrome]
        if check_positions is not None:
            check_positions = [tuple(p) for p in check_positions]
        self._inner = _RustDetectorGraph(
            c2q,
            syn,
            check_positions,
            check_types,
            base_weights,
            n_qubits,
        )

    def update_syndrome(self, syndrome):
        self._inner.update_syndrome([int(bit) for bit in syndrome])

    @property
    def n_nodes(self):
        return self._inner.n_nodes

    @property
    def n_edges(self):
        return self._inner.n_edges

    @property
    def node_features(self):
        return self._inner.node_features

    @property
    def edge_features(self):
        return self._inner.edge_features

    @property
    def edge_qubit_id(self):
        return self._inner.edge_qubit_id

    @property
    def edge_src(self):
        return self._inner.edge_src

    @property
    def edge_dst(self):
        return self._inner.edge_dst


class GNNTrainer:
    """End-to-end GNN training pipeline with Blossom teacher model.

    Generates random syndromes, computes optimal corrections via BlossomDecoder,
    extracts target edge weights, and trains the GNN via SGD.
    """

    def __init__(self, check_to_qubits, n_qubits, error_rate=0.1):
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        self._inner = _RustGNNTrainer(c2q, n_qubits, error_rate)

    def train(self, gnn, n_samples, n_epochs):
        """Train a GNNPredecoder and return the final MSE loss."""
        gnn._sync_to_rust()
        loss = self._inner.train(gnn._inner, n_samples, n_epochs)
        gnn._sync_to_torch()
        return loss

    def train_bp(self, gnn, n_samples, n_epochs, max_bp_iter=20):
        """Train a GNNPredecoder with BP marginal targets and return the final MSE loss."""
        gnn._sync_to_rust()
        loss = self._inner.train_bp(gnn._inner, n_samples, n_epochs, max_bp_iter)
        gnn._sync_to_torch()
        return loss

    def generate_dataset(self, n_samples):
        """Generate a training dataset and return its size."""
        return self._inner.generate_dataset(n_samples)


class HybridDecoder:
    """GNN Pre-Decoder + SparseBlossom hybrid decoder.

    Uses a lightweight MPNN to estimate dynamic edge weights, then passes
    them to SparseBlossom for enriched region-growing decoding.
    """

    def __init__(
        self,
        check_to_qubits,
        n_qubits=None,
        check_positions=None,
        check_types=None,
        base_weights=None,
        gnn_hidden_size=64,
        gnn_n_layers=3,
    ):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustHybridDecoder(
            c2q, nq, check_positions, check_types, base_weights, gnn_hidden_size, gnn_n_layers
        )

    def save_weights(self, path):
        """Save model weights in SafeTensors format."""
        self._inner.save_weights(str(path))

    def load_weights(self, path):
        """Load model weights from a SafeTensors file."""
        self._inner.load_weights(str(path))

    def decode_hybrid(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode_hybrid(syndrome)

    def decode_heuristic(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode_heuristic(syndrome)

    def decode_standard(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode_standard(syndrome)

    def decode(self, syndrome):
        """Alias for ``decode_standard``.

        This provides API consistency with all other QECTOR decoders.
        """
        return self.decode_standard(syndrome)

    def batch_decode_hybrid(self, syndromes):
        """Batch decode multiple syndromes using the GNN-enhanced pipeline.

        Args:
            syndromes: _np.ndarray of shape (batch, n_checks) or list of lists.

        Returns:
            _np.ndarray of shape (batch, n_qubits) with corrections.
        """
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            raise TypeError(f"Syndromes must be dtype uint8, got {syndromes.dtype}")
        if syndromes.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {syndromes.shape}")
        return self._inner.batch_decode_hybrid(syndromes)

    def batch_decode_standard(self, syndromes):
        """Batch decode multiple syndromes using standard SparseBlossom.

        Args:
            syndromes: _np.ndarray of shape (batch, n_checks) or list of lists.

        Returns:
            _np.ndarray of shape (batch, n_qubits) with corrections.
        """
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            raise TypeError(f"Syndromes must be dtype uint8, got {syndromes.dtype}")
        if syndromes.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {syndromes.shape}")
        return self._inner.batch_decode_standard(syndromes)

    def train(self, n_samples, n_epochs, error_rate=0.1):
        """Train the internal GNN using a Blossom teacher model.

        Generates random syndromes, computes optimal corrections via Blossom,
        and trains the GNN via SGD to predict edge weights.

        Args:
            n_samples: Number of training examples to generate.
            n_epochs: Number of training epochs.
            error_rate: Physical error rate for syndrome generation.

        Returns:
            Final MSE loss after training.
        """
        return self._inner.train(n_samples, n_epochs, error_rate)

    def train_bp(self, n_samples, n_epochs, error_rate=0.1, max_bp_iter=20):
        """Train the internal GNN using BP marginal probability targets.

        Generates random syndromes, computes marginal error probabilities via
        Belief Propagation (min-sum), and trains the GNN to predict these
        probabilities as edge weights.

        Args:
            n_samples: Number of training examples to generate.
            n_epochs: Number of training epochs.
            error_rate: Physical error rate for syndrome generation.
            max_bp_iter: Number of BP iterations for marginal computation.

        Returns:
            Final MSE loss after training.
        """
        return self._inner.train_bp(n_samples, n_epochs, error_rate, max_bp_iter)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class LookupTableDecoder:
    """Exact lookup-table decoder with UnionFind fallback.

    Pre-computes all syndrome → correction mappings for small codes
    (n_qubits ≤ 20, exhaustive; otherwise low-weight enumeration).
    Decoding is O(1) for precomputed syndromes, fallback to UnionFind otherwise.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustLookupTableDecoder(c2q, nq)

    def build_table(self, max_entries):
        """Populate the lookup table by enumerating errors up to max_entries."""
        self._inner.build_table(int(max_entries))

    def decode(self, syndrome):
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode(syndromes)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks

    @property
    def table_size(self):
        return self._inner.table_size


def check_to_edges(check_to_qubits):
    """Convert check_to_qubits to edge list."""
    c2q = [[int(q) for q in check] for check in check_to_qubits]
    return py_check_to_edges(c2q)


def generate_surface_code_checks(distance):
    """Generate compact periodic (toric) surface-code checks.

    Note: these produce codes with 4-qubit checks (hyperedges) and rank-deficient
    parity-check matrices. UnionFindDecoder and FastUnionFindDecoder will
    explicitly reject them (since v0.6.2). Use BlossomDecoder or BPOSDDecoder
    for these codes.
    """
    return py_generate_surface_code_checks(int(distance))


def generate_toy_code_checks(distance):
    """Generate a toy code (not a proper quantum code) with d*d qubits for backward compatibility.

    Kept for reference. Prefer generate_surface_code_checks for real QEC tests.
    """
    return py_generate_toy_code_checks(int(distance))


def generate_ring_code_checks(distance):
    """Generate a simple 1D ring code for testing."""
    return py_generate_ring_code_checks(int(distance))


def generate_repetition_code_checks(distance):
    """Generate a 1D repetition/chain code for testing."""
    return py_generate_repetition_code_checks(int(distance))


class BenchmarkSuite:
    """Production benchmark suite. Wraps Rust-native benchmarking."""

    def __init__(self, check_to_qubits, n_qubits=None, n_samples=10000, seed=42):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustBenchmarkSuite(c2q, nq, n_samples, seed)
        self.n_samples = n_samples

    def run(self):
        import json

        raw = self._inner.run()
        return json.loads(raw)

    def save(self, path, results):
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")


class LERBenchmark:
    """Logical Error Rate benchmark suite.

    Compares QECTOR decoder LER against PyMatching baselines across
    multiple code distances and physical error rates.
    """

    def __init__(self):
        if _RustLERBenchmark is None:
            raise RuntimeError(
                "LERBenchmark requires the full Rust source (not available in the public stub). "
                "Install the pre-built wheel from PyPI: pip install qector-decoder-v3"
            )
        self._inner = _RustLERBenchmark()

    def run_all(self):
        """Run all benchmark configurations and return JSON results.

        Returns:
            JSON string with LER comparison data for each configuration.
        """
        if _RustLERBenchmark is None:
            raise RuntimeError("LERBenchmark not available with the current build")
        return self._inner.run_all()


__all__ = [
    "UnionFindDecoder",
    "FastUnionFindDecoder",
    "BlossomDecoder",
    "SlidingWindowDecoder",
    "StreamingDecoder",
    "BatchDecoder",
    "CPUBatchDecoder",
    "OpenCLBatchDecoder",
    "CUDABatchDecoder",
    "SparseBlossomDecoder",
    "BPOSDDecoder",
    "NeuralPredecoder",
    "DetectorGraph",
    "GNNPredecoder",
    "GNNTrainer",
    "HybridDecoder",
    "LookupTableDecoder",
    "BenchmarkSuite",
    "LERBenchmark",
    "check_to_edges",
    "generate_surface_code_checks",
    "generate_toy_code_checks",
    "generate_ring_code_checks",
    "generate_repetition_code_checks",
    "start_metrics_server",
    "run_mcp_server",
    "cuda_is_available",
    "opencl_is_available",
    "run_grpc_server",
]


# Ecosystem / tooling layer (pure-Python, built on the compiled core)
from . import (
    codes,
    dem,
    result,
    backend,
    pymatching_compat,
    benchmarking,
    belief_matching,
    bposd,
    predecoder,
)
from . import workbench
from .backend import AutoDecoder, BackendConfig, Backend
from .result import DecodeResult, decode_with_diagnostics
from .belief_matching import BeliefMatching
from .bposd import BpOsdDecoder
from .predecoder import PredecodedDecoder
from .workbench import Workbench

# New v0.6.4 features: DecoderPool, get_decoder, decode_mmap
from .decoder_pool import DecoderPool
from .decoder_cache import get_decoder, clear_decoder_cache, get_decoder_pool
from .decode_mmap import decode_mmap

# sinter_compat imports `sinter` lazily; tolerate its absence.
try:
    from . import sinter_compat
except Exception:  # pragma: no cover
    sinter_compat = None  # type: ignore[assignment]

__all__ += [
    "codes",
    "dem",
    "result",
    "backend",
    "pymatching_compat",
    "benchmarking",
    "belief_matching",
    "bposd",
    "predecoder",
    "sinter_compat",
    "AutoDecoder",
    "BackendConfig",
    "Backend",
    "DecodeResult",
    "decode_with_diagnostics",
    "BeliefMatching",
    "BpOsdDecoder",
    "PredecodedDecoder",
    "workbench",
    "Workbench",
    "DecoderPool",
    "get_decoder",
    "clear_decoder_cache",
    "get_decoder_pool",
    "decode_mmap",
]

# Optional ecosystem integrations (tolerate missing third-party deps)
try:
    from . import qiskit_plugin
except Exception:  # pragma: no cover
    qiskit_plugin = None  # type: ignore[assignment]

try:
    from . import stim_compat
except Exception:  # pragma: no cover
    stim_compat = None  # type: ignore[assignment]

try:
    from . import rest_api
except Exception:  # pragma: no cover
    rest_api = None  # type: ignore[assignment]

__all__ += [
    "qiskit_plugin",
    "stim_compat",
    "rest_api",
]

# ---------------------------------------------------------------------------
# GPU backend abstraction + batched belief propagation (additive, CuPy-optional).
#
# ``gpu_backend`` only needs NumPy + stdlib to import (CuPy is probed lazily and
# guarded internally), and ``bp_cupy`` builds purely on in-tree pure-Python
# modules, so neither requires CuPy to be installed. They are still wrapped in a
# tolerant guard so that — per the package contract — a missing optional path can
# never break ``import qector_decoder_v3``. The selected detection helpers
# (``has_cupy`` / ``has_cuda_rust`` / ``gpu_available`` / ``get_backend``) and the
# batched-BP entry points are re-exported at top level for convenience; the full
# foundation API remains available as ``qector_decoder_v3.gpu_backend``.
# ---------------------------------------------------------------------------
try:
    from . import gpu_backend as _gpu_backend_module

    has_cupy = _gpu_backend_module.has_cupy
    has_cuda_rust = _gpu_backend_module.has_cuda_rust
    gpu_available = _gpu_backend_module.gpu_available
    get_backend = _gpu_backend_module.get_backend
    gpu_backend = _gpu_backend_module

    __all__.extend(
        [
            "gpu_backend",
            "has_cupy",
            "has_cuda_rust",
            "gpu_available",
            "get_backend",
        ]
    )
except Exception:  # pragma: no cover - defensive; gpu_backend has no hard deps
    gpu_backend = None  # type: ignore[assignment]
    has_cupy = None  # type: ignore[assignment]
    has_cuda_rust = None  # type: ignore[assignment]
    gpu_available = None  # type: ignore[assignment]
    get_backend = None  # type: ignore[assignment]
    __all__.extend(
        [
            "gpu_backend",
            "has_cupy",
            "has_cuda_rust",
            "gpu_available",
            "get_backend",
        ]
    )

try:
    from . import bp_cupy as _bp_cupy_module

    BatchedBpDecoder: object = _bp_cupy_module.BatchedBpDecoder
    batched_bp_decode = _bp_cupy_module.batched_bp_decode
    bp_cupy = _bp_cupy_module

    __all__.extend(
        [
            "bp_cupy",
            "BatchedBpDecoder",
            "batched_bp_decode",
        ]
    )
except Exception:  # pragma: no cover - defensive; bp_cupy has no hard deps
    bp_cupy = None  # type: ignore[assignment]
    BatchedBpDecoder = None
    batched_bp_decode = None  # type: ignore[assignment]
    __all__.extend(
        [
            "bp_cupy",
            "BatchedBpDecoder",
            "batched_bp_decode",
        ]
    )

# ---------------------------------------------------------------------------
# Intelligent decoder routing + high-level streaming orchestration (additive).
#
# Both are pure-Python layers built only on already-imported in-tree modules
# (``gpu_backend`` / ``backend`` and the exported decoder classes); they degrade
# to NumPy/CPU when no CuPy/device is present. ``routing`` answers "which decoder
# *family* is even correct here?" (and forces BP-OSD for hyperedge codes so every
# correction satisfies ``H·c == s``); ``streaming`` orchestrates multi-round
# sliding-window decoding on top of the single-shot decoders without shadowing the
# compiled ``StreamingDecoder`` / ``SlidingWindowDecoder``. Wrapped in tolerant
# guards so a missing optional path can never break ``import qector_decoder_v3``.
# ---------------------------------------------------------------------------
try:
    from . import routing as _routing_module

    recommend_decoder = _routing_module.recommend_decoder
    recommend = _routing_module.recommend
    AutoRouter: object = _routing_module.AutoRouter
    Recommendation: object = _routing_module.Recommendation
    DecoderName: object = _routing_module.DecoderName
    HardwareProfile: object = _routing_module.HardwareProfile
    detect_hardware = _routing_module.detect_hardware
    routing = _routing_module

    __all__.extend(
        [
            "routing",
            "recommend_decoder",
            "recommend",
            "AutoRouter",
            "Recommendation",
            "DecoderName",
            "HardwareProfile",
            "detect_hardware",
        ]
    )
except Exception:  # pragma: no cover - defensive; routing has no hard deps
    routing = None  # type: ignore[assignment]
    recommend_decoder = None  # type: ignore[assignment]
    recommend = None  # type: ignore[assignment]
    AutoRouter = None
    Recommendation = None
    DecoderName = None
    HardwareProfile = None
    detect_hardware = None  # type: ignore[assignment]
    __all__.extend(
        [
            "routing",
            "recommend_decoder",
            "recommend",
            "AutoRouter",
            "Recommendation",
            "DecoderName",
            "HardwareProfile",
            "detect_hardware",
        ]
    )

try:
    from . import streaming as _streaming_module

    StreamingSession: object = _streaming_module.StreamingSession
    sliding_window_decode = _streaming_module.sliding_window_decode
    StreamingResult: object = _streaming_module.StreamingResult
    StreamingTelemetry: object = _streaming_module.StreamingTelemetry
    streaming = _streaming_module

    __all__.extend(
        [
            "streaming",
            "StreamingSession",
            "sliding_window_decode",
            "StreamingResult",
            "StreamingTelemetry",
        ]
    )
except Exception:  # pragma: no cover - defensive; streaming has no hard deps
    streaming = None  # type: ignore[assignment]
    StreamingSession = None
    sliding_window_decode = None  # type: ignore[assignment]
    StreamingResult = None
    StreamingTelemetry = None
    __all__.extend(
        [
            "streaming",
            "StreamingSession",
            "sliding_window_decode",
            "StreamingResult",
            "StreamingTelemetry",
        ]
    )

# Clean public namespace to reduce leakage (per v0.6.2 todo.md).
# Only public names should remain.
for _name in ("os", "sys", "subprocess"):
    globals().pop(_name, None)
# Note: numpy kept as _np internally; public API uses explicit numpy where needed in docs.
