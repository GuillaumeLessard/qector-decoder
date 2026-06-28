"""
QECTOR Decoder v3 — Source-available Rust/Python QEC decoders with reproducible, artifact-hashed benchmark evidence.
Rust core + PyO3 bindings. Zero-copy NumPy. GIL-free decode.

Independently validated (v0.5.3, 2026-06-25, PyPI install, isolated venv):
    Platform:  Windows 10, AMD Ryzen (16 cores), Python 3.11, NumPy 2.2.6
    GPU:       NVIDIA GeForce GTX 1660 Ti (CUDA 7.5)
    Reference: PyMatching 2.4.0, stim 1.16.0, sinter 1.16.0
    Result:    87/87 automated checks PASS (primary 20k + re-test 100k shots/pt)

    Workbench single-decode (repetition d=5, Blossom, 1000 trials):
        throughput: 208,334 decodes/s
        p50: 4.80 µs  |  p90: 8.30 µs  |  p99: 20.30 µs  |  max: 69.8 µs
        syndrome_faithful: True

    CPU batch throughput (repetition d=9):
        UnionFindDecoder:   ~1.06M shots/s
        BlossomDecoder:     ~2.70M shots/s
        SparseBlossomDecoder: ~1.80M shots/s
        CPUBatchDecoder:    ~0.34M shots/s
        BatchDecoder:       ~2.67M shots/s (parallel_batch_decode)

    CUDA GPU vs CPU (100k shots, GTX 1660 Ti):
        repetition_code(d=9):       7.67× faster than CPU batch
        rotated_surface_code(d=5):  6.93× faster than CPU batch
        GPU output: 100% valid, 100% CPU-agreeing
"""

try:
    # Attempt to import compiled Rust bindings from the installed wheel package
    from qector_decoder_v3.qector_decoder_v3 import (
        UnionFindDecoder as _RustUnionFindDecoder,
        FastUnionFindDecoder as _RustFastUnionFindDecoder,
        BlossomDecoder as _RustBlossomDecoder,
        SlidingWindowDecoder as _RustSlidingWindowDecoder,
        StreamingDecoder as _RustStreamingDecoder,
        BatchDecoder as _RustBatchDecoder,
        CPUBatchDecoder as _RustCPUBatchDecoder,
        OpenCLBatchDecoder as _RustOpenCLBatchDecoder,
        BenchmarkSuite as _RustBenchmarkSuite,
        LookupTableDecoder as _RustLookupTableDecoder,
        BPOSDDecoder as _RustBPOSDDecoder,
        NeuralPredecoder as _RustNeuralPredecoder,
        DetectorGraph as _RustDetectorGraph,
        GNNPredecoder as _RustGNNPredecoder,
        GNNTrainer as _RustGNNTrainer,
        SparseBlossomDecoder as _RustSparseBlossomDecoder,
        HybridDecoder as _RustHybridDecoder,
        py_check_to_edges,
        py_generate_surface_code_checks,
        py_generate_toy_code_checks,
        py_generate_ring_code_checks,
        py_generate_repetition_code_checks,
        run_mcp_server,
    )
except Exception:
    # Fallback to the source version (editable install) if wheel is not available
    from .qector_decoder_v3 import (
        UnionFindDecoder as _RustUnionFindDecoder,
        FastUnionFindDecoder as _RustFastUnionFindDecoder,
        BlossomDecoder as _RustBlossomDecoder,
        SlidingWindowDecoder as _RustSlidingWindowDecoder,
        StreamingDecoder as _RustStreamingDecoder,
        BatchDecoder as _RustBatchDecoder,
        CPUBatchDecoder as _RustCPUBatchDecoder,
        OpenCLBatchDecoder as _RustOpenCLBatchDecoder,
        BenchmarkSuite as _RustBenchmarkSuite,
        LookupTableDecoder as _RustLookupTableDecoder,
        BPOSDDecoder as _RustBPOSDDecoder,
        NeuralPredecoder as _RustNeuralPredecoder,
        DetectorGraph as _RustDetectorGraph,
        GNNPredecoder as _RustGNNPredecoder,
        GNNTrainer as _RustGNNTrainer,
        SparseBlossomDecoder as _RustSparseBlossomDecoder,
        HybridDecoder as _RustHybridDecoder,
        py_check_to_edges,
        py_generate_surface_code_checks,
        py_generate_toy_code_checks,
        py_generate_ring_code_checks,
        py_generate_repetition_code_checks,
        run_mcp_server,
    )


try:
    from .qector_decoder_v3 import (
        CUDABatchDecoder as _RustCUDABatchDecoder,
        cuda_is_available,
    )
except ImportError:
    _RustCUDABatchDecoder = None  # type: ignore[assignment]

    def cuda_is_available():
        return False


try:
    from .qector_decoder_v3 import opencl_is_available
except ImportError:

    def opencl_is_available():
        return False


try:
    from .qector_decoder_v3 import run_grpc_server, start_metrics_server
except ImportError:

    def start_metrics_server(*_, **__):
        raise NotImplementedError("start_metrics_server is not available in this build")

    def run_grpc_server(*_, **__):
        raise NotImplementedError("run_grpc_server is not available in this build")


import numpy as np


try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("qector-decoder-v3")
except PackageNotFoundError:
    __version__ = "0.5.5"


class UnionFindDecoder:
    """Production-ready Union-Find quantum error correction decoder.

    Rust core with PyO3 bindings. Zero-copy NumPy interop.
    GIL is released during decode for true parallelism.

    Performance (independently validated, Windows 10, AMD Ryzen, Python 3.11):
        - repetition_code(d=5):  ~9.3 µs/decode,  ~1.06M shots/s (batch)
        - repetition_code(d=9):  ~10.0 µs/decode
        - rotated_surface_code(d=3): ~12.2 µs/decode
        - rotated_surface_code(d=5): ~10.1 µs/decode

    Accuracy note:
        UnionFind LER averages ~3.1× that of Blossom/MWPM across d=3–9.
        All corrections are 100% syndrome-valid. Use when throughput dominates
        over accuracy; use BlossomDecoder when LER matters.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        # Convert Python list-of-lists to Vec<Vec<u32>>
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustUnionFindDecoder(c2q, nq)

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
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
    """SIMD-accelerated zero-allocation Union-Find decoder.

    Uses pre-allocated reusable buffers, AVX2 runtime dispatch, and FFI.
    Same API as UnionFindDecoder but with lower overhead.

    Performance (independently validated, Windows 10, AMD Ryzen, Python 3.11):
        - repetition_code(d=5):  ~9.5 µs/decode
        - repetition_code(d=9):  ~10.2 µs/decode
        - rotated_surface_code(d=3): ~11.4 µs/decode
        - rotated_surface_code(d=5): ~12.1 µs/decode
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustFastUnionFindDecoder(c2q, nq)

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
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

    Performance (independently validated, Windows 10, AMD Ryzen, Python 3.11):
        - repetition_code(d=5):  ~10.6 µs/decode,  ~2.70M shots/s (batch)
        - repetition_code(d=9):  ~10.6 µs/decode
        - rotated_surface_code(d=3): ~14.8 µs/decode
        - rotated_surface_code(d=5): ~16.8 µs/decode

    Accuracy (100k shots/pt, independent validation vs PyMatching 2.4.0):
        Repetition code (d=3–9):  LER within 0.00% of PyMatching
        Surface code (d=3–7):     LER within 1.78% of PyMatching
        pymatching_compat layer:  bit-identical to PyMatching 2.4.0

    Sample LER data (repetition code, Blossom = PyMatching at all distances):
        d=3, p=0.05: LER=0.0069  d=5, p=0.05: LER=0.0011
        d=7, p=0.05: LER=0.0002  d=9, p=0.05: LER<0.0001
    """

    def __init__(self, check_to_qubits, n_qubits=None, edge_weights=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustBlossomDecoder(c2q, nq, edge_weights)

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
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

    def __init__(
        self, check_to_qubits, n_qubits=None, window_size=10, decay_factor=0.8
    ):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustSlidingWindowDecoder(c2q, nq, window_size, decay_factor)

    def update(self, round_syndrome):
        if not isinstance(round_syndrome, np.ndarray):
            round_syndrome = np.array(round_syndrome, dtype=np.uint8)
        if round_syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {round_syndrome.dtype}")
        return self._inner.update(round_syndrome)

    def flush(self):
        self._inner.flush()

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
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
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustStreamingDecoder(c2q, nq, history_size)

    def update(self, round_syndrome):
        if not isinstance(round_syndrome, np.ndarray):
            round_syndrome = np.array(round_syndrome, dtype=np.uint8)
        if round_syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {round_syndrome.dtype}")
        return self._inner.update(round_syndrome)

    def flush(self):
        self._inner.flush()

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
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

    Performance (independently validated, Windows 10, AMD Ryzen, Python 3.11):
        - repetition_code(d=9):  ~2.67M shots/s  (parallel_batch_decode)
                                  ~1.88M shots/s  (batch_decode alias)
        Use parallel_batch_decode for maximum throughput.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustBatchDecoder(c2q, nq)

    def parallel_batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.parallel_batch_decode(syndromes)

    def decode(self, syndrome):
        """Single-syndrome decode. Wraps a 1-row batch for API consistency."""
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            syndrome = syndrome.astype(np.uint8)
        if syndrome.ndim != 1:
            raise ValueError(f"syndrome must be 1D, got shape {syndrome.shape}")
        return self.parallel_batch_decode(syndrome.reshape(1, -1))[0]

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
    """SIMD-friendly CPU batch decoder with pooled buffers and SoA transposition.

    Performance (independently validated, Windows 10, AMD Ryzen, Python 3.11):
        - repetition_code(d=5):  ~11.2 µs/decode
        - repetition_code(d=9):  ~9.7 µs/decode,  ~0.34M shots/s
        - rotated_surface_code(d=3): ~9.5 µs/decode
        - rotated_surface_code(d=5): ~10.7 µs/decode
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustCPUBatchDecoder(c2q, nq)

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
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
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustOpenCLBatchDecoder(c2q, nq)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
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
    def device_name(self) -> str:
        return str(self._inner.device_name)

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

    @staticmethod
    def is_available():
        """Return True if an OpenCL GPU is available on this system."""
        return _RustOpenCLBatchDecoder.is_available()


class CUDABatchDecoder:
    """GPU-accelerated native CUDA batch decoder.

    Uses a compiled CUDA kernel loaded through the CUDA Driver API. Falls back
    to CPU UnionFind for tiny batches or after repeated CUDA failures.

    Always call ``CUDABatchDecoder.is_available()`` before constructing.
    A descriptive ``RuntimeError`` is raised immediately if no CUDA GPU or
    driver is present -- no opaque Rust-level crash.

    Performance (independently validated, NVIDIA GTX 1660 Ti, CUDA 7.5, 100k shots):
        - repetition_code(d=9):  ~3.85M shots/s  (7.67× faster than CPU batch)
        - rotated_surface_code(d=5): ~3.40M shots/s (6.93× faster than CPU batch)
        - GPU output is 100% CPU-agreeing and 100% syndrome-valid at all
          tested batch sizes.

    Example::

        from qector_decoder_v3 import CUDABatchDecoder
        if CUDABatchDecoder.is_available():
            dec = CUDABatchDecoder(check_to_qubits, n_qubits)
            corrections = dec.batch_decode(syndromes)
        else:
            # Fall back to CPU batch decoding
            from qector_decoder_v3 import BatchDecoder
            dec = BatchDecoder(check_to_qubits, n_qubits)
            corrections = dec.parallel_batch_decode(syndromes)
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        if _RustCUDABatchDecoder is None:
            raise RuntimeError(
                "CUDA is not available: this build of qector-decoder-v3 was compiled "
                "without the 'cuda' feature. Install a CUDA-enabled wheel or use "
                "CUDABatchDecoder.is_available() to check before constructing."
            )
        if not CUDABatchDecoder.is_available():
            raise RuntimeError(
                "CUDA is not available on this system: no CUDA-capable GPU or driver "
                "was detected. Use CUDABatchDecoder.is_available() to check before "
                "constructing, or use BatchDecoder for CPU-based batch decoding."
            )
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustCUDABatchDecoder(c2q, nq)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
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
    def device_name(self) -> str:
        return str(self._inner.device_name)

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

    Performance (independently validated, Windows 10, AMD Ryzen, Python 3.11):
        - repetition_code(d=5):  ~11.8 µs/decode,  ~1.80M shots/s (batch)
        - repetition_code(d=9):  ~10.6 µs/decode
        - rotated_surface_code(d=3): ~11.5 µs/decode
        - rotated_surface_code(d=5): ~29.2 µs/decode

    Note on degeneracy:
        batch_decode may return different (but equally valid) corrections than
        single-shot decode on degenerate syndromes. Both are 100% syndrome-valid.
        This is benign matching degeneracy, not an error.
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustSparseBlossomDecoder(c2q, nq)

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def decode_with_weights(self, syndrome, weights):
        """Decode with per-qubit dynamic weight overrides.

        Args:
            syndrome: np.ndarray of shape (n_checks,) with dtype uint8.
            weights: List of (qubit_id, weight) tuples.

        Returns:
            np.ndarray of shape (n_qubits,) with correction.
        """
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        if not isinstance(weights, list):
            weights = list(weights)
        return self._inner.decode_with_weights(syndrome, weights)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
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
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustBPOSDDecoder(c2q, nq, error_rate)

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def bp_decode(self, syndrome, max_iterations=20):
        """Run Belief Propagation and return log-likelihood ratios (LLRs) for each qubit.

        Args:
            syndrome: np.ndarray of shape (n_checks,) with dtype uint8.
            max_iterations: Number of BP iterations (default: 20).

        Returns:
            np.ndarray of shape (n_qubits,) with LLR values.
            Positive LLR -> more likely 0, Negative LLR -> more likely 1.
        """
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
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
        # Ensure inputs are NumPy arrays of dtype uint8 and contiguous in memory
        syndromes = np.ascontiguousarray(np.asarray(syndromes, dtype=np.uint8))
        corrections = np.ascontiguousarray(np.asarray(corrections, dtype=np.uint8))
        # Validate dimensions: expect 2‑D arrays (samples, features)
        if syndromes.ndim != 2:
            raise ValueError(
                f"syndromes must be a 2‑D array, got shape {syndromes.shape}"
            )
        if corrections.ndim != 2:
            raise ValueError(
                f"corrections must be a 2‑D array, got shape {corrections.shape}"
            )
        try:
            self._inner.train(syndromes, corrections, n_epochs, learning_rate)
        except TypeError as exc:
            if (
                "not an instance of 'ndarray'" in str(exc)
                and int(np.__version__.split(".")[0]) >= 2
            ):
                raise RuntimeError(
                    "NeuralPredecoder.train() is not usable with numpy "
                    f"{np.__version__}. The compiled qector_decoder_v3 extension's "
                    "train() binding does a strict native array-type check that is "
                    "incompatible with numpy>=2.0 (this is a binary ABI issue in the "
                    "compiled wheel, not something fixable from Python -- passing "
                    "lists or rebuilding the array in pure Python does not help). "
                    "predict() and decode() are unaffected and work normally on this "
                    "numpy version. To train a model right now, use an environment "
                    "with 'numpy<2' installed; this is tracked for a native fix in a "
                    "future qector-decoder-v3 wheel rebuild."
                ) from exc
            raise

    def predict(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        return self._inner.predict(syndrome)

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
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

    # Standard dimensions matching DetectorGraph
    NODE_FEAT_DIM = 10
    EDGE_FEAT_DIM = 8

    def __init__(
        self, node_feat_dim=None, edge_feat_dim=None, hidden_size=16, n_layers=2
    ):
        """Create a GNNPredecoder.

        If node_feat_dim and edge_feat_dim are not provided, uses the standard
        dimensions matching DetectorGraph (10 and 8).
        """
        nfd = node_feat_dim if node_feat_dim is not None else self.NODE_FEAT_DIM
        efd = edge_feat_dim if edge_feat_dim is not None else self.EDGE_FEAT_DIM
        self._inner = _RustGNNPredecoder(nfd, efd, hidden_size, n_layers)

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

    def forward(self, graph):
        """Predict adjusted edge weights for a DetectorGraph."""
        return self._inner.forward(
            graph._inner if isinstance(graph, DetectorGraph) else graph
        )

    def train(self, graphs, targets, n_epochs):
        """Train the GNN on a list of graphs and target edge weights."""
        inner_graphs = [g._inner if isinstance(g, DetectorGraph) else g for g in graphs]
        return self._inner.train(inner_graphs, targets, n_epochs)

    def predict_with_node_probs(self, graph):
        """Predict edge weights and node error probabilities."""
        return self._inner.predict_with_node_probs(
            graph._inner if isinstance(graph, DetectorGraph) else graph
        )


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
        return self._inner.train(gnn._inner, n_samples, n_epochs)

    def train_bp(self, gnn, n_samples, n_epochs, max_bp_iter=20):
        """Train a GNNPredecoder with BP marginal targets and return the final MSE loss."""
        return self._inner.train_bp(gnn._inner, n_samples, n_epochs, max_bp_iter)

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
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustHybridDecoder(
            c2q,
            nq,
            check_positions,
            check_types,
            base_weights,
            gnn_hidden_size,
            gnn_n_layers,
        )

    def decode_hybrid(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode_hybrid(syndrome)

    def decode_heuristic(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode_heuristic(syndrome)

    def decode_standard(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode_standard(syndrome)

    def batch_decode_hybrid(self, syndromes):
        """Batch decode multiple syndromes using the GNN-enhanced pipeline.

        Args:
            syndromes: np.ndarray of shape (batch, n_checks) or list of lists.

        Returns:
            np.ndarray of shape (batch, n_qubits) with corrections.
        """
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            raise TypeError(f"Syndromes must be dtype uint8, got {syndromes.dtype}")
        if syndromes.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {syndromes.shape}")
        return self._inner.batch_decode_hybrid(syndromes)

    def batch_decode_standard(self, syndromes):
        """Batch decode multiple syndromes using standard SparseBlossom.

        Args:
            syndromes: np.ndarray of shape (batch, n_checks) or list of lists.

        Returns:
            np.ndarray of shape (batch, n_qubits) with corrections.
        """
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
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

    Performance (independently validated, Windows 10, AMD Ryzen, Python 3.11):
        - repetition_code(d=5):  ~8.7 µs/decode  (fastest single-shot decoder tested)
        - repetition_code(d=9):  ~10.7 µs/decode
        - rotated_surface_code(d=3): ~9.5 µs/decode
        - table_size for repetition_code(d=5): 64 entries
    """

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustLookupTableDecoder(c2q, nq)

    def build_table(self, max_entries):
        """Populate the lookup table by enumerating errors up to max_entries."""
        self._inner.build_table(int(max_entries))

    def decode(self, syndrome):
        if not isinstance(syndrome, np.ndarray):
            syndrome = np.array(syndrome, dtype=np.uint8)
        if syndrome.dtype != np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def batch_decode(self, syndromes):
        if not isinstance(syndromes, np.ndarray):
            syndromes = np.array(syndromes, dtype=np.uint8)
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
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
    """Generate the compact periodic surface-code checks used by this API."""
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
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
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
