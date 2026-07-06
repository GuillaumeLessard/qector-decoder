import re
import os

target_file = r"C:\Users\Admin\Desktop\qector-decoder-v3-0.5.0-frozen\python\qector_decoder_v3\__init__.py"

with open(target_file, "r", encoding="utf-8", newline="") as f:
    content = f.read()

# 1. Update version to 0.6.2 and insert validation function
old_version_block = """__fallback_version__ = "0.6.1"

try:
    from .qector_decoder_v3 import __version__
except (ImportError, AttributeError):
    __version__ = __fallback_version__

from typing import Optional

_OPENCL_HEALTH_CACHE: Optional[bool] = None"""

new_version_block = """__fallback_version__ = "0.6.2"

try:
    from .qector_decoder_v3 import __version__
except (ImportError, AttributeError):
    __version__ = __fallback_version__

from typing import Optional

_OPENCL_HEALTH_CACHE: Optional[bool] = None


def _validate_check_to_qubits(check_to_qubits, n_qubits=None, *, reject_hyperedges=False):
    \"\"\"Validate and normalize check_to_qubits for all decoders.

    Args:
        check_to_qubits: List of lists of qubit indices.
        n_qubits: Optional total qubit count.
        reject_hyperedges: If True, raise ValueError when any qubit appears in
            more than 2 checks (Union-Find / FastUnionFind limitation).

    Returns:
        Tuple of (normalized_c2q, resolved_n_qubits).

    Raises:
        ValueError: On empty input, empty checks, negative indices, out-of-range
            qubits, duplicate qubits in a check, or hyperedges when rejected.
        TypeError: On non-integer qubit indices.
    \"\"\"
    if not check_to_qubits:
        raise ValueError("check_to_qubits must be non-empty")

    normalized = []
    max_q = -1
    for i, check in enumerate(check_to_qubits):
        if not check:
            raise ValueError(f"Check {i} is empty")
        cleaned = []
        seen = set()
        for q in check:
            if not isinstance(q, (int, np.integer)):
                raise TypeError(
                    f"Qubit index must be integer, got {type(q).__name__} in check {i}"
                )
            qi = int(q)
            if qi < 0:
                raise ValueError(f"Negative qubit index {qi} in check {i}")
            if qi in seen:
                raise ValueError(f"Duplicate qubit {qi} in check {i}")
            seen.add(qi)
            cleaned.append(qi)
            if qi > max_q:
                max_q = qi
        normalized.append(cleaned)

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
        qubit_degree = {}
        for ci, check in enumerate(normalized):
            for q in check:
                qubit_degree[q] = qubit_degree.get(q, 0) + 1
        for q, deg in qubit_degree.items():
            if deg > 2:
                raise ValueError(
                    f"UnionFindDecoder / FastUnionFindDecoder only support stabilizer codes "
                    f"with checks of weight \u2264 2.\\n"
                    f"Check {q} has {deg} qubits (hyperedge).\\n"
                    f"Use BlossomDecoder, SparseBlossomDecoder, or BPOSDDecoder instead.\\n"
                    f"(Codes from generate_surface_code_checks() contain weight-4 checks)"
                )

    return normalized, nq"""

# Replace with normalized newlines
content = content.replace(old_version_block.replace("\n", "\r\n"), new_version_block.replace("\n", "\r\n"))

# Define other class replacements
class_replacements = [
    # UnionFindDecoder
    (
        """class UnionFindDecoder:
    \"\"\"Production-ready Union-Find quantum error correction decoder.

    Rust core with PyO3 bindings. Zero-copy NumPy interop.
    GIL is released during decode for true parallelism.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        # Convert Python list-of-lists to Vec<Vec<u32>>
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustUnionFindDecoder(c2q, nq)""",
        """class UnionFindDecoder:
    \"\"\"Production-ready Union-Find quantum error correction decoder.

    Rust core with PyO3 bindings. Zero-copy NumPy interop.
    GIL is released during decode for true parallelism.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        self._inner = _RustUnionFindDecoder(c2q, nq)"""
    ),
    # FastUnionFindDecoder
    (
        """class FastUnionFindDecoder:
    \"\"\"SIMD-accelerated zero-allocation Union-Find decoder.

    Uses pre-allocated reusable buffers, AVX2 runtime dispatch, and FFI.
    Same API as UnionFindDecoder but with lower overhead.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustFastUnionFindDecoder(c2q, nq)""",
        """class FastUnionFindDecoder:
    \"\"\"SIMD-accelerated zero-allocation Union-Find decoder.

    Uses pre-allocated reusable buffers, AVX2 runtime dispatch, and FFI.
    Same API as UnionFindDecoder but with lower overhead.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        self._inner = _RustFastUnionFindDecoder(c2q, nq)"""
    ),
    # BlossomDecoder
    (
        """class BlossomDecoder:
    \"\"\"Minimum-Weight Perfect Matching (MWPM) decoder via Edmonds' Blossom algorithm.

    Supports weighted edges for higher decoding accuracy on realistic codes.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, edge_weights=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustBlossomDecoder(c2q, nq, edge_weights)""",
        """class BlossomDecoder:
    \"\"\"Minimum-Weight Perfect Matching (MWPM) decoder via Edmonds' Blossom algorithm.

    Supports weighted edges for higher decoding accuracy on realistic codes.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, edge_weights=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustBlossomDecoder(c2q, nq, edge_weights)"""
    ),
    # SlidingWindowDecoder
    (
        """class SlidingWindowDecoder:
    \"\"\"Sliding-window decoder with exponential decay weighting.

    Maintains a window of the last W rounds. Each round's syndrome is weighted
    by ``decay_factor ** age`` so that more recent rounds contribute more.
    The weighted cumulative syndrome is thresholded at 0.5 and decoded with
    the standard Union-Find decoder.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, window_size=10, decay_factor=0.8):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustSlidingWindowDecoder(c2q, nq, window_size, decay_factor)""",
        """class SlidingWindowDecoder:
    \"\"\"Sliding-window decoder with exponential decay weighting.

    Maintains a window of the last W rounds. Each round's syndrome is weighted
    by ``decay_factor ** age`` so that more recent rounds contribute more.
    The weighted cumulative syndrome is thresholded at 0.5 and decoded with
    the standard Union-Find decoder.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, window_size=10, decay_factor=0.8):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustSlidingWindowDecoder(c2q, nq, window_size, decay_factor)"""
    ),
    # StreamingDecoder
    (
        """class StreamingDecoder:
    \"\"\"Streaming decoder that accumulates syndromes over multiple rounds.

    Rust core with circular history buffer and OR accumulation.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, history_size=10):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustStreamingDecoder(c2q, nq, history_size)""",
        """class StreamingDecoder:
    \"\"\"Streaming decoder that accumulates syndromes over multiple rounds.

    Rust core with circular history buffer and OR accumulation.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, history_size=10):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustStreamingDecoder(c2q, nq, history_size)"""
    ),
    # BatchDecoder
    (
        """class BatchDecoder:
    \"\"\"Parallel batch decoder using Rayon (Rust data parallelism).

    Distributes batch decoding across all CPU cores.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustBatchDecoder(c2q, nq)""",
        """class BatchDecoder:
    \"\"\"Parallel batch decoder using Rayon (Rust data parallelism).

    Distributes batch decoding across all CPU cores.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        self._inner = _RustBatchDecoder(c2q, nq)"""
    ),
    # CPUBatchDecoder
    (
        """class CPUBatchDecoder:
    \"\"\"SIMD-friendly CPU batch decoder with pooled buffers and SoA transposition.\"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustCPUBatchDecoder(c2q, nq)""",
        """class CPUBatchDecoder:
    \"\"\"SIMD-friendly CPU batch decoder with pooled buffers and SoA transposition.\"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        self._inner = _RustCPUBatchDecoder(c2q, nq)"""
    ),
    # OpenCLBatchDecoder
    (
        """    def __init__(self, check_to_qubits, n_qubits=None):
        if os.environ.get("QECTOR_OPENCL_PROBE_CHILD") != "1" and not _opencl_health_check():
            raise RuntimeError(
                "OpenCL backend is unavailable or failed its health check; "
                "use CPUBatchDecoder/AutoDecoder fallback or set QECTOR_DISABLE_OPENCL=1"
            )
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustOpenCLBatchDecoder(c2q, nq)""",
        """    def __init__(self, check_to_qubits, n_qubits=None):
        if os.environ.get("QECTOR_OPENCL_PROBE_CHILD") != "1" and not _opencl_health_check():
            raise RuntimeError(
                "OpenCL backend is unavailable or failed its health check; "
                "use CPUBatchDecoder/AutoDecoder fallback or set QECTOR_DISABLE_OPENCL=1"
            )
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        self._inner = _RustOpenCLBatchDecoder(c2q, nq)"""
    ),
    # CUDABatchDecoder
    (
        """    def __init__(self, check_to_qubits, n_qubits=None):
        if _RustCUDABatchDecoder is None:
            raise RuntimeError("qector-decoder-v3 was built without the 'cuda' feature")
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustCUDABatchDecoder(c2q, nq)""",
        """    def __init__(self, check_to_qubits, n_qubits=None):
        if _RustCUDABatchDecoder is None:
            raise RuntimeError("qector-decoder-v3 was built without the 'cuda' feature")
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        self._inner = _RustCUDABatchDecoder(c2q, nq)"""
    ),
    # SparseBlossomDecoder
    (
        """class SparseBlossomDecoder:
    \"\"\"Region-growing Sparse Blossom decoder with RadixHeap.

    Supports dynamic weight overrides from GNN Pre-Decoder for enriched decoding.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustSparseBlossomDecoder(c2q, nq)""",
        """class SparseBlossomDecoder:
    \"\"\"Region-growing Sparse Blossom decoder with RadixHeap.

    Supports dynamic weight overrides from GNN Pre-Decoder for enriched decoding.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustSparseBlossomDecoder(c2q, nq)"""
    ),
    # BPOSDDecoder
    (
        """class BPOSDDecoder:
    \"\"\"Belief Propagation + Ordered Statistics Decoding.

    Min-sum BP with OSD stage for improved LER on complex codes.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, error_rate=0.1):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustBPOSDDecoder(c2q, nq, error_rate)""",
        """class BPOSDDecoder:
    \"\"\"Belief Propagation + Ordered Statistics Decoding.

    Min-sum BP with OSD stage for improved LER on complex codes.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, error_rate=0.1):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustBPOSDDecoder(c2q, nq, error_rate)"""
    ),
    # HybridDecoder
    (
        """    def __init__(
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
            c2q, nq, check_positions, check_types, base_weights, gnn_hidden_size, gnn_n_layers
        )""",
        """    def __init__(
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
        )"""
    ),
    # LookupTableDecoder
    (
        """class LookupTableDecoder:
    \"\"\"Exact lookup-table decoder with UnionFind fallback.

    Pre-computes all syndrome \u2192 correction mappings for small codes
    (n_qubits \u2264 20, exhaustive; otherwise low-weight enumeration).
    Decoding is O(1) for precomputed syndromes, fallback to UnionFind otherwise.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustLookupTableDecoder(c2q, nq)""",
        """class LookupTableDecoder:
    \"\"\"Exact lookup-table decoder with UnionFind fallback.

    Pre-computes all syndrome \u2192 correction mappings for small codes
    (n_qubits \u2264 20, exhaustive; otherwise low-weight enumeration).
    Decoding is O(1) for precomputed syndromes, fallback to UnionFind otherwise.
    \"\"\"

    def __init__(self, check_to_qubits, n_qubits=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustLookupTableDecoder(c2q, nq)"""
    ),
    # BenchmarkSuite
    (
        """class BenchmarkSuite:
    \"\"\"Production benchmark suite. Wraps Rust-native benchmarking.\"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, n_samples=10000, seed=42):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        c2q = [[int(q) for q in check] for check in check_to_qubits]
        nq = None if n_qubits is None else int(n_qubits)
        self._inner = _RustBenchmarkSuite(c2q, nq, n_samples, seed)""",
        """class BenchmarkSuite:
    \"\"\"Production benchmark suite. Wraps Rust-native benchmarking.\"\"\"

    def __init__(self, check_to_qubits, n_qubits=None, n_samples=10000, seed=42):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustBenchmarkSuite(c2q, nq, n_samples, seed)"""
    ),
]

for orig, repl in class_replacements:
    orig_crlf = orig.replace("\n", "\r\n")
    repl_crlf = repl.replace("\n", "\r\n")
    if orig_crlf not in content:
        # Fallback to check LF in case some text wasn't mapped
        print(f"Warning: exact CRLF match not found for block starting with: {orig.splitlines()[0]}")
    content = content.replace(orig_crlf, repl_crlf)

with open(target_file, "w", encoding="utf-8", newline="") as f:
    f.write(content)

print("Replacement complete successfully!")
