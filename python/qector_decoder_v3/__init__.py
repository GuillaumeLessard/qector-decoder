"""
QECTOR Decoder v3 - Source-available Rust/Python QEC decoders with reproducible, artifact-hashed benchmark evidence.
Rust core + PyO3 bindings. Zero-copy NumPy. GIL-free decode.
"""

# pyright: reportMissingImports=false
# The compiled Rust extension provides this submodule at runtime in the installed package.
import importlib as _importlib
import math as _math


def _load_native_module():
    try:
        return _importlib.import_module(".qector_decoder_v3", __name__)  # pyright: ignore[reportMissingImports]
    except (ImportError, ModuleNotFoundError):
        return _importlib.import_module("qector_decoder_v3.qector_decoder_v3")


_native_module = _load_native_module()


def _guard(name):
    try:
        return getattr(_native_module, name)
    except (AttributeError, ImportError):

        class _Unavailable:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    f"{name} is not available in this build. Install a wheel with the required features enabled."
                )

            def __repr__(self):
                return f"<unavailable {name}>"

        _Unavailable.__name__ = name
        _Unavailable.__qualname__ = name
        _Unavailable.__module__ = __name__
        return _Unavailable


_RustUnionFindDecoder = _guard("UnionFindDecoder")
_RustFastUnionFindDecoder = _guard("FastUnionFindDecoder")
_RustBlossomDecoder = _guard("BlossomDecoder")
_RustSlidingWindowDecoder = _guard("SlidingWindowDecoder")
_RustStreamingDecoder = _guard("StreamingDecoder")
_RustBatchDecoder = _guard("BatchDecoder")
_RustCPUBatchDecoder = _guard("CPUBatchDecoder")
_RustBenchmarkSuite = _guard("BenchmarkSuite")
_RustLookupTableDecoder = _guard("LookupTableDecoder")
_RustBPOSDDecoder = _guard("BPOSDDecoder")
_RustNeuralPredecoder = _guard("NeuralPredecoder")
_RustDetectorGraph = _guard("DetectorGraph")
_RustGNNPredecoder = _guard("GNNPredecoder")
_RustGNNTrainer = _guard("GNNTrainer")
_RustLERBenchmark = _guard("LERBenchmark")
_RustSparseBlossomDecoder = _guard("SparseBlossomDecoder")
_RustHybridDecoder = _guard("HybridDecoder")
_RustHybridCascadeDecoder = _guard("HybridCascadeDecoder")
_RustAutoDecoder = _guard("AutoDecoder")
_RustTwoStageDecoder = _guard("TwoStageDecoder")
_RustAmbiguityClusterDecoder = _guard("AmbiguityClusterDecoder")
# 3D space-time decoder. `qector_decoder_v3.pyi` has declared this class since
# the stub was written and lib.rs registers it, but `__init__.py` never bound
# it -- so type checkers accepted `qd.SpaceTimeDecoder(...)` and the runtime
# raised AttributeError.
SpaceTimeDecoder = _guard("SpaceTimeDecoder")

# The Rust-native routing decoder is exported as `NativeAutoDecoder`; the
# public `AutoDecoder` name stays with the 7-tier self-debugging fallback
# controller imported from `.backend` below (F811: no redefinition).
NativeAutoDecoder = _RustAutoDecoder


def set_license_key(key: str) -> None:
    """Set the global QECTOR license key for tier enforcement.

    Raises ``ValueError`` when the native verifier rejects the key. This used to
    swallow the rejection and set the environment variable anyway, so an
    expired, revoked, or malformed key looked like it had been accepted and the
    caller only found out later, as an unexplained Community-tier distance cap.
    """
    import os

    fn = getattr(_native_module, "py_set_license_key", None)
    if fn is not None:
        try:
            fn(key)
        except Exception as exc:  # native verifier rejected the key
            raise ValueError(f"license key rejected: {exc}") from exc

    os.environ["QECTOR_LICENSE_KEY"] = key


def set_license_key_file(path) -> None:
    """Load a license key from *path* and activate it.

    Mirrors the core's ``QECTOR_LICENSE_FILE`` discovery, including stripping a
    UTF-8 BOM and trailing newline, so a key file written by PowerShell
    redirection activates unmodified.
    """
    import os

    with open(path, encoding="utf-8") as fh:
        key = fh.read().lstrip("﻿").strip()
    if not key:
        raise ValueError(f"license key file is empty: {path}")
    set_license_key(key)
    os.environ["QECTOR_LICENSE_FILE"] = str(path)


def get_license_info() -> dict:
    """Get active license info dictionary."""
    try:
        fn = getattr(_native_module, "py_get_license_info", None)
        if fn is not None:
            return fn()
    except Exception:
        pass
    # Resolve the key the same way the core does (env var, then
    # QECTOR_LICENSE_FILE, then ~/.qector/license.key) so this fallback cannot
    # report Community for a deployment the core reads as Enterprise.
    from .license import _configured_key

    k = _configured_key()
    tier = "Pro" if k.startswith("QECT-PRO-") else ("Enterprise" if k.startswith("QECT-ENT-") else "Community")
    return {
        "tier": tier,
        "customer_id": "free_tier",
        # int, matching the native LicenseManager and license._TIER_MAX_DISTANCE.
        # This was a string, so get_license_info()["max_distance"] changed type
        # depending on whether the compiled core was present.
        "max_distance": {"Community": 7, "Pro": 19, "Enterprise": 63}[tier],
    }


def enforce_unlocked() -> None:
    """Hard-gate decode path when no valid license is active (QECTOR_ENFORCE=1)."""
    try:
        fn = getattr(_native_module, "py_enforce_unlocked", None)
        if fn is not None:
            fn()
    except Exception:
        pass


def enforce_distance_cap(distance) -> None:
    """Enforce that *distance* does not exceed the active tier's cap."""
    from . import license as _license_module

    _license_module.enforce_distance_cap(int(distance))


def record_shots(count: int) -> None:
    """Record decoding shots into telemetry buffer."""
    try:
        fn = getattr(_native_module, "py_record_shots", None)
        if fn is not None:
            fn(count)
    except Exception:
        pass


def get_accumulated_shots() -> int:
    """Get total accumulated decode shots."""
    try:
        fn = getattr(_native_module, "py_get_accumulated_shots", None)
        if fn is not None:
            return fn()
    except Exception:
        pass
    return 0


def flush_usage(customer_id: "str | None" = None, api_key: "str | None" = None) -> dict:
    """Flush accumulated decode shots to the Stripe Billing Meter Events API.

    ``customer_id`` falls back to ``QECTOR_STRIPE_CUSTOMER_ID`` and ``api_key``
    to ``STRIPE_SECRET_KEY``. Returns the receipt dict on success.

    Unlike :func:`record_shots`, failures are **not** swallowed: a failed flush
    leaves the shots pending for a later retry, and the caller has to know that
    happened in order to retry. Raises ``RuntimeError`` when the core is absent
    or the Stripe call fails.
    """
    fn = getattr(_native_module, "py_flush_usage", None)
    if fn is None:
        raise RuntimeError("flush_usage requires the compiled core; this build does not provide it.")
    return fn(customer_id, api_key)


# ---------------------------------------------------------------------------
# Utility functions: native when available, exact pure-Python fallback when not.
#
# Release wheels compile the Rust core from the RUST_SRC_B64_* secrets bundle,
# which can lag the Python layer in this repo. v0.6.9's first tag build died at
# import because `py_generate_parity_check_matrix` was registered in the repo's
# lib.rs but absent from the bundle's older core -- the same failure mode that
# made every v0.6.7 wheel unimportable. Import must never depend on the native
# module's exact symbol set.
#
# Unlike the class guards above (which stub and raise on use), these six have
# exact pure-Python equivalents, so a wheel with an older core keeps FULL
# functionality. Each fallback mirrors src/utils.rs one-to-one, including edge
# cases (out-of-range qubits dropped, n_qubits inferred as max+1, surface
# distance >= 2, repetition distance == 0 -> empty). Equivalence against the
# native implementations is locked by python/tests/test_native_fallbacks.py.
# ---------------------------------------------------------------------------


def _py_check_to_edges(check_to_qubits):
    edges = []
    for qubits in check_to_qubits:
        if len(qubits) < 2:
            continue
        for i in range(len(qubits) - 1):
            edges.append((int(qubits[i]), int(qubits[i + 1])))
    return edges


def _py_generate_ring_code_checks(distance):
    n_qubits = distance * distance
    return [[i, (i + 1) % n_qubits] for i in range(n_qubits)], n_qubits


def _py_generate_surface_code_checks(distance):
    if distance < 2:
        raise ValueError("distance must be >= 2")
    d = distance

    def q(row, col):
        return (row % d) * d + (col % d)

    checks = []
    for row in range(d):
        for col in range(d):
            checks.append([q(row, col), q(row, col + 1), q(row + 1, col), q(row + 1, col + 1)])
    for row in range(d):
        for col in range(d):
            checks.append([q(row, col), q(row, col + d - 1), q(row + d - 1, col), q(row + d - 1, col + d - 1)])
    return checks, d * d


def _py_generate_toy_code_checks(distance):
    n = distance
    checks = []
    for _ in range(2):
        for row in range(n):
            for col in range(n):
                q0 = row * n + col
                q1 = row * n + ((col + 1) % n)
                q2 = ((row + 1) % n) * n + col
                q3 = ((row + 1) % n) * n + ((col + 1) % n)
                checks.append([q0, q1, q2, q3])
    return checks, n * n


def _py_generate_repetition_code_checks(distance):
    if distance == 0:
        return [], 0
    return [[i, i + 1] for i in range(distance - 1)], distance


def _py_generate_parity_check_matrix(check_to_qubits, n_qubits=None):
    import numpy as np

    n_checks = len(check_to_qubits)
    max_q = max((int(q) for qs in check_to_qubits for q in qs), default=0)
    if n_qubits is None:
        n_qubits = max_q + 1
    h = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    for ci, qs in enumerate(check_to_qubits):
        for q in qs:
            if int(q) < n_qubits:
                h[ci, int(q)] = 1
    return h


def _native_or(name, fallback):
    """Prefer the compiled implementation; fall back without losing function."""
    return getattr(_native_module, name, fallback)


py_check_to_edges = _native_or("py_check_to_edges", _py_check_to_edges)
py_generate_surface_code_checks = _native_or("py_generate_surface_code_checks", _py_generate_surface_code_checks)
py_generate_toy_code_checks = _native_or("py_generate_toy_code_checks", _py_generate_toy_code_checks)
py_generate_ring_code_checks = _native_or("py_generate_ring_code_checks", _py_generate_ring_code_checks)
py_generate_repetition_code_checks = _native_or(
    "py_generate_repetition_code_checks", _py_generate_repetition_code_checks
)
# `py_generate_parity_check_matrix` is registered in lib.rs and
# `_py_generate_parity_check_matrix` has always existed as its pure-Python
# fallback, but neither was ever bound to a module-level name -- so the native
# function was unreachable from Python and the fallback was dead code.
py_generate_parity_check_matrix = _native_or("py_generate_parity_check_matrix", _py_generate_parity_check_matrix)


def _py_estimate_distance(check_to_qubits, n_qubits=None):
    if not check_to_qubits:
        return 0
    if n_qubits is None:
        max_q = 0
        for qs in check_to_qubits:
            for q in qs:
                max_q = max(max_q, q)
        nq = max_q + 1
    else:
        nq = n_qubits
    if nq == 0:
        return 0
    n_checks = len(check_to_qubits)
    if all(len(qs) == 2 for qs in check_to_qubits) and n_checks > 1:
        deg = {}
        for qs in check_to_qubits:
            deg[qs[0]] = deg.get(qs[0], 0) + 1
            deg[qs[1]] = deg.get(qs[1], 0) + 1
        if max(deg.values(), default=0) <= 2 and nq == n_checks + 1:
            return nq
    if nq > 2 * n_checks and n_checks > 1:
        # DEM columns are fault mechanisms, not physical qubits.  Infer the
        # spatial stride only from detector incidences of graphlike mechanisms;
        # mechanism ids in a check row are arbitrary enumeration details.
        mech_to_dets = {}
        for det_idx, qs in enumerate(check_to_qubits):
            for mech in qs:
                mech_to_dets.setdefault(mech, []).append(det_idx)
        diff_counts = {}
        for dets in mech_to_dets.values():
            if len(dets) == 2:
                diff = abs(dets[0] - dets[1])
                if diff > 0:
                    diff_counts[diff] = diff_counts.get(diff, 0) + 1
        if diff_counts:
            stride, count = max(diff_counts.items(), key=lambda x: x[1])
            if count >= 3 and stride > 0:
                return stride
        # An arbitrary DEM's detector numbering need not encode geometry.
        # Return a documented lower bound instead of turning its fault count
        # into a spurious licence rejection; callers may supply a validated
        # structural distance hint when one exists.
        return 1
    return max(_math.ceil(_math.sqrt(nq)), 1)


estimate_distance = _native_or("py_estimate_distance", _py_estimate_distance)
try:
    run_mcp_server = _native_module.run_mcp_server
except AttributeError:

    def run_mcp_server(*args, **kwargs):
        raise RuntimeError(
            "The native core does not export run_mcp_server. This is likely a build mismatch — rebuild with 'maturin develop' to include the MCP server."
        )


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
    get_latency_quantiles = _native_module.get_latency_quantiles
except (AttributeError, ImportError):
    get_latency_quantiles = None  # type: ignore[assignment]

try:
    import importlib.util as _importlib_util

    if _importlib_util.find_spec("torch") is not None:
        import torch.nn.functional as _F
    else:
        _F = None  # type: ignore[assignment]
except Exception:
    _F = None  # type: ignore[assignment]

import numpy as _np

# os / sys / subprocess are imported locally inside the functions that need them
# to avoid polluting the public module namespace (see end of file)

# The authoritative runtime version is whatever the compiled Rust core reports.
# ``__fallback_version__`` is the Python-packaging version (pyproject/Cargo) and is
# used ONLY when the compiled module does not expose ``__version__`` (e.g. an old
# wheel). We never overwrite a real compiled ``__version__`` with it - doing so
# would falsely claim a version the loaded binary is not - so after a version bump
# ``__version__`` keeps reporting the *built* value until the Rust wheel is rebuilt.
__fallback_version__ = "0.7.0"

try:
    __version__ = _native_module.__version__
except (AttributeError, ImportError):
    __version__ = __fallback_version__

import logging as _logging
import os as _os_mod
import sys as _sys_mod

from . import license
from .license import verify_license_token

__license__ = "LicenseRef-QECTOR-Source-Available"

# Prevent logger warnings
_logging.getLogger("qector_decoder_v3").addHandler(_logging.NullHandler())

# Core threading configuration
MAX_WORKERS: int = _os_mod.cpu_count() or 1


def _is_license_active() -> bool:
    """Checks if a valid license signature or override is present."""
    token = _os_mod.environ.get("QECTOR_LICENSE", "").strip()
    if not token:
        return False
    return verify_license_token(token)


def _emit_startup_notice() -> None:
    """Emit the licensing notice once per process when unlicensed.

    Previously this fired only when `sys.ps1` existed or IPython was loaded --
    i.e. only in a REPL or notebook. Commercial users run this library from
    scripts, services and job schedulers, so the audience most likely to owe a
    licence was the one audience that never saw the notice.

    It now emits for non-interactive runs too. Deliberate boundaries:

    * It is a single stderr message, written once at import. It never repeats,
      never blocks, never delays, and never degrades or gates any functionality
      -- the library behaves identically licensed or not.
    * `QECTOR_SILENT=1` suppresses it, and a valid `QECTOR_LICENSE` removes it.
    * CI is skipped: a build log is not a purchasing decision, and noise there
      just trains people to filter the output.
    * It goes to stderr, so it never contaminates piped stdout.
    """
    if _os_mod.environ.get("QECTOR_SILENT") or _is_license_active():
        return

    ci_vars = {"CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "CIRCLECI", "JENKINS_URL", "TF_BUILD"}
    if any(var in _os_mod.environ for var in ci_vars):
        return

    is_interactive = hasattr(_sys_mod, "ps1") or "ipykernel" in _sys_mod.modules or "IPython" in _sys_mod.modules

    lines = [
        f"[QECTOR v{__version__}] Source-available QEC decoder - Rust/PyO3 core.",
        f"[QECTOR v{__version__}] Free for academic, personal and non-commercial research.",
        f"[QECTOR v{__version__}] Commercial use requires a licence: https://qector.store/pricing",
    ]
    if not is_interactive:
        # A script or service is the commercial-use signal; point at the
        # 60-day evaluation, which is creditable toward any annual licence.
        lines.append(
            f"[QECTOR v{__version__}] 60-day commercial evaluation (creditable): "
            f"https://qector.store/pricing | Set QECTOR_SILENT=1 to hide this."
        )
    else:
        lines.append(f"[QECTOR v{__version__}] Set QECTOR_SILENT=1 to hide this notice.")

    try:
        _sys_mod.stderr.write("\n".join(lines) + "\n")
        _sys_mod.stderr.flush()
    except (OSError, ValueError, UnicodeEncodeError):
        # stderr can be closed, redirected to a dead pipe, or attached to a
        # console whose encoding rejects a character. A licensing notice must
        # never take down the import, but the catch is narrowed to those cases
        # rather than swallowing everything.
        pass


_emit_startup_notice()


from typing import Optional as _Optional

_OPENCL_HEALTH_CACHE: _Optional[bool] = None


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
        max_q = max(max_q, check_max_q)

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
        local_max_q = max(local_max_q, qi)
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


def _normalise_edge_weights(edge_weights, n_qubits):
    """Validate optional per-mechanism weights and return a plain list, or None.

    Raises rather than silently dropping bad input: a decoder that quietly
    ignored its weights is exactly the defect UF-01 exists to fix.
    """
    if edge_weights is None:
        return None
    w = _np.asarray(edge_weights, dtype=_np.float64).reshape(-1)
    if w.shape[0] != n_qubits:
        raise ValueError(f"edge_weights has length {w.shape[0]}, expected n_qubits={n_qubits}")
    if not _np.all(_np.isfinite(w)):
        bad = int(_np.flatnonzero(~_np.isfinite(w))[0])
        raise ValueError(f"edge_weights[{bad}] is not finite ({w[bad]}); weights must be finite")
    return w.tolist()


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
        "out = np.asarray(dec.batch_decode(np.array([[1]], dtype=np.uint8)), dtype=np.uint8)\n"
        "assert out.shape == (1, 2)\n"
    )
    try:
        proc = _subprocess_local.run(
            [_sys_local.executable, "-c", code],
            check=False,  # probe: a non-zero exit means "no OpenCL", not an error
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

    Args:
        check_to_qubits: per check, the qubit / mechanism indices it covers.
        n_qubits: total qubit count (inferred from the max index when omitted).
        edge_weights: optional per-mechanism matching weights ``log((1-p)/p)``
            (UF-01). When given, clusters grow in proportion to weight and the
            peel runs over a minimum-weight spanning forest, so the decoder can
            distinguish a likely fault mechanism from an unlikely one. Omit for
            the topology-only behaviour, which is what every release before
            v0.7.0 did unconditionally.

    A weighted decoder is markedly more accurate under circuit-level noise,
    where mechanism probabilities span orders of magnitude; on code-capacity
    noise with a single uniform ``p`` the weights carry no information and the
    two agree.
    """

    # T2.4 of PERF_AUDIT: ``__slots__`` removes the per-instance ``__dict__``
    # and the associated hashing cost on every attribute access. On a
    # 1-µs Rust decode this is the dominant Python-side overhead.
    __slots__ = ("_inner",)

    def __init__(self, check_to_qubits, n_qubits=None, edge_weights=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        w = _normalise_edge_weights(edge_weights, nq)
        self._inner = _RustUnionFindDecoder(c2q, nq, None, w)

    def set_edge_weights(self, edge_weights):
        """Replace the per-mechanism weights in place (no graph rebuild)."""
        self._inner.set_edge_weights(_normalise_edge_weights(edge_weights, self.n_qubits))

    @property
    def is_weighted(self):
        """True iff cluster growth uses per-mechanism weights (UF-01)."""
        return bool(self._inner.is_weighted)

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

    def decode_batch_flat(self, flat_syndromes, num_shots, stride=None):
        """Flat-batch decode — accepts a 1D array ``(num_shots × stride,)``
        and returns a 1D correction vector ``(num_shots × n_qubits,)``.

        Args:
            flat_syndromes: 1D ``uint8`` array of ``num_shots × stride`` bytes.
            num_shots: Number of shots in the batch.
            stride: Byte offset between consecutive shots in the input.
                When ``None``, defaults to ``n_checks`` (tight packing).
        """
        if not isinstance(flat_syndromes, _np.ndarray):
            flat_syndromes = _np.array(flat_syndromes, dtype=_np.uint8)
        if flat_syndromes.dtype != _np.uint8:
            flat_syndromes = flat_syndromes.astype(_np.uint8)
        if flat_syndromes.ndim != 1:
            raise ValueError(f"flat_syndromes must be 1D, got shape {flat_syndromes.shape}")
        if stride is None:
            stride = self.n_checks
        return self._inner.decode_batch_flat(flat_syndromes, num_shots, stride)

    def decode_batch_into_flat(self, flat_syndromes, num_shots, corrections, stride=None):
        """Zero-copy flat-batch decode — writes corrections into a caller-owned
        1D buffer ``(num_shots × n_qubits,)``.

        Args:
            flat_syndromes: 1D ``uint8`` array of ``num_shots × stride`` bytes.
            num_shots: Number of shots in the batch.
            corrections: Writable 1D ``uint8`` array ``(num_shots × n_qubits,)``.
            stride: Byte offset between consecutive shots in the input.
                When ``None``, defaults to ``n_checks`` (tight packing).
        """
        if not isinstance(flat_syndromes, _np.ndarray):
            flat_syndromes = _np.array(flat_syndromes, dtype=_np.uint8)
        if flat_syndromes.dtype != _np.uint8:
            flat_syndromes = flat_syndromes.astype(_np.uint8)
        if flat_syndromes.ndim != 1:
            raise ValueError(f"flat_syndromes must be 1D, got shape {flat_syndromes.shape}")
        if not isinstance(corrections, _np.ndarray):
            corrections = _np.array(corrections, dtype=_np.uint8)
        if corrections.dtype != _np.uint8:
            corrections = corrections.astype(_np.uint8)
        if corrections.ndim != 1:
            raise ValueError(f"corrections must be 1D, got shape {corrections.shape}")
        if stride is None:
            stride = self.n_checks
        self._inner.decode_batch_into_flat(flat_syndromes, num_shots, stride, corrections)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class FastUnionFindDecoder:
    """Union-Find with a zero-allocation hot path, reusable buffers and a C ABI.

    Routes to the same :class:`UnionFindDecoder` core, so corrections are
    bit-identical; the two differ only in batching / FFI ergonomics. Accepts the
    same ``edge_weights`` (UF-01).

    Note:
        This class is **not** measurably faster than :class:`UnionFindDecoder`
        per shot — both dispatch to the same ``UfGraph`` growth and peel. Its
        advantage is in the batch and FFI paths, where buffers are reused across
        shots. (Earlier releases documented "1.1M shots/s, consistently faster";
        that figure was measured on a ring code and did not survive measurement
        against either a real detector error model or its own sibling.)
    """

    __slots__ = ("_inner",)

    def __init__(self, check_to_qubits, n_qubits=None, edge_weights=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits, reject_hyperedges=True)
        w = _normalise_edge_weights(edge_weights, nq)
        self._inner = _RustFastUnionFindDecoder(c2q, nq, None, w)

    def set_edge_weights(self, edge_weights):
        """Replace the per-mechanism weights in place (no graph rebuild)."""
        self._inner.set_edge_weights(_normalise_edge_weights(edge_weights, self.n_qubits))

    @property
    def is_weighted(self):
        """True iff cluster growth uses per-mechanism weights (UF-01)."""
        return bool(self._inner.is_weighted)

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

    def decode_batch_flat(self, flat_syndromes, num_shots, stride=None):
        if not isinstance(flat_syndromes, _np.ndarray):
            flat_syndromes = _np.array(flat_syndromes, dtype=_np.uint8)
        if flat_syndromes.dtype != _np.uint8:
            flat_syndromes = flat_syndromes.astype(_np.uint8)
        if flat_syndromes.ndim != 1:
            raise ValueError(f"flat_syndromes must be 1D, got shape {flat_syndromes.shape}")
        if stride is None:
            stride = self.n_checks
        return self._inner.decode_batch_flat(flat_syndromes, num_shots, stride)

    def decode_batch_into_flat(self, flat_syndromes, num_shots, corrections, stride=None):
        if not isinstance(flat_syndromes, _np.ndarray):
            flat_syndromes = _np.array(flat_syndromes, dtype=_np.uint8)
        if flat_syndromes.dtype != _np.uint8:
            flat_syndromes = flat_syndromes.astype(_np.uint8)
        if flat_syndromes.ndim != 1:
            raise ValueError(f"flat_syndromes must be 1D, got shape {flat_syndromes.shape}")
        if not isinstance(corrections, _np.ndarray):
            corrections = _np.array(corrections, dtype=_np.uint8)
        if corrections.dtype != _np.uint8:
            corrections = corrections.astype(_np.uint8)
        if corrections.ndim != 1:
            raise ValueError(f"corrections must be 1D, got shape {corrections.shape}")
        if stride is None:
            stride = self.n_checks
        self._inner.decode_batch_into_flat(flat_syndromes, num_shots, stride, corrections)

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

    __slots__ = ("_inner",)

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

    def set_edge_weights(self, weights):
        weights = _np.asarray(weights, dtype=_np.float64)
        return self._inner.set_edge_weights(weights)

    def batch_decode_weighted(self, syndromes, weights):
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        if not isinstance(weights, _np.ndarray):
            weights = _np.array(weights, dtype=_np.float64)
        if weights.dtype != _np.float64:
            weights = weights.astype(_np.float64)
        if weights.ndim != 2:
            raise ValueError(f"weights must be 2D, got shape {weights.shape}")
        if syndromes.shape[0] != weights.shape[0]:
            raise ValueError(f"syndromes batch {syndromes.shape[0]} != weights batch {weights.shape[0]}")
        return self._inner.batch_decode_weighted(syndromes, weights)

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
    """Sliding-window decoder with exponential decay weighting or fault-tolerant space-time decoding.

    Maintains a window of the last ``window_size`` rounds. Pass ``check_types``,
    ``p_data``, and ``p_meas`` together for exact 3D space-time decoding; omit them
    for legacy decay-weighted accumulation.
    """

    def __init__(
        self,
        check_to_qubits,
        n_qubits=None,
        window_size=10,
        decay_factor=0.8,
        check_types=None,
        p_data=None,
        p_meas=None,
    ):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustSlidingWindowDecoder(c2q, nq, window_size, decay_factor, check_types, p_data, p_meas)

    def update(self, round_syndrome):
        if not isinstance(round_syndrome, _np.ndarray):
            round_syndrome = _np.array(round_syndrome, dtype=_np.uint8)
        if round_syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {round_syndrome.dtype}")
        return self._inner.update(round_syndrome)

    def flush(self):
        self._inner.flush()

    def decode(self, syndrome=None):
        if syndrome is None:
            return self._inner.decode()
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def detector_history(self):
        if hasattr(self._inner, "detector_history"):
            return self._inner.detector_history()
        return []

    @property
    def is_space_time(self):
        return getattr(self._inner, "is_space_time", False)

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

    Pass ``check_types``, ``p_data``, and ``p_meas`` together for exact 3D space-time
    decoding; omit them for legacy OR accumulation.

    Instead of T separate ``update()`` calls (which re-decodes the full window
    every round), use ``decode_rounds(rounds)`` to ingest T rounds and decode
    once — O(T·W) cheaper.
    """

    def __init__(
        self,
        check_to_qubits,
        n_qubits=None,
        history_size=10,
        check_types=None,
        p_data=None,
        p_meas=None,
    ):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustStreamingDecoder(c2q, nq, history_size, check_types, p_data, p_meas)

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

    def detector_history(self):
        if hasattr(self._inner, "detector_history"):
            return self._inner.detector_history()
        return []

    def decode_rounds(self, rounds):
        """Decode a sequence of rounds in a single call.

        Accepts a 2D array of shape ``(T, n_checks)`` where T is the number
        of rounds. Flushes any accumulated history, ingests all rounds, and
        returns the combined correction in a single decode pass.

        This is O(T·W) cheaper than T separate ``update`` calls when
        the decoder's internal history would otherwise grow and shift.
        """
        if not isinstance(rounds, _np.ndarray):
            rounds = _np.array(rounds, dtype=_np.uint8)
        if rounds.dtype != _np.uint8:
            rounds = rounds.astype(_np.uint8)
        if rounds.ndim != 2:
            raise ValueError(f"rounds must be 2D, got shape {rounds.shape}")
        return self._inner.decode_rounds(rounds)

    @property
    def is_space_time(self):
        return getattr(self._inner, "is_space_time", False)

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

    def decode(self, syndrome):
        """Decode a single 1D syndrome vector."""
        syn = _np.asarray(syndrome, dtype=_np.uint8).reshape(1, -1)
        return self.parallel_batch_decode(syn)[0]

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class CPUBatchDecoder:
    """SIMD-friendly CPU batch decoder with pooled buffers and SoA transposition."""

    __slots__ = ("_inner",)

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


try:
    from .qector_decoder_v3 import CUDABatchDecoder as _CudaReal

    CUDABatchDecoder = _CudaReal
except (ImportError, AttributeError):

    class CUDABatchDecoder:  # type: ignore[no-redef]
        """CUDA batch decoder - build feature cuda, runtime needs NVIDIA driver.

        Public wheels are built with --no-default-features --features cuda;
        this stub is used when the native module is absent.
        """

        @classmethod
        def is_available(cls):
            """Return True if CUDA was compiled and a CUDA-capable driver+device is present.

            A False return may mean the build lacked the cuda feature, or no
            CUDA driver/device was detected at runtime.
            """
            return False

        def __init__(self, *a, **k):
            raise RuntimeError("CUDABatchDecoder not available - wheel built without CUDA or no driver.")


try:
    from .qector_decoder_v3 import CUDABpOsdDecoder as _CudaBpOsdReal

    CUDABpOsdDecoder = _CudaBpOsdReal
except (ImportError, AttributeError):

    class CUDABpOsdDecoder:  # type: ignore[no-redef]
        """CUDA BP-OSD decoder - build feature cuda, runtime needs NVIDIA driver.

        Registered in lib.rs since the CUDA BP-OSD kernel landed, but never
        re-exported here, so the class was unreachable from the package. This
        stub is used when the native module is absent.
        """

        @classmethod
        def is_available(cls):
            """Return True if CUDA BP-OSD was compiled and a driver+device is present."""
            return False

        def __init__(self, *a, **k):
            raise RuntimeError("CUDABpOsdDecoder not available - wheel built without CUDA or no driver.")


try:
    from .qector_decoder_v3 import OpenCLBatchDecoder as _OclReal

    OpenCLBatchDecoder = _OclReal
except (ImportError, AttributeError):

    class OpenCLBatchDecoder:  # type: ignore[no-redef]
        """OpenCL batch decoder - requires source build with --features opencl.

        Public PyPI wheels are CUDA-only, so this returns False by design.
        """

        @classmethod
        def is_available(cls):
            """Return True if OpenCL was compiled and the driver+device pass a live decode probe.

            A False return may mean the build lacked the opencl feature, or the
            driver/device is absent or failed the probe.
            """
            return False

        def __init__(self, *a, **k):
            raise RuntimeError(
                "OpenCLBatchDecoder not available in CUDA-only wheel. "
                "Build from source: maturin develop --features opencl"
            )


class SparseBlossomDecoder:
    """Region-growing Sparse Blossom decoder with RadixHeap.

    Supports dynamic weight overrides from GNN Pre-Decoder for enriched decoding.
    """

    __slots__ = ("_inner",)

    def __init__(self, check_to_qubits, n_qubits=None, edge_weights=None):
        # `edge_weights` was missing from this shim while the Rust class has
        # accepted it since UF-01. Two consequences, both fixed here:
        #   1. `DemModel.make_decoder("sparse_blossom")` — the *default* kind —
        #      calls `SparseBlossomDecoder(c2q, nq, weights)` and raised
        #      TypeError, so the whole DEM -> decoder default path was dead.
        #   2. Constructing directly silently produced an *unweighted* matcher.
        #      Under circuit-level noise that discards the per-mechanism
        #      log-likelihood ratios and measurably costs accuracy.
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        if edge_weights is None:
            self._inner = _RustSparseBlossomDecoder(c2q, nq)
        else:
            self._inner = _RustSparseBlossomDecoder(c2q, nq, list(edge_weights))

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

    Exact log-domain sum-product BP (default) with an OSD stage for improved
    LER on complex codes. ``bp_method="min_sum"`` selects the classical
    min-sum approximation; ``bp_method="relay"`` selects layered (serial)
    scheduling for faster convergence on loopy qLDPC graphs (C1-01);
    ``osd_order`` >= 1 enables the OSD-W combination sweep (single/pair
    flips around the base solution, min-weight wins).
    """

    __slots__ = ("_inner",)

    def __init__(self, check_to_qubits, n_qubits=None, error_rate=0.1, bp_method=None, osd_order=None):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustBPOSDDecoder(c2q, nq, error_rate, bp_method=bp_method, osd_order=osd_order)

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

    def batch_decode(self, syndromes):
        """Batch decode a 2D array of syndromes.

        Args:
            syndromes: np.ndarray of shape (batch, n_checks) with dtype uint8.

        Returns:
            np.ndarray of shape (batch, n_qubits) with dtype uint8.
        """
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            syndromes = syndromes.astype(_np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {syndromes.shape}")
        return self._inner.batch_decode(syndromes)

    def decode_batch_flat(self, flat_syndromes, num_shots, stride=None):
        """Flat-batch decode — accepts a 1D aligned array.

        Args:
            flat_syndromes: 1D uint8 array of ``num_shots × stride`` bytes.
            num_shots: Number of shots in the batch.
            stride: Byte offset between shots (default: n_checks, tight).

        Returns:
            np.ndarray of shape (num_shots × n_qubits,) with dtype uint8.
        """
        if not isinstance(flat_syndromes, _np.ndarray):
            flat_syndromes = _np.array(flat_syndromes, dtype=_np.uint8)
        if flat_syndromes.dtype != _np.uint8:
            flat_syndromes = flat_syndromes.astype(_np.uint8)
        if flat_syndromes.ndim != 1:
            raise ValueError(f"flat_syndromes must be 1D, got shape {flat_syndromes.shape}")
        if stride is None:
            stride = self.n_checks
        return self._inner.decode_batch_flat(flat_syndromes, num_shots, stride)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks


class TwoStageDecoder:
    """C1-03: Correlated two-stage matching for CSS codes.

    Decodes X-type checks first, feeds the inferred X corrections into the
    Z-type syndrome (some X flips also affect Z stabilizers), then decodes the
    modified Z syndrome.

    Each stage can use any decoder: ``"blossom"`` (default), ``"unionfind"``,
    ``"bposd[:rate[:osd_order]]"``, or ``"sparse_blossom"``.

    Args:
        check_to_qubits: Full check matrix (X checks, then Z checks).
        check_types: ``True`` for X checks, ``False`` for Z checks.
        n_qubits: Total qubit count (inferred if omitted).
        x_decoder: Decoder type for the X stage (default: ``"blossom"``).
        z_decoder: Decoder type for the Z stage (default: ``"blossom"``).
    """

    def __init__(self, check_to_qubits, check_types, n_qubits=None, x_decoder="blossom", z_decoder="blossom"):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustTwoStageDecoder(c2q, check_types, nq, x_decoder, z_decoder)

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


class AmbiguityClusterDecoder:
    """C1-02: BP + ambiguity-clustered exact decoding for qLDPC codes.

    Runs Belief Propagation to get soft LLRs, identifies ambiguous qubits
    (|LLR| < threshold), clusters them by Tanner-graph connectivity, and
    solves each cluster exactly via enumeration.

    Args:
        check_to_qubits: Full check matrix.
        n_qubits: Total qubit count (inferred if omitted).
        error_rate: Physical error rate for BP prior (default: 0.05).
        ambig_threshold: |LLR| threshold for ambiguity (default: 0.5).
        max_cluster_size: Max cluster size for exact enumeration (default: 12).
    """

    def __init__(self, check_to_qubits, n_qubits=None, error_rate=0.05, ambig_threshold=0.5, max_cluster_size=12):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustAmbiguityClusterDecoder(c2q, nq, error_rate, ambig_threshold, max_cluster_size)

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


class HybridCascadeDecoder:
    """Hybrid cascading decoder: Union-Find pre-filter + Blossom/BP-OSD escalation.

    Runs the fast Union-Find decoder (~85,000 dec/s even at d=13) as a
    pre-filter over every syndrome and only escalates non-trivial or
    high-weight patterns to the slower, more accurate escalation decoder.
    This keeps total pipeline throughput high while preserving near-zero
    logical error rates for the hard cases.

    Args:
        check_to_qubits: For each check, the list of qubits it measures.
        n_qubits: Total number of qubits (inferred if omitted).
        edge_weights: Optional per-qubit weights (Blossom escalation only).
        max_accept_weight: Optional cap on the pre-filter correction weight;
            heavier UF results escalate. Default: derived from defect count.
        escalation: "blossom" (default) or "bposd" (wall-clock-capable).
        error_rate: Physical error rate for the BP-OSD escalation decoder.
    """

    def __init__(
        self,
        check_to_qubits,
        n_qubits=None,
        edge_weights=None,
        max_accept_weight=None,
        escalation=None,
        error_rate=None,
    ):
        c2q, nq = _validate_check_to_qubits(check_to_qubits, n_qubits)
        self._inner = _RustHybridCascadeDecoder(c2q, nq, edge_weights, max_accept_weight, escalation, error_rate)

    def decode(self, syndrome):
        """Decode one syndrome: UF pre-filter first, escalation on decline."""
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode(syndrome)

    def decode_timed(self, syndrome, max_latency_ms=10.0):
        """Decode with a wall-clock deadline (ms) on the escalation stage."""
        if not isinstance(syndrome, _np.ndarray):
            syndrome = _np.array(syndrome, dtype=_np.uint8)
        if syndrome.dtype != _np.uint8:
            raise TypeError(f"Syndrome must be dtype uint8, got {syndrome.dtype}")
        return self._inner.decode_timed(syndrome, max_latency_ms)

    def batch_decode(self, syndromes):
        """Batch cascade decode - the fast path for the pre-decoder strategy.

        Runs the Union-Find pre-filter in parallel (Rayon) across the whole
        batch and escalates only the syndromes the pre-filter declines.

        Args:
            syndromes: np.ndarray of shape (batch, n_checks), dtype uint8.

        Returns:
            np.ndarray of shape (batch, n_qubits), dtype uint8.
        """
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            raise TypeError(f"Syndromes must be dtype uint8, got {syndromes.dtype}")
        if syndromes.ndim != 2:
            raise ValueError(f"Syndromes must be 2D (batch, n_checks), got {syndromes.ndim}D")
        return self._inner.batch_decode(syndromes)

    def batch_decode_timed(self, syndromes, max_latency_ms=10.0):
        """Batch cascade decode with a wall-clock deadline (ms) per escalation."""
        if not isinstance(syndromes, _np.ndarray):
            syndromes = _np.array(syndromes, dtype=_np.uint8)
        if syndromes.dtype != _np.uint8:
            raise TypeError(f"Syndromes must be dtype uint8, got {syndromes.dtype}")
        if syndromes.ndim != 2:
            raise ValueError(f"Syndromes must be 2D (batch, n_checks), got {syndromes.ndim}D")
        return self._inner.batch_decode_timed(syndromes, max_latency_ms)

    @property
    def n_qubits(self):
        return self._inner.n_qubits

    @property
    def n_checks(self):
        return self._inner.n_checks

    @property
    def prefilter_hits(self):
        """Number of syndromes resolved by the Union-Find pre-filter."""
        return self._inner.prefilter_hits

    @property
    def escalations(self):
        """Number of syndromes escalated to Blossom/BP-OSD."""
        return self._inner.escalations

    @property
    def prefilter_hit_rate(self):
        """Fraction of decoded syndromes resolved by the pre-filter."""
        return self._inner.prefilter_hit_rate


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
            import os
            import tempfile

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
            import os
            import tempfile

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
            if _F is None:
                raise ImportError("PyTorch is required to train GNNPredecoder")
            import torch
            from torch import optim

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
                    loss = _F.mse_loss(preds, target_t)
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


def from_circuit(circuit, decoder_type="blossom", **kwargs):
    """Build a decoder from a Stim circuit in one call.

    Extracts the detector error model, infers check-to-qubit adjacency, and
    returns a ready-to-use decoder instance.

    Args:
        circuit: A ``stim.Circuit`` (or anything with a
            ``detector_error_model()`` method).
        decoder_type: One of ``"blossom"``, ``"sparse_blossom"``,
            ``"bposd"``, ``"belief_match"``, ``"colour_code"``, ``"auto"``.
            Use ``"colour_code"`` for colour codes: matching-based types
            cannot decode them (C1-04).
        **kwargs: Passed to the decoder constructor.

    Returns:
        A decoder instance (BlossomDecoder, SparseBlossomDecoder,
        BPOSDDecoder, BeliefMatching, or backend.AutoDecoder).

    Example::

        import stim, qector_decoder_v3 as qector

        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=0.001
        )
        decoder = qector.from_circuit(circuit, decoder_type="belief_match")
        samples = circuit.compile_detector_sampler().sample(100)[0]
        predictions = decoder.decode_batch(samples)
    """
    # Imported for its side effect: `from_circuit` is documented to require
    # stim, so a missing install must fail here with a clear ImportError rather
    # than deeper inside a decoder branch that happens to touch it first.
    import stim  # noqa: F401

    dt = decoder_type.lower().replace("_", "").replace("-", "")

    if dt in ("colourcode", "colorcode"):
        # C1-04: colour codes have non-graphlike error mechanisms, so the
        # decomposed DEM built below cannot represent them — Stim raises
        # outright at d>=5. This branch must run before any
        # decompose_errors=True call.
        from .colour_code import ColourCodeDecoder

        return ColourCodeDecoder.from_stim_circuit(circuit, **kwargs)

    dem = circuit.detector_error_model(decompose_errors=True)
    c2q = _dem_to_check_to_qubits(dem)
    nq = max((q for c in c2q for q in c), default=-1) + 1

    if dt in ("blossom",):
        return BlossomDecoder(c2q, nq, **kwargs)
    if dt in ("sparseblossom",):
        return SparseBlossomDecoder(c2q, nq, **kwargs)
    if dt in ("bposd",):
        return BPOSDDecoder(c2q, nq, **kwargs)
    if dt in ("beliefmatch", "belief_matching"):
        from .belief_matching import BeliefMatching

        return BeliefMatching.from_detector_error_model(dem, **kwargs)
    if dt in ("auto",):
        from .backend import AutoDecoder

        return AutoDecoder(c2q, nq, **kwargs)
    if dt in ("twostage", "two_stage", "correlated", "c1_03"):
        # Default to blossom for both stages; pass check_types via kwargs or
        # require the caller to pass it explicitly.
        if "check_types" not in kwargs and "_check_types" not in kwargs:
            raise ValueError(
                "TwoStageDecoder requires check_types=(list of bool, True=X, False=Z). "
                "Pass check_types=... to from_circuit()."
            )
        ct = kwargs.pop("check_types", kwargs.pop("_check_types", None))
        return TwoStageDecoder(c2q, ct, nq, **kwargs)
    if dt in ("ambigcluster", "ambiguitycluster", "ambig_cluster", "c1_02"):
        return AmbiguityClusterDecoder(c2q, nq, **kwargs)
    raise ValueError(f"Unknown decoder_type={decoder_type!r}")


def _dem_to_check_to_qubits(dem):
    """Extract check_to_qubits from a stim DetectorErrorModel."""
    import stim as _stim

    if isinstance(dem, str):
        dem = _stim.DetectorErrorModel(dem)
    nD = dem.num_detectors
    c2q: list[list[int]] = [[] for _ in range(nD)]
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        targets = inst.targets_copy()
        dets = []
        for t in targets:
            if t.is_relative_detector_id():
                dets.append(t.val)
        if len(dets) == 2:
            c2q[dets[0]].append(dets[1])
            c2q[dets[1]].append(dets[0])
    return [sorted(set(c)) for c in c2q]


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


def generate_parity_check_matrix(check_to_qubits, n_qubits=None):
    """Return the binary parity-check matrix ``H`` for *check_to_qubits*.

    ``H[c, q] == 1`` iff check ``c`` touches qubit ``q``. Shape is
    ``(len(check_to_qubits), n_qubits)``; ``n_qubits`` defaults to one past the
    largest qubit index referenced by any check.
    """
    c2q = [[int(q) for q in check] for check in check_to_qubits]
    if n_qubits is None:
        return py_generate_parity_check_matrix(c2q, None)
    return py_generate_parity_check_matrix(c2q, int(n_qubits))


def compute_detector_differences(history):
    """Computes XOR detector differences across consecutive syndrome rounds: detector_t = raw_t ^ raw_{t-1}."""
    fn = getattr(_native_module, "py_compute_detector_differences", None)
    if fn is not None:
        return fn(history)
    if not history:
        return []
    n_checks = len(history[0])
    prev = [0] * n_checks
    diffs = []
    for round_syn in history:
        diff = [(int(round_syn[i]) & 1) ^ (prev[i] & 1) for i in range(min(n_checks, len(round_syn)))]
        if len(round_syn) == n_checks:
            prev = [int(x) & 1 for x in round_syn]
        else:
            prev = [0] * n_checks
            for i in range(min(n_checks, len(round_syn))):
                prev[i] = int(round_syn[i]) & 1
        diffs.append(diff)
    return diffs


def generate_space_time_surface_code_checks(distance, rounds):
    """Generate 3D space-time surface code checks across multiple rounds."""
    fn = getattr(_native_module, "py_generate_space_time_surface_code_checks", None)
    if fn is not None:
        return fn(int(distance), int(rounds))
    spatial_checks, n_qubits = generate_surface_code_checks(distance)
    st_checks = []
    for r in range(rounds):
        offset = r * n_qubits
        for check in spatial_checks:
            st_checks.append([q + offset for q in check])
    return st_checks, n_qubits * rounds


def generate_triangular_color_code_4_8_8_checks(distance):
    """Generate 4.8.8 triangular color code parity check matrix."""
    fn = getattr(_native_module, "py_generate_triangular_color_code_4_8_8_checks", None)
    if fn is not None:
        return fn(int(distance))
    d = int(distance)
    if d < 3 or d % 2 == 0:
        raise ValueError("distance must be odd and >= 3 for triangular color code")
    n_qubits = (d * (d + 1)) // 2
    check_to_qubits = []
    idx = lambda r, c: (r * (r + 1)) // 2 + c
    for r in range(max(0, d - 2)):
        for c in range(r + 1):
            if c + 2 <= r + 2:
                check_to_qubits.append([idx(r, c), idx(r + 1, c), idx(r + 1, c + 1), idx(r + 2, c + 1)])
    if not check_to_qubits:
        check_to_qubits.append([0, 1, 2])
    return check_to_qubits, n_qubits


def generate_biconnected_qldpc_checks(n_qubits, degree):
    """Generate a bipolar / biconnected qLDPC Tanner graph parity check matrix."""
    fn = getattr(_native_module, "py_generate_biconnected_qldpc_checks", None)
    if fn is not None:
        return fn(int(n_qubits), int(degree))
    n = int(n_qubits)
    k = int(degree)
    if n < 4 or k < 2:
        raise ValueError("n_qubits must be >= 4 and degree must be >= 2")
    check_to_qubits = []
    for i in range(n):
        check = sorted(list(set((i + d * 3) % n for d in range(k))))
        if len(check) >= 2:
            check_to_qubits.append(check)
    return check_to_qubits, n


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
    "BPOSDDecoder",
    "BatchDecoder",
    "BenchmarkSuite",
    "BlossomDecoder",
    "CPUBatchDecoder",
    "CUDABatchDecoder",
    "CUDABpOsdDecoder",
    "DetectorGraph",
    "FastUnionFindDecoder",
    "GNNPredecoder",
    "GNNTrainer",
    "HybridCascadeDecoder",
    "HybridDecoder",
    "LERBenchmark",
    "LookupTableDecoder",
    "NeuralPredecoder",
    "OpenCLBatchDecoder",
    "SlidingWindowDecoder",
    "SpaceTimeDecoder",
    "SparseBlossomDecoder",
    "StreamingDecoder",
    "UnionFindDecoder",
    "check_to_edges",
    "compute_detector_differences",
    "cuda_is_available",
    "enforce_distance_cap",
    "enforce_unlocked",
    "estimate_distance",
    "flush_usage",
    "from_circuit",
    "generate_biconnected_qldpc_checks",
    "generate_parity_check_matrix",
    "generate_repetition_code_checks",
    "generate_ring_code_checks",
    "generate_space_time_surface_code_checks",
    "generate_surface_code_checks",
    "generate_toy_code_checks",
    "generate_triangular_color_code_4_8_8_checks",
    "get_latency_quantiles",
    "opencl_is_available",
    "run_grpc_server",
    "run_mcp_server",
    "set_license_key_file",
    "start_metrics_server",
]


# Ecosystem / tooling layer (pure-Python, built on the compiled core)
from . import (
    backend,
    belief_matching,
    benchmarking,
    bposd,
    codes,
    colour_code,
    dem,
    predecoder,
    pymatching_compat,
    result,
    workbench,
)
from .backend import AutoDecoder, Backend, BackendConfig
from .belief_matching import BeliefMatching, GNNBeliefMatcher, decode_with_gnn
from .bposd import BpOsdDecoder
from .colour_code import ColourCodeDecoder
from .decode_mmap import decode_mmap
from .decoder_cache import clear_decoder_cache, get_decoder, get_decoder_pool

# New v0.6.4 features: DecoderPool, get_decoder, decode_mmap
from .decoder_pool import DecoderPool
from .predecoder import PredecodedDecoder
from .result import DecodeResult, decode_with_diagnostics
from .workbench import Workbench

# sinter_compat imports `sinter` lazily; tolerate its absence.
try:
    from . import sinter_compat
except Exception:  # pragma: no cover
    sinter_compat = None  # type: ignore[assignment]

__all__ += [
    "AmbiguityClusterDecoder",
    "AutoDecoder",
    "Backend",
    "BackendConfig",
    "BeliefMatching",
    "BpOsdDecoder",
    "ColourCodeDecoder",
    "DecodeResult",
    "DecoderPool",
    "GNNBeliefMatcher",
    "NativeAutoDecoder",
    "PredecodedDecoder",
    "TwoStageDecoder",
    "Workbench",
    "backend",
    "belief_matching",
    "benchmarking",
    "bposd",
    "clear_decoder_cache",
    "codes",
    "colour_code",
    "decode_mmap",
    "decode_with_diagnostics",
    "decode_with_gnn",
    "dem",
    "get_decoder",
    "get_decoder_pool",
    "predecoder",
    "pymatching_compat",
    "result",
    "sinter_compat",
    "workbench",
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

try:
    from . import stripe_integration
except Exception:  # pragma: no cover
    stripe_integration = None  # type: ignore[assignment]

try:
    from . import bench_quick
except Exception:  # pragma: no cover
    bench_quick = None  # type: ignore[assignment]

try:
    from . import cli
except Exception:  # pragma: no cover
    cli = None  # type: ignore[assignment]

try:
    from . import doctor
except Exception:  # pragma: no cover
    doctor = None  # type: ignore[assignment]

try:
    from . import ler
except Exception:  # pragma: no cover
    ler = None  # type: ignore[assignment]

__all__ += [
    "MAX_WORKERS",
    "get_accumulated_shots",
    "get_license_info",
    "license",
    "qiskit_plugin",
    "record_shots",
    "rest_api",
    "set_license_key",
    "stim_compat",
    "stripe_integration",
    "verify_license_token",
]

# ---------------------------------------------------------------------------
# GPU backend abstraction + batched belief propagation (additive, CuPy-optional).
#
# ``gpu_backend`` only needs NumPy + stdlib to import (CuPy is probed lazily and
# guarded internally), and ``bp_cupy`` builds purely on in-tree pure-Python
# modules, so neither requires CuPy to be installed. They are still wrapped in a
# tolerant guard so that - per the package contract - a missing optional path can
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
            "get_backend",
            "gpu_available",
            "gpu_backend",
            "has_cuda_rust",
            "has_cupy",
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
            "get_backend",
            "gpu_available",
            "gpu_backend",
            "has_cuda_rust",
            "has_cupy",
        ]
    )

try:
    from . import bp_cupy as _bp_cupy_module

    BatchedBpDecoder: object = _bp_cupy_module.BatchedBpDecoder
    batched_bp_decode = _bp_cupy_module.batched_bp_decode
    bp_cupy = _bp_cupy_module

    __all__.extend(
        [
            "BatchedBpDecoder",
            "batched_bp_decode",
            "bp_cupy",
        ]
    )
except Exception:  # pragma: no cover - defensive; bp_cupy has no hard deps
    bp_cupy = None  # type: ignore[assignment]
    BatchedBpDecoder = None
    batched_bp_decode = None  # type: ignore[assignment]
    __all__.extend(
        [
            "BatchedBpDecoder",
            "batched_bp_decode",
            "bp_cupy",
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
            "AutoRouter",
            "DecoderName",
            "HardwareProfile",
            "Recommendation",
            "detect_hardware",
            "recommend",
            "recommend_decoder",
            "routing",
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
            "AutoRouter",
            "DecoderName",
            "HardwareProfile",
            "Recommendation",
            "detect_hardware",
            "recommend",
            "recommend_decoder",
            "routing",
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
            "StreamingResult",
            "StreamingSession",
            "StreamingTelemetry",
            "sliding_window_decode",
            "streaming",
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
            "StreamingResult",
            "StreamingSession",
            "StreamingTelemetry",
            "sliding_window_decode",
            "streaming",
        ]
    )

# Clean public namespace to reduce leakage (per v0.6.2 todo.md).
# Only public names should remain.
for _name in ("os", "sys", "subprocess"):
    globals().pop(_name, None)
# Note: numpy kept as _np internally; public API uses explicit numpy where needed in docs.


# ===========================================================================
# v0.6.9 release notes (public)
# ===========================================================================
__changelog__ = {
    "0.7.0": [
        (
            "Security: real Ed25519 license verification in the Rust core "
            "(v2 + QECT-PRO/QECT-ENT signed payload formats, production key "
            "matches the Python PEM), offline CRL ~/.qector/revoked.txt, "
            "expiry enforcement, QECTOR_ENFORCE=1 hard gate."
        ),
        (
            "Security: MCP stdin reader enforces a 10 MB max_content_length; "
            "oversized frames are rejected with -32600 instead of buffered."
        ),
        (
            "Feature: real MCP decoder cache keyed by (code, n_qubits, "
            "decoder_type, tuning env) with honest cache_hit reporting and a "
            "working clear_decoder_cache; per-tier decode timeout budgets; "
            "async startup pre-warm with 5s cap."
        ),
        (
            "Feature: decode_syndrome decoder_type='auto' routes through the "
            "native AutoDecoder; benchmark_decoder strictly validates "
            "decoder_type and supports SparseBlossom/BP-OSD/LookupTable/"
            "Cascade with Wilson 95% CI on the unfaithful-decode fraction."
        ),
        (
            "Feature: AutoDecoder learns CPUBatch routing for batch_size > "
            "1024, and select_optimal_with_license for test isolation."
        ),
        (
            "Feature: Stripe metered billing flushes real usage to the "
            "Meter Events API with 1s/2s/4s backoff retry (py_flush_usage)."
        ),
        (
            "Fix: UnionFindDecoder.decode/decode_into return Result instead "
            "of panicking; production unwraps removed in sparse_blossom and "
            "mwpm; NaN-safe latency sort in benchmarks."
        ),
        (
            "Perf: thread-local UF scratch plus arena-backed "
            "decode_into_arena in FastUnionFindDecoder; AVX2 bitmask clear "
            "with runtime detection and O(words) popcount."
        ),
    ],
    "0.6.9": [
        "Feature: Single-precision f32 neural predecoders for 2x SIMD vectorization density.",
        "Feature: Deterministic seeded GNN initialization (new_with_seed) for reproducible training.",
        "Feature: Per-check noise decay vectors in SlidingWindowDecoder.",
        "Feature: Non-blocking gRPC launch handle (start_grpc_server) with explicit stop().",
        "Feature: Threaded async MCP stdio server processing with atomic stdout writing.",
        "Feature: CUDA explicit device_id selection (CUDABatchDecoder::new_with_device) & QECTOR_CUDA_DEVICE_ID.",
        "Feature: cuMemGetInfo_v2 VRAM pre-check and Unified Memory fallback for d >= 17.",
        "Feature: Unrotated surface code & Color code topology generators.",
    ],
    "0.6.8": [
        (
            "Hotfix: v0.6.7 wheel was completely unimportable - "
            "_native_module.HybridCascadeDecoder does not exist in the compiled .so. "
            "All 18 native-module lookups now guarded via _guard() helper."
        ),
        "CI: smoke test before publish catches unimportable wheels.",
        "CI: fixed YAML syntax error in release-build.yml smoke-test run step.",
    ],
    "0.6.7": [
        (
            "Bugfix: SparseBlossomDecoder::grow_regions no longer collapses the "
            "compressed edge set - the previous version drained edges through a "
            "RadixHeap and back, which silently zeroed the `source_defect` field. "
            "All decoded syndromes are now bit-identical to the Blossom decoder."
        ),
        (
            "Bugfix: BPOSDDecoder.bp_decode_timed now initializes the wall-clock "
            "deadline before the iteration loop, not inside it, so the latency "
            "budget is honored on the first iteration."
        ),
        (
            "Bugfix: LER benchmark's rotated-surface generator now emits a proper "
            "two-half (X + Z) graphlike code, so the logical operator and weight "
            "gap statistics are meaningful."
        ),
        (
            "Feature: SparseBlossomDecoder.k_nearest_via_radix - public event-driven "
            "candidate-edge discovery backed by a new RadixHeap<u32, HeapEvent> "
            "structure exposed to downstream callers that need fine control over "
            "the candidate set."
        ),
        (
            "Feature: MCP server (mcp_server) now exposes 5 new tools: "
            "decode_syndrome_blossom, batch_decode_blossom, run_ler_benchmark, "
            "plus expanded get_decoder_info listing all 11 decoder families."
        ),
        (
            "Quality: cross-decoder syndrome-validity test suite "
            "src/cross_decoder_tests.rs covers UF / FastUF / LookupTable / "
            "SparseBlossom / BP-OSD / SlidingWindow / Streaming / Hybrid."
        ),
        (
            "Quality: SafeTensors loader now has a full round-trip test suite "
            "covering generic + runtime dispatch, dtype mismatch, missing tensors, "
            "and shape round-trip."
        ),
        "Quality: dead-code warnings eliminated across the crate (8 → 0).",
    ],
    "0.6.6": [
        (
            "Rust core upgraded with cluster-growth memory reuse and Sparse Blossom "
            "shattering. Python 3.9-3.13 wheel support, smarter numpy cap in the "
            "[all] extra."
        ),
    ],
}


def changelog() -> str:
    """Return a human-readable changelog string for the loaded package."""
    lines = [f"QECTOR Decoder v{__version__}", "=" * 40]
    for ver, items in __changelog__.items():
        lines.append(f"\n## v{ver}")
        for it in items:
            lines.append(f"  - {it}")
    return "\n".join(lines)


# v0.6.7: Expose the new SparseBlossom event-driven k-NN candidate discovery
# as a thin Python convenience for callers that already use the Rust decoder.
def sparse_blossom_radix_neighbors(decoder, defects, k=8):
    """Return the k-nearest candidate edges (sorted by distance) for `defects`.

    Backed by the new Rust `SparseBlossomDecoder::k_nearest_via_radix` method
    introduced in v0.6.7. Use this when you want to build a custom MWPM
    candidate set or analyze the region-growth candidate distribution.

    Parameters
    ----------
    decoder : qector_decoder_v3.SparseBlossomDecoder
        The decoder (must be a `SparseBlossomDecoder`).
    defects : list[int]
        Check indices that fired (1 in the syndrome).
    k : int, default 8
        Number of nearest neighbours to keep per defect.

    Returns
    -------
    list[tuple[int, int, int, int]]
        ``(source_defect, target_defect, distance, observables_bitmask)``
        sorted by ascending distance.
    """
    return list(decoder.k_nearest_via_radix(list(defects), int(k)))


# PyMatching-compatible migration shim
from . import pymatching_compat as pymatching
from .pymatching_compat import Matching

__all__ += [
    "Matching",
    "pymatching",
]
# Names that have always been importable from this package but were never listed
# in `__all__`, so `from qector_decoder_v3 import *` did not provide them and the
# API inventory did not cover them (todo8 8.1). Declaring them documents the
# surface that already ships rather than adding to it; `test_public_symbols.py`
# now locks it in both directions. (The routing / streaming / gpu_backend /
# bp_cupy names are already appended by the optional-import blocks above.)
__all__ += [
    "changelog",
    "decoder_cache",
    "decoder_pool",
    "sparse_blossom_radix_neighbors",
]
