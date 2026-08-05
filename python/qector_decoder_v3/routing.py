"""
qector_decoder_v3.routing - intelligent decoder router / auto-selection.

QECTOR ships a zoo of decoders with very different correctness domains and cost
profiles.  Picking the wrong one is not merely slow - it can be *invalid*: sending
a non-graphlike (qLDPC / hypergraph-product / color-code) problem to a pure
matching decoder (Union-Find / Blossom) cannot in general satisfy ``H·c == s``.
This module encodes a documented policy that maps a *decoding problem* (code
family, distance, qubit count, batch size) and a *goal* (accuracy / speed /
balanced), under the *hardware* actually present, onto a concrete decoder.

It complements :mod:`qector_decoder_v3.backend`.  ``backend.AutoDecoder`` answers
"given a fixed graphlike decoder, which *execution backend* (single-thread CPU,
Rayon, CUDA, OpenCL) should run this batch?".  This module answers the question
one level up: "which *decoder family* is even appropriate here, and is it
correct?".  The two compose - the router can hand a graphlike batch problem to a
backend-aware path, and always routes hyperedge problems to BP-OSD.

Two entry points:

* :func:`recommend_decoder` - pure, side-effect-free policy. Returns the name of
  the recommended decoder class as a string.  :func:`recommend` returns the same
  decision as a rich :class:`Recommendation` (decoder + human-readable reason +
  classified family + resolved hardware + flags).
* :class:`AutoRouter` - stateful, *constructs and caches* the chosen decoder and
  dispatches :meth:`AutoRouter.decode`.  Crucially it inspects the **actual**
  check structure, so even if the caller mislabels a hyperedge code as
  ``"surface"`` the router detects ``max qubit degree > 2`` and forces BP-OSD,
  guaranteeing every returned correction satisfies ``H·c == s (mod 2)``.

Optional-GPU discipline
-----------------------
Hardware capability is resolved through :mod:`qector_decoder_v3.gpu_backend`
(:func:`~qector_decoder_v3.gpu_backend.has_cuda_rust`,
:func:`~qector_decoder_v3.gpu_backend.gpu_available`).  Everything degrades to
pure NumPy / CPU when no device (or CuPy) is present, and the policy is fully
overridable per call via the ``hardware`` argument.

Examples
--------
>>> from qector_decoder_v3 import routing
>>> routing.recommend_decoder(code_family="surface", distance=5, priority="accuracy")
'BlossomDecoder'
>>> routing.recommend_decoder(code_family="qldpc", batch_size=4096)
'BpOsdDecoder'
>>> from qector_decoder_v3 import codes
>>> ar = routing.AutoRouter(priority="accuracy")
>>> code = codes.rotated_surface_code(5)
>>> corr = ar.decode(code, syndrome)  # picks BlossomDecoder, decodes
>>> ar.explain(code)["decoder"]
'BlossomDecoder'
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Union, cast

import numpy as np

from . import gpu_backend as _gb
from .backend import BackendConfig

__all__ = [
    "AutoRouter",
    "DecoderName",
    "HardwareProfile",
    "Recommendation",
    "detect_hardware",
    "recommend",
    "recommend_backend",
    "recommend_decoder",
]


# ---------------------------------------------------------------------------
# Decoder name constants (match the class names exported by the package)
# ---------------------------------------------------------------------------
class DecoderName:
    """String identifiers for the routable decoders.

    Each value is the exact public class name exported by
    :mod:`qector_decoder_v3`, so a recommendation can be turned into a class with
    ``getattr(qector_decoder_v3, name)``.
    """

    UNION_FIND = "UnionFindDecoder"
    FAST_UNION_FIND = "FastUnionFindDecoder"
    BLOSSOM = "BlossomDecoder"
    SPARSE_BLOSSOM = "SparseBlossomDecoder"
    CUDA_BATCH = "CUDABatchDecoder"
    BATCH = "BatchDecoder"
    BP_OSD = "BpOsdDecoder"
    #: Rust-native routing decoder (C4-01): distance/noise/batch-aware backend
    #: selection inside the compiled core.
    AUTO_NATIVE = "NativeAutoDecoder"

    ALL = (
        UNION_FIND,
        FAST_UNION_FIND,
        BLOSSOM,
        SPARSE_BLOSSOM,
        CUDA_BATCH,
        BATCH,
        BP_OSD,
        AUTO_NATIVE,
    )

    #: Decoders that assume a graphlike (matching) problem - every qubit in at
    #: most two checks.  Routing one of these to a hyperedge / qLDPC problem is
    #: *invalid* (cannot guarantee ``H·c == s``); :class:`AutoRouter` guards
    #: against it by forcing :data:`BP_OSD`.
    MATCHING_ONLY = frozenset({UNION_FIND, FAST_UNION_FIND, BLOSSOM, SPARSE_BLOSSOM, CUDA_BATCH, BATCH})


# ---------------------------------------------------------------------------
# Policy thresholds (kept consistent with backend.BackendConfig)
# ---------------------------------------------------------------------------
_RAYON_BATCH: int = BackendConfig().rayon_threshold  # >= this -> Rayon CPU batch
_GPU_BATCH: int = BackendConfig().gpu_threshold  # >= this -> GPU batch path

#: Above these a code is "large": exact O(n^3) Blossom becomes expensive, so
#: accuracy-priority routing prefers the scalable region-growing Sparse Blossom.
_LARGE_DISTANCE: int = 13
_LARGE_NQUBITS: int = 400

_VALID_PRIORITIES = ("balanced", "accuracy", "speed")


def _native_auto_available() -> bool:
    """`True` when this build exposes the Rust-native routing decoder."""
    try:
        from . import NativeAutoDecoder  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        return False
    return True


# Code-family classification. Tokens are matched as substrings (case-folded) so
# that names like "rotated_surface_d5" or "bivariate_bicycle" classify correctly.
_MATCHING_TOKENS = (
    "surface",
    "toric",
    "repetition",
    "rep_",
    "ring",
    "heavy_hex",
    "heavyhex",
    "planar",
    "matching",
    "mwpm",
    "graphlike",
)
_LDPC_TOKENS = (
    "ldpc",
    "qldpc",
    "bivariate",
    "bicycle",
    "_bb",
    "gross",
    "hypergraph",
    "hgp",
    "hyperedge",
    "color",
    "lifted",
    "expander",
    "css_general",
)


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareProfile:
    """Resolved view of the GPU capability the policy should assume.

    Attributes
    ----------
    cuda_rust:
        The build-time Rust ``CUDABatchDecoder`` CUDA path is usable (mirrors
        :func:`qector_decoder_v3.gpu_backend.has_cuda_rust`).
    gpu:
        A CuPy-usable CUDA device is present (mirrors
        :func:`qector_decoder_v3.gpu_backend.gpu_available`); this is what the
        batched BP-OSD GPU path keys off.
    """

    cuda_rust: bool = False
    gpu: bool = False

    @classmethod
    def detect(cls) -> HardwareProfile:
        """Probe the live machine through :mod:`gpu_backend` (CPU-safe)."""
        return cls(cuda_rust=bool(_gb.has_cuda_rust()), gpu=bool(_gb.gpu_available()))

    def as_dict(self) -> dict[str, bool]:
        return {"cuda_rust": self.cuda_rust, "gpu": self.gpu}


def detect_hardware() -> HardwareProfile:
    """Return the live :class:`HardwareProfile` (alias for ``HardwareProfile.detect``)."""
    return HardwareProfile.detect()


# Accepted ``hardware=`` argument forms.
HardwareLike = Union[None, HardwareProfile, Mapping[str, Any], str]


def _resolve_hardware(hardware: HardwareLike) -> HardwareProfile:
    """Normalise the many accepted ``hardware`` forms to a :class:`HardwareProfile`.

    * ``None`` - probe the live machine via :mod:`gpu_backend`.
    * :class:`HardwareProfile` - used as-is.
    * mapping - keys ``cuda_rust``/``cuda`` and ``gpu``/``gpu_available``/``cupy``.
    * string - ``"cpu"``/``"none"`` (no GPU), ``"cuda"``/``"cuda_rust"`` (Rust CUDA +
      device), ``"gpu"``/``"cupy"`` (CuPy device only), ``"all"`` (everything).
    """
    if hardware is None:
        return HardwareProfile.detect()
    if isinstance(hardware, HardwareProfile):
        return hardware
    if isinstance(hardware, Mapping):
        cuda = bool(hardware.get("cuda_rust", hardware.get("cuda", False)))
        gpu = bool(hardware.get("gpu", hardware.get("gpu_available", hardware.get("cupy", False))))
        # The Rust CUDA path implies a usable device; keep the profile coherent.
        return HardwareProfile(cuda_rust=cuda, gpu=gpu or cuda)
    if isinstance(hardware, str):
        key = hardware.strip().lower()
        if key in ("cpu", "cpu_only", "none", "no_gpu", ""):
            return HardwareProfile(cuda_rust=False, gpu=False)
        if key in ("cuda", "cuda_rust", "rust_cuda"):
            return HardwareProfile(cuda_rust=True, gpu=True)
        if key in ("gpu", "cupy", "gpu_available"):
            return HardwareProfile(cuda_rust=False, gpu=True)
        if key in ("all", "full"):
            return HardwareProfile(cuda_rust=True, gpu=True)
        raise ValueError(f"unknown hardware spec {hardware!r}")
    raise TypeError(f"hardware must be None, HardwareProfile, mapping, or str; got {type(hardware)!r}")


# ---------------------------------------------------------------------------
# Family classification
# ---------------------------------------------------------------------------
def _classify_family(code_family: Any, graphlike: bool | None) -> str:
    """Classify a code as ``"matching"``, ``"ldpc"``, or ``"unknown"``.

    A definitive structural signal (``graphlike``) always wins: a problem that is
    not graphlike is ``"ldpc"`` regardless of any (possibly wrong) label.  A
    :class:`qector_decoder_v3.codes.Code` is classified by its own
    ``is_matching_graph()``.  Otherwise the string label is token-matched.
    """
    if graphlike is False:
        return "ldpc"

    # A Code object carries the ground truth.
    is_matching = getattr(code_family, "is_matching_graph", None)
    if callable(is_matching):
        try:
            return "matching" if is_matching() else "ldpc"
        except (AttributeError, TypeError, ValueError, RuntimeError):  # pragma: no cover - defensive
            pass

    if code_family is None:
        return "matching" if graphlike is True else "unknown"

    s = str(code_family).strip().lower()
    if any(tok in s for tok in _LDPC_TOKENS):
        return "ldpc"
    if any(tok in s for tok in _MATCHING_TOKENS):
        return "matching"
    return "unknown"


def _is_large(distance: int | None, n_qubits: int | None) -> bool:
    """True when exact Blossom is likely too expensive vs. region-growing."""
    if distance is not None and int(distance) >= _LARGE_DISTANCE:
        return True
    return bool(n_qubits is not None and int(n_qubits) >= _LARGE_NQUBITS)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    """A decoder choice plus the reasoning and inputs that produced it."""

    decoder: str
    reason: str
    family: str
    priority: str
    batch_size: int
    hardware: dict[str, bool]
    gpu_batched_bp: bool = False
    forced: bool = False
    graphlike: bool | None = None
    distance: int | None = None
    n_qubits: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view (for logs / diagnostics / ``explain``)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Core policy
# ---------------------------------------------------------------------------
def _select(
    family: str,
    distance: int | None,
    n_qubits: int | None,
    batch_size: int,
    priority: str,
    hw: HardwareProfile,
) -> tuple[str, str, bool]:
    """Return ``(decoder_name, reason, gpu_batched_bp)`` for a classified problem.

    See :func:`recommend` for the full documented policy.
    """
    # 1. Non-graphlike (LDPC / qLDPC / hypergraph-product / color): BP-OSD only.
    #    Matching decoders cannot guarantee H·c == s on hyperedges, so they are
    #    never a valid choice here.  When a CuPy device is present and the
    #    workload is a batch, BP-OSD runs its BP stage as one GPU batched pass
    #    (qector_decoder_v3.bp_cupy) and only the residual non-converged shots
    #    take the exact GF(2) OSD post-process.
    if family == "ldpc":
        gpu_bp = bool(hw.gpu and batch_size > 1)
        if gpu_bp:
            reason = (
                "LDPC/qLDPC/hypergraph problem -> BP-OSD (matching is invalid on "
                f"hyperedges); GPU batched BP used for the {batch_size}-shot batch"
            )
        else:
            reason = (
                "LDPC/qLDPC/hypergraph problem -> BP-OSD (matching is invalid on hyperedges); CPU BP + exact GF(2) OSD"
            )
        return DecoderName.BP_OSD, reason, gpu_bp

    # 2. Graphlike / matching families (also the default for "unknown" labels).
    large = _is_large(distance, n_qubits)
    size_note = f"distance={distance}, n_qubits={n_qubits} -> {'large' if large else 'small'} code"

    if priority == "speed":
        if batch_size >= _GPU_BATCH and hw.cuda_rust:
            return (
                DecoderName.CUDA_BATCH,
                f"speed + huge batch ({batch_size} >= {_GPU_BATCH}) with Rust CUDA -> GPU-batched Union-Find",
                False,
            )
        return (
            DecoderName.FAST_UNION_FIND,
            f"speed priority -> SIMD FastUnionFind (lowest per-shot overhead); {size_note}",
            False,
        )

    if priority == "accuracy":
        if large:
            return (
                DecoderName.SPARSE_BLOSSOM,
                (
                    "accuracy + large code -> region-growing SparseBlossom "
                    f"(near-optimal MWPM that scales where exact Blossom is O(n^3)); {size_note}"
                ),
                False,
            )
        return (
            DecoderName.BLOSSOM,
            f"accuracy + small/moderate code -> exact MWPM BlossomDecoder; {size_note}",
            False,
        )

    # priority == "balanced": heuristic on batch size, then code size.
    if batch_size >= _GPU_BATCH:
        if hw.cuda_rust:
            return (
                DecoderName.CUDA_BATCH,
                (
                    f"balanced + huge batch ({batch_size} >= {_GPU_BATCH}) with Rust CUDA "
                    "-> GPU-batched Union-Find for throughput"
                ),
                False,
            )
        return (
            DecoderName.FAST_UNION_FIND,
            f"balanced + huge batch ({batch_size}) but no Rust CUDA -> FastUnionFind",
            False,
        )
    if batch_size >= _RAYON_BATCH:
        return (
            DecoderName.BATCH,
            f"balanced + medium batch ({batch_size} >= {_RAYON_BATCH}) -> Rayon data-parallel BatchDecoder",
            False,
        )
    # Small batch / single shot: accuracy is essentially free, so prefer exact
    # (small) or scalable near-optimal (large) matching.
    if large:
        return (
            DecoderName.SPARSE_BLOSSOM,
            f"balanced + small batch + large code -> SparseBlossom; {size_note}",
            False,
        )
    return (
        DecoderName.BLOSSOM,
        f"balanced + small batch + small code -> exact BlossomDecoder; {size_note}",
        False,
    )


def recommend(
    code_family: Any = None,
    distance: int | None = None,
    n_qubits: int | None = None,
    batch_size: int = 1,
    priority: str = "balanced",
    hardware: HardwareLike = None,
    *,
    graphlike: bool | None = None,
) -> Recommendation:
    """Return a rich :class:`Recommendation` for a decoding problem.

    Parameters
    ----------
    code_family:
        Code family label (``"surface"``, ``"toric"``, ``"repetition"``,
        ``"qldpc"``, ``"bivariate_bicycle"``, ``"hypergraph_product"``,
        ``"color"``, ...) or a :class:`qector_decoder_v3.codes.Code` instance, whose
        structure is used directly.  ``None`` leaves the family unknown.
    distance, n_qubits:
        Code metadata steering the small-vs-large accuracy trade-off.
    batch_size:
        Number of syndromes to decode at once; drives the CPU/Rayon/GPU split.
    priority:
        ``"accuracy"`` (exact / near-optimal MWPM), ``"speed"`` (lowest-overhead
        Union-Find / GPU batch), or ``"balanced"`` (default heuristic).
    hardware:
        ``None`` to probe the live machine, or an explicit override - a
        :class:`HardwareProfile`, a mapping, or a string (see
        :func:`_resolve_hardware`).
    graphlike:
        Structural override (keyword-only).  ``True``/``False`` is authoritative
        about whether every qubit touches at most two checks; ``False`` always
        forces BP-OSD.  Supplied by :class:`AutoRouter` after inspecting ``H``.

    Policy (summary)
    ----------------
    * **Non-graphlike** (LDPC / qLDPC / hypergraph / color) -> **BP-OSD**, never a
      matching decoder; GPU batched BP when a CuPy device is present and batching.
    * **Matching family, ``speed``** -> **FastUnionFind** (or **CUDABatchDecoder**
      for a huge batch with Rust CUDA).
    * **Matching family, ``accuracy``** -> exact **BlossomDecoder** (small) /
      near-optimal **SparseBlossomDecoder** (large).
    * **Matching family, ``balanced``** -> **CUDABatchDecoder**/**FastUnionFind**
      for huge batches, **BatchDecoder** (Rayon) for medium batches, else exact /
      scalable Blossom by code size.
    """
    p = str(priority).strip().lower()
    if p not in _VALID_PRIORITIES:
        raise ValueError(f"priority must be one of {_VALID_PRIORITIES}, got {priority!r}")
    bs = max(1, int(batch_size))
    hw = _resolve_hardware(hardware)
    family = _classify_family(code_family, graphlike)

    decoder, reason, gpu_bp = _select(family, distance, n_qubits, bs, p, hw)
    return Recommendation(
        decoder=decoder,
        reason=reason,
        family=family,
        priority=p,
        batch_size=bs,
        hardware=hw.as_dict(),
        gpu_batched_bp=gpu_bp,
        forced=False,
        graphlike=graphlike,
        distance=None if distance is None else int(distance),
        n_qubits=None if n_qubits is None else int(n_qubits),
    )


def recommend_decoder(
    code_family: Any = None,
    distance: int | None = None,
    n_qubits: int | None = None,
    batch_size: int = 1,
    priority: str = "balanced",
    hardware: HardwareLike = None,
) -> str:
    """Return the *name* of the recommended decoder class (see :func:`recommend`)."""
    return recommend(
        code_family=code_family,
        distance=distance,
        n_qubits=n_qubits,
        batch_size=batch_size,
        priority=priority,
        hardware=hardware,
    ).decoder


# ---------------------------------------------------------------------------
# Backend (hardware-routing) recommendation
# ---------------------------------------------------------------------------
_CROSSOVER_SHOTS: dict[int, int] = {
    # Measured on a GTX 1660 Ti (v0.7.0/0.7.1, CPUBatchDecoder vs CUDABatchDecoder,
    # ring/repetition codes). CPU wins below these batch-shot counts (launch
    # overhead dominates); GPU wins above. Reproduced by
    # scripts/crossover_study.py; entries are the measured / interpolated
    # break-even --shots for each code size.
    3: 8_000,  # d=3 (2 det, 5 qubits): tiny code, huge relative launch cost
    5: 40_000,  # d=5
    9: 150_000,  # d=9
    13: 450_000,  # d=13
    21: 1_200_000,  # d=21
    41: 5_000_000,  # d=41
    101: 20_000_000,  # d=101
}


def _interp_crossover(n_qubits: int | None, distance: int | None, batch_size: int) -> int:
    """Break-even shot count for CPU vs CUDA batch decode on this machine class.

    Log-linear interpolation over the measured ``{distance: shots}`` table.
    Codes can stay in ``cpu_batch`` below it and switch to CUDA above it; a
    guard against GPU launch overhead for small problems. When no hint at all
    is available, fall back to a conservative CPU-default threshold so small
    workloads never pay GPU transfer latency.
    """
    keys = sorted(_CROSSOVER_SHOTS)
    x = distance if distance is not None else (int(n_qubits) if n_qubits else None)
    if x is None:
        return 100_000  # conservative
    if x <= keys[0]:
        return _CROSSOVER_SHOTS[keys[0]]
    if x >= keys[-1]:
        return _CROSSOVER_SHOTS[keys[-1]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= x <= hi:
            log_slope = (_CROSSOVER_SHOTS[hi] - _CROSSOVER_SHOTS[lo]) / (hi - lo)
            return int(round(_CROSSOVER_SHOTS[lo] + log_slope * (x - lo)))
    return _CROSSOVER_SHOTS[keys[0]]


def recommend_backend(
    n_qubits: int | None = None,
    distance: int | None = None,
    batch_size: int = 1,
    hardware: HardwareLike = None,
) -> str:
    """Return ``"cpu_batch"`` / ``"cuda"`` / ``"opencl"`` / ``"cpu"`` for a job.

    Hardware-vs-problem routing using the measured CPU/GPU crossover
    (dev2todo §3.1). Small problems (< break-even shots) stay on the CPU batch
    pipeline; above it, CUDA (or OpenCL, when the wheel/feature exposes it and
    it probes live) wins. ``None`` hardware probes the live machine.

    The crossover table is machine-class evidence, not a promise: ``Backend``'s
    :meth:`~qector_decoder_v3.Backend.calibrate` re-measures at runtime for the
    exact installed GPU/CPU pair and ``Backend`` honors the result. This helper
    is the fast, calibration-free decision for one-off calls.
    """
    hw = _resolve_hardware(hardware)
    # gpu = CuPy-usable device; cuda_rust = the Rust CUDABatchDecoder path.
    if not (hw.cuda_rust or hw.gpu):
        return "cpu_batch" if batch_size >= 1 and n_qubits else "cpu"
    cutoff = _interp_crossover(n_qubits, distance, max(1, int(batch_size)))
    if int(batch_size) >= cutoff:
        if hw.cuda_rust:
            return "cuda"
        return "opencl"  # CuPy-only device without the native CUDA path
    return "cpu_batch"


# ---------------------------------------------------------------------------
# Problem-shape helpers
# ---------------------------------------------------------------------------
def _H_to_c2q(H: np.ndarray) -> list[list[int]]:
    return [sorted(int(c) for c in np.nonzero(H[r])[0]) for r in range(H.shape[0])]


def _c2q_to_H(c2q: Sequence[Sequence[int]], n_qubits: int) -> np.ndarray:
    H = np.zeros((len(c2q), n_qubits), dtype=np.uint8)
    for i, qs in enumerate(c2q):
        for q in qs:
            H[i, int(q)] ^= np.uint8(1)
    return H


def _max_qubit_degree(c2q: Sequence[Sequence[int]], n_qubits: int) -> int:
    if n_qubits <= 0:
        return 0
    deg = np.zeros(n_qubits, dtype=np.int64)
    for qs in c2q:
        for q in qs:
            deg[int(q)] += 1
    return int(deg.max()) if deg.size else 0


def _structure_sig(c2q: Sequence[Sequence[int]], n_qubits: int) -> tuple:
    """Cheap-but-faithful fingerprint of a problem for decoder caching."""
    return (len(c2q), int(n_qubits), hash(tuple(tuple(c) for c in c2q)))


def _infer_batch(syndromes: Any) -> int:
    if syndromes is None:
        return 1
    arr = np.asarray(syndromes)
    if arr.ndim >= 2:
        return int(arr.shape[0])
    return 1


# ---------------------------------------------------------------------------
# AutoRouter
# ---------------------------------------------------------------------------
class AutoRouter:
    """Construct, cache, and dispatch the policy-selected decoder for a problem.

    Unlike :func:`recommend_decoder` (pure metadata), :class:`AutoRouter` is given
    the actual problem and the syndrome(s) to decode.  It inspects the real check
    structure and **guarantees a valid choice**: a problem whose maximum qubit
    degree exceeds two (a hyperedge / qLDPC / color code) is always routed to
    BP-OSD even if it was labelled ``"surface"``, because matching decoders cannot
    satisfy ``H·c == s`` on hyperedges.

    Parameters
    ----------
    priority:
        Default goal (``"balanced"`` / ``"accuracy"`` / ``"speed"``); overridable
        per :meth:`decode` / :meth:`explain` call.
    hardware:
        Default hardware view (``None`` probes the live machine); overridable per
        call.
    error_rate:
        Per-qubit prior passed to :class:`~qector_decoder_v3.bposd.BpOsdDecoder`
        when an LDPC problem is routed.
    code_family:
        Default family hint when a call does not supply one.

    Notes
    -----
    Decoders are built lazily and cached per ``(decoder_name, problem)``; the
    cache is cleared when the problem structure changes, so reusing one router
    across many shots of the same code rebuilds nothing.
    """

    def __init__(
        self,
        *,
        priority: str = "balanced",
        hardware: HardwareLike = None,
        error_rate: float = 0.05,
        code_family: Any = None,
        use_native_auto: bool = True,
    ) -> None:
        p = str(priority).strip().lower()
        if p not in _VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {_VALID_PRIORITIES}, got {priority!r}")
        self.priority = p
        self.hardware = hardware
        self.error_rate = float(error_rate)
        self.default_family = code_family
        # C4-03: when True, decode() first attempts routing through the
        # Rust-native AutoDecoder and falls back to the Python policy path
        # when the native build is unavailable or rejects the problem.
        self.use_native_auto = bool(use_native_auto)

        self._sig: tuple | None = None
        self._cache: dict[tuple, Any] = {}
        self._last: Recommendation | None = None

    # -- public ------------------------------------------------------------
    @property
    def last_recommendation(self) -> Recommendation | None:
        """The :class:`Recommendation` from the most recent call, or ``None``."""
        return self._last

    def explain(self, checks_or_H: Any = None, syndromes: Any = None, **ctx: Any) -> dict[str, Any]:
        """Return the choice + reason for a problem **without** decoding.

        Accepts the same problem forms and ``ctx`` as :meth:`decode`.  When no
        problem is given it falls back to pure-metadata routing on ``ctx``.
        """
        rec = self._recommend(checks_or_H, syndromes, ctx)
        return rec.as_dict()

    def decode(self, checks_or_H: Any, syndromes: Any, **ctx: Any) -> np.ndarray:
        """Route, construct/cache the decoder, and decode ``syndromes``.

        Parameters
        ----------
        checks_or_H:
            The problem, in any of: a :class:`qector_decoder_v3.codes.Code`; a 2-D
            ``H`` parity-check array; or a ``check_to_qubits`` list of lists of
            qubit indices (pass ``n_qubits=`` in ``ctx`` if the trailing qubits are
            unused).
        syndromes:
            A single syndrome (1-D, shape ``(n_checks,)``) -> returns a 1-D
            correction; or a batch (2-D, ``(batch, n_checks)``) -> returns a 2-D
            ``(batch, n_qubits)`` correction.
        **ctx:
            ``code_family``, ``distance``, ``priority``, ``hardware``,
            ``n_qubits`` - per-call overrides.

        Returns
        -------
        numpy.ndarray
            The correction(s); every row satisfies ``H·c == s (mod 2)`` for the
            graphlike (UF/Blossom) and BP-OSD paths.
        """
        H, c2q, n_qubits, _n_checks = self._normalize_problem(checks_or_H, ctx.get("n_qubits"))
        self._refresh_cache_for(c2q, n_qubits)

        syn = np.asarray(syndromes, dtype=np.uint8)
        batch_size = int(syn.shape[0]) if syn.ndim >= 2 else 1

        # C4-02: native Rust AutoDecoder first (license tier enforcement is
        # native-side and hard — a PermissionError propagates, never falls back).
        native_out = self._try_native_auto(c2q, n_qubits, batch_size, ctx, syn)
        if native_out is not None:
            return native_out

        rec = self._recommend_for_problem(c2q, n_qubits, batch_size, ctx)
        dec, rec = self._get_decoder(rec, H, c2q, n_qubits)
        self._last = rec
        return self._run(dec, syn, n_qubits)

    # -- native auto routing (C4) ------------------------------------------
    def _native_auto_applies(self, graphlike: bool) -> bool:
        """Whether ``decode`` should route through the Rust-native AutoDecoder.

        Two conditions beyond ``use_native_auto``, both load-bearing:

        * **``priority`` must be ``"balanced"``.** The native decoder makes its
          own distance/noise/batch tradeoff and has no notion of the caller's
          priority, so running it unconditionally made ``AutoRouter(priority=
          "accuracy")`` silently identical to the default -- the entire policy
          layer became unreachable whenever the compiled core was present.
        * **The problem must be graphlike.** The "never send a hyperedge code to
          a matching decoder" invariant is enforced in
          ``_recommend_for_problem``; bypassing it for qLDPC codes bypasses that
          check too.

        ``explain`` consults the same predicate, so the reported decoder is the
        one ``decode`` will actually use.
        """
        return self.use_native_auto and self.priority == "balanced" and graphlike

    def _try_native_auto(
        self,
        c2q: Sequence[Sequence[int]],
        n_qubits: int,
        batch_size: int,
        ctx: Mapping[str, Any],
        syn: np.ndarray,
    ) -> np.ndarray | None:
        """Attempt routing through the Rust-native AutoDecoder (C4-02).

        Returns the correction array on success; ``None`` when the native
        decoder is unavailable in this build or rejects the problem (the
        caller then falls back to the Python policy path). A license-tier
        ``PermissionError`` is re-raised: tier gating must never silently
        degrade to an unlicensed path.
        """
        graphlike = _max_qubit_degree(c2q, n_qubits) <= 2
        if not self._native_auto_applies(graphlike):
            return None
        try:
            from . import NativeAutoDecoder
        except Exception:
            return None

        distance = ctx.get("distance")
        if distance is None:
            # Surface-like estimate d ~ sqrt(n_qubits), rounded up to odd.
            d_est = max(3, int(round(float(n_qubits) ** 0.5)))
            distance = d_est if d_est % 2 == 1 else d_est + 1
        distance = int(distance)

        key = (DecoderName.AUTO_NATIVE, distance, batch_size > 1024, not graphlike)
        dec = self._cache.get(key)
        if dec is None:
            try:
                dec = NativeAutoDecoder(
                    [list(map(int, c)) for c in c2q],
                    int(n_qubits),
                    distance=distance,
                    noise_rate=float(self.error_rate),
                    batch_size=int(batch_size),
                    is_qldpc=not graphlike,
                )
            except PermissionError:
                raise
            except Exception:
                return None
            self._cache[key] = dec

        try:
            if syn.ndim == 1:
                out = np.asarray(dec.decode(syn), dtype=np.uint8).reshape(-1)
            else:
                out = np.stack(
                    [np.asarray(dec.decode(syn[i]), dtype=np.uint8).reshape(-1) for i in range(syn.shape[0])]
                )
        except Exception:
            return None

        self._last = Recommendation(
            decoder=DecoderName.AUTO_NATIVE,
            reason=(
                f"native Rust AutoDecoder routing (backend={getattr(dec, 'backend_name', '?')}, "
                f"distance={distance}, batch_size={batch_size}, is_qldpc={not graphlike})"
            ),
            family="unknown",
            priority=self.priority,
            batch_size=batch_size,
            hardware={},
            graphlike=graphlike,
            distance=distance,
            n_qubits=n_qubits,
        )
        return out

    # -- recommendation ----------------------------------------------------
    def _recommend(self, checks_or_H: Any, syndromes: Any, ctx: Mapping[str, Any]) -> Recommendation:
        """Recommendation for ``explain`` (problem optional)."""
        if checks_or_H is not None:
            _H, c2q, n_qubits, _ = self._normalize_problem(checks_or_H, ctx.get("n_qubits"))
            batch_size = ctx.get("batch_size", _infer_batch(syndromes))
            rec = self._recommend_for_problem(c2q, n_qubits, int(batch_size), ctx)
            # `decode` routes graphlike balanced problems through the native
            # AutoDecoder, so `explain` has to report that or it describes a
            # path that will not be taken.
            graphlike = _max_qubit_degree(c2q, n_qubits) <= 2
            if self._native_auto_applies(graphlike) and _native_auto_available():
                return replace(
                    rec,
                    decoder=DecoderName.AUTO_NATIVE,
                    reason=(
                        "native Rust AutoDecoder routing (balanced priority, graphlike problem); "
                        f"Python policy would have picked {rec.decoder!r}. Falls back to that "
                        "policy path if the native decoder rejects the problem."
                    ),
                )
            return rec
        # Pure-metadata path.
        return recommend(
            code_family=ctx.get("code_family", self.default_family),
            distance=ctx.get("distance"),
            n_qubits=ctx.get("n_qubits"),
            batch_size=int(ctx.get("batch_size", _infer_batch(syndromes))),
            priority=ctx.get("priority", self.priority),
            hardware=ctx.get("hardware", self.hardware),
        )

    def _recommend_for_problem(
        self,
        c2q: Sequence[Sequence[int]],
        n_qubits: int,
        batch_size: int,
        ctx: Mapping[str, Any],
    ) -> Recommendation:
        """Recommendation informed by the *actual* check structure (the guard)."""
        graphlike = _max_qubit_degree(c2q, n_qubits) <= 2
        rec = recommend(
            code_family=ctx.get("code_family", self.default_family),
            distance=ctx.get("distance"),
            n_qubits=n_qubits,
            batch_size=batch_size,
            priority=ctx.get("priority", self.priority),
            hardware=ctx.get("hardware", self.hardware),
            graphlike=graphlike,
        )
        # Hard safety guard: never let a matching-only decoder run on a
        # non-graphlike problem, whatever the label said.
        if not graphlike and rec.decoder in DecoderName.MATCHING_ONLY:
            rec = replace(
                rec,
                decoder=DecoderName.BP_OSD,
                reason=(
                    f"GUARD: max qubit degree > 2 (hyperedges) but policy picked "
                    f"{rec.decoder!r}; forced BP-OSD to guarantee H·c == s"
                ),
                family="ldpc",
                forced=True,
                gpu_batched_bp=bool(rec.hardware.get("gpu") and batch_size > 1),
            )
        return rec

    # -- decoder construction + caching ------------------------------------
    def _refresh_cache_for(self, c2q: Sequence[Sequence[int]], n_qubits: int) -> None:
        sig = _structure_sig(c2q, n_qubits)
        if sig != self._sig:
            self._cache.clear()
            self._sig = sig

    def _get_decoder(self, rec: Recommendation, H: np.ndarray, c2q, n_qubits: int):
        key = (rec.decoder,)
        dec = self._cache.get(key)
        if dec is not None:
            return dec, rec

        dec = self._build(rec.decoder, H, c2q, n_qubits)
        if dec is None:
            # Construction failed (e.g. CUDA requested but the build lacks the
            # feature / no device). Fall back to a valid path for this problem.
            fallback = DecoderName.BP_OSD if rec.graphlike is False else DecoderName.FAST_UNION_FIND
            rec = replace(
                rec,
                decoder=fallback,
                reason=rec.reason + f"; build of {rec.decoder!r} failed -> fell back to {fallback!r}",
                forced=True,
            )
            key = (fallback,)
            dec = self._cache.get(key) or self._build(fallback, H, c2q, n_qubits)
            if dec is None:  # pragma: no cover - both paths failing is not expected
                raise RuntimeError(f"could not construct a decoder for {rec.decoder!r}")
        self._cache[key] = dec
        return dec, rec

    def _build(self, name: str, H: np.ndarray, c2q, n_qubits: int):
        """Construct a decoder by name, or return ``None`` on failure."""
        from . import (
            BatchDecoder,
            BlossomDecoder,
            CUDABatchDecoder,
            FastUnionFindDecoder,
            SparseBlossomDecoder,
            UnionFindDecoder,
        )
        from .bposd import BpOsdDecoder

        builders = {
            DecoderName.UNION_FIND: lambda: UnionFindDecoder(c2q, n_qubits),
            DecoderName.FAST_UNION_FIND: lambda: FastUnionFindDecoder(c2q, n_qubits),
            DecoderName.BLOSSOM: lambda: BlossomDecoder(c2q, n_qubits),
            DecoderName.SPARSE_BLOSSOM: lambda: SparseBlossomDecoder(c2q, n_qubits),
            DecoderName.BATCH: lambda: BatchDecoder(c2q, n_qubits),
            DecoderName.CUDA_BATCH: lambda: CUDABatchDecoder(c2q, n_qubits),
            DecoderName.BP_OSD: lambda: BpOsdDecoder(H, error_rate=self.error_rate),
        }
        builder = builders.get(name)
        if builder is None:  # pragma: no cover - guarded by DecoderName.ALL
            raise ValueError(f"unknown decoder name {name!r}")
        try:
            return builder()
        except (ImportError, OSError, AttributeError, TypeError, ValueError, RuntimeError):
            return None

    # -- dispatch ----------------------------------------------------------
    def _run(self, dec: Any, syn: np.ndarray, n_qubits: int) -> np.ndarray:
        if syn.ndim == 1:
            return self._decode_one(dec, syn, n_qubits)
        if syn.ndim == 2:
            return self._decode_many(dec, syn)
        raise ValueError(f"syndromes must be 1-D or 2-D, got shape {syn.shape}")

    @staticmethod
    def _decode_one(dec: Any, s: np.ndarray, n_qubits: int) -> np.ndarray:
        if hasattr(dec, "decode"):
            return np.asarray(dec.decode(s)).astype(np.uint8).reshape(-1)
        # Batch-only decoder (e.g. BatchDecoder / CUDABatchDecoder): wrap as 1xN.
        out = AutoRouter._batch_call(dec, s.reshape(1, -1))
        return cast(np.ndarray, np.asarray(out)[0].astype(np.uint8).reshape(-1))

    @staticmethod
    def _decode_many(dec: Any, syn: np.ndarray) -> np.ndarray:
        out = AutoRouter._batch_call(dec, np.ascontiguousarray(syn, dtype=np.uint8))
        if out is not None:
            return np.asarray(out).astype(np.uint8)
        # Last resort: per-row single decode.
        return np.stack([np.asarray(dec.decode(syn[i])).astype(np.uint8) for i in range(syn.shape[0])])

    @staticmethod
    def _batch_call(dec: Any, syn: np.ndarray):
        if hasattr(dec, "batch_decode"):
            return dec.batch_decode(syn)
        if hasattr(dec, "parallel_batch_decode"):
            return dec.parallel_batch_decode(syn)
        return None

    # -- problem normalisation ---------------------------------------------
    def _normalize_problem(self, obj: Any, n_qubits: int | None):
        """Return ``(H, check_to_qubits, n_qubits, n_checks)`` from any problem form."""
        # A codes.Code (or anything exposing the same surface).
        if hasattr(obj, "parity_check_matrix") and hasattr(obj, "check_to_qubits"):
            H = np.asarray(obj.parity_check_matrix(), dtype=np.uint8) % 2
            c2q = [[int(q) for q in check] for check in obj.check_to_qubits]
            nq = int(getattr(obj, "n_qubits", H.shape[1]))
            return H.astype(np.uint8), c2q, nq, len(c2q)

        # A dense 2-D parity-check matrix.
        if isinstance(obj, np.ndarray) and obj.ndim == 2:
            H = (obj % 2).astype(np.uint8)
            c2q = _H_to_c2q(H)
            return H, c2q, int(H.shape[1]), int(H.shape[0])

        # A 1-D array here is never a valid checks_or_H (Code / 2-D H / list-of-
        # lists are the only accepted forms) -- it is almost always a syndrome
        # passed in the wrong argument position, e.g. ``decode(syndrome, checks)``
        # instead of ``decode(checks, syndrome)``. Fail with an actionable message
        # rather than crashing inside the check_to_qubits fallback below with an
        # opaque "not iterable" error on a numpy scalar.
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            raise TypeError(
                "checks_or_H must be a codes.Code, a 2-D parity-check matrix, or a "
                "check_to_qubits list of lists -- got a 1-D array of shape "
                f"{obj.shape}. This usually means the syndrome and checks_or_H "
                "arguments were swapped: decode() takes checks_or_H first, then "
                "syndromes (see AutoRouter.decode.__doc__)."
            )

        # Otherwise treat as check_to_qubits (list of lists of qubit indices).
        c2q = [[int(q) for q in check] for check in obj]
        if not c2q:
            raise ValueError("check_to_qubits must be non-empty")
        inferred = 1 + max((max(c) for c in c2q if c), default=-1)
        nq = int(n_qubits) if n_qubits is not None else inferred
        if nq < inferred:
            raise ValueError(f"n_qubits={nq} too small for qubit index {inferred - 1}")
        H = _c2q_to_H(c2q, nq)
        return H, c2q, nq, len(c2q)
