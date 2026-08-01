"""
qector_decoder_v3.sinter_compat - plug QECTOR into Sinter.

`sinter <https://github.com/quantumlib/Stim/tree/main/glue/sample>`_ is the
standard harness for Monte-Carlo logical-error-rate sampling of Stim circuits.
Exposing QECTOR through Sinter's decoder interface makes QECTOR's accuracy
**externally verifiable with the community-standard tool** - the same harness
people use to benchmark PyMatching, fusion-blossom, etc.

Usage
-----
>>> import sinter, stim
>>> from qector_decoder_v3.sinter_compat import qector_sinter_decoders
>>> tasks = [sinter.Task(circuit=c, json_metadata={"d": d}) for ...]
>>> samples = sinter.collect(
...     num_workers=4,
...     tasks=tasks,
...     decoders=["qector_belief"],
...     custom_decoders=qector_sinter_decoders(),
... )

Decoders provided: ``qector_blossom`` (weighted exact MWPM),
``qector_belief`` (belief-matching), ``qector_unionfind`` (fast, unweighted).
"""

from __future__ import annotations

import numpy as np

__all__ = ["QectorSinterDecoder", "qector_sinter_decoders"]

try:
    import sinter

    _SINTER_BASE: type = sinter.Decoder
    _COMPILED_BASE: type = sinter.CompiledDecoder
    _HAS_SINTER = True
except (ImportError, RuntimeError):  # pragma: no cover - sinter optional
    _SINTER_BASE = object
    _COMPILED_BASE = object
    _HAS_SINTER = False


class _CompiledQectorDecoder(_COMPILED_BASE):  # type: ignore[misc,valid-type]
    """A compiled QECTOR decoder bound to one detector error model."""

    def __init__(self, matcher, num_detectors: int, num_observables: int):
        self.matcher = matcher
        self.num_detectors = int(num_detectors)
        self.num_observables = int(num_observables)

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        # Sinter/Stim use little-endian bit packing.
        dets = np.unpackbits(
            np.ascontiguousarray(bit_packed_detection_event_data),
            axis=1,
            count=self.num_detectors,
            bitorder="little",
        ).astype(np.uint8)
        preds = np.asarray(self.matcher.decode_batch(dets), dtype=np.uint8)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if preds.shape[1] != self.num_observables:
            # pad/truncate defensively to the declared observable count
            fixed = np.zeros((preds.shape[0], self.num_observables), np.uint8)
            k = min(self.num_observables, preds.shape[1])
            fixed[:, :k] = preds[:, :k]
            preds = fixed
        return np.packbits(preds, axis=1, bitorder="little")


class QectorSinterDecoder(_SINTER_BASE):  # type: ignore[misc,valid-type]
    """A Sinter ``Decoder`` backed by QECTOR.

    ``kind`` selects the backend: ``"blossom"`` (weighted exact MWPM),
    ``"belief"`` (belief-matching), or ``"unionfind"`` (fast, unweighted).
    """

    def __init__(self, kind: str = "belief"):
        if not _HAS_SINTER:  # pragma: no cover
            raise ImportError("sinter is not installed (pip install sinter)")
        self.kind = kind

    def compile_decoder_for_dem(self, *, dem) -> _CompiledQectorDecoder:
        matcher = _build_matcher(self.kind, dem)
        return _CompiledQectorDecoder(matcher, dem.num_detectors, dem.num_observables)


def _build_matcher(kind: str, dem):
    kind = kind.lower()
    if kind in ("belief", "belief_matching", "bp"):
        from .belief_matching import BeliefMatching

        return BeliefMatching.from_detector_error_model(dem)
    if kind in ("blossom", "mwpm", "matching"):
        from .pymatching_compat import Matching

        return Matching.from_detector_error_model(dem)
    if kind in ("unionfind", "uf", "union_find"):
        return _UnionFindSinter(dem)
    if kind in ("unionfind_unweighted", "uf_unweighted", "union_find_unweighted"):
        return _UnionFindSinter(dem, weighted=False)
    if kind in ("bposd", "bp_osd", "bp-osd"):
        return _BpOsdSinter(dem)
    if kind in ("cuda", "cuda_batch", "gpu"):
        return _GpuBatchSinter(dem, "cuda")
    if kind in ("cuda_weighted", "cuda_batch_weighted", "gpu_weighted"):
        return _GpuBatchSinter(dem, "cuda", weighted=True)
    if kind in ("opencl", "opencl_batch"):
        return _GpuBatchSinter(dem, "opencl")
    if kind in ("opencl_weighted", "opencl_batch_weighted"):
        return _GpuBatchSinter(dem, "opencl", weighted=True)
    raise ValueError(f"unknown QECTOR sinter decoder kind: {kind!r}")


class _GpuBatchSinter:
    """CUDA / OpenCL batch decoder on a DEM, with observable-space output.

    Exists so the GPU kernels can be measured through the *same* pipeline as
    every other decoder - one DEM, one sample set, one ``decode_batch`` - rather
    than being quoted from a separate harness. Historically the GPU throughput
    figures in this project came from a different measurement than the LER
    figures beside them, which is precisely the mismatch that got earlier
    benchmark artifacts withdrawn.

    ``weighted`` selects which of the two GPU paths is measured. The kernels do
    accept ``edge_weights``; omitting it is a *choice*, and an expensive one -
    an unweighted kernel cannot tell a ``p = 1e-4`` mechanism from a ``p = 1e-2``
    one, so at circuit level its logical error rate stops improving as ``d``
    grows. Both variants are exposed here so that trade is measurable rather
    than asserted, and so the weighted GPU LER - which the README notes has
    never been backed by a surviving artifact - can finally be produced.

    Quote a GPU throughput figure together with its LER, never alone.
    """

    def __init__(self, dem, backend: str = "cuda", weighted: bool = False):
        from . import CUDABatchDecoder, OpenCLBatchDecoder
        from .dem import from_stim

        model = from_stim(dem)
        if model.is_graphlike:
            model = model.collapse_to_graph()
        self._L = model.observables_matrix()
        cls = CUDABatchDecoder if backend == "cuda" else OpenCLBatchDecoder
        if weighted:
            self._dec = cls(model.check_to_qubits(), model.num_errors, model.weights().tolist())
        else:
            self._dec = cls(model.check_to_qubits(), model.num_errors)

    def decode_batch(self, shots):
        corr = np.asarray(self._dec.batch_decode(np.asarray(shots, np.uint8)), dtype=np.uint8)
        return ((self._L @ corr.T) & 1).T.astype(np.uint8)


class _UnionFindSinter:
    """Weighted UF path with observable mapping (for Sinter).

    ``weighted=False`` reproduces the pre-v0.7.0 topology-only behaviour, kept so
    the accuracy effect of UF-01 can be measured against the decoder it replaced
    on an identical DEM.
    """

    def __init__(self, dem, weighted: bool = True):
        from . import UnionFindDecoder
        from .dem import from_stim

        model = from_stim(dem)
        if model.is_graphlike:
            model = model.collapse_to_graph()
        self._L = model.observables_matrix()
        self._dec = UnionFindDecoder(
            model.check_to_qubits(),
            model.num_errors,
            edge_weights=model.weights().tolist() if weighted else None,
        )

    def decode_batch(self, shots):
        corr = np.asarray(self._dec.batch_decode(np.asarray(shots, np.uint8)), dtype=np.uint8)
        return ((self._L @ corr.T) & 1).T.astype(np.uint8)


class _BpOsdSinter:
    """BP-OSD path with observable mapping (E5: circuit-level DEM for BP-OSD)."""

    def __init__(self, dem):
        from .dem import from_stim

        model = from_stim(dem)
        self._L = model.observables_matrix()
        self._nq = model.num_errors
        mean_p = float(model.priors().mean()) if model.num_errors else 0.05
        from . import BPOSDDecoder

        self._dec = BPOSDDecoder(model.check_to_qubits(), model.num_errors, max(mean_p, 1e-3))

    def decode_batch(self, shots):
        corr = np.zeros((shots.shape[0], self._nq), dtype=np.uint8)
        for i in range(shots.shape[0]):
            corr[i] = np.asarray(self._dec.decode(shots[i]), dtype=np.uint8)
        return ((self._L @ corr.T) & 1).T.astype(np.uint8)


def qector_sinter_decoders() -> dict[str, QectorSinterDecoder]:
    """Return the ``custom_decoders`` mapping to pass to ``sinter.collect``."""
    if not _HAS_SINTER:  # pragma: no cover
        raise ImportError("sinter is not installed (pip install sinter)")
    return {
        "qector_blossom": QectorSinterDecoder("blossom"),
        "qector_belief": QectorSinterDecoder("belief"),
        "qector_unionfind": QectorSinterDecoder("unionfind"),
        "qector_bposd": QectorSinterDecoder("bposd"),
    }
