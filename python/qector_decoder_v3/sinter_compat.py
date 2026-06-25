"""
qector_decoder_v3.sinter_compat — plug QECTOR into Sinter.

`sinter <https://github.com/quantumlib/Stim/tree/main/glue/sample>`_ is the
standard harness for Monte-Carlo logical-error-rate sampling of Stim circuits.
Exposing QECTOR through Sinter's decoder interface makes QECTOR's accuracy
**externally verifiable with the community-standard tool** — the same harness
people use to benchmark PyMatching, fusion-blossom, etc.

Usage
-----
>>> import sinter, stim
>>> from qector_decoder_v3.sinter_compat import qector_sinter_decoders
>>> tasks = [sinter.Task(circuit=c, json_metadata={"d": d}) for ...]
>>> samples = sinter.collect(
...     num_workers=4, tasks=tasks,
...     decoders=["qector_belief"],
...     custom_decoders=qector_sinter_decoders(),
... )

Decoders provided: ``qector_blossom`` (weighted exact MWPM),
``qector_belief`` (belief-matching), ``qector_unionfind`` (fast, unweighted).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    pass  # sinter types used only at runtime

__all__ = ["QectorSinterDecoder", "qector_sinter_decoders", "QectorDecoderWrapper"]

# Backward-compat alias — some docs and older examples used this name
# (defined after QectorSinterDecoder class below)

try:
    import sinter

    _SINTER_BASE: type = sinter.Decoder
    _COMPILED_BASE: type = sinter.CompiledDecoder
    _HAS_SINTER = True
except Exception:  # pragma: no cover - sinter optional
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

    Also usable standalone via ``.decode(syndrome, dem)`` for non-Sinter testing.
    """

    def __init__(self, kind: str = "belief"):
        if not _HAS_SINTER:  # pragma: no cover
            raise ImportError("sinter is not installed (pip install sinter)")
        self.kind = kind

    def compile_decoder_for_dem(self, *, dem) -> "_CompiledQectorDecoder":
        matcher = _build_matcher(self.kind, dem)
        return _CompiledQectorDecoder(matcher, dem.num_detectors, dem.num_observables)

    def decode(self, syndrome, dem=None) -> np.ndarray:
        """Decode a single syndrome vector.

        Parameters
        ----------
        syndrome : array-like
            Binary detection event vector of length ``num_detectors``.
        dem : stim.DetectorErrorModel, optional
            DEM to build the decoder from. Required on the first call; cached
            for subsequent calls with the same instance.

        Returns
        -------
        np.ndarray
            Predicted observable flips, shape ``(num_observables,)``.
        """
        if dem is not None:
            self._cached_dem = dem
        if not hasattr(self, "_cached_dem") or self._cached_dem is None:
            raise ValueError("dem must be provided on the first call to .decode()")
        compiled = self.compile_decoder_for_dem(dem=self._cached_dem)
        s = np.asarray(syndrome, dtype=np.uint8).reshape(1, -1)
        # pack -> decode -> unpack
        n_det = compiled.num_detectors
        packed = np.packbits(
            np.pad(s, ((0, 0), (0, (8 - n_det % 8) % 8)), mode="constant"),
            axis=1,
            bitorder="little",
        )
        result_packed = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
        result = np.unpackbits(result_packed, axis=1, count=compiled.num_observables, bitorder="little")
        return result[0].astype(np.uint8)


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
    raise ValueError(f"unknown QECTOR sinter decoder kind: {kind!r}")


class _UnionFindSinter:
    """Fast unweighted UF path with observable mapping (for Sinter)."""

    def __init__(self, dem):
        from . import UnionFindDecoder
        from .dem import from_stim

        model = from_stim(dem)
        if model.is_graphlike:
            model = model.collapse_to_graph()
        self._L = model.observables_matrix()
        self._dec = UnionFindDecoder(model.check_to_qubits(), model.num_errors)

    def decode_batch(self, shots):
        corr = np.asarray(self._dec.batch_decode(np.asarray(shots, np.uint8)), dtype=np.uint8)
        return ((self._L @ corr.T) & 1).T.astype(np.uint8)


def qector_sinter_decoders() -> Dict[str, "QectorSinterDecoder"]:
    """Return the ``custom_decoders`` mapping to pass to ``sinter.collect``."""
    if not _HAS_SINTER:  # pragma: no cover
        raise ImportError("sinter is not installed (pip install sinter)")
    return {
        "qector_blossom": QectorSinterDecoder("blossom"),
        "qector_belief": QectorSinterDecoder("belief"),
        "qector_unionfind": QectorSinterDecoder("unionfind"),
    }


# Backward-compatibility alias — older docs and examples used QectorDecoderWrapper
QectorDecoderWrapper = QectorSinterDecoder
