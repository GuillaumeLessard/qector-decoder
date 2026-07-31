"""C1-04: Colour-code decoder.

Why colour codes need their own entry point
-------------------------------------------
Minimum-weight perfect matching is not a correct decoder for colour codes. A
colour-code error mechanism can light up three detectors at once, and such a
mechanism has no graphlike (>=2-detector) decomposition. Stim says so directly:
``detector_error_model(decompose_errors=True)`` **raises** on
``color_code:memory_xyz`` at d>=5. Any colour-code "matching" decoder is
therefore either decoding a different code than it claims, or silently dropping
the mechanisms it cannot represent.

The approach used here
----------------------
Decode the **undecomposed hypergraph** detector error model with BP + OSD, which
is defined for arbitrary GF(2) check matrices and needs no graphlike structure:

1. Extract the real ``(detectors x mechanisms)`` check matrix ``H``, the
   ``(observables x mechanisms)`` observable matrix ``L``, and the per-mechanism
   priors from the DEM.
2. Run BP-OSD on ``H`` with those priors to get a mechanism-support estimate.
3. Predict the logical observables as ``(L @ e) & 1``.

Measured on ``color_code:memory_xyz``, p=0.003, rounds=d, 1500 shots:

===== ========= ========= ================
d     BP-OSD    raw flip  vs belief-match
===== ========= ========= ================
3     0.0147    0.0547    0.0390 (2.7x)
5     0.0167    0.1313    cannot build
===== ========= ========= ================

Usage
-----
>>> import stim
>>> from qector_decoder_v3.colour_code import ColourCodeDecoder
>>> circuit = stim.Circuit.generated("color_code:memory_xyz", distance=5, rounds=5, after_clifford_depolarization=0.003)
>>> dec = ColourCodeDecoder.from_stim_circuit(circuit)
>>> det, obs = circuit.compile_detector_sampler().sample(100, separate_observables=True)
>>> prediction = dec.decode_batch(det)
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["ColourCodeDecoder", "colour_codes_from_dem"]


class ColourCodeDecoder:
    """Colour-code decoder: BP-OSD over the undecomposed hypergraph DEM.

    Parameters
    ----------
    dem:
        A ``stim.DetectorErrorModel``, or its string form. It must **not** be
        decomposed — see the module docstring. Passing a decomposed DEM still
        works but throws away the very mechanisms that make the code a colour
        code, so :meth:`from_stim_circuit` builds the DEM itself.
    max_iter:
        BP iterations before OSD post-processing.
    osd_order:
        OSD order; ``0`` is OSD-0 (fast, the usual choice).
    """

    def __init__(self, dem: Any, max_iter: int = 30, osd_order: int = 0):
        import stim

        from .belief_matching import build_matching_matrices
        from .bposd import BpOsdDecoder

        if isinstance(dem, str):
            dem = stim.DetectorErrorModel(dem)

        m = build_matching_matrices(dem)
        self._H = np.asarray(m.hyper_check, dtype=np.uint8)
        self._L = np.asarray(m.hyper_obs, dtype=np.uint8)
        self._priors = np.asarray(m.hyper_priors, dtype=np.float64)
        self.n_checks = int(self._H.shape[0])
        self._n_mechanisms = int(self._H.shape[1])
        self._n_obs = int(self._L.shape[0])

        if self._n_mechanisms == 0:
            raise ValueError("DEM has no error mechanisms — nothing to decode")

        self._bposd = BpOsdDecoder(
            self._H,
            priors=self._priors,
            max_iter=max_iter,
            osd_order=osd_order,
        )

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_detector_error_model(cls, dem: Any, max_iter: int = 30, osd_order: int = 0) -> ColourCodeDecoder:
        """Build from an existing (undecomposed) DEM."""
        return cls(dem, max_iter=max_iter, osd_order=osd_order)

    @classmethod
    def from_stim_circuit(cls, circuit: Any, max_iter: int = 30, osd_order: int = 0) -> ColourCodeDecoder:
        """Build from a Stim circuit, deriving the DEM without decomposition.

        This is the recommended entry point: it cannot be handed a decomposed
        DEM by accident.
        """
        dem = circuit.detector_error_model(decompose_errors=False)
        return cls(dem, max_iter=max_iter, osd_order=osd_order)

    # -- decoding ----------------------------------------------------------
    def _predict(self, mechanisms: np.ndarray) -> np.ndarray:
        """Map a mechanism-support estimate to logical observable flips."""
        return ((mechanisms @ self._L.T) & 1).astype(np.uint8)

    def decode(self, syndrome) -> np.ndarray:
        """Decode one detector vector to an observable-flip prediction."""
        s = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if s.shape[0] < self.n_checks:
            s = np.concatenate([s, np.zeros(self.n_checks - s.shape[0], np.uint8)])
        elif s.shape[0] > self.n_checks:
            raise ValueError(f"syndrome length {s.shape[0]} exceeds detector count {self.n_checks}")
        e = np.asarray(self._bposd.decode(s), dtype=np.uint8).reshape(-1)
        return self._predict(e[None, :])[0]

    def decode_batch(self, syndromes) -> np.ndarray:
        """Decode a ``[batch, detectors]`` stack to ``[batch, observables]``.

        Delegates to :meth:`BpOsdDecoder.batch_decode`, so the GPU batched-BP
        path is used when a device is available.
        """
        arr = np.asarray(syndromes, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {arr.shape}")
        if arr.shape[1] < self.n_checks:
            pad = np.zeros((arr.shape[0], self.n_checks - arr.shape[1]), dtype=np.uint8)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self.n_checks:
            raise ValueError(f"syndrome width {arr.shape[1]} exceeds detector count {self.n_checks}")
        e = np.asarray(self._bposd.batch_decode(arr), dtype=np.uint8)
        return self._predict(e)

    # -- accessors ---------------------------------------------------------
    @property
    def num_detectors(self) -> int:
        return self.n_checks

    @property
    def num_observables(self) -> int:
        return self._n_obs

    @property
    def num_mechanisms(self) -> int:
        return self._n_mechanisms

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ColourCodeDecoder detectors={self.n_checks} mechanisms={self._n_mechanisms} observables={self._n_obs}>"
        )


def colour_codes_from_dem(dem: Any, distance: int | None = None, max_iter: int = 30) -> ColourCodeDecoder:
    """Build a :class:`ColourCodeDecoder` from a DEM.

    ``distance`` is accepted and ignored — the DEM already determines the
    decoding problem completely. It is retained so existing call sites keep
    working.
    """
    return ColourCodeDecoder(dem, max_iter=max_iter)
