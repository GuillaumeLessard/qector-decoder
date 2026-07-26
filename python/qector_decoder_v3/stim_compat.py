"""
Compatibilité Stim - Conversion et wrappers pour l'écosystème Stim.

.. deprecated::
    This module is deprecated since v0.6.4. Use ``qector_decoder_v3.dem`` instead.
    See https://qector.store/docs/migration-v0.6.4 for the migration guide.

Stim (https://github.com/quantumlib/stim) est un simulateur de circuits QEC
trépidant. Ce module permet d'utiliser QECTOR comme back-end de décodage
pour des modèles d'erreurs produits par Stim.

Usage (deprecated) ::

    import stim
    from qector_decoder_v3.stim_compat import (
        from_stim_detector_error_model,  # DEPRECATED
        to_stim_decoder,  # DEPRECATED
        stim_decoder_from_dem,  # DEPRECATED
    )

New API (preferred)::

    import stim
    from qector_decoder_v3 import dem

    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, before_round_data_depolarization=0.01)
    model = dem.from_stim(circuit.detector_error_model(decompose_errors=True))
    decoder = model.make_decoder("blossom")
    correction = decoder.decode(syndrome)
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np

from . import BatchDecoder, UnionFindDecoder

_DEPRECATION_MSG = (
    "stim_compat is deprecated since v0.6.4. "
    "Use qector_decoder_v3.dem.from_stim() instead. "
    "See the migration guide at https://qector.store/docs/migration-v0.6.4"
)

# Import optionnel de stim - le module reste importable sans Stim
# ----------------------------------------------------------------
try:
    import stim as _stim

    _HAS_STIM = True
except ImportError:  # pragma: no cover
    _stim = None
    _HAS_STIM = False


def from_stim_detector_error_model(dem: Any) -> tuple[list[list[int]], int]:
    """
    .. deprecated::
        Use ``qector_decoder_v3.dem.from_stim()`` instead.

    Convertir un ``stim.DetectorErrorModel`` en ``check_to_qubits`` pour QECTOR.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    #
    # The earlier implementation here conflated detector indices with qubit
    # indices (``detector_to_qubits[d].add(d)``), producing an incorrect ``H``.
    # The correct detector graph treats each DEM *error mechanism* as a column
    # (a "qubit") and each *detector* as a row (a check):
    # ``check_to_qubits[detector]`` lists the mechanism indices that flip it.
    from .dem import from_stim, parse_dem

    if isinstance(dem, str):
        model = parse_dem(dem)
    elif hasattr(dem, "num_detectors"):
        model = from_stim(dem)
    else:
        raise TypeError(f"dem doit être un stim.DetectorErrorModel (ou un texte .dem), reçu {type(dem).__name__}")

    check_to_qubits = model.check_to_qubits()
    n_qubits = model.num_errors
    return check_to_qubits, n_qubits


def to_stim_decoder(
    check_to_qubits: list[list[int]],
    n_qubits: int | None = None,
    use_batch: bool = False,
):
    """
    .. deprecated::
        Use ``qector_decoder_v3.dem.from_stim()`` + ``model.make_decoder()`` instead.

    Retourner un wrapper compatible avec l'API de ``stim.Decoder``.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    decoder_cls = BatchDecoder if use_batch else UnionFindDecoder
    inner = decoder_cls(check_to_qubits, n_qubits=n_qubits)

    if n_qubits is None:
        n_qubits = max(max(c) for c in check_to_qubits) + 1

    class QECTORStimDecoder:
        """Wrapper QECTOR compatible avec l'interface ``stim.Decoder``-like."""

        def __init__(self, _inner, c2q, nq):
            self._inner = _inner
            self.check_to_qubits = c2q
            self.n_qubits = nq
            self.n_checks = len(c2q)

        def decode(self, syndrome: Any) -> np.ndarray:
            """
            Décoder un syndrome.

            Paramètres
            ----------
            syndrome : array-like
                Syndrome binaire de longueur ``n_checks``.

            Retourne
            -------
            np.ndarray
                Correction de longueur ``n_qubits``.
            """
            if not isinstance(syndrome, np.ndarray):
                syndrome = np.array(syndrome, dtype=np.uint8)
            if syndrome.dtype != np.uint8:
                syndrome = syndrome.astype(np.uint8)
            # BatchDecoder utilise ``parallel_batch_decode``
            if hasattr(self._inner, "parallel_batch_decode"):
                return cast(np.ndarray, self._inner.parallel_batch_decode(syndrome.reshape(1, -1))[0])
            return cast(np.ndarray, self._inner.decode(syndrome))

        def __repr__(self) -> str:
            return f"<QECTORStimDecoder n_qubits={self.n_qubits} n_checks={self.n_checks}>"

    return QECTORStimDecoder(inner, check_to_qubits, n_qubits)


def stim_decoder_from_dem(dem: Any, use_batch: bool = False):
    """
    .. deprecated::
        Use ``qector_decoder_v3.dem.from_stim()`` + ``model.make_decoder()`` instead.

    Pipeline complet : ``stim.DetectorErrorModel`` -> QECTOR decoder.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    c2q, nq = from_stim_detector_error_model(dem)
    return to_stim_decoder(c2q, n_qubits=nq, use_batch=use_batch)
