"""
Compatibilité Stim — Conversion et wrappers pour l'écosystème Stim.

Stim (https://github.com/quantumlib/stim) est un simulateur de circuits QEC
trépidant. Ce module permet d'utiliser QECTOR comme back-end de décodage
pour des modèles d'erreurs produits par Stim.

Usage ::

    import stim
    from qector_decoder_v3.stim_compat import (
        from_stim_detector_error_model,
        to_stim_decoder,
        stim_decoder_from_dem,
    )

    # 1. Convertir un DEM Stim en check_to_qubits QECTOR
    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5)
    dem = circuit.detector_error_model(decompose_errors=True)
    c2q, nq = from_stim_detector_error_model(dem)

    # 2. Créer un wrapper QECTOR compatible stim.Decoder
    decoder = stim_decoder_from_dem(dem)
    correction = decoder.decode(syndrome)
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from . import UnionFindDecoder, BatchDecoder

# Import optionnel de stim — le module reste importable sans Stim
# ----------------------------------------------------------------
try:
    import stim as _stim
    _HAS_STIM = True
except ImportError:  # pragma: no cover
    _stim = None
    _HAS_STIM = False


def from_stim_detector_error_model(dem: Any) -> Tuple[List[List[int]], int]:
    """
    Convertir un ``stim.DetectorErrorModel`` en ``check_to_qubits`` pour QECTOR.

    Cette fonction effectue un parsing textuel du DEM pour maximiser la
    compatibilité inter-versions de l'API Python de Stim.

    Paramètres
    ----------
    dem : stim.DetectorErrorModel
        Modèle d'erreur détecteur produit par Stim.

    Retourne
    -------
    tuple(check_to_qubits, n_qubits)
        * ``check_to_qubits`` : liste de listes d'indices de qubits.
        * ``n_qubits`` : nombre total de qubits déduit.
    """
    # Delegate to the correct DEM loader in :mod:`qector_decoder_v3.dem`.
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
        raise TypeError(
            f"dem doit être un stim.DetectorErrorModel (ou un texte .dem), "
            f"reçu {type(dem).__name__}"
        )

    check_to_qubits = model.check_to_qubits()
    n_qubits = model.num_errors
    return check_to_qubits, n_qubits


def to_stim_decoder(
    check_to_qubits: List[List[int]],
    n_qubits: int | None = None,
    use_batch: bool = False,
):
    """
    Retourner un wrapper compatible avec l'API de ``stim.Decoder``.

    Le wrapper expose une méthode ``decode(syndrome)`` qui délègue à QECTOR.

    Paramètres
    ----------
    check_to_qubits : list[list[int]]
        Checks au format QECTOR.
    n_qubits : int, optionnel
        Nombre de qubits. Déduit automatiquement si absent.
    use_batch : bool, default False
        Si True, utilise ``BatchDecoder`` pour le décodage interne.

    Retourne
    -------
    QECTORStimDecoder
        Objet avec méthode ``decode(syndrome)`` et attributs ``check_to_qubits``.
    """
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
                return self._inner.parallel_batch_decode(syndrome.reshape(1, -1))[0]
            return self._inner.decode(syndrome)

        def __repr__(self) -> str:
            return (
                f"<QECTORStimDecoder n_qubits={self.n_qubits} "
                f"n_checks={self.n_checks}>"
            )

    return QECTORStimDecoder(inner, check_to_qubits, n_qubits)


def stim_decoder_from_dem(dem: Any, use_batch: bool = False):
    """
    Pipeline complet : ``stim.DetectorErrorModel`` -> QECTOR decoder.

    Paramètres
    ----------
    dem : stim.DetectorErrorModel
        Modèle d'erreur produit par Stim.
    use_batch : bool, default False
        Utiliser ``BatchDecoder`` en interne.

    Retourne
    -------
    QECTORStimDecoder
    """
    c2q, nq = from_stim_detector_error_model(dem)
    return to_stim_decoder(c2q, n_qubits=nq, use_batch=use_batch)
