"""
QECTOR Qiskit Plugin — intégration optionnelle avec l'écosystème Qiskit.

Permet d'utiliser QECTOR comme decoder par défaut pour les résultats
de circuits de codes de surface Qiskit.

Usage (avec Qiskit installé) ::

    from qiskit import QuantumCircuit
    from qector_decoder_v3.qiskit_plugin import decode_qiskit_result, create_qiskit_decoder

    decoder = create_qiskit_decoder(code_distance=5)
    result = job.result()           # qiskit.result.Result
    decoded = decoder(result)

Usage (sans Qiskit — mode dict brut) ::

    raw = {"counts": {"0x0": 400, "0x3": 100}}
    out = decode_qiskit_result(raw, code_distance=3)
    # out["correction"] -> np.ndarray (n_shots, n_qubits)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

from . import UnionFindDecoder, generate_surface_code_checks

# Import optionnel de Qiskit — le plugin reste importable sans Qiskit
# ------------------------------------------------------------------------
try:
    from qiskit.result import Result as _QiskitResult

    _HAS_QISKIT = True
except ImportError:  # pragma: no cover
    _QiskitResult = None
    _HAS_QISKIT = False


def _normalize_counts(result: Any) -> Dict[str, int]:
    """Extraire les comptes bruts depuis un Result Qiskit ou un dict."""
    if isinstance(result, dict):
        counts = result.get("counts")
        if counts is None:
            raise ValueError("Dict result must contain a 'counts' key")
        return {str(k): int(v) for k, v in counts.items()}

    if _HAS_QISKIT and isinstance(result, _QiskitResult):
        # get_counts() retourne un dict {bitstring: count}
        return {str(k): int(v) for k, v in result.get_counts().items()}

    raise TypeError(f"result must be a dict or qiskit.result.Result, got {type(result).__name__}")


def _bitstring_to_syndrome(bitstring: str, n_checks: int) -> List[int]:
    """Convertir une chaîne Qiskit (binaire ou hex) en liste de bits syndrome."""
    if bitstring.startswith("0x"):
        val = int(bitstring, 16)
        return [(val >> i) & 1 for i in range(n_checks)]

    # Chaîne binaire : '0101...' — on inverse pour que le LSB soit en index 0
    bits = [int(c) for c in bitstring][::-1]
    if len(bits) < n_checks:
        bits += [0] * (n_checks - len(bits))
    return bits[:n_checks]


def decode_qiskit_result(
    result: Any,
    code_distance: int,
    shots: int | None = None,
    *,
    n_qubits: int | None = None,
) -> Dict[str, Any]:
    """
    Décoder un résultat Qiskit (ou dictionnaire brut) avec QECTOR.

    Paramètres
    ----------
    result : qiskit.result.Result | dict
        Résultat d'un job Qiskit. Si Qiskit n'est pas installé, un dict
        avec la clé ``counts`` est accepté.
    code_distance : int
        Distance du code de surface (ex: 3, 5, 7, …).
    shots : int, optionnel
        Nombre de shots. Déduit automatiquement de ``result`` si absent.
    n_qubits : int, optionnel
        Nombre de qubits. Déduit automatiquement du code de surface si absent.

    Retourne
    -------
    dict
        {
            "correction": np.ndarray — correction pour chaque shot,
            "syndrome": np.ndarray — syndrome déduit,
            "metadata": {
                "decoder": "QECTOR UnionFind",
                "code_distance": int,
                "n_qubits": int,
                "n_checks": int,
                "shots": int,
                "unique_outcomes": int,
            }
        }
    """
    if not _HAS_QISKIT:
        warnings.warn(
            "Qiskit n'est pas installé. L'intégration fonctionne en mode "
            "'dict brut'. Pour l'usage complet : pip install qiskit",
            stacklevel=2,
        )

    counts = _normalize_counts(result)
    if shots is None:
        shots = sum(counts.values())

    # Génération des checks pour le code de surface demandé
    check_to_qubits, auto_n_qubits = generate_surface_code_checks(code_distance)
    if n_qubits is None:
        n_qubits = auto_n_qubits
    n_checks = len(check_to_qubits)

    decoder = UnionFindDecoder(check_to_qubits, n_qubits=n_qubits)

    # Extraction des syndromes à partir des comptes
    syndrome_list: List[np.ndarray] = []
    for bitstring, count in counts.items():
        bits = _bitstring_to_syndrome(bitstring, n_checks)
        syndrome = np.array(bits, dtype=np.uint8)
        for _ in range(count):
            syndrome_list.append(syndrome)

    if not syndrome_list:
        return {
            "correction": np.zeros((0, n_qubits), dtype=np.uint8),
            "syndrome": np.zeros((0, n_checks), dtype=np.uint8),
            "metadata": {
                "decoder": "QECTOR UnionFind",
                "code_distance": code_distance,
                "n_qubits": n_qubits,
                "n_checks": n_checks,
                "shots": 0,
                "warning": "Aucun compte détecté dans le résultat.",
            },
        }

    syndromes = np.stack(syndrome_list)
    corrections = decoder.batch_decode(syndromes)

    return {
        "correction": corrections,
        "syndrome": syndromes,
        "metadata": {
            "decoder": "QECTOR UnionFind",
            "code_distance": code_distance,
            "n_qubits": n_qubits,
            "n_checks": n_checks,
            "shots": shots,
            "unique_outcomes": len(counts),
        },
    }


def create_qiskit_decoder(
    code_distance: int,
    n_qubits: int | None = None,
) -> Any:
    """
    Factory retournant un callable compatible avec l'API Qiskit.

    Le callable retourné accepte un ``Result`` Qiskit (ou un dict) et retourne
    le résultat décodé.

    Exemple ::

        from qector_decoder_v3.qiskit_plugin import create_qiskit_decoder

        decoder = create_qiskit_decoder(code_distance=5)
        raw_result = sampler.run(circuit).result()
        decoded = decoder(raw_result)

    Paramètres
    ----------
    code_distance : int
        Distance du code de surface.
    n_qubits : int, optionnel
        Nombre de qubits. Déduit automatiquement si absent.

    Retourne
    -------
    callable
        Fonction ``(result) -> dict`` avec attribut ``_inner_decoder``
        pour accéder directement à l'instance ``UnionFindDecoder``.
    """
    check_to_qubits, auto_n_qubits = generate_surface_code_checks(code_distance)
    if n_qubits is None:
        n_qubits = auto_n_qubits

    inner_decoder = UnionFindDecoder(check_to_qubits, n_qubits=n_qubits)

    def _decode(result: Any) -> Dict[str, Any]:
        return decode_qiskit_result(
            result,
            code_distance=code_distance,
            n_qubits=n_qubits,
        )

    _decode._inner_decoder = inner_decoder  # type: ignore[attr-defined]
    return _decode
