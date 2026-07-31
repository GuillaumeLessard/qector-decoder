"""
QECTOR Qiskit Plugin - optional integration with the Qiskit ecosystem.

Provides a QECTOR decoder for surface code Qiskit circuit results.

Usage (with Qiskit installed)::

    from qiskit import QuantumCircuit
    from qector_decoder_v3.qiskit_plugin import decode_qiskit_result, create_qiskit_decoder

    decoder = create_qiskit_decoder(code_distance=5)
    result = job.result()  # qiskit.result.Result
    decoded = decoder(result)

Usage (without Qiskit - raw dict mode)::

    raw = {"counts": {"0x0": 400, "0x3": 100}}
    out = decode_qiskit_result(raw, code_distance=3)
    # out["correction"] -> np.ndarray (n_shots, n_qubits)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import BlossomDecoder, generate_surface_code_checks

# Optional Qiskit import - the plugin remains importable without Qiskit
# ------------------------------------------------------------------------
try:
    from qiskit.result import Result as _QiskitResult

    _HAS_QISKIT = True
except ImportError:  # pragma: no cover
    _QiskitResult = None
    _HAS_QISKIT = False


def _normalize_counts(result: Any) -> dict[str, int]:
    """Extract raw counts from a Qiskit Result or a dict."""
    if isinstance(result, dict):
        counts = result.get("counts")
        if counts is None:
            raise ValueError("Dict result must contain a 'counts' key")
        return {str(k): int(v) for k, v in counts.items()}

    if _HAS_QISKIT and isinstance(result, _QiskitResult):
        # get_counts() returns a dict {bitstring: count}
        return {str(k): int(v) for k, v in result.get_counts().items()}

    raise TypeError(f"result must be a dict or qiskit.result.Result, got {type(result).__name__}")


def _bitstring_to_syndrome(bitstring: str, n_checks: int) -> list[int]:
    """Convert a Qiskit (binary or hex) bitstring to a syndrome bit list."""
    if bitstring.startswith("0x"):
        val = int(bitstring, 16)
        return [(val >> i) & 1 for i in range(n_checks)]

    # Binary string: '0101...' - reversed so LSB is at index 0
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
) -> dict[str, Any]:
    """
    Decode a Qiskit result (or raw dict) with QECTOR.

    Parameters
    ----------
    result : qiskit.result.Result | dict
        Result of a Qiskit job. If Qiskit is not installed, a dict
        with a ``counts`` key is accepted.
    code_distance : int
        Surface code distance (e.g. 3, 5, 7, ...).
    shots : int, optional
        Number of shots. Auto-detected from ``result`` if absent.
    n_qubits : int, optional
        Number of qubits. Auto-detected from the surface code if absent.

    Returns
    -------
    dict
        {
            "correction": np.ndarray - correction for each shot,
            "syndrome": np.ndarray - inferred syndrome,
            "metadata": {
                "decoder": "QECTOR Blossom",
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
            "Qiskit is not installed. Integration works in 'raw dict' mode. For full usage: pip install qiskit",
            stacklevel=2,
        )

    counts = _normalize_counts(result)
    if shots is None:
        shots = sum(counts.values())

    # Generate checks for the requested surface code
    check_to_qubits, auto_n_qubits = generate_surface_code_checks(code_distance)
    if n_qubits is None:
        n_qubits = auto_n_qubits
    n_checks = len(check_to_qubits)

    decoder = BlossomDecoder(check_to_qubits, n_qubits=n_qubits)

    # Extract syndromes from counts
    syndrome_list: list[np.ndarray] = []
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
                "decoder": "QECTOR Blossom",
                "code_distance": code_distance,
                "n_qubits": n_qubits,
                "n_checks": n_checks,
                "shots": 0,
                "warning": "No counts detected in the result.",
            },
        }

    syndromes = np.stack(syndrome_list)
    corrections = decoder.batch_decode(syndromes)

    return {
        "correction": corrections,
        "syndrome": syndromes,
        "metadata": {
            "decoder": "QECTOR Blossom",
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
    Factory returning a callable compatible with the Qiskit API.

    The returned callable accepts a Qiskit ``Result`` (or a dict) and returns
    the decoded result.

    Example::

        from qector_decoder_v3.qiskit_plugin import create_qiskit_decoder

        decoder = create_qiskit_decoder(code_distance=5)
        raw_result = sampler.run(circuit).result()
        decoded = decoder(raw_result)

    Parameters
    ----------
    code_distance : int
        Surface code distance.
    n_qubits : int, optional
        Number of qubits. Auto-detected if absent.

    Returns
    -------
    callable
        Function ``(result) -> dict`` with ``_inner_decoder`` attribute
        for direct access to the ``BlossomDecoder`` instance.
    """
    check_to_qubits, auto_n_qubits = generate_surface_code_checks(code_distance)
    if n_qubits is None:
        n_qubits = auto_n_qubits

    inner_decoder = BlossomDecoder(check_to_qubits, n_qubits=n_qubits)

    def _decode(result: Any) -> dict[str, Any]:
        return decode_qiskit_result(
            result,
            code_distance=code_distance,
            n_qubits=n_qubits,
        )

    _decode._inner_decoder = inner_decoder  # type: ignore[attr-defined]
    return _decode
