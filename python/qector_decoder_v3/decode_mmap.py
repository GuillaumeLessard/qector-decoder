"""
qector_decoder_v3.decode_mmap — Out-of-core batch decoding via memory-mapped arrays.

Decode datasets larger than RAM by streaming syndromes from a memory-mapped
file and writing corrections incrementally. Each chunk is flushed to disk
before the next is loaded, keeping peak memory proportional to ``batch_size``.

Example
-------
>>> from qector_decoder_v3 import decode_mmap, generate_repetition_code_checks
>>> import numpy as np
>>> checks, nq = generate_repetition_code_checks(distance=5)
>>> n_shots = 10000
>>> # Create a memory-mapped syndrome file
>>> syn = np.memmap("syndromes.bin", dtype=np.uint8, mode="w+", shape=(n_shots, len(checks)))
>>> syn[:] = (np.random.random((n_shots, len(checks))) < 0.1).astype(np.uint8)
>>> syn.flush()
>>> decode_mmap("syndromes.bin", "corrections.bin", checks, nq, batch_size=1024, n_shots=n_shots)
"""

from __future__ import annotations

import os as _os
from typing import Optional

import numpy as _np

__all__ = ["decode_mmap"]


def decode_mmap(
    syndrome_path: str,
    output_path: str,
    check_to_qubits,
    n_qubits: int,
    decoder_type: str = "cpu_batch",
    batch_size: int = 65536,
    n_shots: Optional[int] = None,
    dtype: _np.dtype = _np.uint8,
    verbose: bool = False,
):
    """Out-of-core batch decoding via memory-mapped arrays.

    Args:
        syndrome_path: Path to the syndrome file (memory-mapped, mode 'r').
        output_path: Path for the correction output file (memory-mapped, mode 'w+').
        check_to_qubits: List of lists of qubit indices per check.
        n_qubits: Number of data qubits.
        decoder_type: Which decoder to use (``"cpu_batch"`` recommended for
            linear scaling with batch size).
        batch_size: Number of syndromes to decode per chunk.
        n_shots: Total number of shots. If None, inferred from file size.
        dtype: NumPy dtype of the syndrome file.
        verbose: If True, print progress information.

    Returns:
        The output path (for chaining).
    """
    c2q = [[int(q) for q in c] for c in check_to_qubits]
    n_checks = len(c2q)
    file_size = _os.path.getsize(syndrome_path)
    bytes_per_shot = n_checks * _np.dtype(dtype).itemsize
    if n_shots is None:
        n_shots = file_size // bytes_per_shot
    if file_size < n_shots * bytes_per_shot:
        raise ValueError(
            f"syndrome file too small: {file_size} bytes, need {n_shots * bytes_per_shot} for {n_shots} shots"
        )

    syndromes = _np.memmap(syndrome_path, dtype=dtype, mode="r", shape=(n_shots, n_checks))
    output = _np.memmap(output_path, dtype=_np.uint8, mode="w+", shape=(n_shots, n_qubits))

    from . import CPUBatchDecoder, UnionFindDecoder

    if decoder_type == "cpu_batch":
        decoder = CPUBatchDecoder(c2q, n_qubits)
        batch_fn = decoder.batch_decode
    else:
        decoder = UnionFindDecoder(c2q, n_qubits)
        batch_fn = lambda batch: _np.array([decoder.decode(batch[i]) for i in range(batch.shape[0])])

    n_chunks = (n_shots + batch_size - 1) // batch_size
    for chunk_idx in range(n_chunks):
        start = chunk_idx * batch_size
        end = min(start + batch_size, n_shots)
        chunk = _np.ascontiguousarray(syndromes[start:end], dtype=_np.uint8)
        output[start:end] = _np.asarray(batch_fn(chunk), dtype=_np.uint8)
        output.flush()
        if verbose:
            pct = (end / n_shots) * 100.0
            print(f"\rdecode_mmap: {end}/{n_shots} ({pct:.0f}%)", end="", flush=True)

    if verbose:
        print()

    return output_path
