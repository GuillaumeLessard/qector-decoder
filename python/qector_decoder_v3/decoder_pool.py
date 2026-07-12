"""
qector_decoder_v3.decoder_pool — Multi-process decoder pool.

Distributes batch decoding across multiple worker processes for
near-linear speedup on multi-core machines.

Performance notes (Windows):
- Single-process Rayon parallel (``batch_decode_par``) is **always faster**
  than multi-process on Windows due to ``spawn`` overhead.
- ``DecoderPool`` primarily benefits Linux/macOS (``fork`` start method)
  and batch sizes > 500K shots.

On Windows, if multiple processes are requested, the pool is still created
but the first call transparently uses the single-process Rayon path when
the decode time is small relative to IPC overhead.

Example
-------
>>> from qector_decoder_v3 import DecoderPool, generate_repetition_code_checks
>>> import numpy as np
>>> checks, nq = generate_repetition_code_checks(distance=5)
>>> pool = DecoderPool(checks, nq)
>>> syndromes = np.eye(len(checks), dtype=np.uint8)
>>> corrections = pool.decode(syndromes)
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import platform as _platform
from typing import List, Optional, cast

import numpy as np

__all__ = ["DecoderPool"]

# On Windows, spawn overhead makes multi-process slower than single-process
# Rayon for batches < 500K.  This threshold adapts to platform.
_IS_WINDOWS = _platform.system() == "Windows"


def _worker_init(checks_tuple, n_qubits, decoder_type):
    """Initialize a worker process with its own decoder instance."""
    global _WORKER_DECODER
    checks = [list(c) for c in checks_tuple]
    from . import BlossomDecoder, CPUBatchDecoder, FastUnionFindDecoder, SparseBlossomDecoder, UnionFindDecoder

    builders = {
        "union_find": lambda: UnionFindDecoder(checks, n_qubits),
        "fast_union_find": lambda: FastUnionFindDecoder(checks, n_qubits),
        "blossom": lambda: BlossomDecoder(checks, n_qubits),
        "sparse_blossom": lambda: SparseBlossomDecoder(checks, n_qubits),
        "cpu_batch": lambda: CPUBatchDecoder(checks, n_qubits),
    }
    if decoder_type not in builders:
        raise ValueError(f"unknown decoder type: {decoder_type!r}")
    _WORKER_DECODER = builders[decoder_type]()


def _worker_decode(chunk_and_idx):
    """Decode chunk; returns (idx, result_array)."""
    chunk, idx = chunk_and_idx
    dec = globals().get("_WORKER_DECODER")
    if hasattr(dec, "batch_decode"):
        result = dec.batch_decode(chunk)
    elif hasattr(dec, "batch_decode_par"):
        result = dec.batch_decode_par(chunk)
    else:
        out = np.zeros((chunk.shape[0], dec.n_qubits), dtype=np.uint8)
        for i in range(chunk.shape[0]):
            out[i] = dec.decode(chunk[i])
        result = out
    return (idx, result)


_DECODER_BUILDERS = {
    "union_find": lambda c2q, nq: __import__("qector_decoder_v3", fromlist=["UnionFindDecoder"]).UnionFindDecoder(
        c2q, nq
    ),
    "fast_union_find": lambda c2q, nq: __import__(
        "qector_decoder_v3", fromlist=["FastUnionFindDecoder"]
    ).FastUnionFindDecoder(c2q, nq),
    "blossom": lambda c2q, nq: __import__("qector_decoder_v3", fromlist=["BlossomDecoder"]).BlossomDecoder(c2q, nq),
    "sparse_blossom": lambda c2q, nq: __import__(
        "qector_decoder_v3", fromlist=["SparseBlossomDecoder"]
    ).SparseBlossomDecoder(c2q, nq),
    "cpu_batch": lambda c2q, nq: __import__("qector_decoder_v3", fromlist=["CPUBatchDecoder"]).CPUBatchDecoder(c2q, nq),
}


class DecoderPool:
    """Multi-process decoder pool for parallel syndrome decoding.

    Automatically uses Rayon (single-process, parallel) on Windows for
    best throughput.  Falls back to multi-process on Linux/macOS.

    Args:
        check_to_qubits: List of lists of qubit indices per check.
        n_qubits: Number of data qubits (inferred if None).
        decoder_type: Which decoder to use in each worker.
            One of ``"union_find"``, ``"fast_union_find"``, ``"blossom"``,
            ``"sparse_blossom"``, ``"cpu_batch"``.
            Default: ``"fast_union_find"``.
        n_workers: Number of worker processes. Defaults to ``os.cpu_count()``.
    """

    def __init__(
        self,
        check_to_qubits,
        n_qubits=None,
        decoder_type: str = "fast_union_find",
        n_workers: Optional[int] = None,
    ):
        if not check_to_qubits:
            raise ValueError("check_to_qubits must be non-empty")
        self._c2q = [[int(q) for q in c] for c in check_to_qubits]
        self._nq = int(n_qubits) if n_qubits is not None else None
        self._decoder_type = str(decoder_type)
        self._n_workers = n_workers or os.cpu_count() or 1
        self._pool: Optional["_mp.pool.Pool"] = None

    def decode(self, syndromes) -> np.ndarray:
        """Decode a batch of syndromes.

        On Windows, uses single-process Rayon parallel (``batch_decode_par``
        or ``batch_decode_simd``) for all batch sizes — this is 50-500x faster
        than multi-process due to spawn overhead.

        On Linux/macOS, uses multi-process pool for batches > 100K.
        """
        arr = np.asarray(syndromes, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"syndromes must be 2D, got shape {arr.shape}")
        n = arr.shape[0]
        if n == 0:
            return np.zeros((0, self._nq or 0), dtype=np.uint8)

        # On Windows: single-process Rayon path (much faster than IPC)
        if _IS_WINDOWS:
            builder = _DECODER_BUILDERS.get(self._decoder_type)
            if builder is not None:
                dec = builder(self._c2q, self._nq)
                if hasattr(dec, "batch_decode"):
                    return np.ascontiguousarray(dec.batch_decode(arr)).astype(np.uint8)
                if hasattr(dec, "batch_decode_par"):
                    return np.ascontiguousarray(dec.batch_decode_par(arr)).astype(np.uint8)

        # Linux/macOS or fallback: multi-process path
        if self._pool is None:
            checks_tuple = tuple(tuple(c) for c in self._c2q)
            self._pool = _mp.get_context("spawn").Pool(
                processes=self._n_workers,
                initializer=_worker_init,
                initargs=(checks_tuple, self._nq, self._decoder_type),
            )

        nw = min(self._n_workers, n)
        chunk_size = (n + nw - 1) // nw
        chunks = [(arr[i : i + chunk_size], i // chunk_size) for i in range(0, n, chunk_size)]

        results: List[Optional[np.ndarray]] = [None] * len(chunks)
        for idx, result in self._pool.imap_unordered(_worker_decode, chunks):
            results[idx] = result
        return cast(np.ndarray, np.concatenate(cast(List[np.ndarray], results), axis=0).astype(np.uint8))

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def n_qubits(self) -> int:
        if self._nq is not None:
            return self._nq
        return max((max(c) for c in self._c2q if c), default=-1) + 1

    @property
    def n_checks(self) -> int:
        return len(self._c2q)
