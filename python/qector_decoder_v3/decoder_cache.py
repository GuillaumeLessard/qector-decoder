"""
qector_decoder_v3.decoder_cache — LRU-decoder cache and factory functions.

Caches decoder instances by ``(checks_tuple, n_qubits, decoder_type)`` so
repeated construction of identical decoders is free after the first call.
Includes a disk-backed cache for :class:`LookupTableDecoder` precomputed tables.

Example
-------
>>> from qector_decoder_v3 import get_decoder, generate_repetition_code_checks
>>> checks, nq = generate_repetition_code_checks(distance=5)
>>> dec = get_decoder(tuple(map(tuple, checks)), nq, "union_find")
>>> dec2 = get_decoder(tuple(map(tuple, checks)), nq, "union_find")  # cache hit
"""

from __future__ import annotations

import functools as _functools
import hashlib as _hashlib
import json as _json
import os as _os
from typing import Optional, Tuple

import numpy as _np

__all__ = ["get_decoder", "clear_decoder_cache", "get_decoder_pool"]


def _normalize_decoder_name(name: str) -> str:
    """Normalize decoder name aliases to canonical form."""
    mapping = {
        "uf": "union_find",
        "unionfind": "union_find",
        "fast_uf": "fast_union_find",
        "fastuf": "fast_union_find",
        "fastunionfind": "fast_union_find",
        "sparse": "sparse_blossom",
        "batch": "cpu_batch",
        "cpu": "cpu_batch",
    }
    return mapping.get(name.lower().replace("-", "_").replace(" ", "_"), name)


@_functools.lru_cache(maxsize=256)
def _build_decoder(checks_tuple: Tuple[Tuple[int, ...]], n_qubits: int, decoder_type: str):
    """Internal LRU-cached decoder factory."""
    checks = [list(c) for c in checks_tuple]
    from . import BlossomDecoder, CPUBatchDecoder, FastUnionFindDecoder, SparseBlossomDecoder, UnionFindDecoder

    builders = {
        "union_find": lambda: UnionFindDecoder(checks, n_qubits),
        "fast_union_find": lambda: FastUnionFindDecoder(checks, n_qubits),
        "blossom": lambda: BlossomDecoder(checks, n_qubits),
        "sparse_blossom": lambda: SparseBlossomDecoder(checks, n_qubits),
        "cpu_batch": lambda: CPUBatchDecoder(checks, n_qubits),
    }
    decoder_type = _normalize_decoder_name(decoder_type)
    if decoder_type not in builders:
        raise ValueError(f"unknown decoder type '{decoder_type}'; choose from {list(builders)}")
    return builders[decoder_type]()


def get_decoder(
    checks_tuple: Tuple[Tuple[int, ...]],
    n_qubits: int,
    decoder_type: str = "union_find",
):
    """Get a decoder, constructing it once and caching for reuse.

    Args:
        checks_tuple: Ideally ``tuple(tuple(c) for c in check_to_qubits)``,
            but a plain list of lists (the format returned directly by
            ``generate_repetition_code_checks`` etc.) is also accepted — it
            is normalized to a canonical hashable tuple here before the
            cache lookup, so the LRU cache still hits on repeated calls with
            equal values either way. Passing a tuple directly on a hot path
            avoids the (cheap) per-call normalization.
        n_qubits: Number of data qubits.
        decoder_type: Which decoder to create.

    Returns:
        A decoder instance with ``decode(syndrome)`` method.
    """
    hashable = tuple(tuple(int(q) for q in c) for c in checks_tuple)
    return _build_decoder(hashable, n_qubits, decoder_type)


@_functools.lru_cache(maxsize=32)
def _build_decoder_pool(
    checks_tuple: Tuple[Tuple[int, ...]],
    n_qubits: int,
    decoder_type: str,
    n_workers: Optional[int],
):
    from .decoder_pool import DecoderPool

    checks = [list(c) for c in checks_tuple]
    return DecoderPool(
        checks,
        n_qubits=n_qubits,
        decoder_type=_normalize_decoder_name(decoder_type),
        n_workers=n_workers,
    )


def get_decoder_pool(
    checks_tuple: Tuple[Tuple[int, ...]],
    n_qubits: int,
    decoder_type: str = "union_find",
    n_workers: Optional[int] = None,
):
    """Get a multi-process :class:`DecoderPool`, constructed once and cached.

    Args:
        checks_tuple: ``tuple(tuple(c) for c in check_to_qubits)`` — hashable.
        n_qubits: Number of data qubits.
        decoder_type: Which decoder to use in each worker.
        n_workers: Number of worker processes (default: ``os.cpu_count()``).

    Returns:
        A :class:`DecoderPool` instance.
    """
    hashable = tuple(tuple(int(q) for q in c) for c in checks_tuple)
    return _build_decoder_pool(hashable, n_qubits, decoder_type, n_workers)


def clear_decoder_cache():
    """Clear the in-memory decoder and decoder pool LRU caches."""
    _build_decoder.cache_clear()
    _build_decoder_pool.cache_clear()


# ---------------------------------------------------------------------------
# Lookup-table disk cache
# ---------------------------------------------------------------------------
_QECTOR_CACHE_DIR = _os.path.join(_os.path.expanduser("~"), ".qector_cache")


def _cache_key(checks_tuple, n_qubits, base_decoder: str) -> str:
    """Deterministic cache key for a lookup-table."""
    raw = _json.dumps({"checks": checks_tuple, "nq": n_qubits, "decoder": base_decoder}, sort_keys=True)
    return _hashlib.sha256(raw.encode()).hexdigest()[:16]


def _lut_cache_path(key: str) -> str:
    return _os.path.join(_QECTOR_CACHE_DIR, f"lut_{key}.npy")


def load_lut_from_cache(checks_tuple, n_qubits, base_decoder: str):
    """Load a precomputed LUT table from disk cache, or return None."""
    key = _cache_key(checks_tuple, n_qubits, base_decoder)
    path = _lut_cache_path(key)
    if _os.path.exists(path):
        return _np.load(path)
    return None


def save_lut_to_cache(table: _np.ndarray, checks_tuple, n_qubits, base_decoder: str):
    """Save a precomputed LUT table to disk cache."""
    _os.makedirs(_QECTOR_CACHE_DIR, exist_ok=True)
    key = _cache_key(checks_tuple, n_qubits, base_decoder)
    path = _lut_cache_path(key)
    _np.save(path, table)
