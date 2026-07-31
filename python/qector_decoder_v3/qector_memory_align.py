"""
QECTOR — buffer conditioning for the PyO3 boundary.

Flattens 2D syndrome matrices into C-contiguous 1D ``uint8`` slices so the Rust
backend reads straight from the raw buffer, optionally forcing a 64-byte start
address.

Usage
-----
    from qector_decoder_v3.qector_memory_align import prepare_syndromes

    syndromes = sampler.sample(shots=1000).astype(np.uint8)   # (shots, detectors)
    flat = prepare_syndromes(syndromes)                       # 1-D, no copy
    # flat.ndim == 1,  flat.shape == (shots * detectors,)

What each guarantee is actually worth
-------------------------------------
Be precise about which of these are load-bearing, because one of them costs a
full pass over the data:

* **C-contiguity — required.** ``PyReadonlyArray*::as_slice()`` on the Rust side
  *fails* on a non-contiguous array. A transposed, strided, or column view has
  to be repacked, and repacking is a copy. Doing it once here beats letting
  every endpoint trip over it.
* **``uint8`` dtype — required.** A ``bool`` or ``int64`` array has a different
  element width and is a type error at a ``uint8`` endpoint. ``bool`` → ``uint8``
  is a free reinterpretation (``.view()``), not a cast.
* **Flat 1-D shape — free.** ``ravel()`` on a contiguous array returns a view, so
  a caller can hand ``(shots, detectors)`` data to a flat endpoint with an
  explicit stride at no cost.
* **64-byte start alignment — cosmetic, and it costs a full copy.** This is the
  claim to be sceptical of. Unaligned AVX-512 loads (``vmovdqu*``) run at the
  same throughput as aligned ones on every AVX-512 part, and LLVM emits the
  unaligned forms for ``u8`` slices regardless of the pointer. Alignment only
  avoids the occasional cache-line split — low single-digit percent on a
  memory-bound kernel. Forcing it means over-allocating and copying everything,
  which on a decode workload usually costs more than it saves.

Therefore: :func:`prepare_syndromes` is the hot-path entry point and never
aligns. :func:`align_and_flatten_syndromes` keeps the alignment behaviour for
callers that want it; pass ``force_alignment=False`` to skip the copy.

NumPy already returns 64-byte-aligned buffers for most fresh allocations of any
size, so the copy branch is usually dead anyway — check with
:func:`buffer_report` before paying for it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SIMD_ALIGN",
    "aligned_stride",
    "alignment_of",
    "is_simd_aligned",
    "prepare_syndromes",
    "align_and_flatten_syndromes",
    "buffer_report",
    "native_buffer_geometry",
    "aligned_syndrome_batch",
]

#: Cache-line / AVX-512 register width in bytes.
SIMD_ALIGN = 64


def _check_pow2(alignment: int) -> None:
    if alignment <= 0 or alignment & (alignment - 1) != 0:
        raise ValueError(f"alignment must be a power of two, got {alignment}")


def _data_address(a: np.ndarray) -> int:
    """Start address of ``a``'s buffer."""
    return a.__array_interface__["data"][0]


def alignment_of(a: np.ndarray, alignment: int = SIMD_ALIGN) -> int:
    """Widest power-of-two alignment of ``a``'s start address, capped at ``alignment``."""
    _check_pow2(alignment)
    addr = _data_address(a)
    if addr == 0:
        return alignment
    got = 1
    while got < alignment and addr % (got * 2) == 0:
        got *= 2
    return got


def is_simd_aligned(a: np.ndarray, alignment: int = SIMD_ALIGN) -> bool:
    """True iff ``a`` starts on an ``alignment``-byte boundary."""
    _check_pow2(alignment)
    return _data_address(a) % alignment == 0


def aligned_stride(n_checks: int, alignment: int = SIMD_ALIGN) -> int:
    """
    Return the padded stride (bytes per shot) needed so each shot starts on an
    ``alignment``-byte boundary when the meaningful data is ``n_checks`` bytes.

    A stride that is *not* a multiple of ``alignment`` means only every
    ``alignment``-th row can start aligned, whatever the base pointer is — so
    this is the half of the alignment story that is actually worth enforcing.

    Example::

        stride = aligned_stride(24)        # -> 64
        stride = aligned_stride(100)       # -> 128
        buf = np.zeros(num_shots * stride, dtype=np.uint8)
        for i, row in enumerate(syndromes):
            buf[i * stride : i * stride + n_checks] = row
    """
    _check_pow2(alignment)
    if n_checks < 0:
        raise ValueError(f"n_checks must be non-negative, got {n_checks}")
    return ((n_checks + alignment - 1) // alignment) * alignment


def _as_uint8(a: np.ndarray) -> np.ndarray:
    """Reinterpret/cast ``a`` to ``uint8``, copying only when the width differs."""
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.bool_:
        # Same element width: a reinterpretation, not a cast.
        return a.view(np.uint8)
    return a.astype(np.uint8, copy=False)


def prepare_syndromes(syndromes, *, flatten: bool = True) -> np.ndarray:
    """
    Return ``syndromes`` as a C-contiguous ``uint8`` array, copying only if needed.

    This is the hot-path entry point: it establishes exactly the two properties
    the native boundary requires (contiguity, dtype) and nothing that costs a
    gratuitous copy.

    Parameters
    ----------
    syndromes:
        Anything array-like; 1-D or 2-D ``(shots, detectors)``.
    flatten:
        If true (default) return a 1-D view. Free on a contiguous array —
        contiguity is established first so the ``reshape`` cannot copy.

    Returns
    -------
    numpy.ndarray
        C-contiguous ``uint8``. **May be a view of the input**, so treat it as
        read-only unless you know you own the buffer.
    """
    a = _as_uint8(np.asarray(syndromes))
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    if flatten and a.ndim != 1:
        a = a.reshape(-1)  # a view: `a` is contiguous by this point
    return a


def align_and_flatten_syndromes(
    syndromes: np.ndarray,
    target_dtype: np.dtype = np.uint8,
    alignment: int = SIMD_ALIGN,
    *,
    force_alignment: bool = True,
) -> np.ndarray:
    """
    Transform a 2D syndrome matrix ``(shots, detectors)`` into a C-contiguous 1D
    ``uint8`` array, by default forced onto an ``alignment``-byte boundary.

    Passing over the PyO3 boundary as ``PyReadonlyArray1<u8>`` is zero-copy for
    any result of this function, aligned or not — contiguity is what makes the
    slice borrow possible, alignment is not required for it.

    Parameters
    ----------
    force_alignment:
        When true (default, preserving this function's original behaviour) a
        misaligned buffer is copied into an over-allocated block sliced at the
        next boundary. That is a full pass over the data and usually costs more
        than the cache-line splits it removes — set false (or call
        :func:`prepare_syndromes`) on the hot path.

    Notes
    -----
    The aligned result keeps its over-allocated backing buffer alive through
    ``.base``, so the pointer stays valid as long as you hold the returned
    array. It is a *snapshot*: later writes to the input are not reflected.
    """
    _check_pow2(alignment)
    if target_dtype != np.uint8:
        # The native endpoints are uint8-only; anything else would be silently
        # misread as bytes on the Rust side.
        raise ValueError(
            f"target_dtype must be np.uint8 for the native boundary, got {target_dtype}"
        )

    flat = prepare_syndromes(syndromes, flatten=True)
    if not force_alignment or is_simd_aligned(flat, alignment):
        return flat

    nbytes = flat.nbytes
    buf = np.empty(nbytes + alignment, dtype=np.uint8)
    shift = (-_data_address(buf)) % alignment
    aligned = buf[shift : shift + nbytes]
    np.copyto(aligned, flat)
    return aligned


def _native():
    """The compiled core, or ``None`` when this build has no extension module."""
    try:
        from . import qector_decoder_v3 as _core  # type: ignore[attr-defined]

        return _core
    except Exception:  # pragma: no cover - defensive
        return None


def native_buffer_geometry(a: np.ndarray) -> dict:
    """Report the geometry of ``a`` **as the Rust boundary actually sees it**.

    :func:`buffer_report` is a Python reimplementation of the same measurement;
    this calls the compiled `syndrome_buffer_geometry` directly, which is the
    only way to confirm the two layers agree rather than assuming they do.

    Raises ``RuntimeError`` when the build has no compiled core.
    """
    core = _native()
    fn = getattr(core, "syndrome_buffer_geometry", None) if core is not None else None
    if fn is None:
        raise RuntimeError(
            "native_buffer_geometry requires the compiled core; use buffer_report() instead."
        )
    arr = np.ascontiguousarray(np.asarray(a).reshape(-1), dtype=np.uint8)
    return fn(arr)


def aligned_syndrome_batch(shots: int, n_checks: int) -> np.ndarray:
    """Allocate a zeroed ``(shots, padded_checks)`` uint8 batch from the core.

    The row stride is padded to a multiple of :data:`SIMD_ALIGN`, so every row
    starts aligned rather than only every 64th one. Falls back to an equivalent
    NumPy allocation when the build has no compiled core, so callers do not need
    to branch on it.
    """
    shots = int(shots)
    n_checks = int(n_checks)
    if shots < 0 or n_checks < 0:
        raise ValueError(f"shots and n_checks must be non-negative, got {shots}, {n_checks}")
    core = _native()
    fn = getattr(core, "aligned_syndrome_buffer", None) if core is not None else None
    if fn is not None:
        return fn(shots, n_checks)
    return np.zeros((shots, aligned_stride(n_checks)), dtype=np.uint8)


def buffer_report(a: np.ndarray) -> dict:
    """
    Diagnostics for ``a`` as the native boundary sees it.

    Mirrors the Rust-side ``syndrome_buffer_geometry`` so both layers can be
    compared directly when tuning -- see :func:`native_buffer_geometry` for the
    compiled core's own answer. Use this to check whether the alignment copy in
    :func:`align_and_flatten_syndromes` would do anything at all.
    """
    arr = np.asarray(a)
    return {
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "c_contiguous": bool(arr.flags["C_CONTIGUOUS"]),
        "writeable": bool(arr.flags["WRITEABLE"]),
        "nbytes": int(arr.nbytes),
        "alignment": alignment_of(arr),
        "simd_aligned": is_simd_aligned(arr),
        "simd_align": SIMD_ALIGN,
        # A row stride that is not a multiple of 64 means only every 64th row can
        # start aligned, whatever the base pointer is.
        "row_stride_aligned": (
            bool(arr.shape[-1] % SIMD_ALIGN == 0) if arr.ndim >= 2 else None
        ),
    }
