"""Tests for qector_decoder_v3.routing — intelligent decoder router.

Two things are checked:

1. **Policy** — ``recommend_decoder`` picks the documented decoder across code
   families, sizes, priorities, and hardware states.  Hardware is exercised both
   via the explicit ``hardware=`` override (deterministic on any machine) and via
   monkeypatching ``gpu_backend.has_cuda_rust`` / ``gpu_backend.gpu_available``
   (the live-detection path).
2. **Correctness** — ``AutoRouter`` actually constructs the chosen decoder and
   decodes real instances, and every returned correction satisfies the only
   validity bar that matters, ``H·c == s (mod 2)`` — including the safety
   guarantee that a non-graphlike (hyperedge) problem is *never* sent to a
   matching-only decoder, even when mislabelled.
"""

from __future__ import annotations

import numpy as np
import pytest
import qector_decoder_v3 as qd
from qector_decoder_v3 import codes, routing
from qector_decoder_v3.routing import (
    AutoRouter,
    DecoderName,
    HardwareProfile,
    recommend,
    recommend_decoder,
)

CPU = {"cuda_rust": False, "gpu": False}
CUPY_ONLY = {"cuda_rust": False, "gpu": True}
CUDA = {"cuda_rust": True, "gpu": True}


# ---------------------------------------------------------------------------
# Non-graphlike LDPC test codes (genuinely have qubits in > 2 checks)
# ---------------------------------------------------------------------------
def _bb_code():
    """The [[72, ...]] bivariate-bicycle X sector (max qubit degree 3)."""
    cx, _ = codes.bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])
    return cx


def _hgp_code():
    """Hypergraph product of a denser seed -> non-graphlike CSS code."""
    seed = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]], dtype=np.uint8)
    cx, _ = codes.hypergraph_product(seed)
    return cx


# ===========================================================================
# Policy: recommend_decoder
# ===========================================================================
def test_invalid_priority_raises():
    with pytest.raises(ValueError):
        recommend_decoder("surface", priority="fastest")
    with pytest.raises(ValueError):
        AutoRouter(priority="nope")


@pytest.mark.parametrize(
    "family,distance,n_qubits,batch,priority,hw,expected",
    [
        # --- LDPC / qLDPC / hypergraph: always BP-OSD, never MWPM -------------
        ("qldpc", None, None, 1, "balanced", CPU, DecoderName.BP_OSD),
        ("ldpc", None, None, 4096, "speed", CUDA, DecoderName.BP_OSD),
        ("bivariate_bicycle", 12, 144, 1, "accuracy", CUDA, DecoderName.BP_OSD),
        ("hypergraph_product", None, None, 256, "balanced", CPU, DecoderName.BP_OSD),
        ("color", 5, None, 1, "accuracy", CPU, DecoderName.BP_OSD),
        # --- matching, accuracy: exact Blossom (small) / Sparse (large) -------
        ("surface", 5, None, 1, "accuracy", CPU, DecoderName.BLOSSOM),
        ("rotated_surface", 7, 49, 1, "accuracy", CUDA, DecoderName.BLOSSOM),
        ("toric", 4, None, 1, "accuracy", CPU, DecoderName.BLOSSOM),
        ("surface", 21, None, 1, "accuracy", CPU, DecoderName.SPARSE_BLOSSOM),
        ("surface", None, 800, 1, "accuracy", CPU, DecoderName.SPARSE_BLOSSOM),
        # --- matching, speed: FastUnionFind, or CUDA for a huge batch ---------
        ("surface", 5, None, 1, "speed", CPU, DecoderName.FAST_UNION_FIND),
        ("repetition", 9, None, 1, "speed", CUDA, DecoderName.FAST_UNION_FIND),
        ("surface", 11, None, 8192, "speed", CUDA, DecoderName.CUDA_BATCH),
        ("surface", 11, None, 8192, "speed", CPU, DecoderName.FAST_UNION_FIND),
        # --- matching, balanced: batch-size heuristic -------------------------
        ("surface", 5, None, 1, "balanced", CPU, DecoderName.BLOSSOM),
        ("surface", 5, None, 64, "balanced", CPU, DecoderName.BATCH),
        ("surface", 5, None, 8192, "balanced", CUDA, DecoderName.CUDA_BATCH),
        ("surface", 5, None, 8192, "balanced", CUPY_ONLY, DecoderName.FAST_UNION_FIND),
        ("surface", 25, None, 1, "balanced", CPU, DecoderName.SPARSE_BLOSSOM),
    ],
)
def test_recommend_policy(family, distance, n_qubits, batch, priority, hw, expected):
    got = recommend_decoder(
        code_family=family,
        distance=distance,
        n_qubits=n_qubits,
        batch_size=batch,
        priority=priority,
        hardware=hw,
    )
    assert got == expected
    assert got in DecoderName.ALL


@pytest.mark.parametrize("family", ["ldpc", "qldpc", "bivariate_bicycle", "hypergraph_product", "color"])
@pytest.mark.parametrize("priority", ["balanced", "accuracy", "speed"])
@pytest.mark.parametrize("hw", [CPU, CUPY_ONLY, CUDA])
def test_ldpc_never_routed_to_matching(family, priority, hw):
    """A hyperedge family must never be handed to a matching-only decoder."""
    for batch in (1, 64, 8192):
        name = recommend_decoder(family, batch_size=batch, priority=priority, hardware=hw)
        assert name == DecoderName.BP_OSD
        assert name not in DecoderName.MATCHING_ONLY


def test_ldpc_gpu_batched_bp_flag():
    """BP-OSD advertises GPU batched BP only when a CuPy device + batch are present."""
    assert recommend("qldpc", batch_size=4096, hardware=CUDA).gpu_batched_bp is True
    assert recommend("qldpc", batch_size=4096, hardware=CUPY_ONLY).gpu_batched_bp is True
    assert recommend("qldpc", batch_size=4096, hardware=CPU).gpu_batched_bp is False
    # Single-shot does not use the batched GPU path.
    assert recommend("qldpc", batch_size=1, hardware=CUDA).gpu_batched_bp is False


def test_code_object_is_classified_structurally():
    """Passing a codes.Code routes by its real structure, not a guess."""
    surf = codes.rotated_surface_code(5)
    assert recommend(surf, priority="accuracy").family == "matching"
    assert recommend(surf, priority="accuracy").decoder == DecoderName.BLOSSOM

    bb = _bb_code()
    assert not bb.is_matching_graph()
    assert recommend(bb).family == "ldpc"
    assert recommend(bb).decoder == DecoderName.BP_OSD


# ---------------------------------------------------------------------------
# Hardware: monkeypatch the live-detection path (both states)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cuda_rust,gpu", [(False, False), (False, True), (True, True)])
def test_detected_hardware_drives_huge_batch(monkeypatch, cuda_rust, gpu):
    # Patch the canonical module by dotted path, NOT via `qd.gpu_backend`.
    # `HardwareProfile.detect()` reads `routing._gb.has_cuda_rust()`, where `_gb`
    # is `qector_decoder_v3.gpu_backend` bound from sys.modules. The package-level
    # attribute `qd.gpu_backend` is not reliably that same object - in CI it is
    # shadowed, so patching it left `detect()` reading the real hardware and these
    # two cases failed on any machine without a GPU while passing on one with.
    monkeypatch.setattr("qector_decoder_v3.gpu_backend.has_cuda_rust", lambda: cuda_rust)
    monkeypatch.setattr("qector_decoder_v3.gpu_backend.gpu_available", lambda: gpu)

    prof = HardwareProfile.detect()
    assert prof.cuda_rust is cuda_rust and prof.gpu is gpu

    # huge batch + speed: CUDA iff the Rust CUDA path was detected, else FastUF.
    name = recommend_decoder("surface", batch_size=8192, priority="speed", hardware=None)
    assert name == (DecoderName.CUDA_BATCH if cuda_rust else DecoderName.FAST_UNION_FIND)

    # LDPC batch advertises GPU batched BP iff a (CuPy) device was detected.
    assert recommend("qldpc", batch_size=512, hardware=None).gpu_batched_bp is gpu


def test_detected_hardware_small_problem_is_cpu(monkeypatch):
    """Hardware never changes the small/accuracy choice (it stays on CPU MWPM)."""
    monkeypatch.setattr(qd.gpu_backend, "has_cuda_rust", lambda: True)
    monkeypatch.setattr(qd.gpu_backend, "gpu_available", lambda: True)
    assert recommend_decoder("surface", distance=5, priority="accuracy") == DecoderName.BLOSSOM
    assert recommend_decoder("surface", distance=5, priority="speed") == DecoderName.FAST_UNION_FIND


# ===========================================================================
# Correctness: AutoRouter actually decodes (H·c == s)
# ===========================================================================
def _all_valid(out, H, syns) -> bool:
    return all(np.array_equal((H @ out[i]) & 1, syns[i]) for i in range(len(syns)))


def test_autorouter_matching_accuracy_single():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    ar = AutoRouter(priority="accuracy")
    rng = np.random.default_rng(7)
    for _ in range(60):
        e = (rng.random(code.n_qubits) < 0.08).astype(np.uint8)
        s = (H @ e) & 1
        corr = ar.decode(code, s.astype(np.uint8))
        assert corr.shape == (code.n_qubits,)
        assert np.array_equal((H @ corr) & 1, s)
    assert ar.last_recommendation.decoder == DecoderName.BLOSSOM


def test_autorouter_matching_speed_batch():
    code = codes.repetition_code(15)
    H = code.parity_check_matrix()
    ar = AutoRouter(priority="speed")
    rng = np.random.default_rng(11)
    errs = (rng.random((40, code.n_qubits)) < 0.1).astype(np.uint8)
    syns = (errs @ H.T) & 1
    out = ar.decode(code, syns.astype(np.uint8))
    assert out.shape == (40, code.n_qubits)
    assert _all_valid(out, H, syns)
    assert ar.last_recommendation.decoder == DecoderName.FAST_UNION_FIND


@pytest.mark.parametrize("make_code", [_bb_code, _hgp_code], ids=["bivariate_bicycle", "hypergraph_product"])
def test_autorouter_ldpc_bposd_single_and_batch(make_code):
    code = make_code()
    assert not code.is_matching_graph()  # genuinely a hyperedge problem
    H = code.parity_check_matrix()
    ar = AutoRouter()
    rng = np.random.default_rng(3)

    # single
    for _ in range(15):
        e = (rng.random(code.n_qubits) < 0.03).astype(np.uint8)
        s = (H @ e) & 1
        corr = ar.decode(code, s.astype(np.uint8))
        assert corr.shape == (code.n_qubits,)
        assert np.array_equal((H @ corr) & 1, s)
    assert ar.last_recommendation.decoder == DecoderName.BP_OSD
    assert ar.last_recommendation.family == "ldpc"

    # batch
    errs = (rng.random((10, code.n_qubits)) < 0.03).astype(np.uint8)
    syns = (errs @ H.T) & 1
    out = ar.decode(code, syns.astype(np.uint8))
    assert out.shape == (10, code.n_qubits)
    assert _all_valid(out, H, syns)


def test_autorouter_guard_mislabeled_hyperedge_code():
    """A hyperedge code mislabelled 'surface' must still decode validly (BP-OSD)."""
    code = _bb_code()
    H = code.parity_check_matrix()
    ar = AutoRouter(priority="speed")  # speed would pick FastUnionFind on a matching code
    rng = np.random.default_rng(5)
    for _ in range(15):
        e = (rng.random(code.n_qubits) < 0.03).astype(np.uint8)
        s = (H @ e) & 1
        corr = ar.decode(code, s.astype(np.uint8), code_family="surface")
        assert np.array_equal((H @ corr) & 1, s)
    rec = ar.last_recommendation
    assert rec.decoder == DecoderName.BP_OSD
    assert rec.decoder not in DecoderName.MATCHING_ONLY


def test_autorouter_accepts_matrix_and_check_list_forms():
    """Problem may be a Code, a dense H matrix, or a check_to_qubits list."""
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    rng = np.random.default_rng(1)
    e = (rng.random(code.n_qubits) < 0.08).astype(np.uint8)
    s = ((H @ e) & 1).astype(np.uint8)

    ar = AutoRouter(priority="accuracy")
    # dense H matrix
    c_mat = ar.decode(H, s)
    assert np.array_equal((H @ c_mat) & 1, s)
    # check_to_qubits list (+ explicit n_qubits)
    c_list = ar.decode(code.check_to_qubits, s, n_qubits=code.n_qubits)
    assert np.array_equal((H @ c_list) & 1, s)


def test_explain_returns_choice_and_reason():
    code = codes.rotated_surface_code(5)
    ar = AutoRouter(priority="accuracy")
    info = ar.explain(code, np.zeros(code.n_checks, np.uint8))
    assert info["decoder"] == DecoderName.BLOSSOM
    assert info["family"] == "matching"
    assert isinstance(info["reason"], str) and info["reason"]
    assert info["graphlike"] is True
    for key in ("decoder", "reason", "family", "priority", "batch_size", "hardware", "gpu_batched_bp"):
        assert key in info

    # metadata-only explain (no problem) still works
    meta = ar.explain(code_family="qldpc", batch_size=4096)
    assert meta["decoder"] == DecoderName.BP_OSD


def test_explain_matches_decode_choice_for_ldpc():
    code = _bb_code()
    ar = AutoRouter()
    s = np.zeros(code.n_checks, np.uint8)
    chosen = ar.explain(code, s)["decoder"]
    ar.decode(code, s)
    assert ar.last_recommendation.decoder == chosen == DecoderName.BP_OSD


def test_invariant_no_nongraphlike_to_matching_decoder():
    """The router's core safety invariant, exercised end-to-end on real codes."""
    for code in (_bb_code(), _hgp_code()):
        s = np.zeros(code.n_checks, np.uint8)
        for priority in ("balanced", "accuracy", "speed"):
            for hw in (CPU, CUPY_ONLY, CUDA):
                ar = AutoRouter(priority=priority, hardware=hw)
                name = ar.explain(code, s)["decoder"]
                assert name == DecoderName.BP_OSD
                assert name not in DecoderName.MATCHING_ONLY
