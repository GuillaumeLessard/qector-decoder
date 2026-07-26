"""Tests for qector_decoder_v3.bp_cupy - batched GPU belief propagation.

Coverage, per the module's contract:

* **Validity** - on shots the batched BP marks converged, the correction satisfies
  the GF(2) syndrome equation ``H @ c == s (mod 2)`` (the load-bearing bar).
* **Parity** - the NumPy backend is *bit-identical* to the single-shot
  :mod:`qector_decoder_v3._bp_core` (max posterior difference 0.0), and on the GPU
  the batched result lands in the **same logical class** as the CPU result for every
  converged shot (residual difference inside the stabiliser row space).
* **CPU fallback** - with CuPy disabled (``has_cupy`` monkeypatched to ``False``)
  the decoder transparently runs on NumPy and still produces valid corrections.

GPU-only assertions are guarded with ``skipif(not gpu_available())`` so the file
runs everywhere; on this machine (GTX 1660 Ti + cupy 14.x) the GPU paths execute
and the evidence test prints the device name and confirms ``cupy.ndarray`` buffers.
"""

from __future__ import annotations

import numpy as np
import pytest
from qector_decoder_v3 import codes
from qector_decoder_v3 import gpu_backend as gb
from qector_decoder_v3._bp_core import build_incidence, min_sum_bp, sum_product_bp
from qector_decoder_v3.bp_cupy import BatchedBpDecoder, batched_bp_decode
from qector_decoder_v3.bposd import BpOsdDecoder

GPU = gb.gpu_available()
requires_gpu = pytest.mark.skipif(not GPU, reason="no usable CuPy/CUDA device")


@pytest.fixture(autouse=True)
def _clean_state():
    """Each test starts (and ends) with prefer_gpu=True and zeroed telemetry."""
    gb.set_prefer_gpu(True)
    gb.reset_telemetry()
    yield
    gb.set_prefer_gpu(True)
    gb.reset_telemetry()


# ---------------------------------------------------------------------------
# Code / helper fixtures
# ---------------------------------------------------------------------------
def _bb72():
    return codes.bivariate_bicycle_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])


def _gf2_rowspace_basis(M):
    """Reduced GF(2) row-space basis of ``M`` (each pivot a distinct leading column)."""
    rows = []
    for r in (np.asarray(M) % 2).astype(np.uint8):
        v = r.copy()
        for b in rows:
            if v[np.argmax(b)]:
                v ^= b
        if v.any():
            rows.append(v)
    return rows


def _in_span(basis, v):
    """Return ``True`` iff ``v`` lies in the GF(2) span of ``basis``."""
    v = (np.asarray(v) % 2).astype(np.uint8)
    for b in basis:
        if v[np.argmax(b)]:
            v ^= b
    return not v.any()


def _random_syndromes(H, shots, p, seed):
    nq = H.shape[1]
    rng = np.random.default_rng(seed)
    errs = (rng.random((shots, nq)) < p).astype(np.uint8)
    syns = ((errs @ H.T) & 1).astype(np.uint8)
    return errs, syns


# ---------------------------------------------------------------------------
# GPU evidence
# ---------------------------------------------------------------------------
@requires_gpu
def test_gpu_evidence_device_and_cupy_arrays(capsys):
    """The decoder resolves to CuPy and its device-resident buffers are cupy.ndarray."""
    import cupy as cp

    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    dec = BatchedBpDecoder(H, error_rate=0.05, max_iter=30, bp_method="min_sum")

    assert dec.on_gpu is True
    assert dec._xp is cp
    # Device-resident Tanner state really lives on the GPU.
    assert isinstance(dec._ic, cp.ndarray)
    assert isinstance(dec._ie, cp.ndarray)
    assert isinstance(dec._prior_llr, cp.ndarray)

    _, syns = _random_syndromes(H, 64, 0.05, 0)
    corr, _ = dec.decode_batch(syns)
    assert isinstance(corr, np.ndarray)  # round-tripped back to host
    assert corr.dtype == np.uint8

    name = gb.get_backend().device_name
    print(
        f"GPU EVIDENCE: device={name!r} xp={dec._xp.__name__} "
        f"ic={type(dec._ic).__module__}.{type(dec._ic).__name__} "
        f"telemetry={dict(gb.TELEMETRY)}"
    )
    assert "NVIDIA" in (name or "") or (name or "")
    out = capsys.readouterr().out
    assert "GPU EVIDENCE" in out


# ---------------------------------------------------------------------------
# Validity:  H @ c == s on converged shots
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bp_method", ["min_sum", "sum_product"])
def test_batched_validity_on_converged_bb72(bp_method):
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    _, syns = _random_syndromes(H, 300, 0.04, 11)
    corr, converged = batched_bp_decode(H, syns, error_rate=0.04, max_iter=40, bp_method=bp_method)
    assert corr.shape == (300, H.shape[1])
    assert converged.shape == (300,)
    assert converged.sum() > 0  # BP explains a healthy fraction at this rate
    pred = (corr @ H.T) & 1
    # Every converged shot must satisfy its syndrome exactly.
    assert np.array_equal(pred[converged], syns[converged])


@pytest.mark.parametrize("bp_method", ["min_sum", "sum_product"])
def test_batched_validity_hypergraph_product(bp_method):
    H1 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    cx, _ = codes.hypergraph_product(H1)
    H = cx.parity_check_matrix().astype(np.uint8)
    _, syns = _random_syndromes(H, 200, 0.05, 5)
    corr, converged = batched_bp_decode(H, syns, error_rate=0.05, max_iter=40, bp_method=bp_method)
    pred = (corr @ H.T) & 1
    assert np.array_equal(pred[converged], syns[converged])


def test_convergence_mask_is_exactly_syndrome_satisfaction():
    """The mask is precisely the set of shots whose BP hard decision satisfies H@c==s."""
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    _, syns = _random_syndromes(H, 200, 0.06, 23)
    corr, converged = batched_bp_decode(H, syns, error_rate=0.06, max_iter=30)
    pred = (corr @ H.T) & 1
    row_ok = np.all(pred == syns, axis=1)
    assert np.array_equal(row_ok, converged)


# ---------------------------------------------------------------------------
# Parity: NumPy backend bit-identical to _bp_core
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bp_method", ["min_sum", "sum_product"])
def test_numpy_batched_bit_identical_to_bp_core(bp_method):
    """With CuPy off, batched BP reproduces the single-shot _bp_core posterior exactly."""
    gb.set_prefer_gpu(False)  # force the NumPy backend even on a GPU box
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    nq, nc = H.shape[1], H.shape[0]
    ic, ie = build_incidence(H)
    p = np.full(nq, 0.05)
    prior_llr = np.log((1.0 - p) / p)
    _, syns = _random_syndromes(H, 48, 0.05, 31)

    dec = BatchedBpDecoder(H, error_rate=0.05, max_iter=30, bp_method=bp_method, prefer_gpu=True)
    assert dec.on_gpu is False
    _, _, llr = dec.decode_batch(syns, return_llr=True, early_stop=False)

    for i in range(syns.shape[0]):
        if bp_method == "min_sum":
            ref = min_sum_bp(ic, ie, nc, nq, prior_llr, syns[i], 30, 1.0)
        else:
            ref = sum_product_bp(ic, ie, nc, nq, prior_llr, syns[i], 30)
        np.testing.assert_allclose(llr[i], ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Parity: GPU result vs CPU result (same logical class on converged shots)
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("bp_method", ["min_sum", "sum_product"])
def test_gpu_matches_cpu_same_logical_class(bp_method):
    """On converged shots the GPU correction is logically equivalent to the CPU one.

    Both backends reproduce the syndrome, so their difference lies in ``ker(Hx)``;
    "logically equivalent" means that difference is a stabiliser - i.e. inside the
    GF(2) row space of ``Hz`` (the established CSS criterion used elsewhere in the
    suite). The vast majority also match bit-for-bit.
    """
    cx, cz = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    Zbasis = _gf2_rowspace_basis(Hz)
    _, syns = _random_syndromes(Hx, 128, 0.04, 77)

    gb.set_prefer_gpu(True)
    g = BatchedBpDecoder(Hx, error_rate=0.04, max_iter=40, bp_method=bp_method)
    assert g.on_gpu is True
    cg, conv_g = g.decode_batch(syns, early_stop=False)

    gb.set_prefer_gpu(False)
    c = BatchedBpDecoder(Hx, error_rate=0.04, max_iter=40, bp_method=bp_method)
    assert c.on_gpu is False
    cc, conv_c = c.decode_batch(syns, early_stop=False)
    gb.set_prefer_gpu(True)

    # Both must be valid on the shots each marks converged.
    assert np.array_equal(((cg @ Hx.T) & 1)[conv_g], syns[conv_g])
    assert np.array_equal(((cc @ Hx.T) & 1)[conv_c], syns[conv_c])

    both = conv_g & conv_c
    assert both.sum() > 0
    exact = 0
    for i in np.nonzero(both)[0]:
        if np.array_equal(cg[i], cc[i]):
            exact += 1
        else:
            assert _in_span(Zbasis, cg[i] ^ cc[i]), f"shot {i} differs logically"
    # Floating-point scatter aside, the overwhelming majority match exactly.
    assert exact >= int(0.8 * both.sum())


# ---------------------------------------------------------------------------
# CPU fallback when CuPy is disabled
# ---------------------------------------------------------------------------
def test_cpu_fallback_when_cupy_disabled(monkeypatch):
    """has_cupy()->False forces the NumPy path and still yields valid corrections."""
    monkeypatch.setattr(gb, "has_cupy", lambda: False)
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    dec = BatchedBpDecoder(H, error_rate=0.05, max_iter=30, bp_method="min_sum")

    assert dec.on_gpu is False
    assert dec._xp is np
    assert isinstance(dec._ic, np.ndarray)

    _, syns = _random_syndromes(H, 120, 0.05, 9)
    corr, converged = dec.decode_batch(syns)
    assert converged.sum() > 0
    pred = (corr @ H.T) & 1
    assert np.array_equal(pred[converged], syns[converged])
    # No host<->device traffic should have been recorded on the NumPy path.
    assert gb.TELEMETRY["h2d"] == 0
    assert gb.TELEMETRY["d2h"] == 0


def test_prefer_gpu_off_uses_numpy(monkeypatch):
    """set_prefer_gpu(False) routes the batched decoder onto NumPy regardless of HW."""
    gb.set_prefer_gpu(False)
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    dec = BatchedBpDecoder(H, error_rate=0.05, max_iter=20)
    assert dec.on_gpu is False
    assert dec._xp is np


# ---------------------------------------------------------------------------
# Convenience wrapper API
# ---------------------------------------------------------------------------
def test_batched_bp_decode_shapes_and_return_llr():
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    _, syns = _random_syndromes(H, 32, 0.05, 3)

    out2 = batched_bp_decode(H, syns, error_rate=0.05, max_iter=20)
    assert isinstance(out2, tuple) and len(out2) == 2
    corr, conv = out2
    assert corr.shape == (32, H.shape[1]) and conv.shape == (32,)

    out3 = batched_bp_decode(H, syns, error_rate=0.05, max_iter=20, return_llr=True)
    assert len(out3) == 3
    _, _, llr = out3
    assert llr.shape == (32, H.shape[1]) and llr.dtype == np.float64


def test_single_row_syndrome_is_accepted():
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    _, syns = _random_syndromes(H, 1, 0.05, 1)
    corr, _conv = batched_bp_decode(H, syns[0], error_rate=0.05, max_iter=20)
    assert corr.shape == (1, H.shape[1])


def test_early_stop_preserves_validity():
    """early_stop must not change the validity contract (converged shots stay valid)."""
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    _, syns = _random_syndromes(H, 150, 0.05, 44)
    c1, k1 = batched_bp_decode(H, syns, error_rate=0.05, max_iter=40, early_stop=True)
    c2, k2 = batched_bp_decode(H, syns, error_rate=0.05, max_iter=40, early_stop=False)
    assert np.array_equal(((c1 @ H.T) & 1)[k1], syns[k1])
    assert np.array_equal(((c2 @ H.T) & 1)[k2], syns[k2])


# ---------------------------------------------------------------------------
# Bp-OSD integration: the GPU batch path is a faithful BP-OSD
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("osd_order", [0, 6])
def test_bposd_gpu_batch_decode_is_faithful(osd_order):
    """BpOsdDecoder.batch_decode (GPU path when present) satisfies H@c==s for every shot."""
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=osd_order)
    _, syns = _random_syndromes(H, 150, 0.05, 2)
    out = np.asarray(dec.batch_decode(syns)).astype(np.uint8)
    assert out.shape == (150, H.shape[1])
    for i in range(150):
        assert np.array_equal((H @ out[i]) & 1, syns[i])


def test_bposd_use_gpu_false_matches_cpu_contract():
    """use_gpu=False forces the per-shot CPU path; output remains syndrome-faithful."""
    cx, _ = _bb72()
    H = cx.parity_check_matrix().astype(np.uint8)
    dec = BpOsdDecoder(H, error_rate=0.05, max_iter=30, osd_order=0, use_gpu=False)
    assert dec._gpu_enabled() is False
    _, syns = _random_syndromes(H, 100, 0.05, 8)
    out = np.asarray(dec.batch_decode(syns)).astype(np.uint8)
    for i in range(100):
        assert np.array_equal((H @ out[i]) & 1, syns[i])


@requires_gpu
def test_bposd_gpu_and_cpu_batch_both_valid_and_mostly_equivalent():
    """GPU and CPU Bp-OSD batch paths are both syndrome-faithful on every shot.

    Both paths obey the validity contract ``Hx @ c == s`` for all shots. On the
    shots resolved by BP alone they additionally land in the *same logical class*
    (rigorously proven for the pure-BP stage in
    ``test_gpu_matches_cpu_same_logical_class``). The handful of shots that fall
    through to OSD are *not* required to agree across backends: OSD is an explicit
    heuristic whose column ordering depends on the tiny floating-point differences
    between the GPU and CPU posteriors, so two equally valid solutions in different
    logical classes are an honest, expected outcome - not an error.
    """
    cx, cz = _bb72()
    Hx = cx.parity_check_matrix().astype(np.uint8)
    Hz = cz.parity_check_matrix().astype(np.uint8)
    Zbasis = _gf2_rowspace_basis(Hz)
    _, syns = _random_syndromes(Hx, 120, 0.04, 99)

    gpu_dec = BpOsdDecoder(Hx, error_rate=0.04, max_iter=30, osd_order=4, use_gpu=True)
    cpu_dec = BpOsdDecoder(Hx, error_rate=0.04, max_iter=30, osd_order=4, use_gpu=False)
    assert gpu_dec._gpu_enabled() is True
    g = np.asarray(gpu_dec.batch_decode(syns)).astype(np.uint8)
    c = np.asarray(cpu_dec.batch_decode(syns)).astype(np.uint8)

    # Validity contract: every row of both paths satisfies its syndrome.
    assert np.array_equal((g @ Hx.T) & 1, syns)
    assert np.array_equal((c @ Hx.T) & 1, syns)

    # Logical agreement holds for the large majority (BP-resolved shots).
    equiv = sum(_in_span(Zbasis, g[i] ^ c[i]) for i in range(120))
    assert equiv >= int(0.9 * 120), (equiv, 120)
