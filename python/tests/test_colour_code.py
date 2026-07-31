"""C1-04: colour-code decoder correctness.

The first cut of `colour_code.py` built all-zero observable matrices, so
`decode()` returned the all-zero prediction for every input — indistinguishable
from "always guess no logical flip". These tests pin the properties that
failure violated: the decoder must actually depend on its input, must beat the
trivial always-zero predictor, and must reproduce the documented shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from qector_decoder_v3.colour_code import ColourCodeDecoder, colour_codes_from_dem


def _circuit(d: int, p: float = 0.003):
    return stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


@pytest.fixture(scope="module")
def d3():
    c = _circuit(3)
    return c, ColourCodeDecoder.from_stim_circuit(c)


# ---------------------------------------------------------------------------
# Shapes and construction
# ---------------------------------------------------------------------------
def test_shapes_match_the_dem(d3):
    circuit, dec = d3
    dem = circuit.detector_error_model(decompose_errors=False)
    assert dec.num_detectors == dem.num_detectors
    assert dec.num_observables == dem.num_observables
    assert dec.num_mechanisms > 0


def test_decode_batch_shape(d3):
    _, dec = d3
    syn = np.zeros((7, dec.num_detectors), dtype=np.uint8)
    out = dec.decode_batch(syn)
    assert out.shape == (7, dec.num_observables)
    assert out.dtype == np.uint8


def test_rejects_oversized_syndrome(d3):
    _, dec = d3
    with pytest.raises(ValueError):
        dec.decode(np.zeros(dec.num_detectors + 5, dtype=np.uint8))
    with pytest.raises(ValueError):
        dec.decode_batch(np.zeros((3, dec.num_detectors + 5), dtype=np.uint8))


def test_rejects_non_2d_batch(d3):
    _, dec = d3
    with pytest.raises(ValueError):
        dec.decode_batch(np.zeros(dec.num_detectors, dtype=np.uint8))


def test_alias_constructor_still_works(d3):
    circuit, _ = d3
    dem = circuit.detector_error_model(decompose_errors=False)
    dec = colour_codes_from_dem(dem, 3)
    assert dec.num_detectors == dem.num_detectors


# ---------------------------------------------------------------------------
# The regression that mattered: output must depend on the input
# ---------------------------------------------------------------------------
def test_observable_matrix_is_not_all_zero(d3):
    """An all-zero L makes every prediction zero regardless of syndrome."""
    _, dec = d3
    assert dec._L.any(), "observable matrix is all zeros — decoder cannot predict"


def test_prediction_is_not_constant_zero(d3):
    """Over real samples the decoder must sometimes predict a logical flip."""
    circuit, dec = d3
    det, _ = circuit.compile_detector_sampler().sample(4000, separate_observables=True)
    pred = dec.decode_batch(np.ascontiguousarray(det.astype(np.uint8)))
    assert pred.any(), "decoder never predicts a flip — it is a no-op"


# ---------------------------------------------------------------------------
# Accuracy: must beat the trivial always-zero predictor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [3, 5])
def test_beats_trivial_predictor(d):
    circuit = _circuit(d)
    dec = ColourCodeDecoder.from_stim_circuit(circuit)
    det, obs = circuit.compile_detector_sampler().sample(1500, separate_observables=True)
    pred = dec.decode_batch(np.ascontiguousarray(det.astype(np.uint8)))
    ler = (pred[:, 0].astype(bool) != obs[:, 0]).mean()
    trivial = obs[:, 0].mean()  # error rate of always predicting zero
    assert ler < trivial, f"d={d}: decoder LER {ler:.4f} is no better than always-zero {trivial:.4f}"


def test_single_and_batch_agree(d3):
    """decode() and decode_batch() must produce identical predictions."""
    circuit, dec = d3
    det, _ = circuit.compile_detector_sampler().sample(24, separate_observables=True)
    det = np.ascontiguousarray(det.astype(np.uint8))
    batch = dec.decode_batch(det)
    single = np.stack([dec.decode(det[i]) for i in range(det.shape[0])])
    np.testing.assert_array_equal(batch, single)


# ---------------------------------------------------------------------------
# The reason this module exists: matching cannot represent these codes
# ---------------------------------------------------------------------------
def test_matching_cannot_decompose_colour_code_at_d5():
    """Documents the premise of C1-04: MWPM is not an option here."""
    circuit = _circuit(5)
    with pytest.raises(Exception):
        circuit.detector_error_model(decompose_errors=True)
