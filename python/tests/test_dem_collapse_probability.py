"""DEM collapse: probability-combination correctness (independent-error XOR rule).

When parallel mechanisms (same detector set) are merged, the combined probability
must follow the independent-error XOR rule ``p = p1(1-p2) + p2(1-p1)`` applied
repeatedly. Covers the todo cases: zero probability, tiny probability, probability
near 0.5, many parallel mechanisms, and nested ``repeat`` / ``shift_detectors``.
"""
import math

import numpy as np

from qector_decoder_v3 import dem


def _xor_combine(ps):
    p = 0.0
    for q in ps:
        p = p * (1.0 - q) + q * (1.0 - p)
    return p


def _merged_prob(model, detset):
    for e in model.errors:
        if e.detectors == tuple(sorted(detset)):
            return e.probability
    raise AssertionError(f"no merged edge for {detset}")


def test_two_parallel_mechanisms_combine_by_xor():
    m = dem.parse_dem("error(0.1) D0 D1\nerror(0.2) D0 D1\nerror(0.05) D1 D2\n")
    c = m.collapse_to_graph()
    assert c.num_errors == 2  # the two D0-D1 mechanisms merged
    assert math.isclose(_merged_prob(c, (0, 1)), _xor_combine([0.1, 0.2]), rel_tol=1e-12)


def test_three_parallel_mechanisms_combine_by_xor():
    m = dem.parse_dem(
        "error(0.05) D0 D1\nerror(0.07) D0 D1\nerror(0.09) D0 D1\n")
    c = m.collapse_to_graph()
    assert c.num_errors == 1
    assert math.isclose(_merged_prob(c, (0, 1)), _xor_combine([0.05, 0.07, 0.09]),
                        rel_tol=1e-12)


def test_zero_probability_member_is_identity():
    m = dem.parse_dem("error(0.0) D0 D1\nerror(0.3) D0 D1\n")
    c = m.collapse_to_graph()
    assert math.isclose(_merged_prob(c, (0, 1)), 0.3, rel_tol=1e-12)


def test_tiny_probability_combines_additively():
    m = dem.parse_dem("error(1e-9) D0 D1\nerror(1e-9) D0 D1\n")
    c = m.collapse_to_graph()
    # p ~ 2e-9 to first order
    assert math.isclose(_merged_prob(c, (0, 1)), _xor_combine([1e-9, 1e-9]), rel_tol=1e-6)


def test_probability_near_half():
    m = dem.parse_dem("error(0.5) D0 D1\nerror(0.5) D0 D1\n")
    c = m.collapse_to_graph()
    # 0.5 XOR 0.5 = 0.5
    assert math.isclose(_merged_prob(c, (0, 1)), 0.5, rel_tol=1e-12)


def test_many_parallel_mechanisms():
    ps = [0.01 * (i + 1) for i in range(8)]
    text = "\n".join(f"error({p}) D0 D1" for p in ps)
    c = dem.parse_dem(text).collapse_to_graph()
    assert c.num_errors == 1
    assert math.isclose(_merged_prob(c, (0, 1)), _xor_combine(ps), rel_tol=1e-10)


def test_weights_follow_combined_probability():
    m = dem.parse_dem("error(0.1) D0 D1\nerror(0.1) D0 D1\n")
    c = m.collapse_to_graph()
    p = _xor_combine([0.1, 0.1])
    expected_w = math.log((1 - p) / p)
    assert math.isclose(c.weights()[0], expected_w, rel_tol=1e-9)


def test_collapse_with_repeat_and_shift_blocks():
    """Parallel mechanisms produced inside repeat/shift blocks still merge."""
    text = """
    error(0.1) D0 D1
    repeat 2 {
        error(0.1) D0 D1
        shift_detectors 0
    }
    """
    m = dem.parse_dem(text)
    assert m.num_errors == 3  # all on D0,D1 (shift 0)
    c = m.collapse_to_graph()
    assert c.num_errors == 1
    assert math.isclose(_merged_prob(c, (0, 1)), _xor_combine([0.1, 0.1, 0.1]),
                        rel_tol=1e-12)
