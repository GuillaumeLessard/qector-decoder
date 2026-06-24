"""Section 15: DecodeResult fallback status.

A default ``DecodeResult`` reports ``fallback is False`` and an empty
``fallback_reason``. Constructing one with ``fallback=True`` and a reason is
reflected in ``to_dict()`` and rendered by ``explain()`` (which surfaces a
"FALLBACK" line).
"""
import numpy as np

from qector_decoder_v3 import result


def _make(**kw):
    return result.DecodeResult(
        correction=np.zeros(3, np.uint8),
        syndrome=np.zeros(2, np.uint8),
        n_qubits=3,
        n_checks=2,
        **kw,
    )


def test_default_has_no_fallback():
    res = _make()
    assert res.fallback is False
    assert res.fallback_reason == ""


def test_fallback_reflected_in_to_dict():
    res = _make(fallback=True, fallback_reason="gpu unavailable")
    d = res.to_dict()
    assert d["fallback"] is True
    assert d["fallback_reason"] == "gpu unavailable"


def test_fallback_rendered_in_explain():
    res = _make(fallback=True, fallback_reason="gpu unavailable")
    text = res.explain()
    assert "FALLBACK" in text
    assert "gpu unavailable" in text


def test_no_fallback_means_no_fallback_line():
    res = _make()
    text = res.explain()
    assert "FALLBACK" not in text
