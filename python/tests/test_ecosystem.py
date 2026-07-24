import numpy as np
import pytest
import qector_decoder_v3 as qd


def test_qiskit_plugin_integration():
    """Qiskit plugin exists and is importable."""
    if qd.qiskit_plugin is None:
        pytest.skip("qiskit_plugin not available")
    from qector_decoder_v3.qiskit_plugin import decode_qiskit_result

    raw = {"counts": {"0x0": 400, "0x3": 100}}
    out = decode_qiskit_result(raw, code_distance=3)
    assert "correction" in out
    assert "metadata" in out


def test_stim_compat_integration():
    """Stim compat module exists and is importable."""
    if qd.stim_compat is None:
        pytest.skip("stim_compat not available")
    from qector_decoder_v3.stim_compat import to_stim_decoder

    c2q = [[0, 1], [1, 2]]
    decoder = to_stim_decoder(c2q)
    correction = decoder.decode(np.array([1, 0], dtype=np.uint8))
    assert len(correction) == 3


def test_rest_api_exists():
    """REST API module exists and create_app works."""
    import qector_decoder_v3 as qd

    if qd.rest_api is None:
        pytest.skip("REST API not available (fastapi/flask not installed)")
    app = qd.rest_api.create_app()
    assert app is not None
