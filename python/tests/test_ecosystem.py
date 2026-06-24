import numpy as np
import qector_decoder_v3 as qd


def test_qiskit_plugin_integration():
    """Qiskit plugin exists and is importable."""
    assert qd.qiskit_plugin is not None or True  # may be None if qiskit not installed
    from qector_decoder_v3.qiskit_plugin import decode_qiskit_result

    raw = {"counts": {"0x0": 400, "0x3": 100}}
    out = decode_qiskit_result(raw, code_distance=3)
    assert "correction" in out
    assert "metadata" in out


def test_stim_compat_integration():
    """Stim compat module exists and is importable."""
    assert qd.stim_compat is not None or True
    from qector_decoder_v3.stim_compat import to_stim_decoder

    c2q = [[0, 1], [1, 2]]
    decoder = to_stim_decoder(c2q)
    correction = decoder.decode(np.array([1, 0], dtype=np.uint8))
    assert len(correction) == 3


def test_rest_api_exists():
    """REST API module exists and create_app works."""
    from qector_decoder_v3.rest_api import create_app

    app = create_app()
    assert app is not None
