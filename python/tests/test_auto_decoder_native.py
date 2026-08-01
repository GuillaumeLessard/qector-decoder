import numpy as np
import pytest
import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


@pytest.mark.skipif(not hasattr(qd, "NativeAutoDecoder"), reason="NativeAutoDecoder not available")
def test_native_auto_decoder_routing_d3():
    code = codes.rotated_surface_code(3)
    H = code.parity_check_matrix()
    ar = qd.NativeAutoDecoder(
        code.check_to_qubits, code.n_qubits, distance=3, noise_rate=0.08, batch_size=1, is_qldpc=False
    )
    rng = np.random.default_rng(42)
    e = (rng.random(code.n_qubits) < 0.08).astype(np.uint8)
    s = (H @ e) & 1
    c = np.asarray(ar.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
    assert c.shape == (code.n_qubits,)
    assert np.array_equal((H @ c) & 1, s)


@pytest.mark.skipif(not hasattr(qd, "NativeAutoDecoder"), reason="NativeAutoDecoder not available")
def test_native_auto_decoder_routing_d5():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    ar = qd.NativeAutoDecoder(
        code.check_to_qubits, code.n_qubits, distance=5, noise_rate=0.08, batch_size=1, is_qldpc=False
    )
    rng = np.random.default_rng(7)
    e = (rng.random(code.n_qubits) < 0.06).astype(np.uint8)
    s = (H @ e) & 1
    c = np.asarray(ar.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
    assert c.shape == (code.n_qubits,)


@pytest.mark.skipif(not hasattr(qd, "NativeAutoDecoder"), reason="NativeAutoDecoder not available")
def test_native_auto_decoder_rejects_qldpc_if_license(monkeypatch):
    # Read the real tier BEFORE the monkeypatch below shadows it. Faking
    # `qd.get_license_info` only moves the Python shim; NativeAutoDecoder asks
    # the Rust LicenseManager, which latches the first licence the process
    # resolves and offers no way back to Community. Under dev.bat's Enterprise
    # token the constructor is therefore allowed and this raises nothing.
    # CI runs unlicensed, so the tier is Community and the assertion holds.
    latched = str((qd.get_license_info() or {}).get("tier", "Community"))
    if latched != "Community":
        pytest.skip(
            f"process has latched tier {latched!r}; the native licence gate cannot be "
            "forced back to Community from Python. Run without QECTOR_LICENSE_KEY."
        )
    monkeypatch.setenv("QECTOR_ENFORCE", "1")
    monkeypatch.setattr(qd, "get_license_info", lambda: {"tier": "Community", "max_distance": 7})
    code = codes.rotated_surface_code(5)
    with pytest.raises((PermissionError, RuntimeError, ValueError)):
        qd.NativeAutoDecoder(
            code.check_to_qubits, code.n_qubits, distance=13, noise_rate=0.08, batch_size=1, is_qldpc=True
        )


@pytest.mark.skipif(not hasattr(qd, "NativeAutoDecoder"), reason="NativeAutoDecoder not available")
def test_native_auto_decoder_syndrome_faithful():
    code = codes.rotated_surface_code(3)
    H = code.parity_check_matrix()
    ar = qd.NativeAutoDecoder(
        code.check_to_qubits, code.n_qubits, distance=3, noise_rate=0.08, batch_size=1, is_qldpc=False
    )
    rng = np.random.default_rng(13)
    for _ in range(20):
        e = (rng.random(code.n_qubits) < 0.08).astype(np.uint8)
        s = (H @ e) & 1
        c = np.asarray(ar.decode(s.astype(np.uint8)), dtype=np.uint8).reshape(-1)
        assert np.array_equal((H @ c) & 1, s)
