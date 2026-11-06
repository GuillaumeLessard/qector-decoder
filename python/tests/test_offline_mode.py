"""QECTOR_OFFLINE=1 guarantees no outbound connection (1.0.1).

The decoders never touch the network. The only egress paths in the package are
explicit billing calls (Stripe metering, checkout, a sales webhook); with
QECTOR_OFFLINE set each of them refuses before opening a socket.
"""
from __future__ import annotations

import socket
import urllib.request

import pytest

import qector_decoder_v3 as qd


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("QECTOR_OFFLINE", "1")

    def _no_network(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("network access attempted while QECTOR_OFFLINE=1")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)


@pytest.mark.parametrize("value,expected", [("1", True), ("yes", True), ("0", False), ("", False), ("off", False)])
def test_offline_flag_parsing(monkeypatch, value, expected):
    from qector_decoder_v3.stripe_integration import offline_mode

    monkeypatch.setenv("QECTOR_OFFLINE", value)
    assert offline_mode() is expected


def test_native_flush_refused_and_shots_kept(offline):
    if getattr(qd._native_module, "py_flush_usage", None) is None:
        pytest.skip("compiled core without py_flush_usage")
    before = int(qd.get_accumulated_shots())
    qd.record_shots(5)
    with pytest.raises(RuntimeError, match="QECTOR_OFFLINE"):
        qd.flush_usage("cus_offline_test", "sk_test_offline")
    assert int(qd.get_accumulated_shots()) == before + 5


def test_sales_webhook_skipped(offline):
    from qector_decoder_v3 import stripe_integration as si

    assert si._notify_labs_bot("offline test") is False


def test_decoding_needs_no_network(offline):
    from qector_decoder_v3 import codes

    code = codes.rotated_surface_code(5)
    dec = qd.BlossomDecoder(code.check_to_qubits, n_qubits=code.n_qubits)
    s = code.syndrome([1] + [0] * (code.n_qubits - 1))
    assert int(sum(dec.decode(s))) == 1
