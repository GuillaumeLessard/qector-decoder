"""
Release import & cryptographic verification test suite for QECTOR Decoder v3 v0.6.7.
Covers tests 1 through 7 specified in gem.md.
"""

import os
import sys
import base64

import pytest

# Allow importing the keygen helper from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import qector_decoder_v3 as qd
from qector_decoder_v3.license import verify_license_token

# ---- inline token creation using the production private key ----
from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed25519
from cryptography.hazmat.primitives import serialization as _ser

_PROD_PRIV_PEM = b"""-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIK1kjPcTlbGSqrFbAE3p1wy/BUvVej8yquSCXqEq8oMR
-----END PRIVATE KEY-----"""

def _create_token(receipt_id: str, email: str = "") -> str:
    priv_key = _ser.load_pem_private_key(_PROD_PRIV_PEM, password=None)
    payload = f"{receipt_id}:{email.lower()}".encode("utf-8")
    sig = priv_key.sign(payload)
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    return f"{receipt_id}.{sig_b64}"


def test_1_version_and_exports():
    assert qd.__version__ == "0.6.7"
    assert hasattr(qd, "verify_license_token")
    assert hasattr(qd, "MAX_WORKERS")
    assert getattr(qd, "__license__") == "LicenseRef-QECTOR-Source-Available"


def test_2_offline_ed25519_verification():
    token = _create_token("rec_98765", "test@domain.com")
    assert verify_license_token(token, "test@domain.com") is True
    assert verify_license_token(token, "TEST@DOMAIN.COM") is True  # case insensitive
    assert verify_license_token(token, "wrong@domain.com") is False


def test_3_invalid_token_rejection():
    assert verify_license_token("") is False
    assert verify_license_token("invalid_token_format") is False
    assert verify_license_token("rec_123.corrupted_sig!!!") is False


def test_4_special_override_tokens():
    assert verify_license_token("academic") is True
    assert verify_license_token("commercial") is True


def test_5_environment_license_check(monkeypatch):
    monkeypatch.delenv("QECTOR_LICENSE", raising=False)
    assert qd._is_license_active() is False

    monkeypatch.setenv("QECTOR_LICENSE", "academic")
    assert qd._is_license_active() is True

    valid_token = _create_token("rec_001", "")
    monkeypatch.setenv("QECTOR_LICENSE", valid_token)
    assert qd._is_license_active() is True


def test_6_startup_notice_suppression(monkeypatch, capsys):
    monkeypatch.setenv("QECTOR_SILENT", "1")
    qd._emit_startup_notice()
    captured = capsys.readouterr()
    assert "[QECTOR" not in captured.err


def test_7_decoder_pool():
    pool = qd.DecoderPool(num_threads=2)
    assert pool.num_threads == min(2, qd.MAX_WORKERS)
    default_pool = qd.DecoderPool()
    assert default_pool.num_threads == qd.MAX_WORKERS
