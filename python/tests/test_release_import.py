"""
Release import & cryptographic verification test suite for QECTOR Decoder v3 v0.6.9.
Covers tests 1 through 7 specified in gem.md.
"""

import base64
import os
import sys

import pytest
import qector_decoder_v3 as qd
import qector_decoder_v3.license as lic
from qector_decoder_v3.license import create_license_token, verify_license_token


@pytest.fixture
def throwaway_key(monkeypatch):
    """Swap the embedded production public key for a throwaway keypair.

    These tests sign with a generated private key, so the verifier has to be
    pointed at the matching public half. Without this the assertions could never
    pass: `verify_license_token` checks against the production key compiled into
    the package, and correctly rejects a token signed by anything else. (That is
    exactly what it should do -- the previous version of this file asserted the
    opposite and failed for the right reason.)

    Patching the key here keeps the real crypto path under test while removing
    any dependency on the production signing secret being present.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    monkeypatch.setattr(lic, "PUBLIC_KEY_PEM", pem)
    monkeypatch.setattr(lic, "_PUBLIC_KEY", lic._load_ed25519_public_key())
    return key


def _create_token(receipt_id: str, email: str = "", key=None) -> str:
    """Sign a test token with `key` (pair it with the `throwaway_key` fixture)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return create_license_token(receipt_id, email, private_key=key or Ed25519PrivateKey.generate())


def test_1_version_and_exports():
    assert qd.__version__ == "0.6.9"
    assert hasattr(qd, "verify_license_token")
    assert hasattr(qd, "MAX_WORKERS")
    assert qd.__license__ == "LicenseRef-QECTOR-Source-Available"


def test_2_offline_ed25519_verification(throwaway_key):
    token = _create_token("rec_98765", "test@domain.com", throwaway_key)
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


def test_5_environment_license_check(monkeypatch, throwaway_key):
    monkeypatch.delenv("QECTOR_LICENSE", raising=False)
    assert qd._is_license_active() is False

    monkeypatch.setenv("QECTOR_LICENSE", "academic")
    assert qd._is_license_active() is True

    valid_token = _create_token("rec_001", "", throwaway_key)
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
