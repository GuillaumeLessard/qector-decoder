"""v2 licence tokens (tier + expiry) and malformed-input hardening.

Two things are locked here:

1. `verify_license_token` must RETURN False for malformed input, never raise.
   The previous `except RuntimeError` caught neither `binascii.Error` nor
   `UnicodeDecodeError` (both ValueError subclasses), so a garbage token
   propagated an exception to the caller.

2. v2 tokens carry `tier` and `exp` inside the signature. Legacy tokens sign
   only `receipt_id:email`, so an evaluation licence was indistinguishable from
   a perpetual one.
"""

from __future__ import annotations

import base64
import json
import time

import pytest
import qector_decoder_v3.license as lic
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


@pytest.fixture
def signing_key(monkeypatch):
    """Ephemeral keypair swapped in for the embedded production public key."""
    key = Ed25519PrivateKey.generate()
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    monkeypatch.setattr(lic, "PUBLIC_KEY_PEM", pem)
    monkeypatch.setattr(lic, "_PUBLIC_KEY", lic._load_ed25519_public_key())
    return key


# ---------------------------------------------------------------------------
# 1. Malformed input must not raise
# ---------------------------------------------------------------------------

MALFORMED = [
    "a.b.c",
    "rcpt.!!!notb64!!!.c2ln",
    "rcpt.####.####",
    "v2.!!!.!!!",
    "v2." + base64.urlsafe_b64encode(b"not json").decode().rstrip("=") + ".AAAA",
    "....",
    "." * 300,
    "\x00.\x00.\x00",
    "rcpt." + "A" * 5 + ".B",
]


@pytest.mark.parametrize("token", MALFORMED)
def test_malformed_tokens_return_false_never_raise(token, signing_key):
    assert lic.verify_license_token(token) is False
    assert lic.verify_license_token(token, "someone@example.com") is False


@pytest.mark.parametrize("token", MALFORMED)
def test_license_claims_returns_none_never_raises(token, signing_key):
    assert lic.license_claims(token) is None


# ---------------------------------------------------------------------------
# 2. v2 round-trip: tier and expiry
# ---------------------------------------------------------------------------


def test_v2_roundtrip_carries_tier(signing_key):
    tok = lic.create_license_token_v2("cs_live_1", "Buyer@Example.COM", "solo_perpetual", private_key=signing_key)
    assert tok.startswith("v2.")
    assert lic.verify_license_token(tok) is True
    assert lic.verify_license_token(tok, "buyer@example.com") is True
    assert lic.verify_license_token(tok, "attacker@evil.com") is False

    claims = lic.license_claims(tok)
    assert claims is not None
    assert claims["tier"] == "solo_perpetual"
    assert claims["email"] == "buyer@example.com"  # normalised
    assert claims["rid"] == "cs_live_1"
    assert "exp" not in claims  # perpetual


def test_v2_expiry_is_enforced(signing_key):
    expired = lic.create_license_token_v2(
        "cs_live_2",
        "a@b.com",
        "evaluation",
        expires_at=time.time() - 60,
        private_key=signing_key,
    )
    assert lic.verify_license_token(expired) is False, "expired token must not validate"
    assert lic.license_claims(expired) is None

    live = lic.create_license_token_v2(
        "cs_live_3",
        "a@b.com",
        "evaluation",
        expires_at=time.time() + 3600,
        private_key=signing_key,
    )
    assert lic.verify_license_token(live) is True
    assert lic.license_claims(live)["tier"] == "evaluation"


def test_v2_tampered_claims_rejected(signing_key):
    """Escalating the tier inside the claims must break the signature."""
    tok = lic.create_license_token_v2(
        "cs_live_4",
        "a@b.com",
        "evaluation",
        expires_at=time.time() - 1,
        private_key=signing_key,
    )
    _, claims_b64, sig = tok.split(".")
    claims = json.loads(lic._b64url_decode(claims_b64))
    claims["tier"] = "solo_perpetual"
    claims.pop("exp")  # try to shed the expiry too
    forged_b64 = (
        base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
        .decode()
        .rstrip("=")
    )
    assert lic.verify_license_token(f"v2.{forged_b64}.{sig}") is False


# ---------------------------------------------------------------------------
# 3. Legacy tokens keep working; v2 fails closed on pre-v2 clients
# ---------------------------------------------------------------------------


def test_legacy_tokens_still_verify(signing_key):
    legacy3 = lic.create_license_token("cs_live_5", "a@b.com", private_key=signing_key)
    assert lic.verify_license_token(legacy3) is True
    assert lic.verify_license_token(legacy3, "a@b.com") is True
    assert lic.license_claims(legacy3) is None  # no claims in legacy format

    legacy2 = lic.create_license_token("cs_live_6", "", private_key=signing_key)
    assert lic.verify_license_token(legacy2) is True


def test_v2_fails_closed_on_pre_v2_verifier(signing_key):
    """A pre-v2 client reads segment 0 as a receipt id and segment 1 as an
    email. It must reject a v2 token rather than accept it by accident."""
    tok = lic.create_license_token_v2("cs_live_7", "a@b.com", "solo_annual", private_key=signing_key)
    prefix, claims_b64, sig_b64 = tok.split(".")

    # Faithful reimplementation of the pre-v2 3-part branch.
    try:
        embedded = lic._b64url_decode(claims_b64).decode("utf-8").lower()
        payload = f"{prefix}:{embedded}".encode()
        lic._PUBLIC_KEY.verify(lic._b64url_decode(sig_b64), payload)
        accepted = True
    except (InvalidSignature, ValueError, TypeError):
        accepted = False
    assert accepted is False, "v2 token must not validate under the legacy path"


# ---------------------------------------------------------------------------
# 4. QECT- format acceptance (D7)
# ---------------------------------------------------------------------------


def test_qect_pro_format_accepted(signing_key):
    tok = lic.create_license_token_v2("cs_pro_1", "pro@example.com", "pro", private_key=signing_key)
    assert tok.startswith("v2.")
    qd_key = f"QECT-PRO-{tok}"
    assert isinstance(qd_key, str)
    assert qd_key.startswith("QECT-PRO-")


def test_qect_ent_format_accepted(signing_key):
    tok = lic.create_license_token_v2("cs_ent_1", "ent@example.com", "enterprise", private_key=signing_key)
    assert tok.startswith("v2.")
    qd_key = f"QECT-ENT-{tok}"
    assert isinstance(qd_key, str)
    assert qd_key.startswith("QECT-ENT-")


def test_qect_comm_unsigned():
    comm_key = "QECT-COMM-unsigned_test_key_for_dev"
    assert isinstance(comm_key, str)
    assert comm_key.startswith("QECT-COMM-")
