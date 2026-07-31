"""
Bulletproof Stripe $0 test sale → Ed25519 license token → activation verification
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import qector_decoder_v3 as qd
from qector_decoder_v3.license import verify_license_token
from qector_decoder_v3.stripe_integration import (
    create_checkout_session,
    handle_stripe_webhook_payload,
)

# Same convention as test_stripe_integration.py: production secrets are absent
# from CI, so the tests that need them skip instead of failing. The import above
# has already run load_dotenv(), so a local .env is visible here.
_has_stripe_key = bool(os.getenv("STRIPE_SECRET_KEY"))
_has_license_key = bool(os.getenv("QECTOR_LICENSE_PRIVATE_KEY_B64"))

_needs_stripe = pytest.mark.skipif(not _has_stripe_key, reason="STRIPE_SECRET_KEY not set")
_needs_signing = pytest.mark.skipif(not _has_license_key, reason="QECTOR_LICENSE_PRIVATE_KEY_B64 not set")


@_needs_stripe
def test_zero_dollar_checkout_session_creation():
    """Test creating a $0 test checkout session via Stripe."""
    with patch("stripe.checkout.Session.create") as mock_create:
        mock_create.return_value.id = "cs_test_zero_dollar_123"
        mock_create.return_value.url = "https://checkout.stripe.com/pay/cs_test_zero_dollar_123"
        mock_create.return_value.payment_status = "paid"

        session = create_checkout_session(
            customer_email="free_trial_user@qector.store",
            license_tier="academic_trial",
            amount_cents=0,
        )
        assert session["session_id"] == "cs_test_zero_dollar_123"
        assert session["url"].startswith("https://checkout.stripe.com")


@_needs_signing
def test_zero_dollar_webhook_issues_valid_token():
    """Simulate Stripe $0 checkout.session.completed → token issuance → Ed25519 verification."""
    test_email = "test_sale_user@qector.store"
    receipt_id = "cs_live_zero_dollar_sale_9999"

    payload_dict = {
        "id": "evt_test_zero_dollar",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": receipt_id,
                "customer_email": test_email,
                "amount_total": 0,
                "payment_status": "paid",
                "metadata": {
                    "license_tier": "commercial_trial",
                    "customer_email": test_email,
                },
            }
        },
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")

    # 1. Process Stripe Webhook payload - token issuance
    webhook_result = handle_stripe_webhook_payload(payload=payload_bytes, sig_header="", webhook_secret="")
    assert webhook_result["issued"] is True
    assert webhook_result["receipt_id"] == receipt_id
    assert webhook_result["customer_email"] == test_email

    issued_token = webhook_result["license_token"]
    assert issued_token is not None
    assert issued_token.startswith(f"{receipt_id}.")

    # 2. Ed25519 cryptographic signature verification - with explicit email
    assert verify_license_token(issued_token, test_email) is True

    # 3. Ed25519 verification - case-insensitive email
    assert verify_license_token(issued_token, "TEST_SALE_USER@QECTOR.STORE") is True

    # 4. Ed25519 verification - wrong email rejected
    assert verify_license_token(issued_token, "wrong_user@qector.store") is False

    # 5. Self-contained 3-part token verification - no email needed (email embedded)
    assert verify_license_token(issued_token) is True


@_needs_signing
def test_license_activates_via_environment(monkeypatch):
    """Verify QECTOR_LICENSE env var activates the license system."""
    test_email = "env_test@qector.store"
    receipt_id = "cs_env_activation_test"

    payload_dict = {
        "id": "evt_env_test",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": receipt_id,
                "customer_email": test_email,
                "amount_total": 0,
                "payment_status": "paid",
                "metadata": {"customer_email": test_email},
            }
        },
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")

    result = handle_stripe_webhook_payload(payload=payload_bytes, sig_header="", webhook_secret="")
    issued_token = result["license_token"]

    # Set environment variable and verify activation
    monkeypatch.setenv("QECTOR_LICENSE", issued_token)
    assert qd._is_license_active() is True


def test_debug_tokens_are_inert_without_qector_debug(monkeypatch):
    """The ``academic``/``commercial`` bypass requires ``QECTOR_DEBUG=1``.

    Without that gate these two words would be a licence for anyone who guessed
    them. Locked in here so the gate cannot be dropped silently.
    """
    monkeypatch.delenv("QECTOR_DEBUG", raising=False)
    for token in ("academic", "commercial"):
        monkeypatch.setenv("QECTOR_LICENSE", token)
        assert qd._is_license_active() is False, f"{token!r} must not activate without QECTOR_DEBUG=1"


def test_autodecoder_operates_with_active_license(monkeypatch):
    """Verify AutoDecoder works bulletproof with an activated license token."""
    # `academic` is a debug-only token; it activates solely under QECTOR_DEBUG=1.
    # The test previously set only QECTOR_LICENSE and asserted activation, which
    # stopped holding once the bypass was gated -- correctly -- behind the debug
    # flag. See test_license_activates_via_environment for the real signed-token
    # path.
    monkeypatch.setenv("QECTOR_DEBUG", "1")
    monkeypatch.setenv("QECTOR_LICENSE", "academic")
    assert qd._is_license_active() is True

    checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
    decoder = qd.AutoDecoder(checks, n_qubits=5)
    syndromes = np.zeros((10, 4), dtype=np.uint8)
    syndromes[0, 1] = 1
    syndromes[3, 0] = 1
    syndromes[3, 3] = 1

    corrections = decoder.batch_decode(syndromes)
    assert corrections.shape == (10, 5)
    assert corrections.dtype == np.uint8


def test_invalid_tokens_rejected():
    """Verify tampered and garbage tokens are rejected."""
    assert verify_license_token("") is False
    assert verify_license_token("garbage_no_dot") is False
    assert verify_license_token("rec.AAAA.BBBB") is False
    assert verify_license_token("rec.tampered_base64.CCCC") is False
