"""
Bulletproof Stripe $0 test sale → Ed25519 license token → activation verification
"""

import json
import os
import numpy as np
import pytest
from unittest.mock import patch

import qector_decoder_v3 as qd
from qector_decoder_v3.stripe_integration import (
    create_checkout_session,
    handle_stripe_webhook_payload,
)
from qector_decoder_v3.license import verify_license_token


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

    # 1. Process Stripe Webhook payload — token issuance
    webhook_result = handle_stripe_webhook_payload(payload=payload_bytes, sig_header="")
    assert webhook_result["issued"] is True
    assert webhook_result["receipt_id"] == receipt_id
    assert webhook_result["customer_email"] == test_email

    issued_token = webhook_result["license_token"]
    assert issued_token is not None
    assert issued_token.startswith(f"{receipt_id}.")

    # 2. Ed25519 cryptographic signature verification — with explicit email
    assert verify_license_token(issued_token, test_email) is True

    # 3. Ed25519 verification — case-insensitive email
    assert verify_license_token(issued_token, "TEST_SALE_USER@QECTOR.STORE") is True

    # 4. Ed25519 verification — wrong email rejected
    assert verify_license_token(issued_token, "wrong_user@qector.store") is False

    # 5. Self-contained 3-part token verification — no email needed (email embedded)
    assert verify_license_token(issued_token) is True


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

    result = handle_stripe_webhook_payload(payload=payload_bytes, sig_header="")
    issued_token = result["license_token"]

    # Set environment variable and verify activation
    monkeypatch.setenv("QECTOR_LICENSE", issued_token)
    assert qd._is_license_active() is True


def test_autodecoder_operates_with_active_license(monkeypatch):
    """Verify AutoDecoder works bulletproof with an activated license token."""
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
