"""
Test suite for Stripe API & License Fulfillment Integration
"""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

import stripe
import qector_decoder_v3 as qd
from qector_decoder_v3.stripe_integration import (
    get_stripe_keys,
    create_checkout_session,
    handle_stripe_webhook_payload,
)
from qector_decoder_v3.license import verify_license_token


def test_stripe_keys_loaded():
    keys = get_stripe_keys()
    assert keys["secret_key_configured"] is True
    assert keys["publishable_key"] != ""
    assert keys["secret_key_prefix"].startswith("sk_live_")


@patch("stripe.checkout.Session.create")
def test_stripe_checkout_session_structure(mock_create):
    mock_session = MagicMock()
    mock_session.id = "cs_live_test_12345"
    mock_session.url = "https://checkout.stripe.com/pay/cs_live_test_12345"
    mock_session.payment_status = "unpaid"
    mock_create.return_value = mock_session

    session_info = create_checkout_session(
        customer_email="test_buyer@qector.store",
        license_tier="commercial",
        amount_cents=49900,
    )
    assert session_info["session_id"] == "cs_live_test_12345"
    assert "https://checkout.stripe.com" in session_info["url"]


def test_stripe_webhook_fulfillment_unauthenticated_fallback():
    mock_payload = json.dumps(
        {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_live_998877665544",
                    "customer_email": "purchaser@qector.store",
                    "payment_status": "paid",
                }
            },
        }
    ).encode("utf-8")

    result = handle_stripe_webhook_payload(payload=mock_payload, sig_header="", webhook_secret="")
    assert result["issued"] is True
    assert result["customer_email"] == "purchaser@qector.store"
    assert result["receipt_id"] == "cs_live_998877665544"

    # Assert issued license token is cryptographically valid Ed25519 token
    token = result["license_token"]
    assert verify_license_token(token, "purchaser@qector.store") is True


@patch("stripe.Webhook.construct_event")
def test_stripe_webhook_fulfillment_authenticated_signature(mock_construct_event):
    mock_event = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_live_112233445566",
                "customer_email": "vip_user@qector.store",
                "payment_status": "paid",
            }
        },
    }
    mock_construct_event.return_value = mock_event

    payload_bytes = json.dumps(mock_event).encode("utf-8")
    test_secret = "whsec_test_secret_for_local_development"
    sig_header = "t=123456,v1=signature_hash_bytes"

    result = handle_stripe_webhook_payload(payload=payload_bytes, sig_header=sig_header, webhook_secret=test_secret)

    assert result["issued"] is True
    assert result["customer_email"] == "vip_user@qector.store"
    assert result["receipt_id"] == "cs_live_112233445566"
    assert verify_license_token(result["license_token"], "vip_user@qector.store") is True
