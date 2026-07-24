"""
test_stripe_missing_dependency.py - Bullet-proof import/use when the optional
`stripe` package is absent. Ensures graceful degradation, clear warnings, and
actionable RuntimeErrors rather than AttributeError crashes.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import patch

import pytest
import qector_decoder_v3.stripe_integration as si
from qector_decoder_v3.stripe_integration import handle_stripe_webhook_payload


def _reload_stripe_integration():
    """Force re-import of stripe_integration so module-level warnings rerun."""
    # Remove from cache to trigger fresh import.
    for key in list(sys.modules):
        if key == "qector_decoder_v3.stripe_integration" or key.endswith(".stripe_integration"):
            del sys.modules[key]
    return __import__("qector_decoder_v3.stripe_integration", fromlist=["si"])


def test_import_succeeds_when_stripe_missing_and_secret_present():
    """Regression: module must import even if STRIPE_SECRET_KEY is set and
    stripe is absent. Previously crashed with AttributeError on stripe.api_key."""
    fake_stripe_module = types.ModuleType("stripe")
    fake_stripe_module.api_key = None
    fake_stripe_module.checkout = None
    fake_stripe_module.Webhook = None

    with patch.dict("sys.modules", {"stripe": fake_stripe_module}):
        mod = _reload_stripe_integration()
        assert mod.stripe is fake_stripe_module
        keys = mod.get_stripe_keys()
        assert keys["secret_key_configured"] is True


def test_create_checkout_session_raises_clear_error_without_stripe():
    with patch.object(si, "stripe", None), pytest.raises(RuntimeError, match="stripe.*package is not installed"):
        si.create_checkout_session("test@example.com")


def test_ensure_products_raises_clear_error_without_stripe():
    with patch.object(si, "stripe", None), pytest.raises(RuntimeError, match="stripe.*package is not installed"):
        si.ensure_qector_products()


def test_webhook_signature_requires_stripe_when_secret_given():
    payload = json.dumps({"type": "checkout.session.completed", "data": {"object": {}}}).encode("utf-8")
    with patch.object(si, "stripe", None), pytest.raises(RuntimeError, match="stripe.*package is not installed"):
        handle_stripe_webhook_payload(payload, sig_header="sig", webhook_secret="whsec_x")


def test_webhook_fulfillment_works_without_stripe_when_secret_disabled():
    """License issuance does not depend on stripe; unauthenticated webhook
    fallback must work with stripe absent."""
    payload = json.dumps(
        {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_no_stripe_001",
                    "customer_email": "offline@example.com",
                }
            },
        }
    ).encode("utf-8")
    with patch.object(si, "stripe", None):
        result = handle_stripe_webhook_payload(payload, sig_header="", webhook_secret="")
    assert result["issued"] is True
    assert result["customer_email"] == "offline@example.com"
    assert result["receipt_id"] == "cs_no_stripe_001"
