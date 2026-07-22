"""
QECTOR Decoder v3 - Full Stripe API & Webhook License Fulfillment Integration
"""

from __future__ import annotations
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load environment variables from .env at repo root or current working dir
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

import stripe
from generate_license_keys import create_license_token

logger = logging.getLogger("qector_decoder_v3.stripe")

# Retrieve keys from environment
STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_SECRET", "")
STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY") or os.getenv("STRIPE_PUBLISHABLE", "")
STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def get_stripe_keys() -> Dict[str, str]:
    """Returns configured Stripe keys snapshot."""
    return {
        "publishable_key": STRIPE_PUBLISHABLE_KEY,
        "secret_key_configured": bool(STRIPE_SECRET_KEY),
        "secret_key_prefix": STRIPE_SECRET_KEY[:8] + "..." if STRIPE_SECRET_KEY else "",
        "webhook_secret_configured": bool(STRIPE_WEBHOOK_SECRET),
    }


def create_checkout_session(
    customer_email: str,
    license_tier: str = "commercial",
    amount_cents: int = 49900,
    currency: str = "usd",
    success_url: str = "https://qector.store/success?session_id={CHECKOUT_SESSION_ID}",
    cancel_url: str = "https://qector.store/pricing",
) -> Dict[str, Any]:
    """
    Creates a Stripe Checkout Session for purchasing a QECTOR Decoder license.
    """
    if not STRIPE_SECRET_KEY:
        raise RuntimeError("STRIPE_SECRET_KEY is not configured in environment or .env file.")

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        customer_email=customer_email,
        line_items=[
            {
                "price_data": {
                    "currency": currency,
                    "product_data": {
                        "name": f"QECTOR Decoder v3 - {license_tier.capitalize()} License",
                        "description": f"Permanent Ed25519-signed {license_tier} license for QECTOR QEC platform.",
                    },
                    "unit_amount": amount_cents,
                },
                "quantity": 1,
            }
        ],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "license_tier": license_tier,
            "customer_email": customer_email,
        },
    )
    return {
        "session_id": session.id,
        "url": session.url,
        "payment_status": session.payment_status,
    }


def handle_stripe_webhook_payload(
    payload: bytes,
    sig_header: str = "",
    webhook_secret: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parses and verifies Stripe webhook events, issuing a signed Ed25519 license token
    upon successful payment.
    """
    secret = webhook_secret or STRIPE_WEBHOOK_SECRET
    if secret and sig_header:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, secret)
        except Exception as e:
            raise ValueError(f"Webhook signature verification failed: {e}")
    else:
        payload_text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        event = json.loads(payload_text)


    event_type = event.get("type", "")
    response_data = {"event_type": event_type, "license_token": None, "issued": False}

    if event_type in ("checkout.session.completed", "payment_intent.succeeded"):
        session_obj = event.get("data", {}).get("object", {})
        receipt_id = session_obj.get("id") or session_obj.get("payment_intent") or "rec_stripe_live"
        
        customer_details = session_obj.get("customer_details") or {}
        customer_email = (
            session_obj.get("customer_email")
            or customer_details.get("email")
            or session_obj.get("metadata", {}).get("customer_email", "")
        )

        token = create_license_token(receipt_id=receipt_id, customer_email=customer_email)
        
        # Save issued license record locally
        _save_issued_license(receipt_id, customer_email, token)

        response_data["license_token"] = token
        response_data["customer_email"] = customer_email
        response_data["receipt_id"] = receipt_id
        response_data["issued"] = True

    return response_data


def _save_issued_license(receipt_id: str, email: str, token: str) -> None:
    """Stores issued licenses in local audit log file."""
    record_file = Path("licenses_issued.json")
    records = []
    if record_file.exists():
        try:
            records = json.loads(record_file.read_text(encoding="utf-8"))
        except Exception:
            records = []
    records.append({
        "receipt_id": receipt_id,
        "customer_email": email,
        "license_token": token,
    })
    record_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
