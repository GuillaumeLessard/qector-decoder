"""
QECTOR Decoder v3 - Full Stripe API & Webhook License Fulfillment Integration

Sales flow (self-serve tiers, see COMMERCIAL.md):
  1. Customer pays via Stripe Checkout / payment link.
  2. Stripe fires ``checkout.session.completed`` at our webhook endpoint.
  3. :func:`handle_stripe_webhook_payload` verifies the signature (when a
     webhook secret is configured), signs an Ed25519 license token with the
     production key (``QECTOR_LICENSE_PRIVATE_KEY_B64``, loaded inside the
     installed package - no repo scripts needed), records the issuance, and
     fires a fire-and-forget notification to the QECTOR Labs bot
     (``QECTOR_LABS_BOT_WEBHOOK_URL``).

Every step logs a clear ``[QECTOR-Stripe]`` message. Bot notification failures
NEVER break fulfillment - a chat outage must not cost a sale.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

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

try:
    import stripe
except ImportError:
    stripe = None

# Signing lives INSIDE the package so fulfillment works in production installs
# (pip-installed wheels do not ship repo-root helper scripts).
from .license import create_license_token

logger = logging.getLogger("qector_decoder_v3.stripe")

# Retrieve keys from environment
# Use the default-argument form of os.getenv so the static type is `str` rather
# than `str | None`; the `or os.getenv(..., "")` pattern confuses mypy.
STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "") or os.getenv("STRIPE_SECRET", "")
STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "") or os.getenv("STRIPE_PUBLISHABLE", "")
STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PURCHASE_LINK: str = os.getenv(
    "STRIPE_PURCHASE_LINK",
    "https://buy.stripe.com/7sY9AVdwlgoyfse9bYeUU00?locale=en&__embed_source=buy_btn_1TsoKxRsa9cg9l8A7ExMmc77",
)
# Discord/Slack-compatible webhook for sales notifications (QECTOR Labs bot).
QECTOR_LABS_BOT_WEBHOOK_URL: str = os.getenv("QECTOR_LABS_BOT_WEBHOOK_URL", "")

if stripe is None:
    logger.warning(
        "[QECTOR-Stripe] The `stripe` Python package is not installed. "
        "Stripe checkout/session creation and signature verification are "
        "unavailable. Install with: pip install 'qector-decoder-v3[stripe]' "
        "(or add `stripe` to your environment). Webhook fulfillment with an "
        "empty/unconfigured webhook secret still works."
    )
elif STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("[QECTOR-Stripe] Live API key configured (prefix %s...).", STRIPE_SECRET_KEY[:8])
else:
    logger.warning(
        "[QECTOR-Stripe] STRIPE_SECRET_KEY is not configured - checkout-session "
        "creation will fail; webhook fulfillment still works."
    )

# ---------------------------------------------------------------------------
# Pricing registry (self-serve tiers - COMMERCIAL.md "Public Pricing Tiers")
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, Any]] = {
    "evaluation": {
        "name": "QECTOR Decoder v3 - Commercial Evaluation License",
        "amount_cents": 49900,
        "currency": "usd",
        "interval": None,  # one-time, 60-day internal evaluation
    },
    "solo_perpetual": {
        "name": "QECTOR Decoder v3 - Solo / Indie Commercial (perpetual)",
        "amount_cents": 89900,
        "currency": "usd",
        "interval": None,  # one-time perpetual internal use
    },
    "solo_annual": {
        "name": "QECTOR Decoder v3 - Solo / Indie Commercial (annual)",
        "amount_cents": 129900,
        "currency": "usd",
        "interval": "year",
    },
}
# Back-compat alias: historical default tier name used by create_checkout_session.
PRICING["commercial"] = PRICING["evaluation"]


def get_stripe_keys() -> dict[str, Any]:
    """Returns configured Stripe keys snapshot."""
    return {
        "publishable_key": STRIPE_PUBLISHABLE_KEY,
        "secret_key_configured": bool(STRIPE_SECRET_KEY),
        "secret_key_prefix": STRIPE_SECRET_KEY[:8] + "..." if STRIPE_SECRET_KEY else "",
        "webhook_secret_configured": bool(STRIPE_WEBHOOK_SECRET),
        "purchase_link": STRIPE_PURCHASE_LINK,
        "labs_bot_configured": bool(QECTOR_LABS_BOT_WEBHOOK_URL),
    }


def ensure_qector_products() -> dict[str, str]:
    """Find-or-create the self-serve QECTOR Products and Prices in Stripe.

    Idempotent: products are located by the ``qector_tier`` metadata key, so
    repeated calls never duplicate. Returns ``{tier: price_id}``. Requires a
    configured secret key; raises RuntimeError otherwise.
    """
    if not STRIPE_SECRET_KEY:
        raise RuntimeError("[QECTOR-Stripe] STRIPE_SECRET_KEY is not configured - cannot ensure products.")
    if stripe is None:
        raise RuntimeError("[QECTOR-Stripe] The `stripe` package is not installed. Install it to ensure products.")
    price_ids: dict[str, str] = {}
    # List-and-filter client-side: works with any key scope (the Search API
    # requires a separate enablement some restricted keys lack).
    existing = stripe.Product.list(active=True, limit=100).data
    for tier, spec in PRICING.items():
        if tier == "commercial":
            continue  # alias, not a real product
        product = None
        for p in existing:
            md = p.metadata
            if hasattr(md, "to_dict"):
                md = md.to_dict()
            if (md or {}).get("qector_tier") == tier:
                product = p
                break
        if product is None:
            product = stripe.Product.create(
                name=spec["name"],
                metadata={"qector_tier": tier},
            )
            logger.info("[QECTOR-Stripe] Created product %s (%s).", product.id, spec["name"])
        price_kwargs: dict[str, Any] = {
            "product": product.id,
            "unit_amount": spec["amount_cents"],
            "currency": spec["currency"],
        }
        if spec["interval"]:
            price_kwargs["recurring"] = {"interval": spec["interval"]}
        price = None
        for pr in stripe.Price.list(product=product.id, active=True, limit=20).data:
            if pr.unit_amount == spec["amount_cents"] and (
                (pr.recurring and pr.recurring.interval == spec["interval"])
                or (not pr.recurring and not spec["interval"])
            ):
                price = pr
                break
        if price is None:
            price = stripe.Price.create(**price_kwargs)
            logger.info(
                "[QECTOR-Stripe] Created price %s for tier %s (%d %s%s).",
                price.id,
                tier,
                spec["amount_cents"],
                spec["currency"],
                f"/{spec['interval']}" if spec["interval"] else " one-time",
            )
        price_ids[tier] = price.id
    return price_ids


def create_checkout_session(
    customer_email: str,
    license_tier: str = "commercial",
    amount_cents: int = 49900,
    currency: str = "usd",
    success_url: str = "https://www.qector.store/success?session_id={CHECKOUT_SESSION_ID}",
    cancel_url: str = "https://www.qector.store/pricing",
) -> dict[str, Any]:
    """
    Creates a Stripe Checkout Session for purchasing a QECTOR Decoder license.
    """
    if not STRIPE_SECRET_KEY:
        raise RuntimeError("STRIPE_SECRET_KEY is not configured in environment or .env file.")
    if stripe is None:
        raise RuntimeError(
            "[QECTOR-Stripe] The `stripe` package is not installed. Install it to create checkout sessions."
        )

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
    logger.info(
        "[QECTOR-Stripe] Checkout session created: id=%s email=%s tier=%s amount=%d %s",
        session.id,
        customer_email,
        license_tier,
        amount_cents,
        currency,
    )
    return {
        "session_id": session.id,
        "url": session.url,
        "payment_status": session.payment_status,
    }


def _notify_labs_bot(message: str, fields: dict[str, Any] | None = None) -> bool:
    """Fire-and-forget sales notification to the QECTOR Labs bot.

    Posts a Discord-compatible JSON payload (also accepted by Slack-compatible
    incoming-webhook bridges) to ``QECTOR_LABS_BOT_WEBHOOK_URL``. NEVER raises:
    any failure is logged as a warning and returns False - a bot outage must
    not interrupt license fulfillment.
    """
    if not QECTOR_LABS_BOT_WEBHOOK_URL:
        logger.warning(
            "[QECTOR-Stripe] QECTOR_LABS_BOT_WEBHOOK_URL not set - skipping Labs-bot "
            "notification. Fulfillment is unaffected."
        )
        return False
    payload: dict[str, Any] = {"content": message}
    if fields:
        payload["embeds"] = [
            {
                "title": "QECTOR License Sale",
                "fields": [{"name": k, "value": str(v), "inline": True} for k, v in fields.items()],
            }
        ]
    try:
        import urllib.request

        req = urllib.request.Request(
            QECTOR_LABS_BOT_WEBHOOK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = 200 <= resp.status < 300
        if ok:
            logger.info("[QECTOR-Stripe] Labs-bot notified: %s", message)
        else:
            logger.warning("[QECTOR-Stripe] Labs-bot responded non-2xx; notification may be lost.")
        return ok
    except (OSError, ValueError) as exc:
        logger.warning("[QECTOR-Stripe] Labs-bot notification FAILED (%s) - fulfillment unaffected.", exc)
        return False


def handle_stripe_webhook_payload(
    payload: bytes, sig_header: str = "", webhook_secret: str | None = None
) -> dict[str, Any]:
    """
    Parses and verifies Stripe webhook events, issuing a signed Ed25519 license token
    upon successful payment.
    """
    # Explicit ``webhook_secret=""`` means "disable signature verification"
    # (local/dev unauthenticated fallback); ``None`` (default) falls back to
    # the configured STRIPE_WEBHOOK_SECRET. A plain ``or`` would conflate the
    # two and break the documented fallback contract.
    secret = webhook_secret if webhook_secret is not None else STRIPE_WEBHOOK_SECRET
    if secret:
        if stripe is None:
            raise RuntimeError(
                "[QECTOR-Stripe] Webhook signature verification requested but "
                "the `stripe` package is not installed. Install it or pass "
                'webhook_secret="" to disable verification.'
            )
        if not sig_header:
            raise ValueError("Stripe signature header ('stripe-signature') is missing.")
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, secret)
        except (stripe.error.SignatureVerificationError, ValueError) as e:
            raise ValueError(f"Webhook signature verification failed: {e}")
        logger.info("[QECTOR-Stripe] Webhook signature verified.")
    else:
        logger.warning(
            "[QECTOR-Stripe] Webhook processed WITHOUT signature verification "
            "(no webhook secret) - acceptable only in local/dev fulfillment."
        )
        payload_text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        event = json.loads(payload_text)

    event_type = event.get("type", "")
    logger.info("[QECTOR-Stripe] Webhook event received: type=%s", event_type)
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
        license_tier = session_obj.get("metadata", {}).get("license_tier", "commercial")
        amount_total = session_obj.get("amount_total")

        token = create_license_token(receipt_id=receipt_id, customer_email=customer_email)

        # Save issued license record locally
        _save_issued_license(receipt_id, customer_email, token)
        logger.info(
            "[QECTOR-Stripe] License issued: receipt=%s email=%s tier=%s amount_total=%s",
            receipt_id,
            customer_email,
            license_tier,
            amount_total,
        )

        # Fire-and-forget Labs-bot sales notification (never breaks fulfillment).
        _notify_labs_bot(
            f"[SALE] QECTOR sale: {license_tier} license issued to {customer_email or 'unknown'}",
            fields={
                "receipt": receipt_id,
                "email": customer_email or "unknown",
                "tier": license_tier,
                "amount_total": amount_total if amount_total is not None else "n/a",
            },
        )

        response_data["license_token"] = token
        response_data["customer_email"] = customer_email
        response_data["receipt_id"] = receipt_id
        response_data["issued"] = True
    else:
        logger.info("[QECTOR-Stripe] Event type %s ignored (not a payment completion).", event_type)

    return response_data


def _save_issued_license(receipt_id: str, email: str, token: str) -> None:
    """Stores issued licenses in local audit log file."""
    record_file = Path("licenses_issued.json")
    records = []
    if record_file.exists():
        try:
            records = json.loads(record_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            records = []
    records.append(
        {
            "receipt_id": receipt_id,
            "customer_email": email,
            "license_token": token,
        }
    )
    record_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
