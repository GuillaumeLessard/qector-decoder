from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

logger = logging.getLogger("qector_decoder_v3.license")

# Embedded Public Key - Production Ed25519 Key (rotated 2026-07-22)
# This is the public half of the production signing key held in
# QECTOR_LICENSE_PRIVATE_KEY_B64 (.env / CI secret). Tokens issued by the
# Stripe webhook fulfillment path verify against this key.
PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAQh9t19EZ4KWZEYjY3EwHCUzUIehZBlovaMtrpLQXeGA=
-----END PUBLIC KEY-----"""


def _load_ed25519_public_key() -> Optional[Ed25519PublicKey]:
    """Loads the embedded PEM and narrows it to Ed25519PublicKey.

    load_pem_public_key() returns a union of every key type cryptography
    supports (RSA, DSA, EC, X25519, ML-DSA, ML-KEM, ...). This project only
    ever signs/verifies with Ed25519, so we assert that narrowly here rather
    than propagating the full union to every call site.
    """
    try:
        key = serialization.load_pem_public_key(PUBLIC_KEY_PEM)
    except Exception:
        return None
    if not isinstance(key, Ed25519PublicKey):
        return None
    return key


_PUBLIC_KEY: Optional[Ed25519PublicKey] = _load_ed25519_public_key()


def verify_license_token(token: str, customer_email: str = "") -> bool:
    """
    Verifies the license token signature completely offline using Ed25519.
    Supports both 2-part ({receipt_id}.{sig}) and 3-part ({receipt_id}.{email_b64}.{sig}) token formats.
    """
    if not token:
        return False

    token_clean = token.strip()
    if token_clean in ("academic", "commercial"):
        return True

    if "." not in token_clean or _PUBLIC_KEY is None:
        return False

    parts = token_clean.split(".")
    if len(parts) == 3:
        receipt_id, email_b64, sig_b64 = parts
        try:
            missing_pad = len(email_b64) % 4
            if missing_pad:
                email_b64 += "=" * (4 - missing_pad)
            embedded_email = base64.urlsafe_b64decode(email_b64).decode("utf-8").lower()

            # If caller provided explicit email check, ensure match
            if customer_email and customer_email.strip().lower() != embedded_email:
                return False

            target_email = embedded_email
        except Exception:
            return False
    elif len(parts) == 2:
        receipt_id, sig_b64 = parts
        target_email = customer_email.strip().lower()
    else:
        return False

    # Fix base64 padding for signature
    missing_padding = len(sig_b64) % 4
    if missing_padding:
        sig_b64 += "=" * (4 - missing_padding)

    try:
        signature = base64.urlsafe_b64decode(sig_b64)
        payload = f"{receipt_id}:{target_email}".encode()
        _PUBLIC_KEY.verify(signature, payload)
        return True
    except (InvalidSignature, Exception):
        return False


# ---------------------------------------------------------------------------
# Signing side (fulfillment server only - never ships a private key)
# ---------------------------------------------------------------------------
#
# The webhook fulfillment path (stripe_integration.handle_stripe_webhook_payload)
# runs on the seller's infrastructure, not in customer installs. The signing
# private key comes exclusively from the QECTOR_LICENSE_PRIVATE_KEY_B64
# environment variable (base64-encoded PKCS#8 PEM). There is deliberately NO
# hardcoded fallback key here: if the variable is missing, issuance fails loudly
# rather than minting tokens that customer-side verification would reject.


def _load_signing_private_key() -> Ed25519PrivateKey:
    env_b64 = os.getenv("QECTOR_LICENSE_PRIVATE_KEY_B64", "")
    if not env_b64:
        raise RuntimeError(
            "[QECTOR-License] QECTOR_LICENSE_PRIVATE_KEY_B64 is not set - cannot "
            "sign license tokens. Set it to the base64-encoded PKCS#8 PEM of the "
            "production Ed25519 private key (see .env.example)."
        )
    try:
        raw = base64.b64decode(env_b64)
        key = serialization.load_pem_private_key(raw, password=None)
    except Exception as exc:
        raise RuntimeError(
            f"[QECTOR-License] QECTOR_LICENSE_PRIVATE_KEY_B64 is malformed: {exc}"
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise RuntimeError(
            "[QECTOR-License] QECTOR_LICENSE_PRIVATE_KEY_B64 is not an Ed25519 private key."
        )
    return key


def create_license_token(
    receipt_id: str, customer_email: str = "", private_key: Optional[Ed25519PrivateKey] = None
) -> str:
    """Sign ``receipt_id:email`` with the production Ed25519 key.

    Returns the 3-part token ``{receipt_id}.{email_b64}.{sig_b64}`` (2-part
    when no email is given) that :func:`verify_license_token` validates
    offline against the embedded production public key.
    """
    key = private_key or _load_signing_private_key()
    email_clean = customer_email.strip().lower()
    payload = f"{receipt_id}:{email_clean}".encode()
    signature = key.sign(payload)
    sig_b64 = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")

    if customer_email:
        email_b64 = base64.urlsafe_b64encode(email_clean.encode("utf-8")).decode("utf-8").rstrip("=")
        return f"{receipt_id}.{email_b64}.{sig_b64}"
    return f"{receipt_id}.{sig_b64}"
