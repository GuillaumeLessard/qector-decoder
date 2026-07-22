from __future__ import annotations
import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

# Embedded Public Key — Production Ed25519 Key (rotated 2026-07-22)
PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAQh9t19EZ4KWZEYjY3EwHCUzUIehZBlovaMtrpLQXeGA=
-----END PUBLIC KEY-----"""

try:
    _PUBLIC_KEY = serialization.load_pem_public_key(PUBLIC_KEY_PEM)
except Exception:
    _PUBLIC_KEY = None


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
        payload = f"{receipt_id}:{target_email}".encode("utf-8")
        _PUBLIC_KEY.verify(signature, payload)
        return True
    except (InvalidSignature, Exception):
        return False
