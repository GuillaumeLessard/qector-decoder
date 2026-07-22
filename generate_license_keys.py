"""
QECTOR Decoder v3 - Key Generation & License Token Signing Utility

The production private key is NEVER hardcoded here. It is loaded from the
QECTOR_LICENSE_PRIVATE_KEY_B64 environment variable (see .env, git-ignored).
This file is safe to publish: it contains no secret material.
"""

from __future__ import annotations
import base64
import os
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

def _load_production_private_key_pem() -> bytes:
    """Loads the production private key PEM from the environment (base64-encoded)."""
    b64 = os.getenv("QECTOR_LICENSE_PRIVATE_KEY_B64", "")
    if not b64:
        raise RuntimeError(
            "QECTOR_LICENSE_PRIVATE_KEY_B64 is not set. Set it in .env (git-ignored) "
            "to sign production license tokens. Never commit the private key."
        )
    return base64.b64decode(b64)


def generate_key_pair() -> tuple[bytes, bytes]:
    """Generates a new Ed25519 private/public keypair in PEM format."""
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    pub_pem = pub.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo
    )
    priv_pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()
    )
    return priv_pem, pub_pem


def create_license_token(
    receipt_id: str,
    customer_email: str = "",
    private_key_pem: bytes | None = None
) -> str:
    """
    Creates an Ed25519 signed license token.
    Format: `{receipt_id}.{email_b64}.{signature_b64}` (with email)
         or `{receipt_id}.{signature_b64}` (without email)
    """
    if private_key_pem is None:
        private_key_pem = _load_production_private_key_pem()
    priv_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(priv_key, ed25519.Ed25519PrivateKey):
        # load_pem_private_key() returns a union of every key type cryptography
        # supports (RSA, DSA, EC, X25519, ML-DSA, ML-KEM, ...). This project only
        # ever signs with Ed25519, so fail loudly if anything else is provided.
        raise TypeError(
            "QECTOR license tokens require an Ed25519 private key; "
            f"got {type(priv_key).__name__}."
        )
    email_clean = customer_email.strip().lower()
    payload = f"{receipt_id}:{email_clean}".encode("utf-8")
    sig = priv_key.sign(payload)
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")

    if email_clean:
        email_b64 = base64.urlsafe_b64encode(email_clean.encode("utf-8")).decode("utf-8").rstrip("=")
        return f"{receipt_id}.{email_b64}.{sig_b64}"
    return f"{receipt_id}.{sig_b64}"


if __name__ == "__main__":
    import sys
    rec = sys.argv[1] if len(sys.argv) > 1 else "rec_001_demo"
    em = sys.argv[2] if len(sys.argv) > 2 else "admin@qector.store"
    tk = create_license_token(rec, em)
    print(f"Generated License Token:\n{tk}")
