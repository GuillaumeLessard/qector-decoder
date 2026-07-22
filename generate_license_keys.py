"""
QECTOR Decoder v3 - Key Generation & License Token Signing Utility
"""

from __future__ import annotations
import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Embedded Production Private Key (matching the embedded PUBLIC_KEY_PEM in license.py)
PRODUCTION_PRIVATE_KEY_PEM = b"""-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIK1kjPcTlbGSqrFbAE3p1wy/BUvVej8yquSCXqEq8oMR
-----END PRIVATE KEY-----"""


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
    private_key_pem: bytes = PRODUCTION_PRIVATE_KEY_PEM
) -> str:
    """
    Creates an Ed25519 signed license token.
    Format: `{receipt_id}.{email_b64}.{signature_b64}` (with email) or `{receipt_id}.{signature_b64}`
    """
    priv_key = serialization.load_pem_private_key(private_key_pem, password=None)
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
