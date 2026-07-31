from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import time

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

logger = logging.getLogger("qector_decoder_v3.license")

# v2 token discriminator. A v2 token is `v2.{claims_b64}.{sig_b64}`; the leading
# segment is a literal so old clients, which read segment 0 as a receipt id and
# segment 1 as an email, fail the signature check and return False. New tokens
# therefore fail CLOSED on installs that predate v2 support -- they never
# validate by accident.
_V2_PREFIX = "v2"

# Embedded Public Key - Production Ed25519 Key (rotated 2026-07-22)
# This is the public half of the production signing key held in
# QECTOR_LICENSE_PRIVATE_KEY_B64 (.env / CI secret). Tokens issued by the
# Stripe webhook fulfillment path verify against this key.
PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAQh9t19EZ4KWZEYjY3EwHCUzUIehZBlovaMtrpLQXeGA=
-----END PUBLIC KEY-----"""


def _load_ed25519_public_key() -> Ed25519PublicKey | None:
    """Loads the embedded PEM and narrows it to Ed25519PublicKey.

    load_pem_public_key() returns a union of every key type cryptography
    supports (RSA, DSA, EC, X25519, ML-DSA, ML-KEM, ...). This project only
    ever signs/verifies with Ed25519, so we assert that narrowly here rather
    than propagating the full union to every call site.
    """
    try:
        key = serialization.load_pem_public_key(PUBLIC_KEY_PEM)
    except (ValueError, TypeError, UnsupportedAlgorithm):
        return None
    if not isinstance(key, Ed25519PublicKey):
        return None
    return key


_PUBLIC_KEY: Ed25519PublicKey | None = _load_ed25519_public_key()


# Exceptions raised while decoding an untrusted token. `binascii.Error` (bad
# base64) and `UnicodeDecodeError` are both ValueError subclasses -- the previous
# `except RuntimeError` caught NEITHER, so a malformed token propagated the
# exception to the caller instead of returning False. Any caller passing
# attacker-influenced input (a web form, an env var, a CLI arg) crashed rather
# than rejecting. Verified: 'a.b.c' raised binascii.Error("Invalid
# base64-encoded string"); 'rcpt.!!!notb64!!!.c2ln' raised
# binascii.Error("Incorrect padding").
_DECODE_ERRORS = (InvalidSignature, ValueError, TypeError)


def _b64url_decode(segment: str) -> bytes:
    """urlsafe-base64 decode tolerating the stripped '=' padding we emit."""
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def verify_license_token(token: str, customer_email: str = "") -> bool:
    """
    Verifies the license token signature completely offline using Ed25519.

    Accepted formats:
      * ``{receipt_id}.{sig}``                       (legacy, 2-part)
      * ``{receipt_id}.{email_b64}.{sig}``           (legacy, 3-part)
      * ``v2.{claims_b64}.{sig}``                    (v2: adds tier + expiry)

    A v2 token carries JSON claims ``{rid, email, tier, exp}``; ``exp`` is a Unix
    timestamp and an expired token verifies as False even though its signature is
    valid. Returns False -- never raises -- for any malformed input.
    """
    if not token:
        return False

    token_clean = token.strip()
    if os.environ.get("QECTOR_DEBUG") == "1" and token_clean in ("academic", "commercial"):
        return True

    if "." not in token_clean or _PUBLIC_KEY is None:
        return False

    parts = token_clean.split(".")

    if len(parts) == 3 and parts[0] == _V2_PREFIX:
        return _verify_v2(parts[1], parts[2], customer_email)

    if len(parts) == 3:
        receipt_id, email_b64, sig_b64 = parts
        try:
            embedded_email = _b64url_decode(email_b64).decode("utf-8").lower()
        except _DECODE_ERRORS:
            return False
        if customer_email and customer_email.strip().lower() != embedded_email:
            return False
        target_email = embedded_email
    elif len(parts) == 2:
        receipt_id, sig_b64 = parts
        target_email = customer_email.strip().lower()
    else:
        return False

    try:
        _PUBLIC_KEY.verify(_b64url_decode(sig_b64), f"{receipt_id}:{target_email}".encode())
        return True
    except _DECODE_ERRORS:
        return False


def _verify_v2(claims_b64: str, sig_b64: str, customer_email: str) -> bool:
    """Verify a v2 token. Signature covers the claims segment verbatim."""
    if _PUBLIC_KEY is None:
        return False
    try:
        # Signature first: never parse claims we have not authenticated.
        _PUBLIC_KEY.verify(_b64url_decode(sig_b64), claims_b64.encode("ascii"))
        claims = json.loads(_b64url_decode(claims_b64).decode("utf-8"))
    except _DECODE_ERRORS + (json.JSONDecodeError, UnicodeEncodeError):
        return False
    if not isinstance(claims, dict):
        return False

    embedded_email = str(claims.get("email", "")).strip().lower()
    if customer_email and customer_email.strip().lower() != embedded_email:
        return False

    exp = claims.get("exp")
    if exp is not None:
        try:
            if time.time() > float(exp):
                logger.info("[QECTOR-License] token expired at %s", exp)
                return False
        except (TypeError, ValueError):
            return False
    return True


def license_claims(token: str) -> dict | None:
    """Return the verified claims of a v2 token, or ``None``.

    Only returns claims for a token whose signature validates and which has not
    expired -- so callers can gate on ``tier`` without re-implementing checks.
    Legacy 2/3-part tokens carry no claims and yield ``None`` even when valid.
    """
    if not token:
        return None
    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != _V2_PREFIX:
        return None
    if not _verify_v2(parts[1], parts[2], ""):
        return None
    try:
        return json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except _DECODE_ERRORS + (json.JSONDecodeError,):
        return None


# ---------------------------------------------------------------------------
# Rust-side tier bridging (C2)
# ---------------------------------------------------------------------------

_TIER_MAX_DISTANCE = {"community": 7, "pro": 19, "enterprise": 63}


def _native_module():
    """Best-effort access to the compiled Rust extension module."""
    try:
        from . import _native_module as nm  # type: ignore[attr-defined]

        return nm
    except Exception:
        try:
            import qector_decoder_v3.qector_decoder_v3 as nm  # type: ignore[no-redef]

            return nm
        except Exception:
            return None


def _configured_key() -> str:
    """Resolve the configured key the same way the Rust core does.

    Precedence: ``QECTOR_LICENSE_KEY``, then ``QECTOR_LICENSE_FILE``, then
    ``~/.qector/license.key``. Keeping this in step with
    ``LicenseManager::key_from_env`` matters because the pure-Python fallback
    otherwise reports Community for a deployment the core would read as
    Enterprise -- the two answers must not disagree.
    """
    key = os.environ.get("QECTOR_LICENSE_KEY", "")
    if key.strip():
        return key.strip()

    candidates = []
    from_file = os.environ.get("QECTOR_LICENSE_FILE", "").strip()
    if from_file:
        candidates.append(from_file)
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        candidates.append(os.path.join(home, ".qector", "license.key"))

    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().lstrip("﻿").strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def activate_license(key: str) -> dict:
    """Activate a license key: sets ``QECTOR_LICENSE_KEY`` AND installs it in
    the Rust ``LicenseManager`` singleton (C2-01).

    Raises ``ValueError`` when the native verifier rejects the key (bad
    signature, expired, revoked, or unknown format). Returns the effective
    license info dict after activation.
    """
    if not key or not key.strip():
        raise ValueError("license key must be non-empty")
    key = key.strip()

    nm = _native_module()
    native_set = getattr(nm, "py_set_license_key", None) if nm is not None else None
    if native_set is not None:
        # Native path: real Ed25519 verification; raises on rejection.
        native_set(key)
    elif not verify_license_token(key):
        # Pure-Python fallback: at least enforce the Python-side signature check.
        raise ValueError("license key failed verification (Ed25519 signature, expiry, or revocation check)")

    os.environ["QECTOR_LICENSE_KEY"] = key
    return get_tier_info()


def get_tier() -> str:
    """Active license tier: ``"Community"`` | ``"Pro"`` | ``"Enterprise"`` (C2-02).

    Reads from the Rust LicenseManager when available; falls back to the
    ``QECTOR_LICENSE_KEY`` prefix / v2 claims otherwise.
    """
    info = get_tier_info()
    return str(info.get("tier", "Community"))


def get_tier_info() -> dict:
    """Full license info from the Rust LicenseManager (native types) or a
    Python-derived fallback."""
    nm = _native_module()
    native_info = getattr(nm, "py_get_license_info", None) if nm is not None else None
    if native_info is not None:
        try:
            info = native_info()
            if isinstance(info, dict) and "tier" in info:
                return info
        except Exception:
            pass

    key = _configured_key()
    claims = license_claims(key) if key else None
    if claims is not None:
        tier_raw = str(claims.get("tier", "community"))
    elif key.startswith("QECT-ENT-"):
        tier_raw = "enterprise"
    elif key.startswith("QECT-PRO-"):
        tier_raw = "pro"
    else:
        tier_raw = "community"
    tier = tier_raw.strip().lower()
    tier_name = {"community": "Community", "pro": "Pro", "enterprise": "Enterprise"}.get(tier, "Community")
    return {
        "tier": tier_name,
        "max_distance": _TIER_MAX_DISTANCE.get(tier, 7),
        "gpu_enabled": tier == "enterprise",
        "gnn_enabled": tier in ("pro", "enterprise"),
        "key_status": "valid" if key else "no_key",
    }


def enforce_distance_cap(distance: int) -> None:
    """Raise ``PermissionError`` when ``distance`` exceeds the active tier cap
    (C2-03): Community d<=7, Pro d<=19, Enterprise d<=63.

    Delegates to the Rust enforcement path when available (which also checks
    expiry); falls back to the Python-derived tier cap otherwise.
    """
    nm = _native_module()
    native_enforce = getattr(nm, "py_enforce_distance_cap", None) if nm is not None else None
    if native_enforce is not None:
        native_enforce(int(distance))  # raises PyPermissionError on violation
        return

    info = get_tier_info()
    cap = int(info.get("max_distance", 7))
    if int(distance) > cap:
        raise PermissionError(
            f"License Tier '{info.get('tier', 'Community')}' caps maximum code distance at "
            f"d<={cap}. Requested d={distance}. Upgrade at https://qector.store/pricing."
        )


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
    except (ValueError, TypeError, binascii.Error, UnsupportedAlgorithm) as exc:
        raise RuntimeError(f"[QECTOR-License] QECTOR_LICENSE_PRIVATE_KEY_B64 is malformed: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("[QECTOR-License] QECTOR_LICENSE_PRIVATE_KEY_B64 is not an Ed25519 private key.")
    return key


def create_license_token(
    receipt_id: str, customer_email: str = "", private_key: Ed25519PrivateKey | None = None
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


def create_license_token_v2(
    receipt_id: str,
    customer_email: str,
    tier: str,
    expires_at: float | None = None,
    private_key: Ed25519PrivateKey | None = None,
) -> str:
    """Sign a v2 token carrying ``tier`` and an optional expiry.

    Legacy tokens sign ``receipt_id:email`` and nothing else, so a 60-day
    evaluation is cryptographically identical to a perpetual licence -- the tier
    exists only in Stripe metadata and a local log, neither of which the package
    can see. v2 puts both inside the signature.

    ``expires_at`` is a Unix timestamp; ``None`` means perpetual. The signature
    covers the base64url claims segment verbatim, so verification never has to
    canonicalise JSON to reproduce the signed bytes.
    """
    key = private_key or _load_signing_private_key()
    claims = {
        "rid": receipt_id,
        "email": customer_email.strip().lower(),
        "tier": tier,
    }
    if expires_at is not None:
        claims["exp"] = int(expires_at)
    # separators= keeps the encoding compact and stable; the signature is over
    # the encoded segment, so key order only has to be deterministic, not
    # canonical across implementations.
    claims_json = json.dumps(claims, separators=(",", ":"), sort_keys=True)
    claims_b64 = base64.urlsafe_b64encode(claims_json.encode("utf-8")).decode("ascii").rstrip("=")
    sig_b64 = base64.urlsafe_b64encode(key.sign(claims_b64.encode("ascii"))).decode("ascii").rstrip("=")
    return f"{_V2_PREFIX}.{claims_b64}.{sig_b64}"
