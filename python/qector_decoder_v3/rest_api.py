"""
Minimal REST API for QECTOR.

Exposes:
- POST /decode - decode a syndrome
- GET  /health - health check
- GET  /version - package version
- GET  /api/license/info - license tier, distance cap, expiry (C8-01)
- POST /api/license/activate - activate a license key at runtime (C8-02)

SECURITY (SEC-01 / P1-17): this API is designed for **localhost use only**.
The default bind address is 127.0.0.1 — do NOT expose it on a public
interface; remote access must go through an SSH tunnel, e.g.::

    ssh -L 8000:127.0.0.1:8000 user@remote-host

Requests are capped at 10 MB (max_content_length), rate-limited per client
IP (default 120 req/min), and every response carries an X-Request-ID header
that is also logged server-side (P1-16).

Optional dependencies (pick one)::

    pip install fastapi uvicorn
    # or
    pip install flask

Quick start::

    python -c "from qector_decoder_v3.rest_api import run_server; run_server()"
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Any, Optional

import numpy as np

from . import (
    BatchDecoder,
    UnionFindDecoder,
    __version__,
    enforce_distance_cap,
    enforce_unlocked,
    estimate_distance,
    get_license_info,
    set_license_key,
)

logger = logging.getLogger("qector_decoder_v3.rest_api")

# SEC-01 (P1-14/C8-03): hard request-size cap shared by both frameworks.
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# P1-16: per-client-IP sliding-window rate limit with bounded LRU eviction
# (A8-01: replaces the unbounded dict + O(n) pop(0)).
_RATE_LIMIT_REQUESTS = 120
_RATE_LIMIT_WINDOW_S = 60.0
_MAX_RATE_LIMIT_CLIENTS = 10_000

# Imported here rather than at the top so the rate-limiter's state and the two
# types it is built from stay in one readable block.
from collections import deque as _deque
from threading import Lock as _Lock

_rate_buckets: dict[str, _deque[float]] = {}
_rate_lock = _Lock()


def _rate_limit_allow(client_ip: str) -> bool:
    """Sliding-window rate limiter: True if the request may proceed.
    Uses a bounded LRU-like eviction scheme: when the total number of
    tracked IPs exceeds _MAX_RATE_LIMIT_CLIENTS, the stalest entries are
    removed (the one with the oldest newest-timestamp).
    """
    now = time.monotonic()
    cutoff = now - _RATE_LIMIT_WINDOW_S
    with _rate_lock:
        bucket = _rate_buckets.get(client_ip)
        if bucket is None:
            # Evict the stalest client if over capacity.
            if len(_rate_buckets) >= _MAX_RATE_LIMIT_CLIENTS:
                # Find the client with the oldest newest-timestamp.
                stalest = min(
                    _rate_buckets,
                    key=lambda ip: _rate_buckets[ip][-1] if _rate_buckets[ip] else 0.0,
                )
                del _rate_buckets[stalest]
            bucket = _deque()
            _rate_buckets[client_ip] = bucket
        # Prune expired timestamps from this bucket.
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_REQUESTS:
            return False
        bucket.append(now)
        return True


# --- Essai d'import FastAPI, fallback Flask -------------------------------
_FRAMEWORK: str | None = None
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    _FRAMEWORK = "fastapi"
except ImportError:  # pragma: no cover
    try:
        from flask import Flask, jsonify, request

        _FRAMEWORK = "flask"
    except ImportError:
        pass


# --- Data models (FastAPI only) ---------------------------------------------
if _FRAMEWORK == "fastapi":

    class DecodeRequest(BaseModel):
        check_to_qubits: list[list[int]]
        syndrome: list[int]
        # `Optional[int]`, not `int | None`. Pydantic resolves model annotations
        # at runtime with `get_type_hints`, and `from __future__ import
        # annotations` only makes that worse here: the annotation reaches
        # pydantic as the *string* "int | None", which it then eval()s, and on
        # 3.9 that raises. It surfaced as six errors telling us to install
        # `eval_type_backport`. `list[list[int]]` above is fine — PEP 585
        # builtin subscripting landed in 3.9; only PEP 604 unions did not.
        n_qubits: Optional[int] = None
        use_batch: bool = False

    class DecodeResponse(BaseModel):
        correction: list[int]
        n_qubits: int
        n_checks: int
        version: str

    class HealthResponse(BaseModel):
        status: str
        decoder: str
        version: str

    class VersionResponse(BaseModel):
        version: str
        framework: str
        decoder_backend: str

    class LicenseActivateRequest(BaseModel):
        key: str


# A8-02: gRPC-style layout-keyed decoder cache for the REST path.
# Maps (decoder_type, layout_key) -> decoder instance.
_DECODER_CACHE: dict[tuple[str, bytes], Any] = {}
_MAX_DECODER_CACHE_SIZE = 64


# A8-03: API-key / bearer auth support. Set QECTOR_API_KEY env var to enable.
def _check_api_auth(authorization: str | None) -> str | None:
    """Validate the Authorization header against QECTOR_API_KEY.
    Returns None on success, or an error string on failure."""
    import os

    api_key = os.environ.get("QECTOR_API_KEY")
    if not api_key:
        return None  # No API key configured — no auth required.
    if not authorization:
        return "API key required: set QECTOR_API_KEY and send Authorization: Bearer <key>"
    if authorization.startswith("Bearer "):
        token = authorization[7:]
        if token == api_key:
            return None
    return "Invalid API key. Check your Authorization: Bearer header."


class _SyndromeValueError(ValueError):
    """Client-supplied syndrome is not binary. Mapped to 422 (fastapi) / 400
    (flask) by the /decode endpoint arms."""


def _decode_impl(check_to_qubits: Any, syndrome: Any, n_qubits: Any, use_batch: bool) -> dict[str, Any]:
    # Licence enforcement (A6-03): hard-gate on missing/invalid key when
    # QECTOR_ENFORCE=1; then check the distance cap.
    enforce_unlocked()
    est_d = estimate_distance(check_to_qubits, n_qubits)
    enforce_distance_cap(est_d)

    # Build a layout key for the decoder cache (A8-02).
    import json

    layout_key = hashlib.sha256(json.dumps(check_to_qubits, sort_keys=True).encode() + str(n_qubits).encode()).digest()
    dec_type = "batch" if use_batch else "unionfind"
    cache_key = (dec_type, layout_key)

    if cache_key not in _DECODER_CACHE:
        if len(_DECODER_CACHE) >= _MAX_DECODER_CACHE_SIZE:
            # Evict the first (oldest) entry.
            _DECODER_CACHE.pop(next(iter(_DECODER_CACHE)))
        if use_batch:
            _DECODER_CACHE[cache_key] = BatchDecoder(check_to_qubits, n_qubits)
        else:
            _DECODER_CACHE[cache_key] = UnionFindDecoder(check_to_qubits, n_qubits)

    dec_any = _DECODER_CACHE[cache_key]
    syndrome_arr = np.array([syndrome], dtype=np.uint8) if use_batch else np.array(syndrome, dtype=np.uint8)
    # SEC-02 trust-boundary hardening (same as the MCP layer): a uint8 value
    # like 2 parses fine but is not a valid detector outcome. Accepting it
    # would silently decode a different syndrome than the caller intended, so
    # reject non-binary input before any decoder touches it. Raised as a plain
    # ValueError subclass so BOTH the fastapi and the flask endpoint arms can
    # map it to a client-error status without NameError.
    if syndrome_arr.size and int(syndrome_arr.max()) > 1:
        raise _SyndromeValueError(
            f"syndrome values must be binary (0/1); found {int(syndrome_arr.max())}"
        )
    correction = dec_any.parallel_batch_decode(syndrome_arr)[0] if use_batch else dec_any.decode(syndrome_arr)
    return {
        "correction": correction.tolist(),
        "n_qubits": dec_any.n_qubits,
        "n_checks": dec_any.n_checks,
        "version": __version__,
    }


# --- Construction FastAPI ---------------------------------------------------
def _create_fastapi_app() -> FastAPI:
    app = FastAPI(
        title="QECTOR REST API",
        description="Quantum error correction decoder as a service (QECTOR) — localhost only, SSH tunnel required for remote access",
        version=__version__,
    )

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = uuid.uuid4().hex[:16]
        client_ip = request.client.host if request.client else "unknown"

        # C8-03/P1-14: max_content_length enforcement.
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_CONTENT_LENGTH:
            logger.warning("[%s] %s rejected: content-length %s > 10MB", request_id, client_ip, content_length)
            raise HTTPException(
                status_code=413, detail=f"Request body exceeds max_content_length ({MAX_CONTENT_LENGTH} bytes)"
            )

        # P1-16: rate limit.
        if not _rate_limit_allow(client_ip):
            logger.warning("[%s] %s rate-limited", request_id, client_ip)
            raise HTTPException(status_code=429, detail="Rate limit exceeded (120 req/min)")

        # Authentication is a service boundary, not a decode-endpoint detail.
        # In particular, activating a licence mutates process-global state and
        # licence information exposes the current entitlement.  Health remains
        # intentionally public for local process supervisors; every other route
        # requires the configured bearer token.
        if request.url.path != "/health":
            auth_err = _check_api_auth(request.headers.get("authorization"))
            if auth_err:
                return JSONResponse(status_code=401, content={"detail": auth_err})

        logger.info("[%s] %s %s from %s", request_id, request.method, request.url.path, client_ip)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.post("/decode", response_model=DecodeResponse)
    async def decode_endpoint(req: DecodeRequest, request: Request) -> dict[str, Any]:  # type: ignore[valid-type]
        if not req.check_to_qubits:
            raise HTTPException(status_code=400, detail="check_to_qubits must be non-empty")
        try:
            return _decode_impl(req.check_to_qubits, req.syndrome, req.n_qubits, req.use_batch)
        except PermissionError as exc:
            # A6-04: licence gate (missing key / distance cap) is a client-side
            # authorisation failure, not a server fault. Without this arm the
            # PermissionError escapes as an unhandled 500.
            raise HTTPException(status_code=403, detail=str(exc))
        except _SyndromeValueError as exc:
            # Non-binary syndrome is a client error (unprocessable entity).
            raise HTTPException(status_code=422, detail=str(exc))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Decode error: {exc}")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:  # type: ignore[valid-type]
        return {
            "status": "ok",
            "decoder": "QECTOR UnionFind",
            "version": __version__,
        }

    @app.get("/version", response_model=VersionResponse)
    async def version() -> dict[str, Any]:  # type: ignore[valid-type]
        return {
            "version": __version__,
            "framework": "fastapi",
            "decoder_backend": "rust-pyo3",
        }

    @app.get("/api/license/info")
    async def license_info() -> dict[str, Any]:  # type: ignore[valid-type]
        """C8-01: current license tier, distance cap, GPU/GNN flags, expiry."""
        return get_license_info()

    @app.post("/api/license/activate")
    async def license_activate(req: LicenseActivateRequest) -> dict[str, Any]:  # type: ignore[valid-type]
        """C8-02: activate a license key at runtime (Ed25519-verified)."""
        if not req.key or not req.key.strip():
            raise HTTPException(status_code=400, detail="key must be non-empty")
        try:
            set_license_key(req.key.strip())
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=403, detail=f"License rejected: {exc}")
        return get_license_info()

    return app


# --- Construction Flask -----------------------------------------------------
def _create_flask_app() -> Flask:
    app = Flask("qector-rest")
    # C8-03/P1-14: werkzeug honours MAX_CONTENT_LENGTH with a 413 itself.
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.before_request
    def _security_gate() -> Any:
        request_id = uuid.uuid4().hex[:16]
        request.request_id = request_id  # type: ignore[attr-defined]
        client_ip = request.remote_addr or "unknown"
        if not _rate_limit_allow(client_ip):
            logger.warning("[%s] %s rate-limited", request_id, client_ip)
            return jsonify({"error": "Rate limit exceeded (120 req/min)"}), 429
        # Keep Flask's policy identical to FastAPI's: health is public for
        # liveness checks; all other endpoints, including licence mutation and
        # inspection, require the configured bearer token.
        if request.path != "/health":
            auth_err = _check_api_auth(request.headers.get("Authorization"))
            if auth_err:
                return jsonify({"error": auth_err}), 401
        logger.info("[%s] %s %s from %s", request_id, request.method, request.path, client_ip)
        return None

    @app.after_request
    def _attach_request_id(response):  # type: ignore[no-untyped-def]
        response.headers["X-Request-ID"] = getattr(request, "request_id", "-")
        return response

    @app.post("/decode")
    def decode_endpoint() -> Any:
        data = request.get_json(force=True, silent=True) or {}  # type: ignore[used-before-def]
        c2q = data.get("check_to_qubits")
        syndrome = data.get("syndrome")
        n_qubits = data.get("n_qubits")
        use_batch = data.get("use_batch", False)

        if not c2q:
            return jsonify({"error": "check_to_qubits must be non-empty"}), 400

        try:
            return jsonify(_decode_impl(c2q, syndrome, n_qubits, use_batch))
        except PermissionError as exc:
            # A6-04: same licence gate as the FastAPI branch — 403, not 500.
            return jsonify({"error": str(exc)}), 403
        except _SyndromeValueError as exc:
            # Non-binary syndrome is a client error (bad request).
            return jsonify({"error": str(exc)}), 400
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": f"Decode error: {exc}"}), 500

    @app.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "decoder": "QECTOR UnionFind",
                "version": __version__,
            }
        )

    @app.get("/version")
    def version() -> Any:
        return jsonify(
            {
                "version": __version__,
                "framework": "flask",
                "decoder_backend": "rust-pyo3",
            }
        )

    @app.get("/api/license/info")
    def license_info() -> Any:
        """C8-01: current license tier, distance cap, GPU/GNN flags, expiry."""
        return jsonify(get_license_info())

    @app.post("/api/license/activate")
    def license_activate() -> Any:
        """C8-02: activate a license key at runtime (Ed25519-verified)."""
        data = request.get_json(force=True, silent=True) or {}
        key = str(data.get("key", "")).strip()
        if not key:
            return jsonify({"error": "key must be non-empty"}), 400
        try:
            set_license_key(key)
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": f"License rejected: {exc}"}), 403
        return jsonify(get_license_info())

    return app


# --- Factory publique -------------------------------------------------------
def create_app() -> Any:
    """
    Create and return the appropriate WSGI/ASGI application.

    Returns FastAPI if available, otherwise Flask.
    """
    if _FRAMEWORK == "fastapi":
        return _create_fastapi_app()
    if _FRAMEWORK == "flask":
        return _create_flask_app()
    raise RuntimeError(
        "No web framework available. Install fastapi+uvicorn or flask:\n"
        "    pip install fastapi uvicorn\n"
        "    # or\n"
        "    pip install flask"
    )


def run_server(host: str = "127.0.0.1", port: int = 8000, **kwargs: Any) -> None:
    """
    Run the REST server.

    SEC-01 (P1-15): binds to localhost by default. Only pass host="0.0.0.0"
    behind a trusted reverse proxy; remote access should use an SSH tunnel.

    * FastAPI: starts via uvicorn.
    * Flask: starts via werkzeug.
    """
    app = create_app()
    if _FRAMEWORK == "fastapi":
        import uvicorn

        uvicorn.run(app, host=host, port=port, **kwargs)
    else:
        # Flask - threaded=True by default for minimal concurrency
        app.run(host=host, port=port, threaded=True, **kwargs)


# Instance globale pour les serveurs WSGI/ASGI standards (uvicorn, gunicorn, etc.)
app: Any = create_app() if _FRAMEWORK is not None else None
