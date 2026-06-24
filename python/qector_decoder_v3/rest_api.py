"""
API REST minimale pour QECTOR.

Expose :
- POST /decode — décodage d'un syndrome
- GET /health — health check
- GET /version — version du package

Dépendances optionnelles (une des deux) ::

    pip install fastapi uvicorn
    # ou
    pip install flask

Lancement rapide ::

    python -c "from qector_decoder_v3.rest_api import run_server; run_server()"

Ou avec uvicorn directement (FastAPI) ::

    uvicorn qector_decoder_v3.rest_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import UnionFindDecoder, BatchDecoder, __version__

# --- Essai d'import FastAPI, fallback Flask -------------------------------
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _FRAMEWORK = "fastapi"
except ImportError:  # pragma: no cover
    try:
        from flask import Flask, request, jsonify
        _FRAMEWORK = "flask"
    except ImportError:
        raise ImportError(
            "Aucun framework web détecté. Installez fastapi+uvicorn ou flask :\n"
            "    pip install fastapi uvicorn\n"
            "    # ou\n"
            "    pip install flask"
        ) from None


# --- Modèles de données (FastAPI uniquement) --------------------------------
if _FRAMEWORK == "fastapi":

    class DecodeRequest(BaseModel):
        check_to_qubits: List[List[int]]
        syndrome: List[int]
        n_qubits: Optional[int] = None
        use_batch: bool = False

    class DecodeResponse(BaseModel):
        correction: List[int]
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


# --- Construction FastAPI ---------------------------------------------------
def _create_fastapi_app() -> FastAPI:
    app = FastAPI(
        title="QECTOR REST API",
        description="Quantum error correction decoder as a service (QECTOR)",
        version=__version__,
    )

    @app.post("/decode", response_model=DecodeResponse)
    async def decode_endpoint(req: DecodeRequest) -> Dict[str, Any]:  # type: ignore[valid-type]
        if not req.check_to_qubits:
            raise HTTPException(status_code=400, detail="check_to_qubits must be non-empty")

        try:
            if req.use_batch:
                dec = BatchDecoder(req.check_to_qubits, req.n_qubits)
                syndrome_arr = np.array([req.syndrome], dtype=np.uint8)
                correction = dec.parallel_batch_decode(syndrome_arr)[0]
            else:
                dec = UnionFindDecoder(req.check_to_qubits, req.n_qubits)
                syndrome_arr = np.array(req.syndrome, dtype=np.uint8)
                correction = dec.decode(syndrome_arr)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Decode error: {exc}")

        return {
            "correction": correction.tolist(),
            "n_qubits": dec.n_qubits,
            "n_checks": dec.n_checks,
            "version": __version__,
        }

    @app.get("/health", response_model=HealthResponse)
    async def health() -> Dict[str, Any]:  # type: ignore[valid-type]
        return {
            "status": "ok",
            "decoder": "QECTOR UnionFind",
            "version": __version__,
        }

    @app.get("/version", response_model=VersionResponse)
    async def version() -> Dict[str, Any]:  # type: ignore[valid-type]
        return {
            "version": __version__,
            "framework": "fastapi",
            "decoder_backend": "rust-pyo3",
        }

    return app


# --- Construction Flask -----------------------------------------------------
def _create_flask_app() -> Flask:
    app = Flask("qector-rest")

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
            if use_batch:
                dec = BatchDecoder(c2q, n_qubits)
                syndrome_arr = np.array([syndrome], dtype=np.uint8)
                correction = dec.parallel_batch_decode(syndrome_arr)[0]
            else:
                dec = UnionFindDecoder(c2q, n_qubits)
                syndrome_arr = np.array(syndrome, dtype=np.uint8)
                correction = dec.decode(syndrome_arr)
        except Exception as exc:
            return jsonify({"error": f"Decode error: {exc}"}), 500

        return jsonify({
            "correction": correction.tolist(),
            "n_qubits": dec.n_qubits,
            "n_checks": dec.n_checks,
            "version": __version__,
        })

    @app.get("/health")
    def health() -> Any:
        return jsonify({
            "status": "ok",
            "decoder": "QECTOR UnionFind",
            "version": __version__,
        })

    @app.get("/version")
    def version() -> Any:
        return jsonify({
            "version": __version__,
            "framework": "flask",
            "decoder_backend": "rust-pyo3",
        })

    return app


# --- Factory publique -------------------------------------------------------
def create_app() -> Any:
    """
    Créer et retourner l'application WSGI/ASGI appropriée.

    Retourne FastAPI si disponible, sinon Flask.
    """
    if _FRAMEWORK == "fastapi":
        return _create_fastapi_app()
    return _create_flask_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
    """
    Lancer le serveur REST.

    * FastAPI : démarrage via uvicorn.
    * Flask : démarrage via werkzeug.
    """
    app = create_app()
    if _FRAMEWORK == "fastapi":
        import uvicorn
        uvicorn.run(app, host=host, port=port, **kwargs)
    else:
        # Flask — threaded=True par défaut pour un minimum de concurrence
        app.run(host=host, port=port, threaded=True, **kwargs)


# Instance globale pour les serveurs WSGI/ASGI standards (uvicorn, gunicorn, etc.)
app = create_app()
