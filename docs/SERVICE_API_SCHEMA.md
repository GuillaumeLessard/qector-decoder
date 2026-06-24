# QECTOR Service API Schema

QECTOR Decoder is safest as a local Rust/Python library. The REST service is a local/demo convenience layer, not an enterprise SaaS product. This document records the current public schema so evaluators can review it without reading the source first.

## Status

| Surface | Status | Notes |
|---|---|---|
| Local Python API | Supported public path | Preferred for research, benchmark reproduction, and commercial evaluation |
| REST API | Experimental local/demo path | No built-in auth, rate limiting, job queue, audit logging, request-size limits, or SLA controls |
| gRPC | Optional feature-gated preview | Requires `--features grpc` / `--features full`; not advertised as stable hosted infrastructure |
| MCP / metrics | Optional preview surfaces | Treat as experimental until a deployment review is complete |
| Hosted API / SaaS | Contact-only beta review | Requires separate commercial agreement and hardening plan |
| OEM / embedded | Contact-only partner validation | Requires hardware/platform scope, support terms, and field-of-use review |

## REST runtime

The REST layer auto-selects FastAPI when available and falls back to Flask.

Install optional REST dependencies only when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn httpx
```

Run locally:

```powershell
.\.venv\Scripts\python.exe -c "from qector_decoder_v3.rest_api import run_server; run_server(host='127.0.0.1', port=8000)"
```

FastAPI direct run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn qector_decoder_v3.rest_api:app --host 127.0.0.1 --port 8000
```

Do not bind to `0.0.0.0` on an untrusted network unless the service is placed behind an authenticated, rate-limited, size-limited gateway.

## Endpoints

### `GET /health`

Purpose: process and package health check.

Response example:

```json
{
  "status": "ok",
  "decoder": "QECTOR UnionFind",
  "version": "0.5.0"
}
```

### `GET /version`

Purpose: expose package version and active web framework.

FastAPI response example:

```json
{
  "version": "0.5.0",
  "framework": "fastapi",
  "decoder_backend": "rust-pyo3"
}
```

Flask response example uses the same fields with `"framework": "flask"`.

### `POST /decode`

Purpose: decode one syndrome using the current REST-layer Union-Find or Batch path.

Request body:

```json
{
  "check_to_qubits": [[0, 1], [1, 2], [2, 3]],
  "syndrome": [0, 1, 0],
  "n_qubits": 4,
  "use_batch": false
}
```

Fields:

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `check_to_qubits` | array of arrays of integers | yes | Parity-check support list |
| `syndrome` | array of integers/bools | yes | Binary syndrome vector |
| `n_qubits` | integer or null | no | Number of qubits when not inferable |
| `use_batch` | boolean | no | Use batch decoder path for one-syndrome batch |

Response example:

```json
{
  "correction": [0, 1, 0, 0],
  "n_qubits": 4,
  "n_checks": 3,
  "version": "0.5.0"
}
```

Error behavior:

```text
400 when check_to_qubits is empty.
500 for decode exceptions in the current minimal service layer.
```

## Current limitations

The REST layer currently does not provide:

```text
authentication
authorization
rate limiting
request-size limits
timeouts / cancellation
job queue
audit logs
structured trace IDs
resource quotas
multi-decoder selection beyond the current UnionFind/Batch path
SLA or uptime controls
```

For commercial hosted use, these must be added or provided by an external gateway and covered by a commercial agreement.

## Safe public wording

Safe:

```text
QECTOR includes a minimal local REST demo API for decode smoke testing and internal workflow prototypes.
```

Unsafe:

```text
QECTOR ships a production-ready hosted API.
QECTOR REST is enterprise SaaS infrastructure.
QECTOR REST provides authenticated commercial API access out of the box.
```

## Related docs

```text
docs/API_STABILITY.md
docs/SECURITY_DEPLOYMENT.md
docs/REPRODUCE.md
COMMERCIAL.md
SECURITY.md
```
