# Security and Deployment Hardening Guide

QECTOR Decoder is safest when used as a local Rust/Python research library. Optional REST, gRPC, MCP, metrics, Docker, CUDA, and OpenCL surfaces require additional deployment review.

## Deployment posture

| Mode | Status | Guidance |
|---|---|---|
| Local CPU library | Supported public path | Preferred default for research and evaluation |
| CUDA/OpenCL batch | Optional local path | Use only on machines with controlled driver/runtime setup |
| Docker REST server | Demo/local service path | Do not expose directly to the public internet |
| gRPC/MCP/metrics | Optional feature-gated service paths | Treat as experimental unless covered by a commercial deployment review |
| SaaS / hosted API | Contact-only beta review | Requires separate commercial agreement and hardening plan |
| OEM / embedded | Contact-only partner validation | Requires hardware/platform scope and support terms |

## Minimum production checklist

Before using QECTOR in a customer-facing or network-accessible system:

```text
[ ] Pin the git commit or release tag.
[ ] Record Cargo.lock and Python dependency versions.
[ ] Generate dependency inventories for Rust and Python.
[ ] Disable unused optional services and GPU features.
[ ] Run local test and import smoke validation.
[ ] Run only the benchmark claims you intend to quote.
[ ] Keep raw JSON/CSV artifacts and SHA-256 hashes.
[ ] Do not expose REST/gRPC directly without authentication, rate limits, and request-size limits.
[ ] Place any service behind TLS termination and a reverse proxy.
[ ] Restrict logs so benchmark inputs, customer data, and proprietary circuits are not leaked.
[ ] Document operational owner, update path, and rollback path.
```

## Dependency inventory / SBOM-lite commands

Rust dependency inventory:

```powershell
cargo metadata --format-version 1 > artifacts\cargo-metadata.json
cargo tree > artifacts\cargo-tree.txt
```

Python dependency inventory:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > artifacts\pip-freeze.txt
.\.venv\Scripts\python.exe -m pip list --format=json > artifacts\pip-list.json
```

Optional vulnerability audit commands:

```powershell
cargo install cargo-audit
cargo audit

.\.venv\Scripts\python.exe -m pip install pip-audit
.\.venv\Scripts\python.exe -m pip_audit
```

If an audit tool reports a vulnerability, record whether the vulnerable package is actually used in the deployed feature set. Do not ignore service-layer vulnerabilities just because the local library path is safe.

## REST service hardening

The REST layer is useful for local and demo workflows. Before exposing it beyond localhost, add or place in front of it:

```text
authentication
authorization
TLS
request-size limits
rate limits
timeouts
structured audit logs
input validation
job queue / cancellation
resource quotas
error redaction
network firewall rules
```

Do not market the REST layer as enterprise SaaS infrastructure unless these controls and support terms are in place.

## gRPC / MCP / metrics hardening

The gRPC, MCP, and metrics paths are optional feature-gated interfaces. Treat them as experimental until a deployment review verifies:

```text
schema stability
authentication boundary
input size and streaming limits
service discovery exposure
metrics cardinality control
log redaction
versioning and compatibility policy
```

## GPU runtime hardening

CUDA and OpenCL paths load local driver/runtime components. For shared infrastructure:

```text
Pin GPU driver/runtime versions.
Separate untrusted workloads by process/container/VM where possible.
Do not run GPU service workers with unnecessary administrator privileges.
Keep CPU fallback enabled for degraded mode when appropriate.
Do not treat GPU bit-identity as GPU throughput proof.
Regenerate throughput and memory artifacts on the exact deployment hardware.
```

## Benchmark artifact integrity

For public or commercial reports:

```text
Use docs/REPRODUCIBILITY_CHECKLIST.md.
Save raw JSON/CSV outputs.
Save git commit and working-tree state.
Save OS / Python / Rust / package / GPU environment.
Hash artifacts with SHA-256.
State what the artifact does and does not prove.
```

## License and commercial boundary

Network use, hosted API use, OEM integration, internal commercial use, product integration, paid consulting, and commercial benchmarking require a written commercial license. See `LICENSE` and `COMMERCIAL.md`.
