# Security Policy

QECTOR Decoder is primarily a local Rust/Python research and validation library, but the repository also contains optional service-facing components such as REST, gRPC, MCP, metrics, Docker packaging, CUDA loading, and OpenCL loading. Treat those components as higher-risk deployment surfaces.

## Supported versions

| Version | Security support |
|---|---|
| `0.5.x` | Supported for source-available review, private vulnerability disclosure, and commercial-license security discussion |
| Older snapshots | Best-effort only |

## Report a vulnerability privately

Send security reports to:

```text
admin@qector.store
```

Use the subject:

```text
[SECURITY] QECTOR Decoder disclosure
```

Include:

```text
1. Affected component or file path
2. Affected version or commit SHA
3. Operating system and Python/Rust versions
4. Reproduction steps
5. Expected impact
6. Whether the issue affects local library use, REST, gRPC, Docker, CUDA, OpenCL, benchmark artifacts, or licensing/security metadata
7. Any proof of concept, kept private
```

Do not publish a public proof of concept before a fix or mitigation is available.

## Scope

In scope:

```text
Rust/PyO3 extension memory safety issues
Python package security issues
Unsafe deserialization or file handling
REST/gRPC/MCP service exposure risks
Docker/runtime hardening issues
CUDA/OpenCL driver-loading issues
Benchmark artifact tampering or misleading security-sensitive metadata
Dependency vulnerabilities in the default build path
```

Out of scope unless it creates a concrete vulnerability:

```text
Performance disagreements
Scientific claim disputes
Expected crashes from intentionally malformed local research inputs
Licensing disagreements
Unsupported GPU driver installation problems
Issues requiring already-compromised administrator access without additional impact
```

## Expected response

For serious private reports, the maintainer will attempt to acknowledge receipt within a reasonable timeframe. Resolution time depends on severity, reproducibility, commercial-license obligations, and whether the issue affects only optional service layers or the default local library path.

## Deployment guidance

The default public install path is a local CPU-safe source build. It is not a hardened internet service.

Before exposing REST, gRPC, MCP, Docker, CUDA, or OpenCL functionality in a production or customer-facing environment, review:

```text
docs/SECURITY_DEPLOYMENT.md
```

Commercial SaaS, hosted API, OEM, or embedded deployments require a separate written commercial agreement.

## Dependency and SBOM guidance

For reproducible security review, record:

```text
git rev-parse HEAD
cargo metadata --format-version 1
python -m pip freeze
python -m pip list
```

Optional audit tools:

```text
cargo audit
python -m pip install pip-audit
python -m pip_audit
```

These tools are advisory; they do not replace source review or deployment hardening.

## License boundary

Security reporting, issue filing, source review, or contribution does not grant commercial use rights. QECTOR remains source-available proprietary software. See `LICENSE` and `COMMERCIAL.md`.
