# Security Policy

QECTOR Decoder is primarily a local Rust/Python research and validation library. However, the repository and optional builds contain service-facing components (e.g., REST, gRPC, MCP, metrics, Docker packaging, CUDA/OpenCL loading). These optional components should be treated as higher-risk deployment surfaces and require appropriate hardening.

## Supported Versions

| Version | Security Support |
| :--- | :--- |
| `0.6.x` (Current: `0.6.6`) | **Active.** Supported for source-available review, private vulnerability disclosure, and commercial-license security discussion. |
| `0.5.x` | **Legacy.** Supported only for private vulnerability disclosure and commercial-license security discussion. |
| Older snapshots | Best-effort only. No guaranteed patches. |

## Report a Vulnerability Privately

We take security seriously. If you discover a vulnerability, please report it privately to allow time for a fix before public disclosure.

Send security reports to:
```text
admin@qector.store
```

Use the following subject line:
```text
[SECURITY] QECTOR Decoder disclosure
```

Please include the following information in your report:
1. Affected component or file path.
2. Affected version or commit SHA.
3. Operating system and Python/Rust versions.
4. Clear reproduction steps.
5. Expected impact and severity.
6. Whether the issue affects local library use, REST, gRPC, Docker, CUDA, OpenCL, benchmark artifacts, or licensing/security metadata.
7. Any proof of concept (must be kept strictly private).

**Do not publish a public proof of concept or exploit before a fix or mitigation is available and coordinated with the maintainer.**

## Scope

**In Scope:**
- Rust/PyO3 extension memory safety issues (e.g., buffer overflows, use-after-free).
- Python package security issues (e.g., unsafe deserialization, path traversal).
- REST/gRPC/MCP service exposure risks (e.g., lack of authentication, injection).
- Docker/runtime hardening issues or container escape vectors.
- CUDA/OpenCL driver-loading or memory isolation issues.
- Benchmark artifact tampering or misleading security-sensitive metadata.
- Dependency vulnerabilities in the default build path.

**Out of Scope** (unless it creates a concrete, exploitable vulnerability):
- Performance disagreements or benchmark disputes.
- Scientific claim or algorithmic accuracy disputes.
- Expected crashes from intentionally malformed local research inputs (e.g., invalid DEM files).
- Licensing disagreements or requests for free commercial use.
- Unsupported GPU driver installation problems.
- Issues requiring already-compromised administrator/root access without additional impact.

## Expected Response

For serious private reports, the maintainer will attempt to acknowledge receipt within 3–5 business days. Resolution time depends on severity, reproducibility, commercial-license obligations, and whether the issue affects only optional service layers or the default local library path. 

*Note: Full security review of the proprietary Rust core is available exclusively under a Non-Disclosure Agreement (NDA) or Commercial License.*

## Deployment Guidance

The default public install path (via PyPI) is a local, CPU-safe binary. It is **not** a hardened internet-facing service.

Before exposing REST, gRPC, MCP, Docker, CUDA, or OpenCL functionality in a production, multi-tenant, or customer-facing environment, you must review:
```text
docs/SECURITY_DEPLOYMENT.md
```
*(If this file is not present in your checkout, contact admin@qector.store for the enterprise deployment guide.)*

Commercial SaaS, hosted API, OEM, or embedded deployments require a separate, written commercial agreement.

## Dependency and SBOM Guidance

For reproducible security review and Software Bill of Materials (SBOM) generation, record your environment state:

```bash
git rev-parse HEAD
cargo metadata --format-version 1
python -m pip freeze
python -m pip list
```

Optional local audit tools (advisory only):
```bash
cargo audit
python -m pip install pip-audit
python -m pip_audit
```
*These tools are advisory; they do not replace formal source review or deployment hardening.*

## License Boundary

**Security reporting, issue filing, source review, or contribution does not grant commercial use rights.** 

QECTOR remains source-available proprietary software. All commercial, institutional, and lab use requires a paid license. See `LICENSE` and `COMMERCIAL.md` for full terms. Provenance and chain of title are protected via timestamped archival.
```
Save this as `SECURITY.md` in the root of your repository. GitHub will automatically detect it and display a "Security" tab on your repo, signaling to enterprise users that you take this seriously.
