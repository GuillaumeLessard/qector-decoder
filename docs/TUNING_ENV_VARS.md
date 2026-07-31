# Tunable Environment Variables

This document lists environment variables that affect decoder performance,
accuracy, and hardware selection.  Most users do not need to set any of these.

## Decoder tuning

| Variable | Default | Description |
|---|---|---|
| `QECTOR_BLOSSOM_K_MULT` | `2.0` | Candidate-neighbour multiplier for sparse MWPM. `k = max(12, ceil(mult * sqrt(n_defects)))`. Higher values increase accuracy (more candidates) at the cost of latency. `2.0` is the tuned minimum that preserves exact-MWPM parity at d >= 15. |
| `QECTOR_BLOSSOM_INTRA_PAR` | auto (enabled >= 64 nodes) | Intra-decode parallelism for candidate discovery. `0` disables, `1` forces. Bit-identical output either way. |
| `QECTOR_BLOSSOM_INTRA_THREADS` | unset (Rayon global pool) | Dedicated thread pool size for the candidate-discovery phase. Performance only. |

## Hardware selection

| Variable | Default | Description |
|---|---|---|
| `QECTOR_CUDA_DEVICE_ID` | `0` | Which CUDA device the native batch and BP-OSD decoders bind to. |
| `QECTOR_OPENCL_DEVICE_ALLOW` | unset (any device) | Comma-separated substrings matched against OpenCL device names (e.g. `nvidia,geforce`). Can change decoder output on non-bit-identical devices. |
| `QECTOR_ENABLE_OPENCL_AUTO` | unset (disabled) | `1` to enable auto-routing to OpenCL backend. |
| `QECTOR_DISABLE_OPENCL` | unset | `1` to skip OpenCL probing entirely. |

## License & security

| Variable | Default | Description |
|---|---|---|
| `QECTOR_LICENSE_KEY` | unset | Ed25519-signed license key (`QECT-PRO-*` or `QECT-ENT-*`). |
| `QECTOR_LICENSE_FILE` | unset | Path to a file containing the key. Used when `QECTOR_LICENSE_KEY` is unset/empty; a UTF-8 BOM and trailing newline are stripped. A path that is set but unreadable is reported as an *invalid* key rather than silently falling back to Community. |
| _(implicit)_ | `~/.qector/license.key` | Read when neither variable above is set. Same format and parsing as `QECTOR_LICENSE_FILE`. |
| `QECTOR_ENFORCE` | `0` | `1` raises `PermissionError` when tier limits are exceeded. |
| `QECTOR_CRL_PATH` | `~/.qector/revoked.txt` | Override path for the Certificate Revocation List. |
| `QECTOR_SILENT` | `0` | `1` suppresses the startup licensing notice. |
| `QECTOR_DEBUG` | `0` | `1` enables debug token bypass (`"academic"`, `"commercial"`). Never set in production. |
| `QECTOR_LICENSE_PRIVATE_KEY_B64` | unset | Base64 Ed25519 private key for token signing (fulfillment server only). |

## Stripe metered billing

| Variable | Default | Description |
|---|---|---|
| `STRIPE_SECRET_KEY` | unset | Stripe API secret key for metered billing. |
| `STRIPE_WEBHOOK_SECRET` | unset | Stripe webhook signing secret. |
| `QECTOR_STRIPE_CUSTOMER_ID` | unset | Fallback customer ID for Rust-side flush. |

## Other

| Variable | Default | Description |
|---|---|---|
| `QECTOR_OPENCL_PROBE_TIMEOUT` | `10` (s) | Timeout for the OpenCL health-check subprocess. |
| `QECTOR_LABS_BOT_WEBHOOK_URL` | unset | Discord/Slack webhook for license-fulfillment notifications. |
| `QECTOR_BENCH_BPOSD_DEADLINE_MS` | `50` (ms) | Per-decode deadline for BP-OSD in benchmark scripts. |
| `QECTOR_REST_HOST` / `QECTOR_REST_PORT` | `127.0.0.1:8000` | REST API bind address (Docker override). |
