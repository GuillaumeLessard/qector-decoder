# Test results

Captured 2026-07-25 on Windows 11, rustc 1.96.0, Python 3.11, NVIDIA GTX 1660 Ti.

| File | What it is |
|---|---|
| `cargo-test.txt` | `cargo test --lib` — full Rust suite, incl. the 4 CUDA tests that exercise a real GPU |
| `cargo-clippy.txt` | `cargo clippy --lib --all-targets` |
| `pytest.txt` | `python -m pytest python/tests -q --durations=10` |
| `token-compat.txt` | `tools/check_token_compat.py` — Cloudflare worker vs Python signer, byte-identity |
| `twine-check.txt` | distribution metadata validation |

The token-compat gate is the load-bearing one: it proves a licence minted by the
deployed worker is byte-identical to `license.py::create_license_token_v2`. If
that ever drifts, every paying customer gets a token their install rejects.
