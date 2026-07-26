# Releasing to PyPI

## Wheels only — never publish an sdist

`pip install qector-decoder-v3` must resolve to a **wheel**. An sdist for this
project is unbuildable by design and publishing one breaks installs.

`.gitignore:5` is `src/*` with `!src/lib.rs` — the proprietary Rust core is not
tracked. `maturin sdist` determines file membership from git, so it honours that
ignore rule and `MANIFEST.in`'s `recursive-include src *.rs` never takes effect.
The resulting tarball carries `Cargo.toml` and `build.rs` but **no `.rs`
sources**, so `cargo build` inside it fails.

Consequence if uploaded: on any platform without a matching wheel, pip falls back
to the sdist, attempts a source build, and fails with a confusing Rust error
instead of a clean "no wheel available". PyPI currently holds 15 wheels and zero
sdists for 0.6.8 — keep it that way.

Verify before every upload:

```bash
ls dist/                       # *.whl only
python -m twine check dist/*
```

## Feature flags in published wheels

`Cargo.toml` defaults to `["opencl", "cuda"]`, but the release workflow builds with:

```
--release --no-default-features --features cuda --out dist
```

so published wheels ship **CUDA support, not OpenCL**. This is deliberate: `ocl`
needs an OpenCL import library at link time, which is not present on the CI
runners. Both backends load their driver at *runtime* via `libloading`/`ocl`, so
a CUDA-enabled wheel still installs and runs on a machine with no GPU —
`cuda_is_available()` simply returns `False`.

`[tool.maturin] features = ["pyo3/extension-module"]` in `pyproject.toml` is
additive to the workflow's `--features`, not a replacement.

If you build locally with plain `maturin develop`, cargo's defaults apply and you
get **both** CUDA and OpenCL — which is why a locally built extension can expose
`OpenCLBatchDecoder` while a PyPI wheel does not.

## Pre-flight checklist

```bash
cargo test --lib                      # expect 203 passed, 0 failed
cargo clippy --lib --all-targets      # expect 0 warnings
python -m pytest python/tests -q      # see test-results/pytest.txt
```

Then, from the repo that owns the fulfilment worker:

```bash
python tools/check_token_compat.py    # CROSS-COMPAT: PASS
```

That gate is load-bearing: it proves a licence minted by the Cloudflare worker is
byte-identical to `license.py::create_license_token_v2`. If the token format ever
drifts, every paying customer receives a token their installed package rejects,
and nothing else in the stack would catch it.

## Version and changelog

Bump in **both** `Cargo.toml` and `pyproject.toml` — they are separate fields and
a mismatch ships a wheel whose `__version__` disagrees with its metadata.

Move the `## [Unreleased]` block in `CHANGELOG.md` to the new version heading.

## Licence-token rollout

`LICENSE_TOKEN_VERSION` in the worker stays `"legacy"` until a release containing
`license._verify_v2` is on PyPI **and** customers have had time to upgrade. A v2
token fails closed on older installs — the safe direction — but flipping early
still breaks everyone who has not updated.
