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

## Refresh the Rust core secrets after ANY `src/*.rs` change

`src/*` is gitignored, so a change to the core reaches CI and the wheels only
through the `RUST_SRC_B64_*` Actions secrets. Editing `src/` and pushing does
**nothing** on its own: the build keeps compiling whatever the secrets last held.
This has bitten the project — a fix once sat local-only for four hours while
every CI run reported green against older source.

```bash
python scripts/pack_rust_core.py pack                 # writes .secrets/ + rust_core.sha256
for i in $(seq 1 12); do
  gh secret set "RUST_SRC_B64_$i" < ".secrets/RUST_SRC_B64_$i.txt"
done
rm -rf .secrets
git add rust_core.sha256 && git commit          # MUST be committed with the change
```

`rust_core.sha256` is the tracked digest of the packed core; it is the only
record of the core's identity that travels with a commit. The
`stale-secrets-check` workflow restores the secrets and runs
`pack_rust_core.py check-manifest` against it, so **forgetting the upload now
fails CI** instead of passing silently. Do not "fix" a red check-manifest by
regenerating the manifest alone — that just re-points the anchor at the drift.
Upload the secrets.

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

### OpenCL distribution decision (v1, dev2todo §3.2)

**Decision: no OpenCL side-wheel for v1.** PyPI wheels ship CUDA only (as above);
OpenCL stays a documented source build:

```bash
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features opencl
.\.venv\Scripts\python.exe -m pytest python/tests/test_opencl_cpu_bit_identical.py -q --tb=short
```

A `qector-decoder-v3-opencl` side-wheel or an OpenCL leg in `release-build.yml`
is deferred *unless* user demand for non-NVIDIA devices materializes (GitHub
issue count / support requests). Revisit at the next minor release.

## Pre-flight checklist

```bash
cargo test --no-default-features        # expect 303 passed, 0 failed
cargo clippy --no-default-features --all-targets   # expect 0 warnings
ruff check python/                      # expect 0 errors
ruff format --check python/             # expect no reformatting needed
dev.bat python -m pytest python/tests -q           # see test-results/pytest.txt
```

**Run pytest through `dev.bat`, never bare.** `dev.bat` exports the Enterprise token that
unlocks the GPU paths. A bare `python -m pytest python/tests` fails 58 CUDA tests
(`test_max_capacity.py::TestGPUCapacity::test_cuda_large_batch[*]` and
`test_syndrome_faithfulness.py::test_cuda_bit_identical_and_faithful[*]`) purely on licensing,
which looks like a broken tree and is not one. Confirm the unlock first with
`dev.bat python scripts\_verify_enterprise_unlock.py` — expect
`PASS: Enterprise/GPU unlock is confirmed working.`

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
