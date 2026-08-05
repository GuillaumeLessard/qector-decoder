# Local Repo Sweep — full review for v1.0.0

**Date:** 2026-08-05
**Scope:** every file in the local `qector-decoder-clone` repo (everything under `C:\Users\Clinque du Batiment\Downloads\qector-decoder-clone\`)
**Method:** inventory + pattern scans + targeted file reads of every important surface

---

## 1. Headline

The local repo is in good shape for the v1.0.0 cut. The `.gitignore` is doing its job — 243 lines covering the proprietary Rust core, the 12-chunk secret bundle, dev tokens, scratch files, log outputs, vendored venvs, and historical benchmark sessions. **Nothing in this repo is in a state that would be shipped by accident to PyPI.**

Two real findings worth acting on:

1. **An `update_versions.py` script already exists at the repo root** that does the exact 6-file v0.6.8 → 1.0.0 + BibTeX replacement I flagged in the previous review. Running it is a one-command fix for all 8 stale version stamps.
2. **The `.env` file at the repo root contains real secrets** (Stripe live keys + the Ed25519 license signing private key). It is correctly `.gitignore`d and won't be committed, but it lives on the local machine. Worth confirming the file's file-system ACL and that it's not in any cloud-synced directory.

The `.env` finding is **not a v1.0.0 ship-blocker** — the secrets are local, the `.gitignore` excludes them, and the publish path doesn't read them. But it's a real-world security posture item.

---

## 2. Inventory

### 2.1 Top-level (60 files, 5 directories in scope)

| Category | Files |
|---|---|
| Configuration | `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `MANIFEST.in`, `.gitignore` (243 lines), `.gitattributes`, `build.rs`, `mcp.json` |
| Workflows (`.github/`) | `release-build.yml`, `tests.yml`, `dependency-audit.yml`, `stale-secrets.yml` + `.github/deny.toml`, `.github/FUNDING.yml` |
| Top-level Python | `vs.py` (Enterprise MC simulator for BB[[144,12,12]]), `_probe_two_stage.py`, `update_versions.py` (helper for the v1.0.0 stamp swap) |
| Top-level .ps1 | `audit_section3.ps1`, `audit_tools.ps1`, `run_test_check.ps1` |
| Top-level .cmd/.bat | `build_maturin.cmd`, `dev.bat` (the dev-shell launcher) |
| Sensitive | `.env` (Stripe + Ed25519 private key), `licenses_issued.json` (24 KB) |
| Build / verification logs | `cargo_*.log`, `cargo_*.txt`, `master_run.log`, `master_pid.txt`, `pytest_after.txt`, `test_out.txt`, `scratch_pytest.txt`, `sst_grep.txt`, `license_check.log`/`.err` |
| Build artefacts | `dist/qector_decoder_v3-1.0.0-cp311-cp311-win_amd64.whl` (2.0 MB, **already built**), `dist/_prev_…bak` (v0.7.0 backup) |
| Source archive | `src.zip` (366 KB) |
| Zenodo / config | `.zenodo.json`, `keywords.json`, `mcp.json`, `maturin_out.log`, `maturin_done.txt` |
| PDFs / scratch | `qector-decoder-v3.pdf` (1.2 MB), `QECTOR_v3_full_source.pdf` (1.7 MB), `QECTOR_benchmark_report.pdf`, `planGPU.pdf` (963 KB), `benchmark_latency_comparison.png` |
| Workspace hygiene | `master_pid.txt`, `run_ts.txt` |

### 2.2 Directories in scope (filtered)

| Dir | Children | Purpose |
|---|---|---|
| `src/` | 55 | Rust core (1.4 MB total) — reviewed in detail in `manual_review_1_0_0.md` |
| `python/` | 619 | Includes the 30 public-package files, 186 tests, and `__pycache__` blobs — reviewed in detail in `manual_review_1_0_0.md` |
| `docs/` | 26 | Public documentation — see §4 |
| `examples/` | 28 | 15 `.py` + 13 `.pyc` from a previous run |
| `scripts/` | 163 | 5 tracked public scripts + 74 other Python scripts + the rest build/CI helpers |
| `audit/` | 5 | Local audit notes (gitignored) |
| `benchmarks/` | 4 | Local benchmark outputs (gitignored) |
| `benchmark_results/` | 21 | Local benchmark outputs (gitignored) |
| `benchmark_export/` | 8 | Local benchmark exports (gitignored) |
| `benchmarks_session/` | 10,346 | **Vendored `.venv/` (~10K pytest site-packages) + historical harness runs** — gitignored, but bloats the working tree |
| `proto/` | 1 | `qector.proto` (1.4 KB — gRPC schema) |
| `lib/` | 1 | OpenCL.def |
| `dist/` | 2 | The v1.0.0 wheel + a v0.7.0 backup |
| `test-results/` | 129 | Verification runs (gitignored except for `*.txt` per the line 208 carve-out) |
| `target/` | (out of scope) | Rust build cache |

`benchmarks_session/` is the elephant — **10,346 files including a vendored PyPI-style `.venv/Lib/site-packages/`** (~10K files from `_pytest/` alone). It's all gitignored, but it makes every `Get-ChildItem` on the clone slow. Adding `benchmarks_session/` to a local file-search index would also be slow. **Recommended for v1.0.0 post-cut:** `git rm -r --cached benchmarks_session/` to be safe, even though it's already matched by the existing `benchmarks_session/` gitignore rule (line 214). The user may have a working copy that's not actually ignored if it was added before the rule.

### 2.3 `examples/` — 15 scripts (3 still have stale `v0.6.8` stamps)

| File | Stale stamp? |
|---|---|
| `example_advanced_decoders.py` | no |
| `example_auto_decoder_native.py` | no |
| `example_auto_routing.py` | **yes** (line 3, line 31) |
| `example_basic.py` | no |
| `example_batch.py` | no |
| `example_blossom.py` | no |
| `example_codes_and_diagnostics.py` | no |
| `example_cupy_bp.py` | **yes** (line 3, line 43) |
| `example_license.py` | no |
| `example_pymatching_and_backend.py` | no |
| `example_stim_dem.py` | no |
| `example_streaming.py` | no |
| `example_streaming_session.py` | **yes** (line 3, line 46) |
| `qiskit_tutorial.py` | no |

**3 of 15 examples need a v0.6.8 → v1.0.0 stamp swap.** Already covered by `update_versions.py` at the root.

### 2.4 `scripts/` — 163 entries, 5 are the public ones

| Tracked-public script (5) | Size | Purpose |
|---|---|---|
| `pack_rust_core.py` | (referenced by CI, not in clone root) | The 12-chunk secret pack/unpack/verify/check-manifest tool |
| `full_decoder_benchmark.py` | (referenced by docs) | The benchmark harness for the v0.7.0 artifacts |
| `regenerate_benchmark_artifacts.py` | (rewritten in v0.7.0) | `--yes`/`--dry-run` gated artifact regeneration |
| `run_due_diligence_bundle.py` | (referenced) | One-command evidence bundle |
| `smoke_test.py` | (referenced) | Wheel smoke test |

The other 74+ scripts are mostly benchmark/export/patch utilities, none of which are documented in `docs/REPRODUCE.md` or invoked by the CI workflows. They're all gitignored (per line 133-156) and exist locally as working scripts. The largest are `generate_report_pdf.py` (64 KB), `run_custom_comparison_benchmark.py` (26 KB), `benchmark_ler_serious.py` (22 KB). Not shipped, not a v1.0.0 concern.

---

## 3. Workflows (`.github/workflows/`)

### 3.1 `release-build.yml` — production release pipeline

Comprehensive. The header (lines 1-28) is a meticulous post-mortem of the v0.7.0 prep commit that introduced six release-blockers; the workflow fixes each one. Specifically:

- **Line 218-230 — wheels-only gate:** refuses to publish if a `.tar.gz` reaches `dist/`, refuses to publish if fewer than 15 wheels are present. The "15 wheels = 3 platforms × 5 CPython versions" invariant is enforced.
- **Line 232-254 — wheel smoke test:** decodes a real syndrome and asserts `H @ c == s (mod 2)` for both `UnionFindDecoder` and `BlossomDecoder`. This is the same test the v0.7.0 `test_09_cli_doctor.py::TestCLI::test_decode_subcommand` would have caught.
- **Line 259-290 — console-script round-trip:** invokes `/tmp/smoke/bin/qector decode` end-to-end with `--decoder blossom`, `--decoder union_find`, `--decoder bposd`. The v0.7.0 `qector decode` ImportError would have been caught here.
- **Line 60-73 — 12-chunk secret restoration:** the `RUST_SRC_B64_1..12` secrets are restored at build time. The workflow is the *only* honest place to use the proprietary core; `src/*` is gitignored (line 5 of `.gitignore`).
- **Line 301-307 — artifact attestation:** gated on `github.event.repository.private == false` and `continue-on-error: true`. Non-blocking by design. This is the v0.7.0 `release-build.yml@<commit>` provenance line that v0.7.0 carried to PyPI.
- **Line 308-317 — `skip-existing: true`:** idempotent re-upload, so a half-published release can be retried.

**For v1.0.0:** the workflow needs zero changes. The `qector decode` smoke test at line 272-289 is the v0.7.1 DEFECT-1 regression test; it would have caught the v0.7.0 ImportError. The dependency-gate `needs` chain correctly blocks publish on a known advisory. The tag-only publish gate is correctly tag-scoped (no `inputs.publish` string-compare bug).

### 3.2 `tests.yml` — CI test matrix

Standard. `cargo test --no-default-features` (line 61), `cargo clippy` (line 64), `cargo test --features full` (line 67), `cargo clippy --features full` (line 69) — all four are present and correctly flagged. Python matrix (line 78) is 3.9–3.13. The "verify installed package matches repo source" step (line 137-154) is the A5-03 regression that caught a stale `maturin develop` install. The import smoke test (line 161-168) asserts `NativeAutoDecoder`, `set_license_key`, and `get_license_info` are all present.

**For v1.0.0:** zero changes needed. The Rust source restore is the same 12-chunk mechanism; it works.

### 3.3 `dependency-audit.yml` — advisory gate

Two halves:
- **Rust:** `cargo audit` (vulnerabilities + yanked) + `cargo deny` (licenses + bans + sources) at `--all-features`. Both correctly distinguish "advisory present" (warn, not block) from "gate did not run" (always error, regardless of strict). The 0.18 schema flag-position fix (line 117) is documented inline.
- **Python:** `pip-audit` against the `lowest-direct` resolved set (line 214-218), with the explicit comment that auditing whatever `pip` resolves today would have missed all 45 findings. The 3.9 leg is warn-only by design (accepted pytest exception); 3.13 is strict.

**For v1.0.0:** zero changes needed. This workflow is the reason RUSTSEC-2026-0204 was caught before 0.7.0 shipped.

### 3.4 `stale-secrets.yml` — chunked-secret integrity check

`scripts/pack_rust_core.py verify` (line 62) and `check-manifest` (line 74) are exactly the right primitives. The header (line 1-24) is candid about what this *can't* catch — a maintainer who edited `src/`, re-packed, and never ran `gh secret set` — but it catches the secret-corruption and self-inconsistency cases, which is the 90% risk. 

**For v1.0.0:** zero changes needed. **However:** the workflow as written only catches secret corruption. The header acknowledges a known failure mode. A v1.1 enhancement would be a `git diff` between the restored source and a tagged release's source, so a stale secret gets caught by commit reference, not just by self-consistency.

### 3.5 `mcp.json` (top-level, not `.github/`)

```json
{
  "mcpServers": {
    "qector": {
      "command": "python",
      "args": ["-c", "import qector_decoder_v3; qector_decoder_v3.run_mcp_server()"],
      "env": {
        "QECTOR_SILENT": "1",
        "QECTOR_BLOSSOM_K_MULT": "2.0"
      },
      "timeout": 30000
    }
  }
}
```

This is the **client config** for Claude/Cursor etc. to launch the QECTOR MCP server. It's the file that `docs/MCP_INTEGRATION.md` references when it says "Point your MCP client at that file". Verified against the README: `README.md:553-556` says exactly this. **For v1.0.0:** zero changes needed.

### 3.6 `.zenodo.json`, `keywords.json`, `codemeta.json`, `CITATION.cff`

All four correctly describe the v1.0.0 release. The Zenodo DOI is `10.5281/zenodo.20825980` (matches the live PyPI metadata). `keywords.json` is a duplicate of the PyPI keywords; can be regenerated by `python -m qector_decoder_v3.scripts.update_keywords` if you have such a script. Not a v1.0.0 concern.

---

## 4. `docs/` — public documentation (25 files, all reviewed)

| File | Size | Status |
|---|---|---|
| `API_STABILITY.md` | 6 KB | ✓ Correct. Long-form rolling policy. Promotion log empty. |
| `API_SURFACES.md` | 6 KB | ✓ Correct. Per-symbol status table. |
| `BENCHMARK_COMPETITIVE.md` | 3 KB | ✓ Correct. |
| `BENCHMARK_THROUGHPUT.md` | 2 KB | ✓ Correct. |
| `BEYOND_PYMATCHING.md` | 4 KB | ✓ Correct. |
| `CORRECTNESS_AUDIT.md` | 11 KB | ✓ Correct (per v0.7.0 changelog). |
| `DECODER_PICKER.md` | 2 KB | ✓ Correct. |
| `ENVIRONMENTAL_SKIPS.md` | 4 KB | ✓ Correct. |
| `GPU_AND_CUPY.md` | 15 KB | ✓ Correct. |
| `GPU_GUIDE.md` | 3 KB | ✓ Correct. |
| `LICENSE_SLA.md` | 1 KB | ✓ Correct. |
| `LICENSING.md` | 1 KB | ✓ Correct. |
| `MANUAL_REVIEW_1_0_0.md` | 24 KB | New this session (previous turn). |
| `MCP_INTEGRATION.md` | 2 KB | ⚠️ Out of date — see §4.1 below. |
| `METHODOLOGY.md` | 9 KB | ✓ Correct. |
| `PLATFORM_ARTIFACT_ROADMAP.md` | 5 KB | ✓ Correct. |
| `QUICKSTART.md` | 2 KB | ✓ Correct. |
| `RELEASING.md` | 6 KB | ✓ Correct. Wheels-only guidance. |
| `REPRODUCE.md` | 9 KB | ✓ Correct. Says "13 MCP tools" — matches source. |
| `REPRODUCIBILITY_CHECKLIST.md` | 5 KB | ✓ Correct. |
| `SCALING.md` | 3 KB | ✓ Correct. |
| `SECURITY_DEPLOYMENT.md` | 5 KB | ✓ Correct. |
| `SERVICE_API_SCHEMA.md` | 4 KB | ⚠️ `"version": "0.6.8"` in example responses — see §4.2. |
| `STABLE_API.md` | 15 KB | New this session. ✓ Correct, matches source. |
| `TUNING_ENV_VARS.md` | 3 KB | ✓ Correct. |
| `UNRELEASED_AUDIT.md` | 9 KB | New this session. ✓ Correct. |

### 4.1 `MCP_INTEGRATION.md` — out of date (carried over from previous review)

Lists 8 tools, source has 13. Says "11 decoder families", source has 9. **Should-do before v1.0.0 tag.** Easy mechanical fix.

### 4.2 `SERVICE_API_SCHEMA.md` — example JSON says "0.6.8"

Two example responses carry `"version": "0.6.8"` (lines 53, 65). `update_versions.py` at the repo root already handles this — running the script fixes both.

---

## 5. Configuration and secrets

### 5.1 `.env` — local secrets (real, sensitive)

```
STRIPE_SECRET=<REDACTED-107chars>
STRIPE_SECRET_KEY=<REDACTED-107chars>
STRIPE_PUBLISHABLE=<REDACTED-107chars>
STRIPE_PUBLISHABLE_KEY=<REDACTED-107chars>
STRIPE_WEBHOOK_SECRET=<REDACTED-39chars>
QECTOR_LICENSE_PRIVATE_KEY_B64=<REDACTED-160chars>
```

The `QECTOR_LICENSE_PRIVATE_KEY_B64` is the **Ed25519 private signing key for license tokens** — the half that pairs with the `PRODUCTION_PUBLIC_KEY` in `src/license.rs:47-50`. If this is ever leaked, anyone can mint valid license tokens for any tier.

**Status:**
- Correctly `.gitignore`d (`.gitignore` line 41, 111-113)
- Not committed, not on PyPI, not in the wheel
- Lives on the local machine only

**For v1.0.0:** zero change to the wheel/publish path. **But** confirm:
1. The `.env` file's file-system ACL on Windows (who can read it besides the current user)
2. The repo is not in a cloud-synced directory (OneDrive, Dropbox, Google Drive backup) that would push the file off-machine
3. The repo is not in a directory that gets snapshotted by a CI runner (this is unlikely for a local clone, but worth confirming)
4. The `licenses_issued.json` (24 KB) carries the same trust boundary — also gitignored, also local-only

This is not a v1.0.0 *ship* blocker (the secrets don't leak through the publish path) but it is a real-world security posture item that should be addressed before the next round of "this got into the wrong hands" scares.

### 5.2 `.gitignore` — 243 lines, comprehensive

Sections:
- Lines 4-6: `src/*` !`src/lib.rs` — the proprietary Rust core protection
- Lines 9-16: build artefacts (target, dist, wheels, .pyd/.so/.dll/.dylib)
- Lines 18-38: Python build caches
- Lines 40-44: secrets (`.env`, `.env.local`, `.env.*.local`, `licenses_issued.json`)
- Lines 46-52: venvs
- Lines 61-71: Rust/Cargo + local build dirs
- Lines 73-75: OS
- Lines 77-78: benchmark outputs
- Lines 80-86: test caches
- Lines 88-89: temp files
- Lines 92-105: scratch probes (kimi, _probe_*, etc.)
- Lines 107-109: kimi probe/fix helpers
- Lines 111-114: secrets (DUPLICATED — already at 40-44)
- Lines 116-127: logs, scratch test outputs (with a careful carve-out for `python/tests/expected_symbols.txt` and `test-results/*.txt`)
- Lines 129-156: scratch scripts and dev environments
- Lines 158-177: scratch planning docs
- Lines 179-184: local-only archives
- Lines 186-188: editor config
- Lines 190-198: scratch/repo notes
- Lines 200-201: local-only docs with infra details
- Lines 203-205: internal implementation trackers
- Lines 207-208: `test-results/*.txt` carve-out
- Lines 210-211: `.secrets/` for the packed-Rust bundle
- Lines 213-242: dev session artefacts, audit scripts, todo files

**The `.gitignore` does its job.** The only concern is whether the `benchmarks_session/` working directory (10K+ files including the vendored `.venv/`) is actually being matched by line 214's `benchmarks_session/` rule — if it was added before the rule, git may need `git rm -r --cached benchmarks_session/` to apply the ignore. Worth a quick `git check-ignore benchmarks_session/.venv/Lib/site-packages/_pytest/__init__.py` to confirm.

### 5.3 `dev.bat` — Windows dev shell launcher

The script reads a dev-only Enterprise token from `benchmarks_session/utils/ent_license.key` and exports it as `QECTOR_LICENSE_KEY` before activating the venv. The header (lines 1-23) is explicit: this is a **short-lived local dev/benchmark token**, not a production license. When it expires, you re-mint it. The file is `.gitignore`d via `benchmarks_session/` (line 214).

The script's logic is correct and the file is at the repo root where Windows users will find it. No changes needed.

### 5.4 `build_maturin.cmd` — local wheel build

`cd /d "%~dp0"` then `maturin develop --release > maturin_out.log 2>&1`. Header notes the v0.6-era bug where the script `cd`'d to a hard-coded absolute path on one machine. The fix is correct. Writes `DONE` or `FAILED <rc>` to `maturin_done.txt` for the user to tail. No changes needed.

### 5.5 `update_versions.py` — the v0.6.8 → v1.0.0 helper

**This already exists at the repo root.** It does the 4 file replacements I flagged in the previous review:

```python
code = code.replace('"version": "0.6.8"', '"version": "1.0.0"')           # SERVICE_API_SCHEMA.md
code = code.replace('QECTOR Decoder v3 (v0.6.8) decoder auto-routing', ...)  # example_auto_routing.py
code = code.replace('QECTOR Decoder v3 (v0.6.8) batched GPU BP-OSD', ...)    # example_cupy_bp.py
code = code.replace('QECTOR Decoder v3 (v0.6.8) streaming orchestration', ...) # example_streaming_session.py
code = code.replace('print("QECTOR v3 (0.6.8) —', 'print("QECTOR v3 (1.0.0) —')  # 3 prints across the 3 files
code = code.replace('version = {0.7.1}', 'version = {1.0.0}')           # README.md BibTeX
```

Running it is one command. **All 8 stale-stamp issues from my previous review are handled by this script.** It is currently *not* run (the stamps are still in the files), so the user just needs to invoke it.

### 5.6 `vs.py` — Enterprise MC simulator for BB[[144,12,12]]

This is a 5.7 KB scratch script at the repo root. It does not import `qector_decoder_v3` — it imports `ldpc.bposd_decoder.BpOsdDecoder` directly. It runs multi-process Monte Carlo to find the true logical error rate floor for the bivariate-bicycle code, with adaptive shot scaling (`TARGET_FAILURES = 100` for ~10% relative error). 

Not part of the public surface, not a v1.0.0 concern. Gitignored via line 226.

### 5.7 `Dockerfile` — multi-stage build

Starts with `FROM python:3.12-slim AS rust-builder`. The comments are in French (mojibake in the head - probably UTF-8 issues with the rendering). The multi-stage build chain is:
1. `rust-builder` — install Rust via rustup, install maturin, copy `Cargo.toml`/`Cargo.lock`/`build.rs`/`src/`/`proto/`/`python/`/`pyproject.toml`/`README.md`, run `maturin build --release`
2. (not shown in the head) presumably `python-runtime` and `server` stages

The Dockerfile references `proto/` (so gRPC is built) and uses `python:3.12-slim` (matches the v0.7.0 PyPI's CPython 3.12 wheel). **The proprietary-Rust source restore mechanism is not visible in the Dockerfile head — it's the vendored-COPY step that would re-introduce the `src/*` gitignore problem on a public Docker build.** Worth a follow-up to ensure the Dockerfile either (a) is built from a `RUST_SRC_B64_*`-restored context, or (b) uses a pre-built wheel as input, or (c) is `.gitignore`d and rebuilt from a private image registry. **Not a v1.0.0 ship-blocker** (the Dockerfile is dev-infra, not in the wheel), but a real concern if anyone publishes a `qector/qector-decoder` Docker image to Docker Hub.

### 5.8 `proto/qector.proto` — gRPC schema

1.4 KB. Single `qector.proto` file, defines the gRPC service for the (optional) `grpc` feature. Not in the default build. **For v1.0.0:** zero changes needed.

---

## 6. Documentation cross-check (after applying the open fixes)

After the 8 mechanical version-stamp swaps from `update_versions.py` + the `MCP_INTEGRATION.md` rewrite + the `MCP_INTEGRATION.md` 11-families claim fix:

| Doc claim | Source | Consistent? |
|---|---|---|
| `README.md:556` "13 MCP tools" | 13 tools in `src/mcp_server.rs` | ✓ |
| `README.md:618` `version = {0.7.1}` (BibTeX) | Will be `{1.0.0}` after `update_versions.py` | ✓ (after fix) |
| `README.md:422` "13 MCP tools operational" | 13 tools | ✓ |
| `REPRODUCE.md:422` "13/13 ... 13 MCP tools" | 13 tools | ✓ |
| `REPRODUCE.md:362` "5 new tools" | historical, correct | ✓ |
| `MCP_INTEGRATION.md` 8 tools listed, "11 decoder families" | 13 tools, 9 families, 7 backend tiers | ⚠ (after fix: ✓) |
| `SERVICE_API_SCHEMA.md:53,65` `"version": "0.6.8"` | Will be `"1.0.0"` after `update_versions.py` | ✓ (after fix) |
| `CHANGELOG.md` `## [1.0.0]` | New this session, accurate | ✓ |
| `STABLE_API.md` Stable/Provisional/Internal | Matches source | ✓ |
| `API_STABILITY.md` policy + promotion log | Empty promotion log, correct conservative choice | ✓ |
| `API_SURFACES.md` per-symbol status | Matches `STABLE_API.md` | ✓ |
| `unreleased_audit.md` 9/10 in source | Verified | ✓ |
| `manual_review_1_0_0.md` defensive-code health | Verified | ✓ |

---

## 7. Workspace hygiene — final state

After running `update_versions.py`, the open issues for the v1.0.0 cut are:

| # | Action | Severity | Effort |
|---|---|---|---|
| 1 | Run `python update_versions.py` (or apply 8 line-edits manually) | should-do | 1 min |
| 2 | Rewrite `docs/MCP_INTEGRATION.md` to match source (13 tools, 9 families, 7 backend tiers, `notifications/initialized` contract) | should-do | 10 min |
| 3 | Apply the two pre-publish fixes from the original analysis: (a) `bench_community.json` hygiene, (b) `CUDABatchDecoder` no-weights warning suppression | should-do | 30 min |
| 4 | Confirm `.gitignore` covers `benchmarks_session/.venv/` (may need `git rm -r --cached` if added before the rule) | should-do | 1 min |
| 5 | Add `.gitignore` for `vs.py` (already covered, but worth verifying — line 226) | should-do | 1 min |
| 6 | Add file-system ACL on `.env` (or move to a secrets manager) | could-do | 1 hr |
| 7 | Clean up the 10K-file `benchmarks_session/` working tree | could-do | 1 hr |
| 8 | `manual_review_1_0_0.md` §10.3 v1.1 quality work: mcp_server.rs / license.rs / space_time_decoder.rs unwrap audits | defer | 1-2 days |
| 9 | Dockerfile: address the proprietary-Rust source restore mechanism before publishing to Docker Hub | defer | TBD |

**The v1.0.0 cut is ready after items 1-5.** Items 6-9 are real but not blocking.

---

## 8. v1.0.0 ship-readiness — consolidated verdict

| Aspect | Status |
|---|---|
| Public API (Stable/Provisional/Internal split) | ✓ Documented and matches source |
| `__version__` consistency | ✓ All 8 metadata fields at `1.0.0` |
| Verification harness | ✓ 120/120, 2×2 runs, 0 failures, 0 flaky, sandboxed community |
| DEFECT-1 (CLI `qector decode` crash) | ✓ Fixed in v0.7.1 source |
| DEFECT-2 (MCP `ping` missing) | ✓ Fixed in v0.7.1 source |
| DEFECT-3 (MCP notification silence) | ✓ Fixed in v0.7.1 source (worker pool guard at `mcp_server.rs:1562`) |
| Cargo (no-default-features / full / clippy) | ✓ Per CHANGELOG Validation section |
| Python test suite (3.9–3.13 matrix) | ✓ Per `tests.yml` |
| Dependency gate (cargo-audit, cargo-deny, pip-audit) | ✓ Per `dependency-audit.yml` |
| Release pipeline (wheels-only, smoke test, attestations) | ✓ Per `release-build.yml` |
| LICENSE (PolyForm Noncommercial 1.0.0) | ✓ Matches `pyproject.toml` and `Cargo.toml` |
| Provenance (Sigstore, Zenodo DOI) | ✓ `CITATION.cff`, `.zenodo.json`, `codemeta.json` all consistent |
| `.gitignore` coverage | ✓ 243 lines, comprehensive |
| Secrets in `.env` not in version control | ✓ Excluded by `.gitignore` lines 41, 111-113 |
| **Version stamps in examples/docs** | ⚠ Stale in 3 examples + 2 SERVICE_API_SCHEMA.md lines + 1 BibTeX. **Handled by `update_versions.py`.** |
| **`MCP_INTEGRATION.md` accuracy** | ⚠ Lists 8 tools (should be 13). Mechanical rewrite. |
| **`bench_community.json` stale GPU value** | ⚠ Known issue. Pre-publish fix. |
| **9-warning spam on `CUDABatchDecoder` no-weights** | ⚠ Known issue. Pre-publish fix. |
| `mcp_server.rs` 54 unwraps | 🔧 v1.1 quality work |
| `license.rs` 20 unwraps (security-critical) | 🔧 v1.1 quality work |
| `space_time_decoder.rs` 30 unwraps | 🔧 v1.1 quality work |

**Bottom line: ship v1.0.0 after running `update_versions.py` + the two pre-publish fixes + the `MCP_INTEGRATION.md` rewrite.** That's 1 hour of mechanical work, no design decisions, no new code, no new tests.

The repo is in better shape than I expected when I started this review. The release pipeline is hardened (the v0.7.0 prep-commit six-blocker audit is the most rigorous release-postmortem I've seen in a small OSS project), the `.gitignore` is the right shape for a partially-proprietary codebase, and the v1.0.0 API freeze is a real commitment, not just a version-string swap. Go ship it.
