# CLAUDE.md — release pipeline rules (STRICT)

Read this before touching **anything** under `.github/workflows/`, `scripts/pack_rust_core.py`,
`src/*.rs`, `Cargo.toml`, `pyproject.toml`, or `rust_core.sha256`.

---

## 0. Prime directive

The pipeline in `.github/workflows/` is the workflow that actually shipped **v0.6.5 → v0.6.9**
to PyPI. It is field-proven. **Do not modernize, refactor, tidy, restructure, or rewrite it.**

Every release outage in this project's history came from the same cause: someone replaced a
working workflow with an untested rewrite that looked cleaner. Commit `3d332eb` did exactly
this and introduced six separate release-breaking faults at once, none of which were visible
until a tag was pushed.

If you believe the pipeline is wrong, **say so and stop**. Do not fix it unilaterally.

A publish to PyPI is **irreversible**. A version number can never be reused — only yanked.
Treat every change that could reach `publish-to-pypi` as one-way.

---

## 1. Absolute prohibitions

Never do any of the following unless the human explicitly names the rule number and tells you
to break it. "Make CI better", "clean this up", or "modernize the actions" is **not** permission.

| # | Never | Why — verified consequence |
|---|---|---|
| 1 | Publish, build, or restore an sdist job | `.gitignore` line 5 is `src/*` with `!src/lib.rs`. `maturin sdist` reads membership from git, so the tarball ships **no `.rs` sources** and cannot build. Any platform without a wheel then fails on a source build. PyPI currently holds **15 wheels, 0 sdists** — keep it exact. |
| 2 | Reduce the Rust-core restore below 12 chunks, or inline `base64 -d \| tar -xzf` | The core is **46 `.rs` files** and packs to **exactly 12 chunks** at `MAX_CHUNK_BYTES = 30_000`. The pre-0.7.0 3-chunk inline restore (`${PART1}${PART2}${PART3}`) would silently truncate to a quarter of the source and fail the build. Verify with `python scripts/pack_rust_core.py pack --out <tmp>` and count. |
| 3 | Add `\|\| inputs.publish` (or any `workflow_dispatch` input) to the publish gate | A dispatch input arrives as a **string**, and every non-empty string is truthy in GitHub expressions — so `publish=false` evaluated **TRUE**. This made publish live on runs explicitly asked not to publish. The gate must remain ref-based only. |
| 4 | Reintroduce the `SKIP_BUILD` / "secret not set → warn and skip" pattern | The v0.6.9 file downgraded a missing secret to `::warning::` and skipped the wheel build. A release could then report **success while producing zero wheels**. A missing secret must fail loudly. |
| 5 | Reference `get_decoder_info` | It does not exist. It appears only inside a docstring at `python/qector_decoder_v3/__init__.py:2819`. A smoke test calling it fails at import. |
| 6 | Add `musllinux`, Linux `aarch64`, or `macos-13` / macos-intel targets | README documents 15 wheels. musllinux and Linux-aarch64 never built or smoke-tested cleanly; macos-intel **hangs** the runner. |
| 7 | Change the wheel count, or weaken the `-lt 15` guard in `publish` | 3 platforms × 5 CPython (3.9–3.13) = **15**. The guard is the only thing that catches a partially-built matrix before upload. |
| 8 | Make `actions/attest-build-provenance` gating | Provenance needs a public repo or GHAS; on a private repo the API returns "Feature not available". It must stay `continue-on-error: true` behind `if: github.event.repository.private == false`. See commit `49ba957`. |
| 9 | Remove `skip-existing: true` from `pypa/gh-action-pypi-publish` | PyPI rejects re-uploading an existing file. Without this, a run that uploaded 9 of 15 wheels then failed can **never be retried** — the release is stuck half-published forever. |
| 10 | Edit `src/*.rs` without repacking the secrets **and** committing `rust_core.sha256` | `src/*` is gitignored. Editing it and pushing does **nothing** to CI — the build keeps compiling whatever the secrets last held. A fix once sat local-only for four hours while every CI run reported green against stale source. |
| 11 | Delete or disable the `python` job in `tests.yml` | It is the **only** CI that imports the built package. It runs pytest, ruff lint + format, the installed-vs-source `filecmp` gate, and the import smoke test. The wheel jobs only compile and upload; they never import anything. |
| 12 | Remove `concurrency` from `tests.yml` | Without it, four quick pushes put ~20 test jobs in the pool at once, starving `release-build` of runners and making healthy runs look hung at 7%. Do **not** copy it onto the release build — cancelling a half-finished release is worse than paying for it to finish. |
| 13 | Write unindented code inside a YAML `run: \|` block | This exact mistake made `tests.yml` unparseable and **silently disabled every trigger** — 0 jobs, `failure` conclusion, no error anyone noticed. Use `python - <<'PY' … PY`, indented to the block. |
| 14 | Change `--no-default-features --features cuda` | `ocl` needs an OpenCL import library at link time, which CI runners lack. Both backends load drivers at **runtime**, so a CUDA wheel still installs and runs GPU-free — `cuda_is_available()` just returns `False`. |
| 15 | Delete or force-push a branch or tag without archiving it first | Archive as `archive/<name>` and push the tag **before** deleting. `archive/*` matches none of the release triggers, so it fires no CI. |

---

## 2. Mandatory verification before any `.github/workflows/**` change

Run **all** of these. A change that skips them is not permitted.

```bash
# 1. every workflow must parse — a ScannerError silently disables all triggers
python -c "import yaml,glob,sys; [yaml.safe_load(open(f,encoding='utf-8')) for f in glob.glob('.github/workflows/*.yml')]; print('YAML OK')"

# 2. the packed core must still match the working tree
python scripts/pack_rust_core.py check-manifest        # must print OK

# 3. chunk count must not exceed the wired secrets (currently 1..12)
python scripts/pack_rust_core.py pack --out /tmp/_chunks && ls /tmp/_chunks | grep -c RUST_SRC_B64_ && rm -rf /tmp/_chunks

# 4. versions must agree — a mismatch ships a wheel whose __version__ disagrees with its metadata
grep -m1 '^version' Cargo.toml pyproject.toml
```

Then prove it on CI **without** a tag:

```bash
gh workflow run Build --ref main     # builds all 15 wheels; cannot publish (publish needs refs/tags/v*)
```

Never validate a workflow change by pushing a `v*` tag. Use `workflow_dispatch`, or a
`test-*` / `ci-*` tag — both build everything and neither can reach PyPI.

---

## 3. Trigger semantics — memorize before editing `on:`

| Ref pushed | Build wheels | `dependency-gate` | `publish-to-pypi` |
|---|---|---|---|
| `main` / `master` | yes | skipped | skipped |
| `test-*`, `ci-*` tag | yes | skipped | skipped |
| **`v*` tag** | yes | **runs strict** | **PUBLISHES — irreversible** |
| pull request | yes | skipped | skipped |
| `workflow_dispatch` | yes | skipped | skipped |
| `archive/*` tag | no | no | no |

`publish` requires **all** of: `startsWith(github.ref, 'refs/tags/v')`,
`github.repository == 'GuillaumeLessard/qector-decoder'`, and
`github.event_name != 'pull_request'`. It also `needs: [linux-x86_64, windows-x64, macos-arm,
dependency-gate]`. Do not relax any term.

The `pypi` environment has **no protection rules** — nothing will pause a publish for human
approval. The tag is the only gate.

---

## 4. The pipeline, as it must remain

- `release-build.yml` — `linux-x86_64`, `windows-x64`, `macos-arm` (5 CPython each) → `dependency-gate` (tag-only, `strict: true`) → `publish` (tag-only).
- `tests.yml` — `rust` (cargo test/clippy, default + `full`) and `python` (5 versions).
- `dependency-audit.yml` — advisories + licence policy. `workflow_call` declares `strict` (**boolean**, default `false`) and optional `SAFETY_API_KEY`. Strict mode caught **RUSTSEC-2026-0204** before it shipped.
- `stale-secrets.yml` — restores the secrets and runs `check-manifest` against the committed `rust_core.sha256`, so forgetting to upload now fails CI instead of passing silently.

`SAFETY_API_KEY` is intentionally **not** configured; that step no-ops by design and `pip-audit`
is the real gate. Do not "fix" its absence.

Do not "fix" a red `check-manifest` by regenerating the manifest alone — that just re-points the
anchor at the drift. **Upload the secrets.**

---

## 5. Releasing

`docs/RELEASING.md` is authoritative; this file does not replace it. In particular:

- Bump the version in **both** `Cargo.toml` and `pyproject.toml`.
- Move the `## [Unreleased]` / `UNRELEASED` block in `CHANGELOG.md` to the new version heading.
- Refresh `RUST_SRC_B64_*` after **any** `src/*.rs` change and commit `rust_core.sha256` in the
  same change.
- Run `python tools/check_token_compat.py` from the fulfilment-worker repo — expect
  `CROSS-COMPAT: PASS`. This is load-bearing: it proves a licence minted by the Cloudflare
  worker is byte-identical to `license.py::create_license_token_v2`. If the token format drifts,
  **every paying customer receives a token their installed package rejects**, and nothing else
  in the stack catches it.
- `LICENSE_TOKEN_VERSION` in the worker stays `"legacy"` until a release containing
  `license._verify_v2` is on PyPI **and** customers have had time to upgrade.

Do not create a `v*` tag on the user's behalf without an explicit, unambiguous instruction to
release. "It's ready", "v0.7.0 is there", or "ship it" is a cue to run the pre-flight and
**report**, not to tag.

---

## 6. If you break something

Do not force-push over it. Report exactly what changed, cite the run URL, and restore from git
(`git checkout HEAD -- .github/workflows/`). The committed state is the proven state.
