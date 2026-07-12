# RELEASE_PROCEDURE.md — QECTOR Decoder v3

**Audience: any LLM/agent (or human) doing dev work, version bumps, or
releases in this repository.** Read this in full before touching version
numbers, `src/`, GitHub secrets, or git tags. It exists because every
mistake described below has actually happened in this repo at least once.

Companion to `AGENTS.md` (general dev conventions, coding style, testing).
This file is specifically about **the release/push/publish pipeline** —
the part with irreversible consequences (PyPI publishes cannot be deleted
or overwritten; GitHub secrets, once wrong, silently break every build).

---

## 0. The one rule that matters most

**`src/*.rs` in this working checkout is the real, proprietary Rust source.
It must never be committed directly to git on `main`.**

This repo's public GitHub history does NOT contain the real Rust source in
plain form. Instead:

- The real source lives only in this local checkout (and wherever else the
  operator keeps it) and as three base64-encoded, chunked GitHub Actions
  secrets: `RUST_SRC_B64_1`, `RUST_SRC_B64_2`, `RUST_SRC_B64_3`.
- CI (`.github/workflows/CI.yml`) reassembles and decodes these secrets at
  build time (`Inject Rust source` step) into `src/` before compiling.
- `git status` in this checkout will normally show `src/lib.rs` and other
  `src/*.rs` files as modified or untracked. **This is expected and correct.**
  Do not `git add src/` and do not include `src/` in release commits.

If you ever need to update the source the secrets decode to, see §5
("Updating the injected Rust source"). Never treat a modified/untracked
`src/` as something to silently stage in a routine commit.

---

## 1. Repo / environment facts (verified, not assumed)

- Local checkout root (Windows): `C:\Users\Admin\Desktop\qector-build-rc-0.6.2`
- Remote: `https://github.com/GuillaumeLessard/qector-decoder.git`, default
  branch `main` (an earlier stray `master` branch existed and was deleted —
  if you ever see a `master` branch again, do not assume it's authoritative;
  verify against `main` first).
- PyPI project: `qector-decoder-v3` (https://pypi.org/project/qector-decoder-v3/)
- Publish mechanism: OIDC Trusted Publisher via `pypa/gh-action-pypi-publish`
  — **no API token is stored or needed.** Publishing is gated on:
  - tag matching `v*`
  - `github.repository == 'GuillaumeLessard/qector-decoder'` (forks can't publish)
  - not a `pull_request` event
  - all of `linux-x86_64`, `windows-x64`, `macos-arm` jobs succeeding first
    (`needs:` in CI.yml) — `macos-intel` is present but disabled
    (`if: false`, long queue times) and is NOT required for publish.
- Build flags actually used in CI (`CI.yml` → `env.MATURIN_ARGS`):
  `--release --no-default-features --features cuda --out dist`
  — **OpenCL is NOT compiled into public wheels.** CUDA is present as a
  feature but loads at runtime via `libloading`/NVRTC, so no CUDA toolkit
  is needed on CI runners. Any code path that unconditionally assumes an
  OpenCL symbol exists on the compiled module will crash on import for
  every user (this exact bug shipped in v0.6.5 — see CHANGELOG).
- Local dev builds (`maturin develop`) commonly use **default features**,
  which include both `opencl` and `cuda`. This means a bug gated behind
  "only present without the opencl feature" will not reproduce locally
  under default `maturin develop` — you must rebuild with the exact CI
  flags to catch it (§6).

---

## 2. Version numbers — every location, no exceptions

A version bump touches **all** of the following. Miss one and you ship an
inconsistent release (this has happened — a stale `0.6.4` citation block
survived two version bumps before being caught).

| File | What to change |
|---|---|
| `pyproject.toml` | `[project] version = "X.Y.Z"` |
| `Cargo.toml` | `[package] version = "X.Y.Z"` |
| `Cargo.lock` | Auto-updates when you run `cargo check`/`cargo build` after bumping `Cargo.toml`. Verify the `qector_decoder_v3` package entry specifically — do NOT hand-edit; do not confuse it with unrelated third-party crate versions in the same lockfile (e.g. `rand_core 0.6.4` is unrelated and must not change). |
| `python/qector_decoder_v3/__init__.py` | `__fallback_version__ = "X.Y.Z"` — **this is a fallback only**, used if the compiled extension's own `__version__` can't be read. Never make this overwrite the real compiled `__version__`; the code should keep reporting the built value until the wheel is actually rebuilt. |
| `CITATION.cff` | `version: X.Y.Z` |
| `codemeta.json` | `"version":"X.Y.Z"` (single-line minified JSON in this repo — use exact string match, don't reformat the whole file) |
| `README.md` | Citation/BibTeX block (`version = {X.Y.Z}`) + add a new `## New in vX.Y.Z` section ahead of the previous one |
| `PYPI_README.md` | Citation/BibTeX block (`version = {X.Y.Z}`) + add a new `## vX.Y.Z Highlights` section ahead of the previous one |
| `CHANGELOG.md` | New `## [X.Y.Z] - YYYY-MM-DD` entry at the top (below `## [Unreleased]`), following Keep a Changelog format already in use |

### Verifying you got them all

Before committing, grep every version-bearing file for the OLD version
string and confirm zero hits outside of historical changelog entries
(which should of course still mention old versions in their own sections):

```powershell
Select-String -Path README.md,PYPI_README.md,CHANGELOG.md,CITATION.cff,codemeta.json,pyproject.toml,Cargo.toml,Cargo.lock -Pattern "OLD\.VERSION\.HERE"
```

Then grep for the NEW version and confirm it appears in every file from
the table above:

```powershell
Select-String -Path README.md,PYPI_README.md,CHANGELOG.md,CITATION.cff,codemeta.json,pyproject.toml,Cargo.toml,Cargo.lock -Pattern "NEW\.VERSION\.HERE"
```

Read the actual diff before committing, not just the file list:

```powershell
git diff -- pyproject.toml Cargo.toml CITATION.cff codemeta.json Cargo.lock python/qector_decoder_v3/__init__.py README.md PYPI_README.md CHANGELOG.md
```

---

## 3. Staging and committing — the safe pattern

**Never `git add -A` or `git add .` in this repo.** The working tree
routinely contains real proprietary `src/*.rs` files (modified or
untracked) plus scratch/debug files that should never be committed
(`*.ps1` one-off scripts, `mypy_*.txt`, `pytest_*.txt`, `ruff_*.txt`,
`cargo_check_*.txt`, `.venv*/`, `benchmark_*.py` ad hoc scripts, etc.).
Blind `-A` staging has caused real incidents in this repo's history.

Correct pattern — stage exactly the files you intend to change, by name:

```powershell
cd "C:\Users\Admin\Desktop\qector-build-rc-0.6.2"
git status --short                     # review everything first
git add pyproject.toml Cargo.toml Cargo.lock CITATION.cff codemeta.json `
        python/qector_decoder_v3/__init__.py README.md PYPI_README.md CHANGELOG.md
git status --short                     # confirm ONLY intended files show staged (M in col 1)
                                        # confirm src/lib.rs is NOT staged
```

If `src/lib.rs` or other `src/*.rs` files show as staged, unstage them
immediately: `git restore --staged src/`. This should not normally happen
if you only `git add` explicit filenames, but always verify.

Write a commit message that documents **root cause and verification**, not
just "bump version" — future sessions (including future you, and future
LLM agents with no memory of this session) need to know *why*, not just
*what*:

```powershell
git commit -m "vX.Y.Z: <one-line summary>" -m "<paragraph: root cause, what was verified, and how>"
```

Push to `main` directly (this repo does not use a PR-gated release
branch model for routine releases):

```powershell
git push origin main
```

---

## 4. Verify CI is green on `main` BEFORE tagging

Tagging triggers the real publish pipeline. Never tag on a red or unverified
`main`. Use the GitHub CLI to watch the exact commit's runs:

```powershell
gh run list --branch main --limit 5
gh run watch <tests-run-id> --exit-status
gh run watch <ci-run-id> --exit-status
```

Confirm in the output:
- `tests` workflow: `ruff-and-mypy`, `smoke-import-py3.9`...`py3.13` all pass
  (the `docker` job is informational only — `continue-on-error: true` — a
  red docker job is expected/harmless when `src/` in the checkout is a stub
  or doesn't match the Dockerfile's expectations; do not treat it as a
  release blocker)
- `CI` workflow: all 15 real matrix jobs (5 Python versions × Linux/Windows,
  plus 5 × macOS ARM) show ✓ Complete job. `macos-x86_64-py${{ matrix.python-version }}`
  will show as **skipped**, not failed — this is the disabled Intel-macOS
  job (`if: false` in CI.yml, pre-existing, not something to fix as part
  of a release). A skip here is normal; a failure anywhere else is not.

Do not proceed to tagging on partial/incomplete runs. Wait for full completion.

---

## 5. Updating the injected Rust source (only when `src/` actually changed)

If your release includes real Rust changes (new decoder, bugfix inside
`src/*.rs`), the GitHub secrets must be regenerated from the verified-correct
local source **before** the CI build that will use them:

1. Confirm the local `src/` tree compiles and passes tests first:
   ```powershell
   cargo check --no-default-features --features cuda   # match CI's exact flags
   cargo test
   ```
2. Round-trip-verify the encoding BEFORE uploading anything — encode, then
   decode back locally, and diff against the original byte-for-byte. Never
   upload a secret you haven't verified decodes to exactly the source you
   intended. A prior incident in this repo involved uploading a stub/wrong
   `src/lib.rs` this way, which silently broke every subsequent build.
3. Chunk the base64 into three parts sized to stay under GitHub's per-secret
   size limit, and upload as `RUST_SRC_B64_1`, `RUST_SRC_B64_2`, `RUST_SRC_B64_3`
   via `gh secret set <name> < chunk_file` (or the GitHub UI).
4. After uploading, trigger a fresh CI run (push a commit or re-run an
   existing workflow) and manually confirm the "Inject Rust source" step
   in the Actions log actually extracted the expected files — check the
   file count and spot-check one distinctive file (e.g. confirm a specific
   new function name grep-matches inside the extracted `src/lib.rs` in the
   Actions log, or that `cargo check` succeeds in that job).
5. Only after confirming a full green CI run against the NEW secrets should
   you proceed to tagging.

**Never assume a secret upload worked. Verify by observing the next real
CI run use it successfully**, not by trusting the upload command's exit code.

---

## 6. Tagging and publishing to PyPI

**This step is irreversible.** PyPI does not allow re-uploading or deleting
a version once published — a broken release stays broken forever at that
version number; the only fix is shipping a new, higher version number
(this happened with v0.6.5 → v0.6.6). Do not tag speculatively "to see
what happens."

Before tagging:
- [ ] All version numbers updated and verified (§2)
- [ ] Docs (`README.md`, `PYPI_README.md`, `CHANGELOG.md`) updated and re-read
      in full, not just diffed (read the actual rendered section — stale
      surrounding prose or broken markdown tables are easy to miss in a diff)
- [ ] Committed and pushed to `main`
- [ ] CI fully green on that exact commit (§4)
- [ ] **Check whether a tag with this version already exists** — if you're
      re-doing a release (e.g. after discovering the version was already
      tagged against a stale/wrong commit), delete and recreate deliberately:
  ```powershell
  git tag -l "vX.Y.Z"                          # check local
  git ls-remote --tags origin | Select-String "vX.Y.Z"   # check remote
  # if it exists and points at the wrong commit:
  git tag -d vX.Y.Z
  git push origin :refs/tags/vX.Y.Z            # delete remote tag
  ```

Create and push the tag against the exact commit you verified in §4:

```powershell
git tag vX.Y.Z <commit-sha>          # or omit sha to tag HEAD, but verify HEAD first
git push origin vX.Y.Z
```

Watch the tag-triggered run through to completion, including the publish job:

```powershell
gh run list --branch main --limit 3
gh run watch <tag-triggered-ci-run-id> --exit-status
```

Confirm in the output that `publish-to-pypi` actually ran (not skipped)
and succeeded. Then verify independently, from a fresh shell, that PyPI
actually has it — don't trust the Actions log alone, and don't trust
`pypi.org`'s JSON API's `info.version` field immediately after (it lags
behind actual file availability due to CDN caching):

```powershell
pip index versions qector-decoder-v3          # or check the Files page directly:
# https://pypi.org/project/qector-decoder-v3/#files
```

---

## 7. Post-publish: install and test the REAL published artifact

**Never consider a release verified based on local dev builds, editable
installs, or `maturin develop` output.** You must install the actual thing
a real user would get from `pip install`.

1. Uninstall any local/editable install first, and confirm it's actually gone:
   ```powershell
   cd "C:\Users\Admin\Desktop\qector-build-rc-0.6.2"
   .\.venv\Scripts\Activate.ps1
   pip uninstall qector-decoder-v3 -y
   python -c "import qector_decoder_v3"   # MUST raise ModuleNotFoundError
   ```
2. Create a **fresh, separate, throwaway venv** — not `.venv` — with no
   cache, so nothing local leaks in:
   ```powershell
   python -m venv .venv_pypi_test
   .\.venv_pypi_test\Scripts\Activate.ps1
   python -m pip install --upgrade pip --quiet
   pip install qector-decoder-v3==X.Y.Z --no-cache-dir
   ```
3. Confirm what actually got installed — look for the absence of an
   "Editable project location" line, and confirm the `Location` path
   points inside the throwaway venv, not this checkout:
   ```powershell
   pip show qector-decoder-v3
   ```
4. Actually run code against it — at minimum: import (catches the exact
   class of bug that broke v0.6.5), GPU-availability checks (these run
   after import and were previously unreachable), and one real decode
   through the public API with real array shapes, not just `assert True`.
   Write this as a disposable script, run it, delete it after (see §8) —
   don't leave ad hoc verification scripts littering the repo root.
5. Only after this passes is the release actually done. "CI is green" and
   "PyPI shows the version" are necessary but not sufficient — a release
   that imports-fails for every user (as v0.6.5 did) can still pass CI,
   because CI's own smoke-import test in `tests.yml` runs on Linux only
   and may not exercise every feature-flag combination a real user hits.

---

## 8. Cleanup discipline

- Delete throwaway venvs (`.venv_pypi_test`, `.venv_clean_test`, etc.) and
  ad hoc verification scripts once you're done with them. They are not
  tracked in git (see `.gitignore`), but they clutter the working directory
  and can confuse the next session (human or LLM) about what's real vs.
  scratch.
- Never delete `todo.md`, `todo2.md`, `howto.md`, or other operator-authored
  planning docs at repo root unless explicitly asked — these are working
  notes, not scratch output, even though they sit alongside actual scratch
  files.
- If you're unsure whether a root-level file is scratch or intentional,
  ask, or check `git log --follow -- <file>` / whether it's tracked at all
  (`git ls-files <file>`) before deleting.

---

## 9. Quick reference — full release checklist

```
[ ] Bump version in all 8 locations (§2), verify with grep
[ ] Update README.md, PYPI_README.md, CHANGELOG.md — read full rendered sections
[ ] git status --short — review everything, confirm no src/ surprises
[ ] git add <explicit filenames only>              -- never -A / .
[ ] git status --short — confirm ONLY intended files staged
[ ] git commit -m "vX.Y.Z: ..." -m "<root cause + verification detail>"
[ ] git push origin main
[ ] gh run watch <tests-run> --exit-status
[ ] gh run watch <ci-run> --exit-status           -- confirm all real jobs green
[ ] (if src/ changed) regenerate + verify RUST_SRC_B64_1/2/3, confirm next
    CI run actually used the new secret successfully
[ ] Check no stale/wrong tag already exists for this version
[ ] git tag vX.Y.Z && git push origin vX.Y.Z
[ ] gh run watch <tag-triggered-run> --exit-status -- confirm publish-to-pypi ran + succeeded
[ ] pip index versions qector-decoder-v3           -- confirm live independent of Actions log
[ ] Uninstall local install, confirm ModuleNotFoundError
[ ] Fresh throwaway venv, pip install <pkg>==X.Y.Z --no-cache-dir
[ ] pip show <pkg> -- confirm no "Editable project location", right venv path
[ ] Run real code against it: import + GPU checks + one real decode
[ ] Delete throwaway venv + ad hoc test script
[ ] Done -- only now is the release actually verified
```

---

*This document describes process, not code. If the actual workflow files
(`.github/workflows/CI.yml`, `tests.yml`) change, re-verify this document
against them rather than trusting it as historical truth — it was last
verified against the live workflow files on 2026-07-12.*
