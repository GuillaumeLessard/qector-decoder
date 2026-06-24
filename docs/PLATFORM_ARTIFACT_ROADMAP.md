# Platform Artifact Roadmap

This roadmap turns the external audit recommendations into explicit public work items. It does not create new benchmark claims by itself.

## Current public artifact status

| Area | Current status | Claim boundary |
|---|---|---|
| Windows source build | Verified from fresh clone on a second Windows PC with Python 3.11 | Public install path is real for that environment |
| Python tests | Local validation report says 832 passed | Treat as local validation unless CI run artifact is cited |
| Rust tests | Local validation report says 87 passed | Treat as local validation unless CI run artifact is cited |
| d=13/d=15 LER parity | Checked-in Windows JSON artifact | Safe as scoped artifact evidence, not universal proof |
| Belief-matching selected workload | Checked-in Windows JSON artifact | Safe as selected-workload lower-observed-LER evidence, not universal superiority |
| GPU bit-identity | Tested configurations and docs | Safe as bit-identity claim, not throughput claim |
| GPU throughput | Not yet a stable public proof asset | Do not claim speedup without regenerated local artifacts |
| Linux/macOS benchmark corpus | Not yet published as comparable artifacts | Do not imply cross-platform benchmark equivalence |
| Prebuilt wheels | Not yet published as release assets | Source build remains the public path |

## Planned artifact classes

### 1. CI run artifacts

Goal: make test status externally legible.

Target outputs:

```text
GitHub Actions badge
pytest summary artifact
cargo test summary artifact
benchmark smoke JSON/CSV artifact
Docker build result
commit SHA attached to each run
```

Do not claim latest CI is green unless the GitHub Actions run is visible and passed for the commit being discussed.

### 2. Cross-platform benchmark corpus

Goal: separate machine-specific evidence from portable conclusions.

Target platforms:

```text
Windows 10/11 + Python 3.11/3.12
Ubuntu LTS + Python 3.11/3.12
macOS arm64 + Python 3.11/3.12
CUDA NVIDIA workstation if available
OpenCL CPU/GPU runtime if available
```

Minimum metadata per artifact:

```text
git commit
working tree clean/dirty state
OS and kernel/build version
CPU model
GPU model and driver/runtime, if applicable
Python version
Rust version
Cargo.lock hash
pip freeze
benchmark command
seed
shot count
batch size
raw JSON/CSV output
SHA-256 hash
safe claim wording
unsafe claim wording
```

### 3. Prebuilt wheels

Goal: reduce install friction once release automation and license packaging are ready.

Candidate wheel targets:

```text
Windows x86_64 CPython 3.10/3.11/3.12 CPU-safe
Linux x86_64 CPython 3.10/3.11/3.12 CPU-safe
macOS arm64 CPython 3.10/3.11/3.12 CPU-safe
```

Constraints:

```text
Do not publish GPU-enabled wheels until CUDA/OpenCL runtime assumptions are documented.
Do not publish commercial-use language that conflicts with LICENSE and COMMERCIAL.md.
Attach license and source-available terms to every release.
Keep source build path documented even if wheels become available.
```

### 4. SBOM and dependency audit outputs

Goal: make security review reproducible.

Suggested generated files:

```text
artifacts/cargo-metadata.json
artifacts/cargo-tree.txt
artifacts/pip-freeze.txt
artifacts/pip-list.json
artifacts/cargo-audit.txt
artifacts/pip-audit.txt
```

These are advisory review artifacts, not guarantees of absence of vulnerabilities.

## Release gate before stronger public claims

Before changing the website or README to stronger claims, require:

```text
[ ] Clean git commit or release tag.
[ ] Fresh clone install succeeds.
[ ] Python smoke import succeeds.
[ ] Relevant tests pass.
[ ] Relevant benchmark artifact regenerated.
[ ] Artifact hash recorded.
[ ] Environment captured.
[ ] Safe wording added.
[ ] Unsafe wording added.
[ ] No conflict with commercial/license boundary.
```

## Safe public roadmap wording

Safe:

```text
QECTOR is expanding its public artifact corpus toward CI-run evidence, cross-platform benchmark reports, SBOM-style dependency inventories, and prebuilt CPU-safe wheels.
```

Unsafe:

```text
QECTOR already has universal cross-platform benchmark proof.
QECTOR already ships production wheels for every platform.
QECTOR GPU throughput is generally faster than PyMatching.
```
