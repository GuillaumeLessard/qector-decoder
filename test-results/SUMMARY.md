# QECTOR test results

Captured 2026-07-26 · Windows 11 · rustc 1.96.0 · Python 3.11 · NVIDIA GTX 1660 Ti

| Gate | Result |
|---|---|
| `cargo test --lib` | **203 passed, 0 failed** (incl. 4 CUDA tests on real hardware) |
| `cargo clippy --lib --all-targets` | **0 warnings** |
| `pytest python/tests -q` | **1237 passed, 1 skipped, 0 failed** (16m25s) |
| `tools/check_token_compat.py` | **CROSS-COMPAT: PASS** (7 cases, incl. byte-identity on v2) |
| `twine check` | PASSED |

## Four failures fixed in this run

Previous state was `4 failed, 1233 passed`. All four were pre-existing (confirmed
by re-running against a stashed tree — identical failure set with and without the
session's changes).

1. `test_release_import.py::test_2_offline_ed25519_verification`
2. `test_release_import.py::test_5_environment_license_check`
   — signed with a throwaway key, then asserted the token verified against the
   **production** public key compiled into the package. Unsatisfiable; the
   verifier was right to reject. Added a `throwaway_key` fixture that swaps the
   public half, keeping the real crypto path under test with no dependency on
   the production signing secret.

3. `test_comprehensive_suite.py::test_decoderpool_basic`
   — the spawn-capability probe used `apply_async(lambda: 42)`. A lambda is
   unpicklable under spawn, so it raised `AttributeError`, which `except
   RuntimeError` did not catch: the probe could neither succeed nor skip.
   Replaced with a module-level task and a wider except.

4. `test_comprehensive_suite.py::test_belief_matching_decode`
   — **a real product bug.** `BeliefMatching.from_numpy_h` built
   `hyper_obs = zeros((0, nH))`, and `decode` returns `hyper_obs @ hard`, so
   **every syndrome returned an empty array**: a decoder built from a raw
   parity-check matrix was silently useless. `edge_obs` had the same defect,
   breaking the matching fallback for precisely the syndromes BP found hardest.
   Both now carry qubit incidence, so `decode` returns a length-`n_qubits`
   correction. Verified faithful on all 16 repetition-code d=5 syndromes.
