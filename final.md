# v1.0.0 Final Release TODO — QECTOR Decoder v3
Generated: 2026-08-05 from workspace Clinque du Batiment clone
Alt clone: Clinique du Batiment (thin, 932 bytes mcp_server) - stale
Verification harness: Desktop/v0.7.1

This file is single source of truth before tagging v1.0.0, push, publish 15 wheels.

## 0) Status snapshot
- Cargo.toml 1.0.0, pyproject.toml 1.0.0, CITATION.cff/codemeta.json 1.0.0, dist wheel 1.0.0 (2859033 bytes 2026-08-05)
- rust_core.sha256 = 7a0593... STALE vs local src/*.rs after SEC-02 hardening
- Trust-boundary re-applied locally but not yet packed:
  src/mcp_server.rs parse_syndrome L333-354 binary-only >1 -> -32602 SEC-02 DONE
  src/mcp_server.rs batch L976-1006 same DONE
  src/grpc_server.rs single L426-435 reject 2/256 DONE
  src/grpc_server.rs batch L515-523 DONE
  rest_api.py _SyndromeValueError DONE (tracked)
- Missing .py sources only .pyc in __pycache__:
  test_mcp_adversarial.py 68204 bytes, test_independent_mcp_cli.py 14772, test_explicit_api_protocols.py 16736, test_mcp_syndrome_validation.py placeholder
- Explicit-API gap: __init__.py 20 wrappers without decode_correction/decode_observables, bposd.py L151, colour_code.py L374, belief_matching.py L321 HAVE them
- Temp files polluting scripts/: _tmp_search.py 4273, _search_tmp.py 894, _decompile_explicit.py 733, _fix_ruff_*.py
- docs/STABLE_API.md L210-224 all [ ] unchecked
- docs/unreleased_audit.md version mismatch fallback_version 0.7.1 vs wheel 1.0.0, test_version_is_071 -> 100

## 1) Trust-boundary hardening SEC-02
.gitignore src/* !src/lib.rs => public repo only lib.rs. Core via 12 secrets RUST_SRC_B64_1..12. Old secrets revert binary check, allowing [0,2,0,0] as valid u8.

File | Line | Fix | Status | Pack?
mcp_server.rs:parse_syndrome 333 any>1 -> Err must be binary | DONE | YES
mcp_server.rs:batch 976 same syndromes_flat | DONE | YES
grpc_server.rs:decode 426 req.syndrome any>1 before as u8 | DONE | YES
grpc_server.rs:batch_decode 515 same | DONE | YES
rest_api.py _decode_impl ~140 max>1 _SyndromeValueError | DONE | NO (py tracked)

Action:
 python scripts/pack_rust_core.py pack --out .secrets
 python scripts/pack_rust_core.py verify --in .secrets
 for i in 1..12: gh secret set RUST_SRC_B64_$i < .secrets/...
 git add rust_core.sha256; rm -rf .secrets
