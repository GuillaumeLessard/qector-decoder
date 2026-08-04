"""A6-04: licence-tier enforcement matrix — {tier} x {under cap, over cap} x {surface}.

Design notes (the previous version of this file was red for all of these reasons):

1. **Distances must be real, not labels.** Every enforcement point caps the
   *estimated* distance (`estimate_distance(check_to_qubits, n_qubits)`), not a
   number the caller supplies. Hand-written matrices therefore prove nothing.
   This file uses `py_generate_repetition_code_checks(d)`, for which
   `estimate_distance` returns exactly `d` (utils.rs treats 1D repetition codes
   as `d == n_qubits`), so every cell's distance is anchored.

2. **Tiers cannot be forced with fake key strings.** `get_tier_info()` prefers
   the Rust `LicenseManager`, which requires a properly signed
   `QECT-{TIER}-{payload}.{signature}` and rejects anything else. Setting
   `QECTOR_LICENSE_KEY=QECT-PRO-...` leaves the tier at Community. To exercise
   Pro/Enterprise deterministically we pin the *Python* enforcement layer by
   forcing its native lookup to miss, which activates the documented
   `QECT-PRO-`/`QECT-ENT-` prefix fallback in `license.get_tier_info()`.

3. **gRPC and MCP have no Python decode entry point.** The native module exposes
   `py_enforce_distance_cap` / `py_enforce_unlocked` and `run_mcp_server` only.
   Both servers gate on that same shared primitive (`grpc_server.rs:183-184`,
   `mcp_server.rs:783-784`), so this file asserts the primitive directly and the
   transport wiring is asserted in Rust (`grpc_server.rs::
   test_grpc_distance_cap_rejects_oversized_code`).

Cells: 3 tiers x 2 (under/over) x 4 surfaces = 24.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import qector_decoder_v3 as q
from qector_decoder_v3 import license as lic

# Caps per license.py `_TIER_MAX_DISTANCE`.
TIER_CAPS = {"Community": 7, "Pro": 19, "Enterprise": 63}

# One distance comfortably inside each cap and one clearly outside it.
STRADDLE = {
    "Community": {"under": 5, "over": 9},
    "Pro": {"under": 15, "over": 25},
    "Enterprise": {"under": 45, "over": 90},
}

TIERS = ["Community", "Pro", "Enterprise"]
SIDES = ["under", "over"]

_PREFIX_KEY = {
    "Community": "",
    "Pro": "QECT-PRO-matrixtest",
    "Enterprise": "QECT-ENT-matrixtest",
}


def _layout(d: int):
    """Repetition-code layout whose `estimate_distance` is exactly `d`."""
    c2q, n_qubits = q._native_module.py_generate_repetition_code_checks(d)
    assert q._native_module.py_estimate_distance(c2q, n_qubits) == d, f"anchor broken: estimate_distance != {d}"
    return c2q, n_qubits


@pytest.fixture
def python_tier(monkeypatch):
    """Force the Python enforcement layer onto its documented prefix fallback.

    Yields a setter; call it with a tier name before exercising a surface.
    """
    monkeypatch.setattr(lic, "_native_module", lambda: None)
    monkeypatch.delenv("QECTOR_API_KEY", raising=False)
    monkeypatch.delenv("QECTOR_ENFORCE", raising=False)

    def _set(tier: str) -> None:
        key = _PREFIX_KEY[tier]
        if key:
            monkeypatch.setenv("QECTOR_LICENSE_KEY", key)
        else:
            monkeypatch.delenv("QECTOR_LICENSE_KEY", raising=False)
        info = lic.get_tier_info()
        assert info["tier"] == tier, f"tier pin failed: wanted {tier}, got {info}"
        assert info["max_distance"] == TIER_CAPS[tier]

    return _set


def _expect(tier: str, side: str) -> bool:
    """True when the cell must be rejected."""
    d = STRADDLE[tier][side]
    over = d > TIER_CAPS[tier]
    assert over == (side == "over"), f"straddle table wrong for {tier}/{side}"
    return over


# ---------------------------------------------------------------------------
# Surface 1/4 — Python API (`license.enforce_distance_cap`)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("side", SIDES)
def test_python_api_enforcement(tier, side, python_tier):
    python_tier(tier)
    d = STRADDLE[tier][side]
    if _expect(tier, side):
        with pytest.raises(PermissionError):
            lic.enforce_distance_cap(d)
    else:
        lic.enforce_distance_cap(d)  # must not raise


# ---------------------------------------------------------------------------
# Surface 2/4 — REST `/decode`
# ---------------------------------------------------------------------------
@pytest.fixture
def rest_client():
    rest_api = pytest.importorskip("qector_decoder_v3.rest_api")
    app = rest_api.create_app()
    if rest_api._FRAMEWORK == "fastapi":
        starlette_testclient = pytest.importorskip("fastapi.testclient")
        return starlette_testclient.TestClient(app, raise_server_exceptions=False)
    return app.test_client()


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("side", SIDES)
def test_rest_enforcement(tier, side, python_tier, rest_client):
    python_tier(tier)
    d = STRADDLE[tier][side]
    c2q, n_qubits = _layout(d)
    payload = {
        "check_to_qubits": c2q,
        "n_qubits": n_qubits,
        "syndrome": np.zeros(len(c2q), dtype=np.uint8).tolist(),
    }
    resp = rest_client.post("/decode", json=payload)
    if _expect(tier, side):
        assert resp.status_code == 403, (
            f"{tier}/{side} d={d}: expected 403 from the licence gate, "
            f"got {resp.status_code} {resp.text if hasattr(resp, 'text') else resp.data!r}"
        )
    else:
        assert resp.status_code == 200, (
            f"{tier}/{side} d={d}: expected 200, got {resp.status_code} "
            f"{resp.text if hasattr(resp, 'text') else resp.data!r}"
        )


# ---------------------------------------------------------------------------
# Surface 3/4 — gRPC (shared native primitive, grpc_server.rs:183-184)
# ---------------------------------------------------------------------------
def _native_cap_cell(tier: str, side: str, monkeypatch) -> None:
    """Assert the native cap primitive both servers call.

    Only Community is reachable natively: the Rust LicenseManager requires a
    production-signed token, so Pro/Enterprise cannot be pinned from a test.
    """
    native = getattr(q._native_module, "py_enforce_distance_cap", None)
    if native is None:
        pytest.skip("native py_enforce_distance_cap not built")
    if tier != "Community":
        pytest.skip(
            f"{tier} needs a production-signed QECT-{tier[:3].upper()}- token; the Rust "
            "LicenseManager rejects unsigned keys by design. Transport wiring is "
            "asserted in Rust (grpc_server.rs / mcp_server.rs)."
        )
    # monkeypatch, not os.environ.pop: a raw pop permanently stripped the key
    # from the pytest process, so every later test spawning a licensed
    # subprocess (e.g. test_examples.py's CUDABatchDecoder) failed on
    # Community-tier licensing - an ordering-dependent suite failure.
    monkeypatch.delenv("QECTOR_LICENSE_KEY", raising=False)
    # Popping the variable is not sufficient once the process has resolved a
    # licence. The native LicenseManager latches the first tier it sees and
    # exposes no way back - `set_license_key("")` raises ValueError, and
    # `get_license_info()` keeps reporting the latched tier - so on a machine
    # running under dev.bat (Enterprise token) the cap below simply does not
    # fire and `pytest.raises(PermissionError)` fails.
    #
    # It only reproduced in a specific order: the latch is set by whichever test
    # first calls get_license_info(), which is test_rest_api_auth's authorised
    # /api/license/info request. Alone, this file passed; after that file, it
    # failed. Skip rather than assert something the process can no longer be put
    # into. CI runs unlicensed, so the tier is Community there and these cells
    # execute exactly as before - the coverage that matters is not lost.
    latched = str((q.get_license_info() or {}).get("tier", "Community"))
    if latched != "Community":
        pytest.skip(
            f"process has already latched tier {latched!r}; the native LicenseManager "
            "caches the first licence it resolves and cannot be returned to Community "
            "in-process. Run without QECTOR_LICENSE_KEY (as CI does) to exercise this."
        )
    d = STRADDLE[tier][side]
    if _expect(tier, side):
        with pytest.raises(PermissionError):
            native(d)
    else:
        native(d)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("side", SIDES)
def test_grpc_enforcement(tier, side, monkeypatch):
    _native_cap_cell(tier, side, monkeypatch)


# ---------------------------------------------------------------------------
# Surface 4/4 — MCP (same shared primitive, mcp_server.rs:783-784)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("side", SIDES)
def test_mcp_enforcement(tier, side, monkeypatch):
    _native_cap_cell(tier, side, monkeypatch)


# ---------------------------------------------------------------------------
# Matrix integrity — the straddle table must actually straddle every cap
# ---------------------------------------------------------------------------
def test_matrix_covers_every_cap_boundary():
    assert set(STRADDLE) == set(TIER_CAPS)
    for tier, cap in TIER_CAPS.items():
        assert STRADDLE[tier]["under"] <= cap < STRADDLE[tier]["over"]
