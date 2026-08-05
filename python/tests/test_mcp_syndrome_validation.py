"""Tests for MCP syndrome length and binary validation (trust boundary SEC-02).

The MCP server must reject syndromes:
  * whose length does not match the decoder's expected number of checks
    (the number of entries in check_to_qubits) -- otherwise the decoder
    would silently decode a different syndrome than the caller intended.
  * whose values are not binary (0/1) -- a uint8 like 2 or 9 parses fine in
    JSON but is not a valid detector outcome.  Previously this was a trust-
    boundary gap that could yield SIGABRT from out-of-bounds slice indexing.

Every rejected request must return JSON-RPC -32602 (Invalid params) with a
human-readable message naming the offending validation, and the worker must
stay alive for subsequent legitimate requests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import pytest

import qector_decoder_v3 as q

SERVER_CODE = "import qector_decoder_v3 as qd; qd.run_mcp_server()"
TIMEOUT = 20


def _has_mcp() -> bool:
    return hasattr(q, "run_mcp_server") and callable(q.run_mcp_server)

class _McpStdio:
    """Minimal line-based stdio JSON-RPC client with read timeouts."""

    def __init__(self, extra_env=None):
        env = dict(os.environ)
        env["QECTOR_SILENT"] = "1"
        if extra_env:
            env.update(extra_env)
        self.proc = subprocess.Popen(
            [sys.executable, "-c", SERVER_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def _read_line(self, timeout=TIMEOUT):
        buf = []

        def _r():
            try:
                buf.append(self.proc.stdout.readline())
            except Exception:
                buf.append("")

        t = threading.Thread(target=_r, daemon=True)
        t.start()
        t.join(timeout)
        return buf[0] if buf else None

    def send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None, rid=1):
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        for _ in range(8):
            line = self._read_line()
            if line is None:
                return None
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("id") == rid:
                return resp
        return None

    def ping_ok(self):
        resp = self.request("ping", {}, rid=900)
        assert resp is not None, "server died: no ping response"
        assert resp.get("result") is not None, f"ping not answered: {resp}"
        return True

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

@pytest.fixture(scope="module")
def server():
    if not _has_mcp():
        pytest.skip("run_mcp_server not available (grpc feature required)")
    client = _McpStdio()
    yield client
    client.close()


_RING3_C2Q = [[0, 1], [1, 2], [2, 0]]
_RING3_NQ = 3


@pytest.fixture(scope="module")
def ring_code():
    return _RING3_C2Q, _RING3_NQ


class TestMcpSyndromeValidation:
    """Focused syndrome-length and binary-value checks on the MCP stdio server."""

    def test_run_mcp_server_is_callable(self):
        assert callable(q.run_mcp_server)
        return False


    def test_syndrome_too_short_rejected(self, server, ring_code):
        c2q, nq = ring_code
        resp = server.request(
            "tools/call",
            {
                "name": "decode_syndrome",
                "arguments": {
                    "check_to_qubits": c2q,
                    "n_qubits": nq,
                    "syndrome": [1, 0],
                },
            },
            rid=11,
        )
        assert resp is not None
        err = resp.get("error", {})
        assert err.get("code") == -32602, f"expected -32602, got: {resp}"
        assert "syndrome length" in err.get("message", "")
        assert server.ping_ok()


    def test_syndrome_too_long_rejected(self, server, ring_code):
        c2q, nq = ring_code
        resp = server.request(
            "tools/call",
            {
                "name": "decode_syndrome",
                "arguments": {
                    "check_to_qubits": c2q,
                    "n_qubits": nq,
                    "syndrome": [1, 0, 1, 0, 1],
                },
            },
            rid=12,
        )
        assert resp is not None
        err = resp.get("error", {})
        assert err.get("code") == -32602, f"expected -32602, got: {resp}"
        assert "syndrome length" in err.get("message", "")
        assert server.ping_ok()



    def test_syndrome_empty_rejected(self, server, ring_code):
        c2q, nq = ring_code
        resp = server.request(
            "tools/call",
            {
                "name": "decode_syndrome",
                "arguments": {
                    "check_to_qubits": c2q,
                    "n_qubits": nq,
                    "syndrome": [],
                },
            },
            rid=13,
        )
        assert resp is not None
        err = resp.get("error", {})
        assert err.get("code") == -32602
        assert "syndrome length" in err.get("message", "")
        assert server.ping_ok()


    def test_syndrome_non_binary_rejected(self, server, ring_code):
        c2q, nq = ring_code
        resp = server.request(
            "tools/call",
            {
                "name": "decode_syndrome",
                "arguments": {
                    "check_to_qubits": c2q,
                    "n_qubits": nq,
                    "syndrome": [1, 2, 3],  # non-binary values
                },
            },
            rid=14,
        )
        assert resp is not None
        err = resp.get("error", {})
        assert err.get("code") == -32602, f"expected -32602, got: {resp}"
        assert "binary" in err.get("message", "").lower(), f"expected 'binary' in error message: {err}"
        assert server.ping_ok()


    def test_syndrome_negative_rejected(self, server, ring_code):
        c2q, nq = ring_code
        resp = server.request(
            "tools/call",
            {
                "name": "decode_syndrome",
                "arguments": {
                    "check_to_qubits": c2q,
                    "n_qubits": nq,
                    "syndrome": [1, -1, 0],  # negative value
                },
            },
            rid=15,
        )
        assert resp is not None
        err = resp.get("error", {})
        assert err.get("code") == -32602, f"expected -32602, got: {resp}"
        assert "binary" in err.get("message", "").lower()
        assert server.ping_ok()


    def test_valid_binary_syndrome_accepted(self, server, ring_code):
        """Valid binary syndrome should succeed (not hang or crash)."""
        c2q, nq = ring_code
        resp = server.request(
            "tools/call",
            {
                "name": "decode_syndrome",
                "arguments": {
                    "check_to_qubits": c2q,
                    "n_qubits": nq,
                    "syndrome": [1, 0, 1],
                },
            },
            rid=16,
        )
        # Should not be an error response
        assert resp is not None
        assert "error" not in resp or resp.get("error") is None
        assert server.ping_ok()
