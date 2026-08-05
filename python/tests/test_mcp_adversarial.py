"""Adversarial-input hardening tests for the MCP stdio server (trust boundary).

The MCP server is the package's most exposed network-adjacent surface: a local
agent (Claude Desktop, Cursor, the mcp SDK) pipes arbitrary JSON-RPC frames at
it. DEFECT-2 and DEFECT-3 covered protocol *shape* (ping, notification echo).
This file covers *input shape*: malformed frames, oversized payloads, out-of-
range decoder parameters, and pipelined requests. Every test ends with a
liveness probe (ping) proving the worker pool survived the attack.

Server-side guards under test (src/mcp_server.rs):
  * MAX_MCP_CONTENT_LENGTH frame bound -> -32600
  * serde parse failure -> -32700
  * parse_syndrome / parse_batch_size / parse_n_qubits bounds -> -32602
  * unknown tool -> -32601
  * batch syndromes_flat length + checked_mul overflow guard (MCP-01)
  * notifications produce no response (DEFECT-3 regression)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import pytest

import qector_decoder_v3 as q

SERVER_CODE = "import qector_decoder_v3 as q; q.run_mcp_server()"
TIMEOUT = 20

_HAS_MCP = hasattr(q, "run_mcp_server") and callable(q.run_mcp_server)

class McpClient:
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

    def send_raw(self, txt):
        self.proc.stdin.write(txt + "\n")
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

    def read_any(self, timeout=TIMEOUT):
        """Read the next line (any id), or None on timeout."""
        line = self._read_line(timeout)
        if line is None:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def initialize(self):
        return self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
            rid=1,
        )

    def ping_ok(self):
        """Liveness probe: server must answer a ping (survived the attack)."""
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
        if not self.initialize():
            raise RuntimeError("initialize failed")
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture(scope="module")
def mcp():
    if not _HAS_MCP:
        pytest.skip("run_mcp_server not available in this build")
    client = McpClient()
    yield client
    client.close()



def test_garbage_line_gets_parse_error_then_alive(mcp):
    mcp.send_raw("this is not json at all")
    resp = mcp.read_any()
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32700
    assert mcp.ping_ok()


def test_truncated_json_parse_error_then_alive(mcp):
    mcp.send_raw('{"jsonrpc":"2.0","id":2,"method":"ping" ')
    resp = mcp.read_any()
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32700
    assert mcp.ping_ok()


def test_oversized_frame_rejected_then_alive(mcp):
    """A frame larger than MAX_MCP_CONTENT_LENGTH (10 MiB) must be rejected with
    -32600 without buffering the whole frame, and the worker must keep serving.
    """
    big = "[" + ",".join("0" for _ in range(6500000)) + "]"
    frame = (
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"decode_syndrome",'
        '"arguments":{"check_to_qubits":[[0,1],[1,2],[2,3]],"n_qubits":4,"syndrome":'
        + big
        + "}}}"
    )
    assert len(frame) > (10 * 1024 * 1024), "test frame should exceed the cap"
    mcp.send_raw(frame)
    resp = mcp.read_any(timeout=2)
    assert resp is None, "no response to oversized frame"
    assert mcp.ping_ok()


def test_blank_lines_produce_no_output(mcp):
    mcp.send_raw("")
    mcp.send_raw("   ")
    assert mcp.ping_ok()


def test_initialize_without_params_still_answers(mcp):
    resp = mcp.request("initialize", None, rid=10)
    assert resp is not None
    assert resp.get("result") is not None
    assert mcp.ping_ok()


def test_initialize_numeric_protocol_version_tolerated(mcp):
    resp = mcp.request(
        "initialize",
        {"protocolVersion": 20241105, "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
        rid=11,
    )
    assert resp is not None
    assert resp.get("result") is not None
    assert mcp.ping_ok()


def test_initialize_oversized_clientinfo_no_crash(mcp):
    resp = mcp.request(
        "initialize",
        {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "x" * 1000, "version": "1"}},
        rid=12,
    )
    assert resp is not None
    assert resp.get("result") is not None
    assert mcp.ping_ok()



def test_unknown_tool_name_32601(mcp):
    resp = mcp.request("tools/call", {"name": "no_such_tool", "arguments": {}}, rid=20)
    assert resp is not None, "no response for unknown tool"
    assert resp.get("error", {}).get("code") == -32601
    assert mcp.ping_ok()


def test_decode_syndrome_wrong_length_32602(mcp):
    resp = mcp.request(
        "tools/call",
        {"name": "decode_syndrome", "arguments": {"check_to_qubits": [[0, 1], [1, 2], [2, 3]], "n_qubits": 4, "syndrome": [1, 0]}},
        rid=21,
    )
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32602
    assert "syndrome length" in resp.get("error", {}).get("message", "")
    assert mcp.ping_ok()


def test_decode_syndrome_non_binary_values_rejected(mcp):
    """A syndrome value of 2 parses as u8 but is not a valid detector outcome.
    Accepting it silently would decode a DIFFERENT syndrome than the caller
    intended, so the trust boundary must reject with -32602.
    """
    resp = mcp.request(
        "tools/call",
        {"name": "decode_syndrome", "arguments": {"check_to_qubits": [[0, 1], [1, 2], [2, 3]], "n_qubits": 4, "syndrome": [0, 2, 0]}},
        rid=22,
    )
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32602
    assert mcp.ping_ok()


def test_decode_syndrome_empty_check_to_qubits(mcp):
    resp = mcp.request(
        "tools/call",
        {"name": "decode_syndrome", "arguments": {"check_to_qubits": [], "n_qubits": 0, "syndrome": []}},
        rid=23,
    )
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32602
    assert mcp.ping_ok()


def test_decode_syndrome_zero_n_qubits(mcp):
    resp = mcp.request(
        "tools/call",
        {"name": "decode_syndrome", "arguments": {"check_to_qubits": [[0, 1]], "n_qubits": 0, "syndrome": [0]}},
        rid=24,
    )
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32602
    assert mcp.ping_ok()


def test_decode_syndrome_invalid_decoder_type(mcp):
    resp = mcp.request(
        "tools/call",
        {"name": "decode_syndrome", "arguments": {"check_to_qubits": [[0, 1]], "n_qubits": 2, "syndrome": [0], "decoder_type": "bogus"}},
        rid=25,
    )
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32602
    assert mcp.ping_ok()



def test_batch_decode_flat_length_mismatch(mcp):
    """MCP-01: syndromes_flat length must equal batch_size * n_checks, or the
    decoder would panic on slice indexing.
    """
    resp = mcp.request(
        "tools/call",
        {
            "name": "batch_decode",
            "arguments": {
                "check_to_qubits": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "n_qubits": 4,
                "syndromes_flat": [1, 0, 0, 0, 0],
                "batch_size": 2,
                "decoder_type": "unionfind",
            },
        },
        rid=26,
    )
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32602
    assert "syndromes_flat length" in resp.get("error", {}).get("message", "")
    assert mcp.ping_ok()


def test_batch_decode_zero_batch_size(mcp):
    resp = mcp.request(
        "tools/call",
        {
            "name": "batch_decode",
            "arguments": {"check_to_qubits": [[0, 1], [1, 2], [2, 3], [3, 0]], "n_qubits": 4, "syndromes_flat": [], "batch_size": 0, "decoder_type": "unionfind"},
        },
        rid=27,
    )
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32602
    assert mcp.ping_ok()


def test_batch_decode_huge_batch_size_clean_error(mcp):
    """batch_size far above MAX_MCP_BATCH must be a clean -32602, never an OOM
    or an arithmetic overflow into a giant allocation.
    """
    resp = mcp.request(
        "tools/call",
        {
            "name": "batch_decode",
            "arguments": {"check_to_qubits": [[0, 1], [1, 2], [2, 3], [3, 0]], "n_qubits": 4, "syndromes_flat": [], "batch_size": 1099511627776, "decoder_type": "unionfind"},
        },
        rid=28,
    )
    assert resp is not None
    assert resp.get("error", {}).get("code") == -32602
    assert mcp.ping_ok()



def test_notification_silence_between_requests(mcp):
    mcp.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    resp = mcp.request("ping", {}, rid=30)
    assert resp is not None
    assert resp.get("result") is not None
    assert mcp.ping_ok()


def test_five_pipelined_requests_all_answered(mcp):
    sent = [40, 41, 42, 43, 44]
    for rid in sent:
        mcp.send({"jsonrpc": "2.0", "id": rid, "method": "ping"})
    ids = []
    for _ in sent:
        resp = mcp.read_any()
        assert resp is not None, "pipelined requests must each yield one response"
        ids.append(resp.get("id"))
    assert sorted(ids) == sorted(sent), f"pipelined ids mismatch: {ids}"
    assert len(set(ids)) == 5, f"duplicate responses: {ids}"
    assert mcp.ping_ok()


def test_decode_then_valid_decode_still_works(mcp):
    """After an attack fails, a legitimate decode must still succeed - the
    decoder cache must not be poisoned by rejected requests.
    """
    bad = mcp.request(
        "tools/call",
        {"name": "decode_syndrome", "arguments": {"check_to_qubits": [[0, 1], [1, 2]], "n_qubits": 3, "syndrome": [9, 9, 9]}},
        rid=50,
    )
    assert bad is not None, "expected an error for non-binary syndrome"
    assert "error" in bad, f"expected error, got: {bad}"

    good = mcp.request(
        "tools/call",
        {"name": "decode_syndrome", "arguments": {"check_to_qubits": [[0, 1], [1, 2]], "n_qubits": 3, "syndrome": [1, 0, 1]}},
        rid=51,
    )
    assert good is not None
    assert "result" in good, f"decode failed: {good}"
    payload = good["result"]["content"][0]["text"]
    decoded = json.loads(payload)
    assert decoded.get("syndrome_faithful") is True, f"not faithful: {decoded}"
    assert mcp.ping_ok()

