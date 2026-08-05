"""Independent test suite for MCP server protocol conformance and CLI subprocess flows (dev2todo §0.4).

Runs under pytest on all supported CPython versions.
"""

import json
import subprocess
import sys

SERVER_CODE = "import qector_decoder_v3 as qd; qd.run_mcp_server()"


def test_cli_decode_subcommand_exits_zero():
    cmd = [
        sys.executable,
        "-m",
        "qector_decoder_v3.cli",
        "decode",
        "--help",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"qector decode --help failed: {res}"
    assert "Decode syndromes" in res.stdout, f"expected help text, got: {res.stdout}"
    assert "usage" in res.stdout.lower(), f"expected usage, got: {res.stdout}"


def test_cli_doctor_exits_zero():
    cmd = [sys.executable, "-m", "qector_decoder_v3.doctor"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"qector doctor failed: {res}"
    assert "native-core" in res.stdout, f"expected native-core check, got: {res.stdout}"
    assert "PASS" in res.stdout, f"expected PASS, got: {res.stdout}"


def test_cli_bench_quick_exits_zero():
    cmd = [sys.executable, "-m", "qector_decoder_v3.bench_quick"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"qector bench_quick failed: {res}"


def test_mcp_ping_conformance():
    """MCP spec: ping MUST be answered with a result (not -32601)."""
    payload = json.dumps({"jsonrpc": "2.0", "id": 100, "method": "ping"}) + "\n"
    proc = subprocess.Popen(
        [sys.executable, "-c", SERVER_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, _err = proc.communicate(input=payload, timeout=5)
        assert proc.returncode == 0, f"server exited nonzero: {proc.returncode}"
        assert '"id":100' in out, f"missing response id, got: {out!r}"
        assert '"result"' in out or '"result":{}' in out or '"result": {}' in out, (
            f"ping did not return a result, got: {out!r}"
        )
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def test_mcp_notification_silence():
    """MCP spec: servers MUST NOT respond to notifications.

    Send a notification followed by a ping; the only response on the wire must
    be the ping's answer (a notification response would shift the stream).
    """
    payload = (
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 200, "method": "ping"})
        + "\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", SERVER_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, _err = proc.communicate(input=payload, timeout=5)
        assert proc.returncode == 0, f"server exited nonzero: {proc.returncode}"
        lines = [line.strip() for line in out.splitlines()]
        assert len(lines) == 1, f"expected exactly one response, got {len(lines)}: {out!r}"
        assert '"id":200' in lines[0], f"response missing ping id, got: {lines[0]!r}"
    finally:
        try:
            proc.kill()
        except Exception:
            pass
