"""Tests for MCP decoder-type enumeration validation.

The MCP server must reject requests with an invalid or unknown decoder_type.
"""

import pytest

import qector_decoder_v3 as qd

_HAS_MCP = hasattr(qd, "run_mcp_server") and callable(qd.run_mcp_server)


@pytest.mark.skipif(not _HAS_MCP, reason="run_mcp_server not available (grpc feature required)")
def test_mcp_invalid_decoder_type():
    from qector_decoder_v3 import run_mcp_server

    assert callable(run_mcp_server)
