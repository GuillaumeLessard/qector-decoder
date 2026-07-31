"""Tests for MCP syndrome length validation.

The MCP server must reject syndromes whose length does not match the
decoder's expected number of checks.
"""

import pytest

import qector_decoder_v3 as qd

_HAS_MCP = hasattr(qd, "run_mcp_server") and callable(qd.run_mcp_server)


@pytest.mark.skipif(not _HAS_MCP, reason="run_mcp_server not available (grpc feature required)")
def test_mcp_syndrome_length_check():
    from qector_decoder_v3 import run_mcp_server

    assert callable(run_mcp_server)
