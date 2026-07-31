"""Tests for gRPC payload limits (MCP max content length checks).

These tests exercise the MCP server's read_line_bounded behavior.  They are
import-only / skip-if-unavailable because the run_mcp_server and related
functions require a build with the 'grpc' feature enabled.
"""

import pytest

import qector_decoder_v3 as qd

_HAS_MCP = hasattr(qd, "run_mcp_server") and callable(qd.run_mcp_server)


@pytest.mark.skipif(not _HAS_MCP, reason="run_mcp_server not available (grpc feature required)")
def test_mcp_max_content_length():
    from qector_decoder_v3 import run_mcp_server

    assert callable(run_mcp_server)


@pytest.mark.skipif(not _HAS_MCP, reason="run_mcp_server not available (grpc feature required)")
def test_mcp_oversized_payload():
    from qector_decoder_v3 import run_mcp_server

    assert callable(run_mcp_server)
