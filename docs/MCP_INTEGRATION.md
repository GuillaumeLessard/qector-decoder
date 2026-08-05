# QECTOR Decoder v3 — Model Context Protocol (MCP) Integration

QECTOR Decoder v3 exposes a fully conformant JSON-RPC 2.0 stdio Model Context Protocol (MCP) server (`qector serve` / `qector_decoder_v3.run_mcp_server`). This enables AI coding agents (Claude Desktop, Cursor, custom agentic frameworks) to query decoders, execute benchmarks, and inspect QEC telemetry programmatically.

The server implements the MCP transport contract correctly: `initialize` returns the protocol version and server info; `ping` is supported as required by the spec (empty-object result); `notifications/initialized` is processed silently per JSON-RPC 2.0 (no response is written — the worker pool guards on `req.id.is_some()` before sending a response, so notifications never desynchronise the response stream).

---

## Launching the MCP Server

From the command line:

```bash
qector serve
```

From Python:

```python
import qector_decoder_v3 as qd
qd.run_mcp_server()
```

A ready-made client configuration lives at `mcp.json` at the repository root (it launches `python -c "import qector_decoder_v3; qector_decoder_v3.run_mcp_server()"` with `QECTOR_SILENT=1` and `QECTOR_BLOSSOM_K_MULT=2.0`). Point your MCP client at that file — e.g. Claude Code supports `mcp.add` with the `qector` server name.

---

## Decoder family and backend surface

| Concept | Count | Notes |
|---|---|---|
| Decoder families (`get_decoder_info`) | 9 | `UnionFind`, `FastUnionFind`, `Blossom`, `SparseBlossom`, `BPOSD` / `BpOsd`, `HybridCascade`, `AmbiguityCluster`, `TwoStage`, `ColourCode` |
| Backend tiers (`get_backend_health`) | 7 | `CUDA_BPOSD` (Enterprise + CUDA), `OPENCL_BATCH`, `CPU_RAYON_BATCH`, `SPARSE_BLOSSOM`, `EXACT_BLOSSOM`, `FAST_UNION_FIND`, `PURE_PYTHON_FALLBACK` |
| Public Sinter entry points | 5 | `qector_blossom`, `qector_belief`, `qector_unionfind`, `qector_bposd`, `qector_unionfind_unweighted` (registered as `[project.entry-points.sinter_decoder]`) |

---

## Exposed MCP Tools (13)

| # | MCP Tool Name | Description |
|---|---|---|
| 1 | `ping` | Protocol keepalive (returns `{}`). Required by the MCP spec for client-side liveness checks. |
| 2 | `decode_syndrome` | Decode a 1D syndrome vector. Args: `check_to_qubits`, `n_qubits?`, `syndrome`, `decoder_type?` (`unionfind`/`fastunionfind`/`blossom`/`sparseblossom`/`bposd`/`cascade`/`hybrid`/`lookuptable`/`slidingwindow`/`streaming`/`auto`), `osd_order?`, `bp_method?` (`"exact"` (default, log-domain sum-product) or `"min_sum"`). |
| 3 | `batch_decode` | Decode a 2D batch array of syndromes. Args: `check_to_qubits`, `n_qubits?`, `syndromes_flat` (row-major), `batch_size`, `decoder_type?`, `osd_order?`, `bp_method?`. |
| 4 | `decode_hyperedge` | Decode non-graphlike hyperedge syndromes (qLDPC) via BP-OSD. Bypasses the graphlike Union-Find restriction. Args: `check_to_qubits`, `n_qubits?`, `syndrome`, `decoder_type?` (`bposd`/`blossom`/`beliefmatching`), `osd_order?`, `bp_method?`. |
| 5 | `decode_syndrome_blossom` | Decode a single syndrome with the exact Blossom (MWPM) decoder. Args: `check_to_qubits`, `n_qubits?`, `syndrome`. |
| 6 | `batch_decode_blossom` | Batch-decode with the Blossom decoder (Rayon-parallel). Args: `check_to_qubits`, `n_qubits?`, `syndromes_flat`, `batch_size`. |
| 7 | `decode_syndrome_cascade` | Decode with the hybrid cascading decoder: fast Union-Find pre-filter that escalates hard syndromes to Blossom. Args: `check_to_qubits`, `n_qubits?`, `syndrome`, `max_accept_weight?`. |
| 8 | `benchmark_decoder` | Run a latency / throughput benchmark on a decoder. Args: `check_to_qubits`, `n_qubits?`, `n_samples?`, `seed?`, `phys_error?`, `decoder_type?`. |
| 9 | `run_ler_benchmark` | Measure logical error rate with 95% Wilson confidence intervals across multiple code distances. Args: none required. |
| 10 | `get_decoder_info` | Return decoder configuration, package version, and the full 9-family listing. Args: none. |
| 11 | `get_backend_health` | Return the 7-tier backend health status (per-tier availability + measured warmup latency in µs). Args: none. |
| 12 | `clear_decoder_cache` | Clear the decoder factory cache to eliminate stale state. Args: none. |
| 13 | `recommend_decoder` | Recommend an optimal decoder family based on code topology, distance, and target priority. Args: `code_family?`, `distance?`, `priority?` (`"accuracy"` / `"speed"` / `"balanced"`). |

---

## Protocol details

### `initialize` result shape

```json
{
  "protocolVersion": "2024-11-05",
  "capabilities": { "tools": {} },
  "serverInfo": {
    "name": "qector-decoder-v3",
    "version": "1.0.0"
  }
}
```

### `notifications/initialized` is silent

Per JSON-RPC 2.0, notifications (messages without an `id`) MUST NOT receive a response. The QECTOR worker pool enforces this: the response is built by `handle_request` for the `notifications/initialized` arm, but the worker pool only writes to stdout when `resp.id.is_some()`. So a strict in-order MCP client (Claude Desktop, Cursor, the mcp SDK) sees the response stream remain correctly aligned after the `initialize` / `notifications/initialized` handshake. **This is the v0.7.0 → v0.7.1 fix** (DEFECT-3 in the verification report).

### `ping`

`ping` returns an empty-object result `{}`. It is required by the MCP spec for keepalive on long-lived sessions; v0.7.0 returned `-32601 Method not found` for `ping` (DEFECT-2 in the verification report), which was a hard protocol error for any conformant client.

### Error codes

| Code | Meaning |
|---|---|
| `-32700` | Parse error (malformed JSON frame) |
| `-32600` | Request exceeds `max_content_length` (10 MB stdin frame limit) |
| `-32601` | Method not found, or "Unknown tool" on `tools/call` |
| `-32602` | Invalid params (syndrome length mismatch, invalid `decoder_type`, etc.) |
| `-32603` | Internal error or "server is busy; request queue is full" |

### Worker pool and concurrency

The MCP server uses a fixed-size worker pool bounded by `available_parallelism()`. Requests are processed in parallel; the dedicated writer thread guarantees atomic line-delimited stdout writes. The stdin reader runs on its own thread so the blocking I/O never holds the GIL. If the queue is full, the server returns `-32603` immediately rather than blocking.

---

## Security and trust boundary

- The server reads from stdin and writes to stdout. Stderr is reserved for license banner output (suppressed with `QECTOR_SILENT=1`).
- All JSON-RPC frames are size-bounded to 10 MB (a frame larger than the bound is rejected with `-32600`, not buffered in memory).
- All `decoder_type` values are validated against a strict enum; invalid values return `-32602` in under 50 ms.
- License tier is enforced: on Community tier, the GPU decoders are skipped at construction time, not at request time.

For a deployment review checklist (auth, rate limiting, audit logging), see `docs/SECURITY_DEPLOYMENT.md`. The MCP server is a **Provisional** API in the v1.0.0 stability contract — see `docs/STABLE_API.md` for the stability tier.
