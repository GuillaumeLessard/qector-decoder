#!/usr/bin/env python3
"""
example_streaming_session.py — QECTOR Decoder v3 (v0.5.7) streaming orchestration

Demonstrates the higher-level Python streaming layer
``qector_decoder_v3.streaming`` (``StreamingSession`` + ``sliding_window_decode``),
which sits *on top of* the ordinary single-shot decoders. (The compiled Rust core
also ships low-level ``StreamingDecoder`` / ``SlidingWindowDecoder`` primitives,
shown in example_streaming.py — this example is the Python orchestration layer and
does not shadow them.)

Honest decoding model: each round carries one spatial syndrome and is decoded
INDEPENDENTLY (phenomenological / perfect-measurement regime). This layer does NOT
claim full space-time circuit-level matching with time-like edges. The window only
controls commit latency, batching granularity, and telemetry. For a stateless inner
decoder (Union-Find family, exact Blossom) the per-round result is window-invariant
and reproduces a single full per-round decode bit-for-bit; every committed
correction satisfies H·c == s.

Run:
    PYTHONPATH=python python examples/example_streaming_session.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from qector_decoder_v3 import codes, gpu_backend as gb
from qector_decoder_v3.streaming import StreamingSession, sliding_window_decode


def make_stream(code, n_rounds, p, rng):
    """Synthesize a (n_rounds, n_checks) stream of independent per-round syndromes."""
    H = code.parity_check_matrix()
    errs = (rng.random((n_rounds, code.n_qubits)) < p).astype(np.uint8)
    syndromes = (errs @ H.T) & 1
    return syndromes


def main():
    print("=" * 70)
    print("QECTOR v3 (0.5.7) — Streaming Session (window + commit) orchestration")
    print("=" * 70)
    print(f"\nGPU backend active module: {gb.get_backend().summary()['active_module']}")

    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    print(f"Code: {code.name}  ({code.n_qubits} qubits, {code.n_checks} checks, "
          f"matching={code.is_matching_graph()})")

    rng = np.random.default_rng(123)
    n_rounds = 40
    p = 0.06
    stream = make_stream(code, n_rounds, p, rng)
    print(f"Stream: {n_rounds} rounds @ per-round p = {p}")

    # ---- 1. Incremental StreamingSession: push one round at a time -------
    print("\n1. StreamingSession.push_round (window_size=4, commit buffer):")
    session = StreamingSession(code, window_size=4)
    print(f"   inner decoder auto-resolved to: {type(session.decoder).__name__}")
    committed_total = 0
    for t, s in enumerate(stream):
        released = session.push_round(s)
        committed_total += len(released)
        if t in (0, 3, 4, n_rounds - 1):
            print(f"   after round {t:2d}: pending={session.pending:2d} "
                  f"committed={session.committed_count:2d} (released this step={len(released)})")
    tail = session.flush()
    print(f"   flush() released the final {len(tail)} buffered rounds")
    print(f"   total committed = {session.committed_count} (== {n_rounds} rounds)")

    # Validity of every committed correction.
    corr = session.committed_corrections()
    syn = session.committed_syndromes()
    valid_session = bool(np.all(((corr @ H.T) & 1) == syn))
    print(f"   H·c == s on all committed rounds ? {valid_session}")
    print(f"   telemetry: rounds={session.telemetry.rounds} windows={session.telemetry.windows} "
          f"committed={session.telemetry.committed}")
    print(f"   measured inner-decode time: total={session.telemetry.decode_seconds*1e3:.3f} ms, "
          f"mean/round={session.telemetry.mean_window_seconds*1e6:.1f} µs")

    # ---- 2. sliding_window_decode: batched, windowed convenience ---------
    print("\n2. sliding_window_decode (single shot, window_size=8):")
    res = sliding_window_decode(stream, code=code, window_size=8)
    print(f"   corrections shape: {res.corrections.shape}")
    print(f"   is_valid(H) ? {res.is_valid(H)}")
    print(f"   logical_flips shape: "
          f"{None if res.logical_flips is None else res.logical_flips.shape}")
    if res.logical_flips is not None:
        print(f"   rounds with a logical flip: {int(res.logical_flips.any(axis=1).sum())}/{n_rounds}")
    print(f"   telemetry: windows={res.telemetry.windows} "
          f"decode_seconds={res.telemetry.decode_seconds*1e3:.3f} ms  gpu={res.telemetry.gpu}")

    # ---- 3. Window-invariance for a stateless decoder --------------------
    print("\n3. Window-invariance (stateless exact decoder): w=8 vs w=1 vs full decode:")
    res_w1 = sliding_window_decode(stream, code=code, window_size=1)
    inner = res.corrections  # window_size=8
    same_w1 = bool(np.array_equal(res_w1.corrections, inner))
    # Independent ground truth: decode each round directly with the same family.
    from qector_decoder_v3 import BlossomDecoder
    direct = BlossomDecoder(code.check_to_qubits, code.n_qubits)
    full = np.stack([np.asarray(direct.decode(s)).astype(np.uint8) for s in stream])
    same_full = bool(np.array_equal(full, inner))
    print(f"   window=8 corrections == window=1 corrections ? {same_w1}")
    print(f"   window=8 corrections == full per-round decode  ? {same_full}")

    # ---- 4. Batched multi-shot stream ------------------------------------
    print("\n4. sliding_window_decode on a BATCH of shots (S, T, C):")
    n_shots = 16
    batch_stream = np.stack(
        [make_stream(code, n_rounds, p, rng) for _ in range(n_shots)]
    )
    res_b = sliding_window_decode(batch_stream, code=code, window_size=8)
    print(f"   input {batch_stream.shape} -> corrections {res_b.corrections.shape}")
    print(f"   is_valid(H) over all shots×rounds ? {res_b.is_valid(H)}")
    print(f"   telemetry: rounds={res_b.telemetry.rounds} windows={res_b.telemetry.windows}")

    ok = (valid_session and res.is_valid(H) and same_w1 and same_full
          and res_b.is_valid(H))
    print("\n" + "=" * 70)
    print(f"Result: {'all committed corrections valid + window-invariant. OK.' if ok else 'FAILURE'}")
    print("=" * 70)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
