from __future__ import annotations

import argparse
import sys
from typing import NoReturn


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qector",
        description="QECTOR Decoder v3 — decode, benchmark, serve",
    )
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("decode", help="Decode a syndrome file")
    d.add_argument("input", help="Path to syndrome file (numpy .npy or CSV)")
    d.add_argument("--check-to-qubits", "-c", required=True, help="Check matrix file (.npy)")
    d.add_argument("--n-qubits", "-n", type=int, required=True)
    d.add_argument(
        "--decoder",
        default="blossom",
        choices=[
            "blossom",
            "sparse_blossom",
            "union_find",
            "fast_union_find",
            "bposd",
            "belief_match",
            "auto",
        ],
    )
    d.add_argument("--output", "-o", default=None, help="Output path for correction")

    b = sub.add_parser("bench", help="Run a quick throughput benchmark")
    b.add_argument("--distance", "-d", type=int, default=5)
    b.add_argument("--rounds", "-r", type=int, default=5)
    b.add_argument("--shots", "-s", type=int, default=10_000)
    b.add_argument("--decoder", default="blossom")
    b.add_argument("--noise", type=float, default=0.001)

    s = sub.add_parser("serve", help="Start the REST server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", "-p", type=int, default=8000)
    s.add_argument("--transport", choices=["rest", "grpc", "mcp"], default="rest")

    return p


def cmd_decode(args: argparse.Namespace) -> None:
    import numpy as np

    from . import (
        AutoDecoder,
        BeliefMatching,
        BlossomDecoder,
        BPOSDDecoder,
        FastUnionFindDecoder,
        SparseBlossomDecoder,
        UnionFindDecoder,
    )

    syndromes = (
        np.load(args.input) if args.input.endswith(".npy") else np.loadtxt(args.input, dtype=np.uint8, delimiter=",")
    )
    if syndromes.ndim == 1:
        syndromes = syndromes.reshape(1, -1)

    # --check-to-qubits is a dense 0/1 check matrix (.npy). Graph decoders take
    # the check_to_qubits adjacency form; convert once here.
    H = np.asarray(np.load(args.check_to_qubits), dtype=np.uint8)
    c2q = [sorted(int(q) for q in np.nonzero(row)[0]) for row in H]
    nq = args.n_qubits

    decoder_map = {
        "blossom": BlossomDecoder,
        "sparse_blossom": SparseBlossomDecoder,
        "union_find": UnionFindDecoder,
        "fast_union_find": FastUnionFindDecoder,
        "bposd": BPOSDDecoder,
    }
    if args.decoder in decoder_map:
        dec = decoder_map[args.decoder](c2q, nq)
    elif args.decoder == "belief_match":
        # from_numpy_h returns a per-qubit correction, same contract as the
        # graph decoders above.
        dec = BeliefMatching.from_numpy_h(H, error_rate=0.05)
    elif args.decoder == "auto":
        dec = AutoDecoder(c2q, nq)
    else:
        raise ValueError(f"unknown decoder: {args.decoder}")

    corr = dec.decode_batch(syndromes) if syndromes.shape[0] > 1 else dec.decode(syndromes[0])
    if args.output:
        np.save(args.output, corr)
    else:
        print(corr)


def cmd_bench(args: argparse.Namespace) -> None:
    import stim

    from . import BlossomDecoder

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=args.distance,
        rounds=args.rounds,
        after_clifford_depolarization=args.noise,
        after_reset_flip_probability=args.noise,
        before_measure_flip_probability=args.noise,
    )
    sampler = circuit.compile_detector_sampler()
    shots = sampler.sample(args.shots)

    from .dem import from_stim

    dem = from_stim(circuit.detector_error_model(decompose_errors=True))
    if dem.is_graphlike:
        dem = dem.collapse_to_graph()
    c2q = dem.check_to_qubits()
    nq = dem.num_errors

    import time

    dec = BlossomDecoder(list(c2q), nq)
    t0 = time.perf_counter()
    dec.decode_batch(shots)
    elapsed = time.perf_counter() - t0
    print(
        f"{args.decoder} @ d={args.distance}, r={args.rounds}: "
        f"{args.shots / elapsed:.0f} shots/s ({elapsed:.2f}s for {args.shots} shots)"
    )


def cmd_serve(args: argparse.Namespace) -> None:
    if args.transport == "rest":
        import uvicorn

        from .rest_api import app as fastapi_app

        uvicorn.run(fastapi_app, host=args.host, port=args.port)
    elif args.transport == "grpc":
        print("gRPC server: use the native qector_decoder_v3 module directly")
    elif args.transport == "mcp":
        print("MCP server: use the native qector_decoder_v3 module directly")


def main(argv: list[str] | None = None) -> NoReturn:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "decode":
        cmd_decode(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
