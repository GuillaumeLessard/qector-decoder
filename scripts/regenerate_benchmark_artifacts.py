"""Regenerate the competitive benchmark artifact on the v0.7.0 methodology.

The six pre-v0.7.0 benchmark artifacts were withdrawn (todo6 A1-03): they compared
**code-capacity** QECTOR against **circuit-level** PyMatching and printed both in one
table. :func:`qector_decoder_v3.ler.run_competitive_suite` fixes that by driving every
decoder through one circuit-level pipeline, and validates the rows with
``assert_comparable`` before returning.

This script wraps that call and records the provenance of the run *inside* the artifact,
so a future reader can tell what produced the numbers without trusting a filename.

Usage::

    python scripts/regenerate_benchmark_artifacts.py --dry-run     # show the plan, run nothing
    python scripts/regenerate_benchmark_artifacts.py --yes         # the full publication run
    python scripts/regenerate_benchmark_artifacts.py --yes --shots 2000 --distances 3,5

A full run is expensive (100k shots x 4 distances x 4 decoders) and must be done on a
**quiesced machine** -- throughput measured while other jobs compete for CPU is not
publishable. The run therefore requires an explicit ``--yes``; without it the script
prints the plan and exits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import ROOT, git_tree_dirty, write_artifact

DEFAULT_DECODERS = ("qector_blossom", "qector_belief", "qector_unionfind", "pymatching")


def _parse_distances(raw: str) -> tuple[int, ...]:
    try:
        vals = tuple(int(x) for x in raw.split(",") if x.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"distances must be comma-separated integers, got {raw!r}")
    if not vals:
        raise argparse.ArgumentTypeError("at least one distance is required")
    if any(v < 3 or v % 2 == 0 for v in vals):
        raise argparse.ArgumentTypeError(f"distances must be odd and >= 3, got {list(vals)}")
    return vals


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="regenerate_benchmark_artifacts.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--p", type=float, default=0.001, help="physical error rate (default: 0.001)")
    ap.add_argument("--shots", type=int, default=100_000, help="shots per cell (default: 100000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument(
        "--distances",
        type=_parse_distances,
        default=(3, 5, 7, 9),
        help="comma-separated odd code distances (default: 3,5,7,9)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="measurement rounds per shot (default: equal to the distance)",
    )
    ap.add_argument(
        "--decoders",
        type=lambda s: tuple(x.strip() for x in s.split(",") if x.strip()),
        default=DEFAULT_DECODERS,
        help=f"comma-separated decoder names (default: {','.join(DEFAULT_DECODERS)})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "competitive_results.json",
        help="output path (default: <repo>/competitive_results.json)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without running")
    ap.add_argument(
        "--yes",
        action="store_true",
        help="actually run; required because a full run costs hours of CPU",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cells = len(args.distances) * len(args.decoders)
    total_shots = cells * args.shots
    print("=== competitive benchmark regeneration ===")
    print("  methodology : circuit_level (all decoders, one pipeline)")
    print(f"  p           : {args.p}")
    print(f"  shots/cell  : {args.shots:,}")
    print(f"  distances   : {list(args.distances)}")
    print(f"  decoders    : {list(args.decoders)}")
    print(f"  cells       : {cells}  ({total_shots:,} decodes total)")
    print(f"  output      : {args.out}")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    if not args.yes:
        print(
            "\nRefusing to run without --yes.\n"
            "A full run is hours of CPU and must be done on a quiesced machine -- "
            "throughput measured under contention is not publishable.\n"
            "Re-run with --yes (or --dry-run to just see this plan)."
        )
        return 2

    try:
        from qector_decoder_v3.ler import run_competitive_suite
    except ImportError as exc:
        print(f"ERROR: qector_decoder_v3 not importable: {exc}")
        print("Run: maturin build --release && pip install --force-reinstall --no-deps target/wheels/*.whl")
        return 1

    if git_tree_dirty():
        print(
            "\nWARNING: the working tree is dirty. The artifact will record "
            "git_tree_dirty=true; it is not reproducible from the commit alone."
        )

    t0 = time.perf_counter()
    rows = run_competitive_suite(
        p=args.p,
        shots=args.shots,
        seed=args.seed,
        distances=args.distances,
        rounds=args.rounds,
        decoders=args.decoders,
    )
    elapsed = time.perf_counter() - t0

    write_artifact(
        args.out,
        rows,
        parameters={
            "p": args.p,
            "shots": args.shots,
            "seed": args.seed,
            "rounds": args.rounds if args.rounds is not None else "distance (d rounds)",
            "distances": list(args.distances),
            "decoders": list(args.decoders),
        },
        elapsed_seconds=elapsed,
    )

    print(f"\n  {len(rows)} rows written to {args.out} in {elapsed:.1f}s")
    print("  Every row carries noise_model='circuit_level' and passed assert_comparable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
