"""Shared provenance stamping for benchmark artifacts.

Every benchmark artifact this repo publishes must say, inside itself, what produced it.
The six pre-v0.7.0 artifacts were withdrawn (todo6 A1-03) because their methodology was not
recorded anywhere in the file: they compared **code-capacity** QECTOR against **circuit-level**
PyMatching, and nothing in the JSON said so. A reader could not tell.

Use :func:`write_artifact` from every generator script so that failure mode cannot recur.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WITHDRAWN_ARTIFACTS = (
    "benchmark_results.json",
    "benchmark_results_empirical.json",
    "chart_competitor.png",
    "chart_throughput.png",
    "competitive_results.json",
    "live_test_results.json",
)

METHODOLOGY_NOTE = (
    "Every decoder is measured on the same Stim circuit, the same detector samples, the same "
    "DEM and the same observable scoring (ler.estimate_ler_circuit_level). Only the decoder "
    "varies between rows. Rows are validated by ler.assert_comparable before being written."
)

CAVEATS = (
    "Throughput figures are only meaningful on an otherwise-idle machine.",
    "LER figures are subject to binomial error; at low p and low shot counts the confidence "
    "interval can exceed the difference between decoders. Check ci95_lo/ci95_hi per row.",
)


def _git(*args: str, default: str = "unknown") -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() or default
    except (OSError, subprocess.SubprocessError):
        return default


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_tree_dirty() -> bool:
    """True if the tree has uncommitted changes, so the run is not reproducible from the commit.

    Errs on the side of ``True``: an artifact wrongly marked reproducible is worse than one
    wrongly marked dirty.
    """
    return bool(_git("status", "--porcelain", default="dirty"))


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import qector_decoder_v3 as q

        versions["qector_decoder_v3"] = getattr(q, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - provenance must never be the thing that fails a run
        versions["qector_decoder_v3"] = "not importable"
    for name in ("stim", "pymatching", "sinter", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing reference package is data, not an error
            versions[name] = "not installed"
    return versions


def build_provenance(
    parameters: dict,
    elapsed_seconds: float | None = None,
    methodology: str = "circuit_level",
    generator: str | None = None,
) -> dict:
    """Everything a reader needs to judge, and to reproduce, a set of benchmark rows."""
    return {
        "methodology": methodology,
        "methodology_note": METHODOLOGY_NOTE,
        "supersedes": (
            "The six pre-v0.7.0 artifacts ("
            + ", ".join(WITHDRAWN_ARTIFACTS)
            + "), which mixed code-capacity and circuit-level measurements and were withdrawn."
        ),
        "generator": generator or Path(sys.argv[0]).name or "unknown",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": None if elapsed_seconds is None else round(elapsed_seconds, 1),
        "git_commit": git_commit(),
        "git_tree_dirty": git_tree_dirty(),
        "parameters": parameters,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
            "versions": dependency_versions(),
        },
        "caveats": list(CAVEATS),
    }


def load_artifact(path: str | Path) -> tuple[list, dict | None]:
    """Read a benchmark artifact, accepting both the stamped and the legacy shapes.

    Returns ``(rows, provenance)``. ``provenance`` is ``None`` for legacy files written as a
    bare JSON list -- those predate stamping, so their methodology is unknown and any number
    read out of them must be treated as unattributed (see todo6 A1-03).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data["results"], data.get("provenance")
    if isinstance(data, list):
        return data, None
    raise ValueError(
        f"{path}: expected a stamped artifact {{'provenance':…, 'results':[…]}} or a legacy "
        f"JSON list, got {type(data).__name__}"
    )


def write_artifact(
    path: str | Path,
    rows: list,
    parameters: dict,
    elapsed_seconds: float | None = None,
    methodology: str = "circuit_level",
    generator: str | None = None,
) -> Path:
    """Write ``{"provenance": ..., "results": rows}`` and return the path written."""
    out = Path(path)
    artifact = {
        "provenance": build_provenance(
            parameters,
            elapsed_seconds=elapsed_seconds,
            methodology=methodology,
            generator=generator,
        ),
        "results": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return out
