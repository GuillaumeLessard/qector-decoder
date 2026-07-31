#!/usr/bin/env python
"""QECTOR vs PyMatching vs ldpc — circuit-level decoder comparison.

Every row comes from :func:`qector_decoder_v3.ler.estimate_ler_circuit_level`,
which is the only sanctioned way to produce a cross-decoder number here:

* one Stim rotated-surface-code circuit per (distance, p);
* one decomposed detector error model;
* one detector/observable sample set per (distance, shots, seed), so every
  decoder in a cell sees byte-identical input;
* one resolver (``ler._dem_observable_decoder``) handing QECTOR's own backends,
  PyMatching and ldpc alike to the same ``decode_batch(dets) -> predicted
  observables`` interface, each in a single native batch call;
* scoring against the circuit's own logical observables.

``ler.assert_comparable`` gates the collected rows before anything is written.

Why this file is written so defensively
---------------------------------------
Six pre-v0.7.0 artifacts were withdrawn (todo6 A1-03) for comparing
code-capacity QECTOR against circuit-level PyMatching with nothing in the file
saying so. The v0.7.0 regeneration reproduced that defect *and* stamped
``_provenance.METHODOLOGY_NOTE`` — which asserts ``estimate_ler_circuit_level``
scoring and ``assert_comparable`` validation — into a file produced by neither.
Specifically it had:

* "LER" computed as ``(H @ correction) % 2 == syndrome``: a syndrome-consistency
  check, not a logical error rate. It asks whether the decoder returned *a*
  valid correction, never whether the logical observable flipped, and duly read
  0.000% for every decoder at every distance — the shape of a tautology.
* QECTOR timed through a native ``batch_decode`` over the full shot count
  against PyMatching timed through a Python ``[dec.decode(s) for s in ...]``
  loop over as few as 20 shots, linearly extrapolated to 100,000 and written
  into ``elapsed_sec`` as though measured.
* An LER chart whose points were not measurements but hardcoded analytic curves,
  ``0.14 * (0.03/0.10) ** ((d + 1) / 2)``, with per-decoder constants set by hand.

So: no extrapolation anywhere below. A cell that does not fit the per-cell time
budget is recorded as skipped, carrying the measured probe rate and the
projected cost, and is absent from the charts. An honest hole beats a
synthesised number.

Usage
-----
    python scripts/run_custom_comparison_benchmark.py \
        --distances 3,5,7,9,11 --shots 1000,5000,10000 --p 0.005
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QECTOR_SILENT", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _provenance  # noqa: E402
import matplotlib  # noqa: E402
from qector_decoder_v3.ler import assert_comparable, estimate_ler_circuit_level  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Label -> decoder kind understood by ler._dem_observable_decoder.
DECODERS = [
    ("QECTOR Sparse Blossom (CPU)", "blossom"),
    ("QECTOR Union-Find (CPU)", "union_find"),
    ("PyMatching v2 (C++)", "pymatching"),
    ("ldpc BP-OSD", "ldpc_bposd"),
]

COLORS = {
    "QECTOR Sparse Blossom (CPU)": "#059669",
    "QECTOR Union-Find (CPU)": "#1a56db",
    "PyMatching v2 (C++)": "#dc2626",
    "ldpc BP-OSD": "#4b5563",
}

PROBE_SHOTS = 256


def _fmt_speedup(sp) -> str:
    """Format a throughput ratio without collapsing small ones to '0.00x'.

    A decoder ~1000x slower than the reference has a ratio of 0.001; printed with
    two decimals that becomes '0.00x', which reads as zero throughput rather than
    as the (accurate, unflattering) number it is.
    """
    if sp is None:
        return "—"
    return f"{sp:.2f}x" if sp >= 0.01 else f"{sp:.3g}x"


def run_grid(distances, shot_list, p, seed, budget_s):
    """Measure every feasible (distance, decoder, shots) cell. Never extrapolate."""
    rows: list[dict] = []
    skipped: list[dict] = []

    for d in distances:
        print(f"\n--- distance d={d} (p={p}) ---", flush=True)
        for label, kind in DECODERS:
            # Probe once so the budget decision rests on a measurement, not a guess.
            try:
                probe = estimate_ler_circuit_level(
                    distance=d, decoder=kind, p=p, shots=PROBE_SHOTS, seed=seed
                )
            except Exception as exc:  # noqa: BLE001 - an absent backend is data, not a crash
                print(f"  {label:30s} unavailable: {type(exc).__name__}: {exc}", flush=True)
                skipped.append(
                    {
                        "decoder": label,
                        "distance": d,
                        "shots": None,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            rate = probe.decodes_per_s
            print(f"  {label:30s} probe {rate:12.1f} dec/s", flush=True)

            for S in shot_list:
                projected = S / rate if rate > 0 else float("inf")
                if projected > budget_s:
                    skipped.append(
                        {
                            "decoder": label,
                            "distance": d,
                            "shots": S,
                            "reason": "over per-cell decode budget",
                            "probe_rate_dec_per_s": rate,
                            "projected_decode_seconds": projected,
                            "budget_seconds": budget_s,
                        }
                    )
                    print(
                        f"      S={S:>7,} SKIPPED (~{projected:,.0f}s > {budget_s:g}s budget)",
                        flush=True,
                    )
                    continue

                res = estimate_ler_circuit_level(
                    distance=d, decoder=kind, p=p, shots=S, seed=seed
                )
                row = res.to_dict()
                row.update({"decoder": label, "decoder_kind": kind, "distance": d})
                rows.append(row)
                lo, hi = res.ci95
                print(
                    f"      S={S:>7,} {res.decodes_per_s:12.1f} dec/s  "
                    f"LER={res.ler:.5f} [{lo:.5f},{hi:.5f}]  errors={res.errors}",
                    flush=True,
                )
    return rows, skipped


def add_speedups(rows):
    """Speedup vs PyMatching within the SAME (distance, shots) cell. Never across cells."""
    ref = {
        (r["distance"], r["shots"]): r["decodes_per_s"]
        for r in rows
        if r["decoder"].startswith("PyMatching")
    }
    for r in rows:
        base = ref.get((r["distance"], r["shots"]))
        r["speedup_vs_pymatching"] = round(r["decodes_per_s"] / base, 3) if base else None
    return rows


FIELDS = [
    "decoder",
    "decoder_kind",
    "code",
    "distance",
    "rounds",
    "noise_model",
    "physical_error_rate",
    "shots",
    "errors",
    "ler",
    "ci95_lo",
    "ci95_hi",
    "seconds",
    "decodes_per_s",
    "speedup_vs_pymatching",
    "seed",
]


def export_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print(f"[Exported] CSV ({len(rows)} rows): {path}")


def _best_per_distance(rows, label):
    """Largest-shot-count row per distance for one decoder (most statistics)."""
    per_d: dict[int, dict] = {}
    for r in rows:
        if r["decoder"] != label:
            continue
        cur = per_d.get(r["distance"])
        if cur is None or r["shots"] > cur["shots"]:
            per_d[r["distance"]] = r
    return per_d


def export_markdown(path: Path, rows, skipped, prov, p, budget_s):
    env = prov.get("environment", {})
    v = env.get("versions", {})
    md: list[str] = []
    md.append("# QECTOR v0.7.0 — circuit-level decoder comparison\n")
    md.append(
        "Every row below is one `ler.estimate_ler_circuit_level` measurement: the same "
        "Stim rotated-surface-code circuit, the same decomposed DEM, the same "
        "detector/observable samples and the same `decode_batch` resolver for every "
        "decoder, scored against the circuit's own logical observables. "
        "`ler.assert_comparable` gated these rows before writing.\n"
    )
    md.append(
        f"- **Noise model**: circuit-level, p = {p} (gate, reset and measurement noise "
        "over d rounds of syndrome extraction)\n"
        f"- **Per-cell decode budget**: {budget_s:g}s. Cells projected to exceed it appear "
        "under *Not measured* — nothing is extrapolated.\n"
        f"- **Git commit**: `{prov.get('git_commit', '')[:10]}` "
        f"(tree dirty: {prov.get('git_tree_dirty')})\n"
        f"- **Platform**: {env.get('platform', '?')} · Python {env.get('python', '?')}\n"
        f"- **Versions**: qector {v.get('qector_decoder_v3', '?')}, "
        f"pymatching {v.get('pymatching', '?')}, ldpc {v.get('ldpc', '?')}, "
        f"stim {v.get('stim', '?')}\n"
    )

    md.append(f"\n## Measured results ({len(rows)} cells)\n")
    md.append(
        "| Decoder | d | Shots | Errors | LER | 95% CI | Throughput (dec/s) | vs PyMatching |"
    )
    md.append("|:---|:---:|---:|---:|---:|:---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (x["distance"], x["shots"], x["decoder"])):
        sp = r.get("speedup_vs_pymatching")
        md.append(
            f"| {r['decoder']} | {r['distance']} | {r['shots']:,} | {r['errors']} | "
            f"{r['ler']:.5f} | [{r['ci95_lo']:.5f}, {r['ci95_hi']:.5f}] | "
            f"{r['decodes_per_s']:,.1f} | {_fmt_speedup(sp)} |"
        )

    if skipped:
        md.append(f"\n## Not measured ({len(skipped)} cells)\n")
        md.append(
            "These cells were **not run** and carry no numbers. They are listed so the gaps "
            "in the grid are explicit rather than quietly filled in.\n"
        )
        md.append("| Decoder | d | Shots | Reason | Probe rate (dec/s) | Projected decode |")
        md.append("|:---|:---:|---:|:---|---:|---:|")
        for s in skipped:
            sh = f"{s['shots']:,}" if s.get("shots") else "—"
            pr = s.get("probe_rate_dec_per_s")
            pj = s.get("projected_decode_seconds")
            rate_cell = f"{pr:,.1f}" if pr else "—"
            proj_cell = f"{pj:,.0f}s" if pj else "—"
            md.append(
                f"| {s['decoder']} | {s['distance']} | {sh} | {s['reason']} | "
                f"{rate_cell} | {proj_cell} |"
            )

    md.append("\n## How to read this\n")
    for c in prov.get("caveats", []):
        md.append(f"- {c}")
    md.append(
        "- Accuracy and speed are independent axes: a lower LER at the same (d, p) is more "
        "accurate, a higher throughput is faster. This table deliberately does not collapse "
        "them into one score."
    )
    md.append(
        "- Throughput counts decode time only — `LerResult.seconds` wraps the single "
        "`decode_batch` call. Circuit construction and sampling are excluded for every "
        "decoder equally."
    )
    md.append(
        "- A cell with 0 errors is not evidence of a zero error rate; read its `ci95_hi` as "
        "an upper bound. The LER chart plots those as open downward markers."
    )
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[Exported] Markdown ({len(rows)} measured, {len(skipped)} skipped): {path}")


def generate_charts(rows, out_dir: Path, p: float):
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    written: list[str] = []
    labels = sorted({r["decoder"] for r in rows})

    # 1. Throughput vs distance.
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    for lab in labels:
        per_d = _best_per_distance(rows, lab)
        xs = sorted(per_d)
        if xs:
            ax.plot(
                xs,
                [per_d[x]["decodes_per_s"] for x in xs],
                "o-",
                label=lab,
                color=COLORS.get(lab),
                linewidth=2.2,
                markersize=5,
            )
    ax.set_yscale("log")
    ax.set_title("Decode throughput vs code distance (circuit-level)", fontweight="bold")
    ax.set_xlabel("Code distance d")
    ax.set_ylabel("Throughput (decodes/s, log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    q = out_dir / "chart_official_throughput.png"
    fig.savefig(q)
    plt.close(fig)
    written.append(q.name)

    # 2. Measured LER vs distance, with Wilson CIs. Zero-error cells are drawn as
    #    95% upper bounds, never as a point estimate of zero.
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    any_point = False
    for lab in labels:
        per_d = _best_per_distance(rows, lab)
        xs = sorted(per_d)
        sx, sy, lo_e, hi_e, bx, by = [], [], [], [], [], []
        for x in xs:
            r = per_d[x]
            if r["errors"] > 0:
                sx.append(x)
                sy.append(r["ler"])
                lo_e.append(max(0.0, r["ler"] - r["ci95_lo"]))
                hi_e.append(max(0.0, r["ci95_hi"] - r["ler"]))
            else:
                bx.append(x)
                by.append(r["ci95_hi"])
        if sx:
            ax.errorbar(
                sx, sy, yerr=[lo_e, hi_e], fmt="o-", capsize=3, label=lab,
                color=COLORS.get(lab), linewidth=2.0, markersize=5,
            )
            any_point = True
        if bx:
            ax.plot(
                bx, by, "v", mfc="none", color=COLORS.get(lab),
                label=f"{lab} (0 errors → 95% upper bound)",
            )
            any_point = True
    if any_point:
        ax.set_yscale("log")
    ax.set_title(
        f"Measured logical error rate vs distance (circuit-level, p={p})", fontweight="bold"
    )
    ax.set_xlabel("Code distance d")
    ax.set_ylabel("Logical error rate per shot")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    q = out_dir / "chart_official_ler.png"
    fig.savefig(q)
    plt.close(fig)
    written.append(q.name)

    # 3. Throughput vs batch size, at the largest distance with more than one point.
    by_d: dict[int, set] = {}
    for r in rows:
        by_d.setdefault(r["distance"], set()).add(r["shots"])
    multi = [d for d, s in by_d.items() if len(s) > 1]
    if multi:
        dsel = max(multi)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
        for lab in labels:
            pts = sorted(
                (
                    (r["shots"], r["decodes_per_s"])
                    for r in rows
                    if r["decoder"] == lab and r["distance"] == dsel
                ),
                key=lambda t: t[0],
            )
            if pts:
                ax.plot(
                    [a for a, _ in pts], [b for _, b in pts], "s--", label=lab,
                    color=COLORS.get(lab), linewidth=2.0, markersize=5,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Throughput vs batch size (d={dsel})", fontweight="bold")
        ax.set_xlabel("Shots per batch (log)")
        ax.set_ylabel("Throughput (decodes/s, log)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        q = out_dir / "chart_official_batch_scaling.png"
        fig.savefig(q)
        plt.close(fig)
        written.append(q.name)

    print(f"[Generated] charts: {', '.join(written)}")
    return written


def export_pdf(path: Path, rows, skipped, prov, charts, out_dir: Path, p: float):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            HRFlowable,
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        print("[Skipped] PDF: reportlab is not installed")
        return

    ss = getSampleStyleSheet()
    st_title = ParagraphStyle("t", parent=ss["Title"], fontSize=17, spaceAfter=4)
    st_sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=9,
                            textColor=colors.HexColor("#475569"))
    st_h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=5)
    cell = ParagraphStyle("c", parent=ss["Normal"], fontSize=7, leading=9)
    cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")

    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        title="QECTOR v0.7.0 circuit-level decoder comparison",
    )
    env = prov.get("environment", {})
    story = [
        Paragraph("QECTOR v0.7.0 — circuit-level decoder comparison", st_title),
        Paragraph(
            "Measured with ler.estimate_ler_circuit_level: identical circuit, DEM, samples "
            "and decode_batch resolver for every decoder, scored against the circuit's "
            "logical observables. Validated by ler.assert_comparable. No extrapolated cells.",
            st_sub,
        ),
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2563eb"), spaceAfter=8),
        Paragraph(
            f"<b>Noise model:</b> circuit-level, p={p} &nbsp;|&nbsp; "
            f"<b>Commit:</b> {prov.get('git_commit', '')[:10]} &nbsp;|&nbsp; "
            f"<b>Generated:</b> {prov.get('generated_utc', '')[:19]}Z<br/>"
            f"<b>Platform:</b> {env.get('platform', '?')} &nbsp;|&nbsp; "
            f"<b>Python:</b> {env.get('python', '?')}",
            cell,
        ),
        Spacer(1, 8),
    ]
    for name in charts:
        story.append(Image(str(out_dir / name), width=6.9 * inch, height=3.55 * inch))
        story.append(Spacer(1, 6))

    story.append(PageBreak())
    story.append(Paragraph(f"Measured results ({len(rows)} cells)", st_h2))
    data = [[Paragraph(x, cellb) for x in
             ("Decoder", "d", "Shots", "Errors", "LER", "95% CI", "dec/s", "vs PM")]]
    for r in sorted(rows, key=lambda x: (x["distance"], x["shots"], x["decoder"])):
        sp = r.get("speedup_vs_pymatching")
        data.append([
            Paragraph(r["decoder"], cell),
            Paragraph(str(r["distance"]), cell),
            Paragraph(f"{r['shots']:,}", cell),
            Paragraph(str(r["errors"]), cell),
            Paragraph(f"{r['ler']:.5f}", cellb),
            Paragraph(f"[{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]", cell),
            Paragraph(f"{r['decodes_per_s']:,.0f}", cellb),
            Paragraph(_fmt_speedup(sp), cell),
        ])
    t = Table(data, colWidths=[138, 22, 48, 38, 52, 94, 60, 40], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t)

    if skipped:
        story.append(PageBreak())
        story.append(Paragraph(f"Not measured ({len(skipped)} cells)", st_h2))
        story.append(Paragraph(
            "Listed so the gaps are explicit. These cells were never run and carry no "
            "numbers; nothing here was extrapolated or synthesised.", cell))
        story.append(Spacer(1, 6))
        sd = [[Paragraph(x, cellb) for x in ("Decoder", "d", "Shots", "Reason", "Projected")]]
        for s in skipped:
            pj = s.get("projected_decode_seconds")
            sd.append([
                Paragraph(s["decoder"], cell),
                Paragraph(str(s["distance"]), cell),
                Paragraph(f"{s['shots']:,}" if s.get("shots") else "—", cell),
                Paragraph(s["reason"], cell),
                Paragraph(f"{pj:,.0f}s" if pj else "—", cell),
            ])
        t2 = Table(sd, colWidths=[145, 25, 55, 190, 60], repeatRows=1)
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c2d12")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t2)

    doc.build(story)
    print(f"[Exported] PDF: {path}")


def main():
    ap = argparse.ArgumentParser(description="Circuit-level decoder comparison benchmark.")
    ap.add_argument("--distances", default="3,5,7,9,11")
    ap.add_argument("--shots", default="1000,5000,10000")
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--budget-seconds", type=float, default=30.0,
        help="Per-cell decode budget; cells projected to exceed it are recorded as "
             "skipped rather than extrapolated.",
    )
    ap.add_argument("--out-dir", default=".")
    ap.add_argument(
        "--from-json", metavar="PATH",
        help="re-render CSV/Markdown/PDF/charts from an existing stamped artifact "
             "instead of measuring. Use for presentation fixes so the numbers stay "
             "byte-identical to what has already been cited.",
    )
    args = ap.parse_args()

    distances = [int(x) for x in args.distances.split(",") if x.strip()]
    shot_list = sorted(int(x) for x in args.shots.split(",") if x.strip())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("QECTOR circuit-level comparison benchmark")
    print(f"  distances={distances}  shots={shot_list}  p={args.p}  seed={args.seed}")
    print(f"  per-cell decode budget: {args.budget_seconds:g}s (no extrapolation)")

    if args.from_json:
        # Re-render the reports from an existing stamped artifact without
        # re-measuring. A presentation fix must not silently change the numbers
        # underneath it: re-running would produce a different (equally valid)
        # sample, and any figure already quoted elsewhere would quietly stop
        # matching the artifact it cites.
        src = Path(args.from_json).resolve()
        rows, prov = _provenance.load_artifact(src)
        if prov is None:
            print(f"{src} carries no provenance block - refusing to re-render it.")
            return 1
        skipped = prov.get("parameters", {}).get("skipped_cells", [])
        p = prov.get("parameters", {}).get("physical_error_rate", args.p)
        budget = prov.get("parameters", {}).get("per_cell_budget_seconds", args.budget_seconds)
        assert_comparable(rows)
        print(f"Re-rendering {len(rows)} measured / {len(skipped)} skipped cells from {src.name}")
        export_csv(out_dir / "official_benchmark_results.csv", rows)
        export_markdown(
            out_dir / "official_benchmark_results.md", rows, skipped, prov, p, budget
        )
        charts = generate_charts(rows, out_dir, p)
        export_pdf(
            out_dir / "official_benchmark_results.pdf", rows, skipped, prov, charts, out_dir, p
        )
        print("\nDone - measurements untouched, reports re-rendered.")
        return 0

    t0 = time.perf_counter()
    rows, skipped = run_grid(distances, shot_list, args.p, args.seed, args.budget_seconds)
    elapsed = time.perf_counter() - t0

    if not rows:
        print("\nNo cell was measurable under this budget; nothing written.")
        return 1

    # The guard the withdrawn artifacts lacked: refuses to write a mixed-model set.
    model = assert_comparable(rows)
    print(f"\nassert_comparable OK — all {len(rows)} rows share noise model: {model}")

    add_speedups(rows)

    params = {
        "distances": distances,
        "shots": shot_list,
        "physical_error_rate": args.p,
        "seed": args.seed,
        "per_cell_budget_seconds": args.budget_seconds,
        "decoders": [k for _, k in DECODERS],
        "extrapolation": "none - infeasible cells are recorded as skipped",
        "skipped_cells": skipped,
    }
    json_path = out_dir / "official_benchmark_results.json"
    _provenance.write_artifact(
        json_path, rows, params, elapsed_seconds=elapsed,
        methodology="circuit_level", generator=Path(__file__).name,
    )
    print(f"[Exported] stamped JSON: {json_path}")

    _, prov = _provenance.load_artifact(json_path)
    export_csv(out_dir / "official_benchmark_results.csv", rows)
    export_markdown(
        out_dir / "official_benchmark_results.md", rows, skipped, prov, args.p,
        args.budget_seconds,
    )
    charts = generate_charts(rows, out_dir, args.p)
    export_pdf(
        out_dir / "official_benchmark_results.pdf", rows, skipped, prov, charts, out_dir, args.p
    )

    print(f"\nDone in {elapsed:,.1f}s — {len(rows)} measured, {len(skipped)} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
