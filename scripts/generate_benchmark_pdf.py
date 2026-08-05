#!/usr/bin/env python
"""Render `full_decoder_benchmark.json` into a PDF report with charts and tables.

Reads only the JSON the benchmark wrote -- it runs no measurements of its own, so
every number in the PDF is traceable to a recorded run. Rows the benchmark marked
`failed` or `unavailable` are reported in their own section rather than dropped,
because a decoder that could not run is a result.

Usage::

    python scripts/generate_benchmark_pdf.py
    python scripts/generate_benchmark_pdf.py --in benchmark_results/full_decoder_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Validated categorical palette, light surface, assigned in fixed slot order and
# never cycled. A 9th series folds into the "other decoders" table rather than
# taking a generated hue.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"

TITLE = "QECTOR Decoder v3 — Decoder Family Benchmark"


def style_axes(ax):
    """Recessive grid and axes; the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def chart_latency_at(rows, distance, path):
    """Latency by decoder at one distance.

    A dot plot on a log axis, not bars: latency here spans four decades
    (about 12 us to 192,000 us), and on a linear bar chart every decoder except
    the slowest collapses to an invisible sliver. Bar *length* is supposed to
    encode magnitude proportionally, which a log axis breaks -- so the mark
    becomes a dot, whose position (not length) carries the value.
    """
    sel = [r for r in rows if r["status"] == "ok" and r["distance"] == distance]
    if not sel:
        return None
    sel.sort(key=lambda r: r["latency_us"], reverse=True)
    names = [r["decoder"] for r in sel]
    vals = [r["latency_us"] for r in sel]

    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(sel) + 1.3))
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    lo = min(vals) / 2.5
    # Leader line from the axis to the dot: keeps the row readable across a wide
    # axis without implying a proportional length.
    ax.hlines(names, lo, vals, color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(vals, names, s=52, color=SERIES[0], zorder=3,
               edgecolors=SURFACE, linewidths=1.2)
    ax.set_xscale("log")
    for name, v in zip(names, vals):
        ax.annotate(f"{v:,.2f}", (v, name), textcoords="offset points", xytext=(9, 0),
                    va="center", ha="left", fontsize=7.5, color=INK)
    ax.set_xlim(lo, max(vals) * 6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("microseconds per shot, log scale (lower is better)",
                  fontsize=8.5, color=INK_2)
    ax.set_title(f"Decode latency at d={distance}", fontsize=11, color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def _series_by_decoder(rows, key):
    out: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r["status"] != "ok" or r.get(key) is None:
            continue
        out.setdefault(r["decoder"], []).append((r["distance"], r[key]))
    for pts in out.values():
        pts.sort()
    return out


def _merge_identical(data, names):
    """Collapse decoders whose series are numerically identical into one entry.

    Several families here produce bit-identical corrections (the weighted
    Union-Find variants; every decoder that falls back to the unweighted
    pre-filter). Drawing them as separate lines hides all but the last one
    painted, so the legend ends up naming lines the reader cannot see -- and
    identity by colour alone is exactly what must not happen. Merging says the
    true thing instead: these decoders scored the same.
    """
    groups: list[tuple[list[str], list[tuple[int, float]]]] = []
    for name in names:
        pts = data.get(name)
        if not pts:
            continue
        for labels, existing in groups:
            if existing == pts:
                labels.append(name)
                break
        else:
            groups.append(([name], pts))
    return groups


def chart_lines(rows, key, names, title, ylabel, path, logy=True):
    """Change-over-distance for a capped set of decoders -> line chart, one axis."""
    data = _series_by_decoder(rows, key)
    groups = _merge_identical(data, names)
    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    style_axes(ax)
    for i, (labels, pts) in enumerate(groups[: len(SERIES)]):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = labels[0] if len(labels) == 1 else " = ".join(labels)
        ax.plot(xs, ys, color=SERIES[i], linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("code distance d", fontsize=8.5, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    xs_all = sorted({d for pts in data.values() for d, _ in pts})
    ax.set_xticks(xs_all)
    # A legend is always present for >= 2 series, so identity is never
    # colour-alone. Placed below the axes rather than "best" so it can never
    # occlude the data it labels.
    leg = ax.legend(frameon=False, fontsize=8, loc="upper center",
                    bbox_to_anchor=(0.5, -0.16), ncol=2)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def fitted_image(path, width_mm):
    """Image scaled to *width_mm*, height derived from the PNG's own aspect ratio.

    reportlab's ``kind="proportional"`` needs both dimensions; passing width
    alone leaves height None and blows up inside the layout engine.
    """
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(path).getSize()
    w = width_mm * mm
    return Image(path, width=w, height=w * ih / iw)


def make_table(header, body, col_widths=None):
    data = [header] + body
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(INK_2)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(GRID)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f3ef")]),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src",
                    default=os.path.join(_REPO, "benchmark_results", "full_decoder_benchmark.json"))
    ap.add_argument("--out", default=os.path.join(_REPO, "benchmark_results", "QECTOR_v3_decoder_benchmark.pdf"))
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        print(f"ERROR: benchmark JSON not found: {args.src}", file=sys.stderr)
        return 1
    with open(args.src, encoding="utf-8") as fh:
        payload = json.load(fh)

    rows = payload["results"]
    ok = [r for r in rows if r["status"] == "ok"]
    not_ok = [r for r in rows if r["status"] != "ok"]
    distances = sorted({r["distance"] for r in rows})
    params = payload["parameters"]
    env = payload["environment"]
    lic = payload.get("license", {})

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # -- charts -------------------------------------------------------------
    max_d = max(distances) if distances else 0
    p_lat = chart_latency_at(ok, max_d, os.path.join(charts_dir, "latency_max_d.png"))

    accuracy_names = [
        "qector:blossom", "qector:sparse_blossom", "competitor:pymatching",
        "qector:union_find", "qector:fast_union_find", "qector:bp_osd",
        "qector:hybrid_cascade", "qector:lookup_table",
    ]
    p_ler = chart_lines(ok, "ler", accuracy_names,
                        "Logical error rate vs code distance", "logical error rate",
                        os.path.join(charts_dir, "ler_vs_distance.png"), logy=True)

    backend_names = [
        "qector:union_find", "gpu:cuda_union_find", "gpu:opencl_union_find", "gpu:cuda_bp_osd",
    ]
    p_gpu = chart_lines(ok, "latency_us", backend_names,
                        "CPU vs GPU decode latency vs code distance", "microseconds per shot",
                        os.path.join(charts_dir, "cpu_vs_gpu.png"), logy=True)

    # -- document -----------------------------------------------------------
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, leading=21,
                        textColor=colors.HexColor(INK), alignment=TA_LEFT, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, leading=15,
                        textColor=colors.HexColor(INK), spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=8.8, leading=12.4,
                          textColor=colors.HexColor(INK_2))
    small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=10.4)

    doc = SimpleDocTemplate(
        args.out, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
        title=TITLE, author="QECTOR",
    )
    S: list = []
    S.append(Paragraph(TITLE, h1))
    S.append(Paragraph(
        f"QECTOR v{payload['qector_version']} &nbsp;·&nbsp; generated {payload['generated_utc'][:19]}Z", small))
    S.append(Spacer(1, 4))
    S.append(HRFlowable(width="100%", color=colors.HexColor(GRID), thickness=0.8))
    S.append(Spacer(1, 8))

    S.append(Paragraph("Method", h2))
    S.append(Paragraph(
        f"Every row is measured against syndromes <b>sampled from a Stim circuit</b> "
        f"(<font face='Courier'>{params['circuit']}</font>, rounds = distance, circuit-level "
        f"depolarising noise p = {params['noise']}), so each syndrome is physically realizable "
        f"and admits a correction. Decoders are built from that circuit's detector error model "
        f"collapsed to a graph, so all of them decode the same problem from the same shots. "
        f"Logical error rate is measured against the circuit's own recorded observable flips, "
        f"with a Wilson 95% interval. Latency is the fastest of {params['repeats']} timed "
        f"repeats over {params['shots']:,} shots, preferring each decoder's batch entry point.", body))
    S.append(Spacer(1, 4))
    S.append(Paragraph(
        "<b>Why this replaces the previous harness numbers.</b> The earlier "
        "<font face='Courier'>benchmarks_session</font> harnesses fed uniform-random bit patterns "
        "to a boundaryless ring code. Roughly 49% of those inputs carry odd defect parity, so no "
        "correction exists; a Union-Find decoder answers by growing clusters until they span the "
        "graph. Every backend measured about 2 ms/shot of that non-decode, which is why GPU paths "
        "appeared slower than CPU. Those figures describe the workload, not the decoders.", body))

    S.append(Paragraph("Environment", h2))
    S.append(make_table(
        ["Field", "Value"],
        [
            ["License tier", f"{lic.get('tier','?')} (key_status={lic.get('key_status','?')}, "
                             f"max_distance={lic.get('max_distance','?')}, gpu={lic.get('gpu_enabled','?')})"],
            ["Platform", env.get("platform", "?")],
            ["Processor", env.get("processor", "?") or "n/a"],
            ["Python / NumPy", f"{env.get('python','?')} / {env.get('numpy','?')}"],
            ["Distances", ", ".join(f"d={d}" for d in distances)],
            ["Shots per point", f"{params['shots']:,}"],
            ["Seed", str(params["seed"])],
            ["Measurements", f"{len(ok)} succeeded, {len(not_ok)} did not run"],
        ],
        col_widths=[34 * mm, 144 * mm],
    ))

    if p_lat:
        S.append(Paragraph(f"Decode latency at d={max_d}", h2))
        S.append(fitted_image(p_lat, 178))

    S.append(PageBreak())
    if p_ler:
        S.append(Paragraph("Accuracy scaling", h2))
        S.append(fitted_image(p_ler, 178))
        S.append(Paragraph(
            "Lower is better. A point at zero logical errors is plotted at the resolution floor of "
            "the shot count and should be read as an upper bound, not a measured rate.", small))
    if p_gpu:
        S.append(Paragraph("CPU versus GPU", h2))
        S.append(fitted_image(p_gpu, 178))

    S.append(PageBreak())
    S.append(Paragraph("Full results", h2))
    for d in distances:
        sel = [r for r in ok if r["distance"] == d]
        if not sel:
            continue
        sel.sort(key=lambda r: r["latency_us"])
        meta = sel[0]
        S.append(Paragraph(
            f"<b>d={d}</b> &nbsp;·&nbsp; {meta['detectors']} detectors, {meta['mechanisms']} "
            f"error mechanisms, {meta['shots']:,} shots", body))
        S.append(Spacer(1, 3))
        S.append(make_table(
            ["Decoder", "LER", "95% CI", "Errors", "Faithful", "µs/shot", "shots/s"],
            [[
                r["decoder"],
                f"{r['ler']:.5f}",
                f"{r['ler_ci95_lo']:.5f}–{r['ler_ci95_hi']:.5f}",
                f"{r['logical_errors']:,}",
                "n/a" if r.get("faithful_frac") is None else f"{r['faithful_frac']*100:.2f}%",
                f"{r['latency_us']:,.2f}",
                f"{r['throughput_shots_s']:,.0f}",
            ] for r in sel],
            col_widths=[42 * mm, 19 * mm, 32 * mm, 16 * mm, 20 * mm, 22 * mm, 27 * mm],
        ))
        S.append(Spacer(1, 8))

    if not_ok:
        S.append(Paragraph("Did not run", h2))
        S.append(Paragraph(
            "Listed rather than omitted: a decoder that could not be constructed or that raised "
            "during decode is a result about this build and this machine.", body))
        S.append(Spacer(1, 3))
        S.append(make_table(
            ["d", "Decoder", "Status", "Reason"],
            [[str(r["distance"]), r["decoder"], r["status"], (r.get("error", "") or "")[:110]]
             for r in sorted(not_ok, key=lambda x: (x["decoder"], x["distance"]))],
            col_widths=[10 * mm, 40 * mm, 22 * mm, 106 * mm],
        ))

    S.append(Spacer(1, 10))
    S.append(HRFlowable(width="100%", color=colors.HexColor(GRID), thickness=0.8))
    S.append(Paragraph(
        f"Source data: <font face='Courier'>{os.path.relpath(args.src, _REPO)}</font> · "
        f"rendered {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} · no measurement is performed by "
        f"this report generator.", small))

    doc.build(S)
    print(f"wrote {args.out}")
    for p in (p_lat, p_ler, p_gpu):
        if p:
            print(f"  chart {os.path.relpath(p, _REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
