from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(r"C:\Users\Clinque du Batiment\Downloads\qector-decoder-clone")
HARNESS = ROOT / "benchmarks_session" / "harnesses"
LOGDIR = HARNESS / "run_logs_20260730_143625"


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = HARNESS / f"export_real_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)

    for name in [
        "bb_results.json",
        "competitive_results.json",
        "mega_results.json",
        "final_h2h_benchmark.csv",
        "TODO_enterprise_harness_test.md",
    ]:
        src = HARNESS / name
        if src.exists():
            shutil.copy2(src, raw / name)

    if LOGDIR.exists():
        dst = raw / LOGDIR.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(LOGDIR, dst)

    records: list[dict] = []
    notes: list[dict] = []

    def add_record(**kwargs) -> None:
        records.append(kwargs)

    bb = HARNESS / "bb_results.json"
    if bb.exists():
        data = json.loads(bb.read_text(encoding="utf-8"))
        notes.append(
            {
                "source": "bb_results.json",
                "status": "completed",
                "rows": len(data.get("rows", [])),
                "note": "Fresh completed bb_bench.py run from this session.",
            }
        )
        for row in data.get("rows", []):
            if row.get("us_per_shot") is None:
                continue
            add_record(
                source="bb_results.json",
                suite="bivariate_bicycle",
                status="real_measured",
                axis=row.get("p"),
                decoder=row.get("decoder"),
                family=row.get("family"),
                shots=row.get("shots"),
                us_per_shot=row.get("us_per_shot"),
                shots_per_s=row.get("shots_per_s"),
                error_rate=row.get("logical_err"),
                faithful=row.get("faithful"),
                note="fresh run",
            )

    competitive = HARNESS / "competitive_results.json"
    if competitive.exists():
        data = json.loads(competitive.read_text(encoding="utf-8"))
        notes.append(
            {
                "source": "competitive_results.json",
                "status": "existing_complete",
                "rows": len(data.get("rows", [])),
                "note": "Complete file already present. Today's competitive.py run was stopped before it rewrote JSON; partial stdout is in raw logs.",
            }
        )
        for row in data.get("rows", []):
            status = "real_measured" if row.get("us_per_shot") is not None else "failed_or_unavailable"
            add_record(
                source="competitive_results.json",
                suite="surface_competitive",
                status=status,
                axis=row.get("distance"),
                decoder=row.get("decoder"),
                family=row.get("family"),
                shots=data.get("meta", {}).get("shots"),
                us_per_shot=row.get("us_per_shot"),
                shots_per_s=row.get("shots_per_s"),
                error_rate=row.get("ler"),
                faithful=None,
                note=row.get("note", ""),
            )

    mega = HARNESS / "mega_results.json"
    if mega.exists():
        data = json.loads(mega.read_text(encoding="utf-8"))
        notes.append(
            {
                "source": "mega_results.json",
                "status": "existing_complete",
                "rows": len(data.get("rows", [])),
                "note": "Existing shot ladder file, not regenerated after stop request.",
            }
        )
        for row in data.get("rows", []):
            status = "real_measured" if row.get("status") == "ok" and row.get("us_per_shot") is not None else "skipped_or_failed"
            add_record(
                source="mega_results.json",
                suite="shot_ladder",
                status=status,
                axis=row.get("distance"),
                decoder=row.get("decoder"),
                family=row.get("family"),
                shots=row.get("shots"),
                us_per_shot=row.get("us_per_shot"),
                shots_per_s=row.get("shots_per_s"),
                error_rate=row.get("ler"),
                faithful=None,
                note=row.get("status", ""),
            )

    h2h = HARNESS / "final_h2h_benchmark.csv"
    if h2h.exists():
        rows = list(csv.DictReader(h2h.open(encoding="utf-8")))
        notes.append(
            {
                "source": "final_h2h_benchmark.csv",
                "status": "existing_complete",
                "rows": len(rows),
                "note": "Existing adaptive h2h file, not regenerated after stop request.",
            }
        )
        for row in rows:
            p_value = float(row["p"])
            for name in ["ldpc", "qector"]:
                add_record(
                    source="final_h2h_benchmark.csv",
                    suite="bb_h2h_adaptive",
                    status="real_measured",
                    axis=p_value,
                    decoder=name,
                    family="competitor" if name == "ldpc" else "qector",
                    shots=int(float(row[f"{name}_shots"])),
                    us_per_shot=float(row[f"{name}_us_per_shot"]),
                    shots_per_s=None,
                    error_rate=float(row[f"{name}_ler"]),
                    faithful=None,
                    note=f"timing_mode={row.get('timing_mode', '')}; failures={row.get(name + '_failures')}",
                )

    log_notes: list[dict] = []
    if LOGDIR.exists():
        for log_file in sorted(LOGDIR.glob("*.log")):
            text = log_file.read_text(encoding="utf-8", errors="replace")
            status = "completed"
            if "Traceback" in text or "TypeError:" in text:
                status = "failed"
            if log_file.name == "competitive.log":
                status = "stopped_partial"
            log_notes.append({"log": log_file.name, "status": status, "bytes": log_file.stat().st_size})
            if "AutoDecoder.__init__() takes" in text:
                notes.append(
                    {
                        "source": log_file.name,
                        "status": "stale_api_failure",
                        "rows": None,
                        "note": "Harness uses old AutoDecoder positional signature against upgraded wheel.",
                    }
                )
            if "NVRTC_ERROR_COMPILATION" in text:
                notes.append(
                    {
                        "source": log_file.name,
                        "status": "cuda_compile_failure",
                        "rows": None,
                        "note": "CUDA availability is true, but runtime kernel compilation failed; no CUDA speed number counted.",
                    }
                )
            if "OpenCLBatchDecoder" in text and "CPU batch_decode_par" in text:
                notes.append(
                    {
                        "source": log_file.name,
                        "status": "opencl_real_measured",
                        "rows": None,
                        "note": "OpenCL executed, but measured slower than CPU reference in this run.",
                    }
                )

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": "real rows only; stale API failures and stopped partial runs are labeled, not converted into benchmark wins",
        "record_count": len(records),
        "notes": notes,
        "logs": log_notes,
        "records": records,
    }
    (out / "real_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = [
        "source",
        "suite",
        "status",
        "axis",
        "decoder",
        "family",
        "shots",
        "us_per_shot",
        "shots_per_s",
        "error_rate",
        "faithful",
        "note",
    ]
    with (out / "real_export_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    with (out / "real_export_notes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "status", "rows", "note"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(notes)

    real = [row for row in records if row["status"] == "real_measured"]
    write_markdown(out, summary, real)
    write_pdf(out, summary, real)

    pdf = out / "real_harness_export.pdf"
    reader = PdfReader(str(pdf))
    (out / "pdf_verify.txt").write_text(f"pages={len(reader.pages)}\nbytes={pdf.stat().st_size}\n", encoding="utf-8")

    print(out)
    print(f"records={len(records)} real={len(real)} pdf_pages={len(reader.pages)}")
    for path in sorted(out.iterdir()):
        print(path.name, path.stat().st_size if path.is_file() else "dir")


def write_markdown(out: Path, summary: dict, real: list[dict]) -> None:
    lines = [
        "# Real Harness Export",
        "",
        f"Generated: {summary['generated']}",
        "",
        "This export keeps real measurements separate from stopped, stale, skipped, or failed paths.",
        "",
        f"- Total normalized records: {summary['record_count']}",
        f"- Real measured records: {len(real)}",
        f"- Export folder: `{out}`",
        "",
        "## Notes",
    ]
    for note in summary["notes"]:
        lines.append(f"- `{note['source']}` [{note['status']}]: {note['note']}")
    lines += ["", "## Fastest Real Rows By Suite"]
    for suite in sorted({row["suite"] for row in real}):
        rows = [row for row in real if row["suite"] == suite and row.get("us_per_shot") is not None]
        rows = sorted(rows, key=lambda row: float(row["us_per_shot"]))[:12]
        lines += [
            f"### {suite}",
            "| axis | decoder | us/shot | error | shots | note |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            error = "" if row.get("error_rate") is None else format(float(row["error_rate"]), ".6g")
            lines.append(
                f"| {row.get('axis')} | {row.get('decoder')} | {float(row['us_per_shot']):.4f} | {error} | {row.get('shots')} | {row.get('note', '')} |"
            )
        lines.append("")
    (out / "real_export_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_pdf(out: Path, summary: dict, real: list[dict]) -> None:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("HarnessH1", parent=styles["Heading1"], fontSize=17, textColor=colors.HexColor("#123a5f"), spaceAfter=8)
    h2 = ParagraphStyle("HarnessH2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#123a5f"), spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("HarnessBody", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("HarnessSmall", parent=styles["BodyText"], fontSize=7, leading=9)

    pdf = out / "real_harness_export.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = [
        Paragraph("Real Harness Export", h1),
        Paragraph(f"Generated {summary['generated']}", body),
        Paragraph(
            "Only real measurements are summarized as data. Stopped runs, stale harness API failures, skipped paths, and runtime failures are labeled separately.",
            body,
        ),
        Spacer(1, 8),
        Paragraph("Run Notes", h2),
    ]
    for note in summary["notes"]:
        story.append(Paragraph(f"<b>{note['source']}</b> [{note['status']}]: {note['note']}", small))

    story += [Spacer(1, 8), Paragraph("Record Counts", h2)]
    counts: dict[tuple[str, str], int] = {}
    for row in summary["records"]:
        key = (row["suite"], row["status"])
        counts[key] = counts.get(key, 0) + 1
    count_table = [["Suite", "Status", "Rows"]] + [[suite, status, str(value)] for (suite, status), value in sorted(counts.items())]
    story.append(styled_table(count_table, [190, 170, 70]))

    for suite in sorted({row["suite"] for row in real}):
        story.append(PageBreak())
        story.append(Paragraph(f"Fastest Real Rows: {suite}", h2))
        rows = [row for row in real if row["suite"] == suite and row.get("us_per_shot") is not None]
        rows = sorted(rows, key=lambda row: float(row["us_per_shot"]))[:25]
        data = [["Axis", "Decoder", "us/shot", "Error", "Shots", "Note"]]
        for row in rows:
            error = "" if row.get("error_rate") is None else f"{float(row['error_rate']):.6g}"
            data.append(
                [
                    str(row.get("axis")),
                    str(row.get("decoder"))[:42],
                    f"{float(row['us_per_shot']):.4f}",
                    error,
                    str(row.get("shots")),
                    str(row.get("note", ""))[:35],
                ]
            )
        story.append(styled_table(data, [45, 190, 65, 65, 65, 95]))
    doc.build(story)


def styled_table(data: list[list[str]], widths: list[int]) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


if __name__ == "__main__":
    main()
