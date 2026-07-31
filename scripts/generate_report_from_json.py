#!/usr/bin/env python
"""Strict, fully-traceable generator for report.md from raw JSON outputs.

Reads benchmark JSON files from the repo root and writes report.md
to ``benchmark_results/`` (repo-local). Missing files are reported
but do not crash.
"""
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT_DIR = os.path.join(_REPO, "benchmark_results")
_REPORT_PATH = os.path.join(_OUT_DIR, "report.md")

# Look in the repo root for benchmark JSON files
_JSON_CANDIDATES = {
    "competitive_results.json": os.path.join(_REPO, "competitive_results.json"),
    "benchmark_results.json": os.path.join(_REPO, "benchmark_results.json"),
}


def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data.get("results", [])
    return data


# Column spec: (header, [accepted result keys], formatter). A column is emitted
# only when at least one row carries one of its keys, and every row then gets a
# cell for it. Deriving the header row and the data cells from one spec is what
# keeps them aligned: the previous version emitted cells by key-presence *per
# row* while taking headers from a caller-supplied list, so any row whose shape
# differed from the caller's assumption silently shifted every value one column
# to the left (e.g. a "rounds"/"detectors" DEM row rendered under the
# "Qubits (n)" header).
_COLUMN_SPEC = [
    ("Distance (d)", ["distance"], lambda v: f"**d={v}**"),
    ("Qubits (n)", ["n_qubits"], str),
    ("Rounds (r)", ["rounds"], str),
    ("Detectors", ["detectors"], str),
    ("Shots", ["shots"], lambda v: f"{v:,}"),
    ("Decoder", ["decoder"], lambda v: f"`{v}`"),
    ("LER (p=0.005)", ["ler"], lambda v: f"{v:.5f}"),
    ("95% Wilson CI", ["ci95"], lambda v: f"({v[0]:.5f}, {v[1]:.5f})"),
    ("Faithfulness %", ["faith_pct", "faithful_pct"], lambda v: f"**{v:.1f}%**"),
    ("Throughput (dec/s)", ["dec_per_s", "decodes_per_s"], lambda v: f"{v:,.0f}"),
    ("Latency (us/dec)", ["latency_us"], lambda v: f"{v:.2f}"),
]


def _cell(result, keys, fmt):
    """Formatted cell for the first present key, or an em dash when absent."""
    for key in keys:
        if key in result:
            try:
                return fmt(result[key])
            except (TypeError, ValueError, IndexError, KeyError):
                return str(result[key])
    return "—"


def format_table(results_list):
    """Render *results_list* as a Markdown table whose header matches its data.

    Columns are chosen from the rows themselves, so heterogeneous result files
    (surface-code rows vs. 3D DEM rows) each render under correct headers.
    """
    rows = [r for r in results_list if r.get("status") != "unsupported" and "error" not in r]
    if not rows:
        return "_No records._"

    active = [(h, keys, fmt) for h, keys, fmt in _COLUMN_SPEC if any(k in r for r in rows for k in keys)]
    if not active:
        return "_No records._"

    lines = [
        "| " + " | ".join(h for h, _, _ in active) + " |",
        "| " + " | ".join([":---:"] * len(active)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_cell(r, keys, fmt) for _, keys, fmt in active) + " |")
    return "\n".join(lines)


def main():
    # Discover available JSON files
    available = {name: load_json(path) for name, path in _JSON_CANDIDATES.items()}
    comp = available.get("competitive_results.json", [])
    emp = available.get("benchmark_results.json", [])

    # Filter successful rows
    comp_valid = [r for r in comp if r.get("status") != "unsupported" and "error" not in r]
    comp_10k = [r for r in comp_valid if r.get("shots") == 10000]
    comp_1k = [r for r in comp_valid if r.get("shots") == 1000]

    total_records = len(emp) + len(comp_valid)

    report = f"""# QECTOR v3 Empirical Benchmark Report (100% Traceable Raw JSON Data)

> **Empirical Traceability Notice**: Every row in the tables below is read directly from JSON
> benchmark result files in the repo root. Unsupported or fast-fail execution paths are strictly excluded.

---

## 1. Head-to-Head Market Competitor Benchmark (10,000 Shots, p=0.005)

{format_table(comp_10k)}

---

## 2. Low-Shot Market Competitor Benchmark (1,000 Shots, p=0.005)

{format_table(comp_1k)}

---

## 3. Standard Benchmark Results

{format_table(emp)}

---

## 4. Execution Traceability & Verification Summary
- **Total Valid JSON Records Loaded**: {total_records}
- **Sources**: {', '.join(k for k, v in available.items() if v)}
"""

    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Generated report.md with {total_records} traceable records at {_REPORT_PATH}")


if __name__ == "__main__":
    main()
