#!/usr/bin/env python
"""Parse competitive_results.json from the repo root and append head-to-head ranking table to report.md."""
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JSON_PATH = os.path.join(_REPO, "competitive_results.json")
_OUT_DIR = os.path.join(_REPO, "benchmark_results")
_REPORT_PATH = os.path.join(_OUT_DIR, "report.md")


def main():
    if not os.path.exists(_JSON_PATH):
        print(f"File {_JSON_PATH} not found. Generate it by running one of the competitive_ranking scripts first.")
        return

    with open(_JSON_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Support both bare list and wrapper dict formats
    if isinstance(data, dict):
        data = data.get("results", data)

    # Sort data by distance, shots, decodes_per_s descending
    data.sort(key=lambda x: (x.get("distance", 0), x.get("shots", 0), -(x.get("dec_per_s") or x.get("decodes_per_s", 0))))

    lines = [
        "",
        "---",
        "",
        "## 4. Head-to-Head Market Competitor Ranking (circuit-level Stim pipeline)",
        "",
        "> **Direct Empirical Benchmark**: Comparison of QECTOR decoders vs PyMatching v2.4 "
        "on an identical circuit-level Stim pipeline (A1 methodology).",
        "",
        "| Distance ($d$) | Qubits ($n$) | Shots | Rank | Decoder & Implementation | LER ($p=0.005$) | Throughput (dec/s) | Latency ($\mu$s/dec) | **Speedup vs PyMatching** |",
        "| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |",
    ]

    current_group = None
    rank = 1

    for r in data:
        if r.get("status") == "unsupported" or "error" in r:
            continue

        group_key = (r.get("distance", 0), r.get("shots", 0))
        if group_key != current_group:
            current_group = group_key
            rank = 1
        else:
            rank += 1

        d = r.get("distance", "?")
        nq = r.get("n_qubits", "?")
        shots = r.get("shots", "?")
        dec = r.get("decoder", "?")
        ler = f"{r.get('ler', 0):.5f}"
        dec_s_val = r.get("dec_per_s") or r.get("decodes_per_s", 0)
        dec_s = f"{dec_s_val:,.0f}"
        lat = f"{r.get('latency_us', 0):.2f}"

        # Find PyMatching throughput in same group
        pm_throughput = next(
            (x.get("dec_per_s") or x.get("decodes_per_s", 0) for x in data
             if x.get("distance") == d and x.get("shots") == shots and "PyMatching" in x.get("decoder", "")),
            None,
        )
        if pm_throughput and pm_throughput > 0:
            speedup = dec_s_val / pm_throughput
            speedup_str = f"**{speedup:.2f}x**" if speedup >= 1.0 else f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        lines.append(f"| **d={d}** | {nq} | {shots:,} | #{rank} | `{dec}` | {ler} | **{dec_s}** | {lat} | {speedup_str} |")

    content_to_append = "\n".join(lines)

    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_REPORT_PATH, "a", encoding="utf-8") as fh:
        fh.write(content_to_append)

    print(f"Appended head-to-head market ranking ({len(data)} entries) to {_REPORT_PATH}")


if __name__ == "__main__":
    main()
