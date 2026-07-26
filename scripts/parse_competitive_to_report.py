#!/usr/bin/env python
"""Parse competitive_results.json and append head-to-head market ranking table to report.md."""
import json
import os

REPORT_PATH = r"C:\Users\Clinque du Batiment\.gemini\antigravity-cli\brain\d4777725-ef6d-4e7d-b7a2-4bd3e9d2bd3d\report.md"
JSON_PATH = r"C:\Users\Clinque du Batiment\Desktop\qector-decoder-clean\competitive_results.json"


def main():
    if not os.path.exists(JSON_PATH):
        print(f"File {JSON_PATH} not found.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Sort data by distance, shots, decodes_per_s descending
    data.sort(key=lambda x: (x["distance"], x["shots"], -x["dec_per_s"]))

    lines = [
        "",
        "---",
        "",
        "## 4. Head-to-Head Market Competitor Ranking (PyMatching v2.4, LDPC, BeliefMatching)",
        "",
        "> **Direct Empirical Benchmark**: Head-to-head execution of QECTOR v3 decoders vs. Google/IBM **PyMatching v2.4 (C++)**, **`ldpc` BP-OSD (C++)** by Joschka Roffe, and **BeliefMatching** on identical error syndromes.",
        "",
        "| Distance ($d$) | Qubits ($n$) | Shots | Rank | Decoder & Implementation | LER ($p=0.005$) | Throughput (dec/s) | Latency ($\mu$s/dec) | **QECTOR Speedup vs PyMatching** |",
        "| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |",
    ]

    current_group = None
    rank = 1

    for r in data:
        group_key = (r["distance"], r["shots"])
        if group_key != current_group:
            current_group = group_key
            rank = 1
        else:
            rank += 1

        d = r["distance"]
        nq = r["n_qubits"]
        shots = r["shots"]
        dec = r["decoder"]
        ler = f"{r['ler']:.5f}"
        dec_s = f"{r['dec_per_s']:,.0f}"
        lat = f"{r['latency_us']:.2f}"

        # Find PyMatching throughput in same group
        pm_throughput = next(
            (x["dec_per_s"] for x in data if x["distance"] == d and x["shots"] == shots and "PyMatching" in x["decoder"]),
            None,
        )
        if pm_throughput and pm_throughput > 0:
            speedup = r["dec_per_s"] / pm_throughput
            speedup_str = f"**{speedup:.2f}x**" if speedup >= 1.0 else f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        lines.append(f"| **d={d}** | {nq} | {shots:,} | #{rank} | `{dec}` | {ler} | **{dec_s}** | {lat} | {speedup_str} |")

    content_to_append = "\n".join(lines)

    with open(REPORT_PATH, "a", encoding="utf-8") as fh:
        fh.write(content_to_append)

    print(f"Appended head-to-head market ranking with {len(data)} entries to {REPORT_PATH}")


if __name__ == "__main__":
    main()
