#!/usr/bin/env python
"""One command that produces the full QECTOR evidence bundle.

Runs the core benchmark/correctness scripts, collects every JSON/CSV/MD/PDF
artifact, the environment snapshot, frozen dependency lists, the git commit, and
a ``sha256sums.txt`` over everything — all into a single folder a third party can
inspect to reproduce the core claims.

    python scripts/run_due_diligence_bundle.py --out qector_evidence_bundle
    python scripts/run_due_diligence_bundle.py --out bundle_smoke --quick

``--quick`` runs small shot counts / low distances so the whole bundle finishes
in a couple of minutes (CI smoke); the default uses buyer-grade shot counts.
Every step is isolated: a failing step is recorded in ``manifest.json`` and does
not abort the bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO, "scripts")
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np  # noqa: E402

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import benchmarking as bm  # noqa: E402
from qector_decoder_v3 import codes  # noqa: E402


def _run(cmd, log_path, timeout):
    """Run a subprocess, tee output to a log, return (ok, returncode)."""
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False, cwd=_REPO)
        return proc.returncode == 0, proc.returncode
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def _correctness_audit(out_path, seed=20260622):
    """Inline correctness audit: every decoder faithful + CPU/GPU bit-identical
    on a fixed set of seeded syndromes across several codes. Real evidence, no
    external deps required."""
    rng = np.random.default_rng(seed)
    results = []
    cuda = qd.cuda_is_available()
    opencl = qd.opencl_is_available()
    for d in (3, 5, 7, 9):
        code = codes.rotated_surface_code(d)
        c2q, nq = code.check_to_qubits, code.n_qubits
        H = code.parity_check_matrix().astype(np.uint8)
        shots = 2000
        err = (rng.random((shots, nq)) < 0.06).astype(np.uint8)
        syn = ((err @ H.T) & 1).astype(np.uint8)

        cpu = qd.CPUBatchDecoder(c2q, nq)
        cpu_out = np.stack([np.asarray(cpu.decode(syn[i]), np.uint8) for i in range(shots)])
        faithful = {
            "cpu_batch": bool(np.array_equal((cpu_out @ H.T) & 1, syn)),
        }
        bit_identical = {}
        for name, ok, Dec in (("cuda", cuda, qd.CUDABatchDecoder),
                              ("opencl", opencl, qd.OpenCLBatchDecoder)):
            if not ok:
                bit_identical[name] = None
                continue
            g = Dec(c2q, nq)
            g_out = np.asarray(g.batch_decode(syn), np.uint8)
            bit_identical[name] = bool(np.array_equal(g_out, cpu_out))
            faithful[name] = bool(np.array_equal((g_out @ H.T) & 1, syn))
        for kind, Dec in (("blossom", qd.BlossomDecoder),
                          ("sparse_blossom", qd.SparseBlossomDecoder),
                          ("union_find", qd.UnionFindDecoder)):
            dec = Dec(c2q, nq)
            o = np.stack([np.asarray(dec.decode(syn[i]), np.uint8) for i in range(shots)])
            faithful[kind] = bool(np.array_equal((o @ H.T) & 1, syn))
        results.append({"code": code.name, "distance": d, "n_qubits": nq,
                        "shots": shots, "syndrome_faithful": faithful,
                        "gpu_bit_identical_to_cpu": bit_identical})
    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    audit = {"environment": env, "seed": seed,
             "all_faithful": all(all(v for v in r["syndrome_faithful"].values())
                                 for r in results),
             "results": results}
    bm.write_json(out_path, audit)
    return audit["all_faithful"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="qector_evidence_bundle")
    ap.add_argument("--quick", action="store_true",
                    help="small shot counts / low distances (CI smoke)")
    ap.add_argument("--step-timeout", type=float, default=3600.0)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    logs = os.path.join(out, "run_logs")
    os.makedirs(logs, exist_ok=True)
    py = sys.executable

    shots = 2000 if args.quick else 40000
    bshots = 1500 if args.quick else 10000
    dists = "3 5 7" if args.quick else "3 5 7 9 11 13 15"
    hi_dists = "9 11" if args.quick else "13 15"
    gpu_dists = "3 5" if args.quick else "3 5 7 9 11 13"
    gpu_batches = "1 64 1024" if args.quick else "1 64 1024 4096 16384 65536"
    wg_dists = "9 11" if args.quick else "13 15 17"
    nm_dists = "5 9" if args.quick else "5 9 13"

    steps = [
        ("competitive_stim_ler",
         [py, os.path.join(_SCRIPTS, "competitive_stim_ler.py"),
          "--distances", *dists.split(), "--shots", str(shots),
          "--out", os.path.join(out, "competitive_stim_ler")]),
        ("stim_ler_d13_d15",
         [py, os.path.join(_SCRIPTS, "competitive_stim_ler.py"),
          "--distances", *hi_dists.split(), "--shots", str(shots),
          "--out", os.path.join(out, "stim_ler_d13_d15")]),
        ("stim_ler_memz",
         [py, os.path.join(_SCRIPTS, "competitive_stim_ler.py"),
          "--task", "surface_code:rotated_memory_z",
          "--distances", *dists.split(), "--shots", str(shots),
          "--out", os.path.join(out, "stim_ler_memz")]),
        ("belief_extended",
         [py, os.path.join(_SCRIPTS, "belief_extended.py"),
          "--ms-distances", "3", "5", "--ms-shots", str(bshots // 2),
          "--ps-shots", str(bshots // 2), "--mz-shots", str(bshots // 2),
          "--out", os.path.join(out, "belief_extended")]),
        ("belief_grid",
         [py, os.path.join(_SCRIPTS, "belief_grid.py"),
          "--shots", str(bshots // 3),
          "--out", os.path.join(out, "belief_grid")]),
        ("weight_gap_analysis",
         [py, os.path.join(_SCRIPTS, "weight_gap_analysis.py"),
          "--distances", *wg_dists.split(), "--shots", str(min(shots, 3000)),
          "--out", os.path.join(out, "weight_gap_analysis")]),
        ("d15_mismatch_audit",
         [py, os.path.join(_SCRIPTS, "d15_mismatch_audit.py"),
          "--distance", "11" if args.quick else "15", "--shots", str(shots),
          "--out", os.path.join(out, "d15_mismatch_audit.csv")]),
        ("native_memory",
         [py, os.path.join(_SCRIPTS, "native_memory_profile.py"),
          "--distances", *nm_dists.split(),
          "--batch", "4096" if args.quick else "16384",
          "--out", os.path.join(out, "native_memory")]),
    ]
    if qd.cuda_is_available() or qd.opencl_is_available():
        steps.append(("gpu_extensive",
                      [py, os.path.join(_SCRIPTS, "gpu_extensive_test.py"),
                       "--distances", *gpu_dists.split(),
                       "--batches", *gpu_batches.split(),
                       "--out", os.path.join(out, "gpu_extensive")]))

    manifest = {"started_unix": int(time.time()), "quick": args.quick, "steps": []}

    # Inline correctness audit (no subprocess).
    try:
        ok = _correctness_audit(os.path.join(out, "correctness_audit.json"))
        manifest["steps"].append({"name": "correctness_audit", "ok": ok, "rc": 0})
        print(f"[correctness_audit] all_faithful={ok}")
    except Exception as exc:
        manifest["steps"].append({"name": "correctness_audit", "ok": False, "rc": str(exc)})
        print(f"[correctness_audit] FAILED: {exc}")

    for name, cmd in steps:
        print(f"[{name}] running...", flush=True)
        t0 = time.perf_counter()
        ok, rc = _run(cmd, os.path.join(logs, name + ".log"), args.step_timeout)
        dt = round(time.perf_counter() - t0, 1)
        manifest["steps"].append({"name": name, "ok": ok, "rc": rc, "seconds": dt})
        print(f"[{name}] ok={ok} rc={rc} ({dt}s)", flush=True)

    # Environment / dependency / provenance artifacts.
    env = bm.capture_environment()
    env["timestamp_unix"] = int(time.time())
    bm.write_json(os.path.join(out, "environment.json"), env)

    with open(os.path.join(out, "git_commit.txt"), "w", encoding="utf-8") as fh:
        fh.write((bm.git_commit() or "unknown") + "\n")

    freeze = subprocess.run([py, "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=False)
    with open(os.path.join(out, "pip_freeze.txt"), "w", encoding="utf-8") as fh:
        fh.write(freeze.stdout)

    cargo = subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"],
                           capture_output=True, text=True, check=False, cwd=_REPO)
    if cargo.returncode == 0 and cargo.stdout.strip():
        with open(os.path.join(out, "cargo_metadata.json"), "w", encoding="utf-8") as fh:
            fh.write(cargo.stdout)
        manifest["cargo_metadata"] = True
    else:
        manifest["cargo_metadata"] = False

    # Optional PDF report (reportlab/matplotlib).
    pdf_ok = False
    try:
        rc = subprocess.run(
            [py, os.path.join(_SCRIPTS, "generate_report_pdf.py"),
             "--out", os.path.join(out, "full_report.pdf"),
             "--stamp", bm.git_commit()],
            capture_output=True, text=True, timeout=600, check=False, cwd=_REPO)
        pdf_ok = rc.returncode == 0 and os.path.exists(os.path.join(out, "full_report.pdf"))
    except Exception:
        pdf_ok = False
    manifest["full_report_pdf"] = pdf_ok

    # sha256sums over every artifact (except the sums file itself).
    sums = []
    for root, _dirs, files in os.walk(out):
        for f in sorted(files):
            if f == "sha256sums.txt":
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            try:
                sums.append(f"{_sha256(p)}  {rel}")
            except Exception:
                pass
    with open(os.path.join(out, "sha256sums.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(sums)) + "\n")

    manifest["finished_unix"] = int(time.time())
    manifest["n_artifacts"] = len(sums)
    bm.write_json(os.path.join(out, "manifest.json"), manifest)

    n_ok = sum(1 for s in manifest["steps"] if s["ok"])
    print(f"\nBundle written to {out}")
    print(f"  steps ok: {n_ok}/{len(manifest['steps'])}   artifacts: {len(sums)}   "
          f"pdf: {pdf_ok}   git: {bm.git_commit()}")
    return 0 if n_ok == len(manifest["steps"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
