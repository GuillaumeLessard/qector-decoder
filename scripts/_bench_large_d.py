"""Focused latency/optimality probe at large d (d=21,25,31) for BlossomDecoder
vs PyMatching. Temporary measurement harness — reuses benchmark_vs_pymatching.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_vs_pymatching import rotated_surface_code, build_H, syndrome_of
from pymatching import Matching
from qector_decoder_v3 import BlossomDecoder


def _per_decode_min(decode_fn, inputs, trials=9):
    """Minimum over `trials` of the per-decode mean time (seconds). Timing the
    whole pass and dividing by N avoids per-call perf_counter granularity noise;
    the MIN over trials is the cleanest estimate of true compute — scheduling,
    GC and turbo transitions only ever *add* time, so the fastest trial is the
    least-perturbed one. (Critical here because PyMatching is so fast that its
    absolute time is small and otherwise swamped by run-to-run noise.)"""
    n = len(inputs)
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        for x in inputs:
            decode_fn(x)
        best = min(best, (time.perf_counter() - t0) / n)
    return best


def bench(d, p=0.06, shots=2000, seed=7, warmup=300, trials=9):
    rng = np.random.default_rng(seed)
    c, n, lg = rotated_surface_code(d)
    H = build_H(c, n)
    q = BlossomDecoder(c, n)
    pm = Matching.from_check_matrix(H)
    errors = (rng.random((shots, n)) < p).astype(np.uint8)
    syndromes = ((errors @ H.T) & 1).astype(np.uint8)
    # Pre-convert inputs to each decoder's native form OUTSIDE the timed region
    # so we measure pure decode work, identical syndromes for both.
    syn_list = [syndromes[s] for s in range(shots)]
    ndef = int(syndromes.sum())

    # warmup (both)
    for s in range(min(warmup, shots)):
        q.decode(syn_list[s])
        pm.decode(syn_list[s])

    # correctness pass (untimed)
    wm = qv = pmv = 0
    for syn in syn_list:
        qc = np.asarray(q.decode(syn)).astype(np.uint8)
        pc = pm.decode(syn).astype(np.uint8)
        qok = np.array_equal(syndrome_of(H, qc), syn)
        pmok = np.array_equal(syndrome_of(H, pc), syn)
        qv += qok
        pmv += pmok
        if qok and pmok and qc.sum() == pc.sum():
            wm += 1

    qt = _per_decode_min(q.decode, syn_list, trials)
    pmt = _per_decode_min(pm.decode, syn_list, trials)

    print(f"d={d:2d} q={n:4d} checks={len(c):4d} avg_defects={ndef/shots:5.1f}")
    print(f"    latency  QECTOR={qt*1e6:8.2f}us  PyMatching={pmt*1e6:8.2f}us  ratio={qt/pmt:5.2f}x")
    print(f"    syndrome QECTOR={100*qv/shots:6.2f}%  PyMatching={100*pmv/shots:6.2f}%   MWPM-weight-agree={100*wm/shots:6.2f}%")


if __name__ == "__main__":
    for d in (21, 25, 31):
        bench(d)
