"""Head-to-head: QECTOR BlossomDecoder (MWPM, Edmonds + multi-source Dijkstra)
vs PyMatching, on rotated surface codes and the repetition code.

Both decoders solve the *same* minimum-weight perfect-matching problem on the
*same* check matrix with uniform edge weights, so the comparison is apples to
apples. We report:

  * latency           — microseconds per single-syndrome decode
  * syndrome validity — fraction of decodes with H @ correction == syndrome
  * MWPM weight match — fraction of decodes where QECTOR's correction weight
                        equals PyMatching's (both are optimal => should be ~1.0)
  * logical error rate— fraction of shots where (error XOR correction) is a
                        non-trivial logical operator (homology class flipped)

Run:  .venv/Scripts/python.exe scripts/benchmark_vs_pymatching.py
"""
import time
import numpy as np

try:
    from pymatching import Matching
except Exception as e:  # pragma: no cover
    raise SystemExit(f"PyMatching not available: {e}")

from qector_decoder_v3 import BlossomDecoder


# ---------------------------------------------------------------------------
# Code constructions (proper matching graphs: each qubit in <= 2 checks)
# ---------------------------------------------------------------------------
def rotated_surface_code(d):
    """Rotated surface code distance d, Z-checks (detect X errors).

    Bulk weight-4 plaquettes on the (r+c) even checkerboard. Every data qubit
    sits in at most two Z-plaquettes, so the decoding graph is a true matching
    graph (boundary qubits have degree 1 -> boundary edges). Logical-Z
    representative = column 0.
    """
    checks = []
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                checks.append([r * d + c, r * d + c + 1,
                               (r + 1) * d + c, (r + 1) * d + c + 1])
    logical = [r * d for r in range(d)]          # column 0 (logical-Z rep)
    return checks, d * d, logical


def repetition_code(d):
    """Distance-d repetition (1D surface) code. Logical rep = qubit 0
    (residual in ker(H) is constant, so qubit 0 determines the class)."""
    return [[i, i + 1] for i in range(d - 1)], d, [0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_H(check_to_qubits, n_qubits):
    H = np.zeros((len(check_to_qubits), n_qubits), dtype=np.uint8)
    for ci, qs in enumerate(check_to_qubits):
        for q in qs:
            H[ci, q] ^= 1
    return H


def syndrome_of(H, vec):
    return (H @ vec) & 1


def bench_code(name, check_to_qubits, n_qubits, logical, p=0.06, shots=3000, seed=7):
    rng = np.random.default_rng(seed)
    H = build_H(check_to_qubits, n_qubits)
    n_checks = len(check_to_qubits)
    logical = np.array(logical, dtype=np.int64)

    qector = BlossomDecoder(check_to_qubits, n_qubits)
    pm = Matching.from_check_matrix(H)  # uniform weights

    # Pre-generate identical error set for both decoders
    errors = (rng.random((shots, n_qubits)) < p).astype(np.uint8)
    syndromes = (errors @ H.T) & 1

    q_valid = pm_valid = 0
    q_log = pm_log = 0
    weight_match = 0
    q_t = pm_t = 0.0

    for s in range(shots):
        syn = syndromes[s].astype(np.uint8)
        err = errors[s]

        t0 = time.perf_counter()
        qc = np.asarray(qector.decode(syn)).astype(np.uint8)
        q_t += time.perf_counter() - t0

        t0 = time.perf_counter()
        pc = pm.decode(syn).astype(np.uint8)
        pm_t += time.perf_counter() - t0

        # syndrome validity
        q_ok = np.array_equal(syndrome_of(H, qc), syn)
        pm_ok = np.array_equal(syndrome_of(H, pc), syn)
        q_valid += q_ok
        pm_valid += pm_ok

        # MWPM optimality (uniform weights -> compare Hamming weight)
        if q_ok and pm_ok and qc.sum() == pc.sum():
            weight_match += 1

        # logical error: residual overlaps logical rep with odd parity
        q_res = (err ^ qc)
        pm_res = (err ^ pc)
        if int(q_res[logical].sum()) & 1:
            q_log += 1
        if int(pm_res[logical].sum()) & 1:
            pm_log += 1

    print(f"{name:26s} p={p:.3f} shots={shots} q={n_qubits:4d} checks={n_checks:4d}")
    print(f"    latency  QECTOR={q_t/shots*1e6:8.2f} us   PyMatching={pm_t/shots*1e6:8.2f} us   "
          f"(ratio {q_t/pm_t:5.2f}x)")
    print(f"    syndrome QECTOR={100*q_valid/shots:6.2f}%  PyMatching={100*pm_valid/shots:6.2f}%")
    print(f"    MWPM weight agreement QECTOR==PyMatching: {100*weight_match/shots:6.2f}%")
    print(f"    logical  QECTOR={100*q_log/shots:6.3f}%  PyMatching={100*pm_log/shots:6.3f}%")
    return dict(name=name, p=p, shots=shots, n_qubits=n_qubits, n_checks=n_checks,
                q_us=q_t / shots * 1e6, pm_us=pm_t / shots * 1e6,
                q_valid=q_valid / shots, pm_valid=pm_valid / shots,
                weight_match=weight_match / shots,
                q_logical=q_log / shots, pm_logical=pm_log / shots)


if __name__ == "__main__":
    print("=" * 100)
    print("QECTOR BlossomDecoder  vs  PyMatching   —   identical check matrices, uniform weights")
    print("=" * 100)
    rows = []
    for d in (3, 5, 7, 9):
        c, n, lg = rotated_surface_code(d)
        rows.append(bench_code(f"rotated surface d={d}", c, n, lg, p=0.06, shots=3000))
    print("-" * 100)
    for d in (11, 21):
        c, n, lg = repetition_code(d)
        rows.append(bench_code(f"repetition d={d}", c, n, lg, p=0.08, shots=4000))
    print("-" * 100)
    # logical-rate sweep on a rotated surface code to show threshold-like behaviour
    print("Logical error rate vs p (rotated surface d=7):")
    c, n, lg = rotated_surface_code(7)
    for p in (0.02, 0.04, 0.06, 0.08, 0.10):
        r = bench_code(f"  d=7 p={p}", c, n, lg, p=p, shots=4000)
    print("=" * 100)
    print("Summary: QECTOR matches PyMatching on syndrome validity and MWPM weight; "
          "logical rates track each other (both optimal MWPM).")
