"""QECTOR v3 — Benchmark robuste avec métriques complètes + LER.

Usage: python scripts/benchmark_ler.py

Sortie JSON sur stdout.
"""

from __future__ import annotations

import json
import platform
import time
import statistics
from typing import Dict, Any, List

import numpy as np

import qector_decoder_v3 as qd


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _cpu_info() -> str:
    """Retourne le nom du CPU si disponible."""
    try:
        # Windows
        import wmi
        c = wmi.WMI()
        for proc in c.Win32_Processor():
            return proc.Name
    except Exception:
        pass
    try:
        # Linux / macOS fallback
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _percentile(sorted_data: List[float], p: float) -> float:
    """Percentile sur liste déjà triée."""
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# ---------------------------------------------------------------------------
# Benchmark latence
# ---------------------------------------------------------------------------

def benchmark_latency(
    decoder,
    n_checks: int,
    n_warmup: int = 1000,
    n_runs: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Benchmark robuste : warm-up + statistiques complètes."""
    rng = np.random.default_rng(seed)
    syndromes = rng.integers(0, 2, size=(n_runs, n_checks), dtype=np.uint8)

    # Warm-up
    for s in syndromes[:n_warmup]:
        decoder.decode(s)

    # Mesure
    times = []
    for s in syndromes:
        t0 = time.perf_counter_ns()
        decoder.decode(s)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    times_sorted = sorted(times)
    total = sum(times) / 1e3  # μs
    mean = total / n_runs
    std = statistics.stdev(times) / 1e3  # μs
    p50 = _percentile(times_sorted, 50) / 1e3
    p90 = _percentile(times_sorted, 90) / 1e3
    p99 = _percentile(times_sorted, 99) / 1e3
    min_t = min(times) / 1e3
    max_t = max(times) / 1e3
    throughput = n_runs / (sum(times) / 1e9)

    return {
        "n_runs": n_runs,
        "n_warmup": n_warmup,
        "latency_mean_us": round(mean, 3),
        "latency_std_us": round(std, 3),
        "latency_p50_us": round(p50, 3),
        "latency_p90_us": round(p90, 3),
        "latency_p99_us": round(p99, 3),
        "latency_min_us": round(min_t, 3),
        "latency_max_us": round(max_t, 3),
        "throughput": round(throughput, 1),
    }


# ---------------------------------------------------------------------------
# Benchmark LER (Logical Error Rate)
# ---------------------------------------------------------------------------

def benchmark_ler(
    check_to_qubits: List[List[int]],
    n_qubits: int,
    error_rate: float = 0.05,
    n_shots: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare le LER de UnionFind vs Blossom sur un modèle d'erreur simple.

    Le LER est estimé en injectant des erreurs aléatoires, mesurant le syndrome,
    décodant, et vérifiant si la correction produit un cycle non-trivial (logical
    operator).

    Pour un code simple (anneau / chaîne), on utilise la parité de la somme des
    corrections comme proxy de l'erreur logique.
    """
    rng = np.random.default_rng(seed)
    n_checks = len(check_to_qubits)

    # Construire la matrice de check (checks x qubits)
    check_matrix = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    for ci, qubits in enumerate(check_to_qubits):
        for q in qubits:
            if q < n_qubits:
                check_matrix[ci, q] = 1

    # Générateurs de décodeurs
    uf_dec = qd.UnionFindDecoder(check_to_qubits, n_qubits)
    bl_dec = qd.BlossomDecoder(check_to_qubits)

    uf_logical_errors = 0
    bl_logical_errors = 0
    mismatch = 0

    for _ in range(n_shots):
        # Erreur aléatoire sur les qubits
        error = rng.random(n_qubits) < error_rate
        error = error.astype(np.uint8)

        # Syndrome = error * check_matrix (mod 2)
        syndrome = (error @ check_matrix.T) % 2
        syndrome = syndrome.astype(np.uint8)

        # Décode
        uf_corr = uf_dec.decode(syndrome)
        bl_corr = bl_dec.decode(syndrome)

        # Vérifier si la correction est égale à l'erreur (mod 2) pour un code
        # simple. Pour un code en anneau/chaîne, l'erreur logique est détectée
        # quand la correction + l'erreur forme un cycle non-trivial.
        # Approximation : on compare la parité de la somme des corrections.
        uf_parity = int(uf_corr.sum()) % 2
        err_parity = int(error.sum()) % 2
        if uf_parity != err_parity:
            uf_logical_errors += 1

        bl_parity = int(bl_corr.sum()) % 2
        if bl_parity != err_parity:
            bl_logical_errors += 1

        if not np.array_equal(uf_corr, bl_corr):
            mismatch += 1

    return {
        "n_shots": n_shots,
        "error_rate": error_rate,
        "uf_ler": round(uf_logical_errors / n_shots, 6),
        "bl_ler": round(bl_logical_errors / n_shots, 6),
        "mismatch_rate": round(mismatch / n_shots, 6),
        "uf_logical_errors": uf_logical_errors,
        "bl_logical_errors": bl_logical_errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cpu = _cpu_info()
    print(f"# CPU: {cpu}")
    print(f"# Python: {platform.python_version()}")
    print(f"# QECTOR: {qd.__version__}")
    print()

    # --- Petit code : anneau 4 qubits --------------------------------------
    print("## Petit code : anneau 4 qubits\n")
    c2q_ring = [[0, 1], [1, 2], [2, 3], [3, 0]]

    ring_results = {}
    for name, dec in [
        ("UnionFind", qd.UnionFindDecoder(c2q_ring)),
        ("FastUnionFind", qd.FastUnionFindDecoder(c2q_ring)),
    ]:
        ring_results[name] = benchmark_latency(dec, len(c2q_ring), n_runs=20000)
        r = ring_results[name]
        print(f"### {name}")
        print(f"  mean={r['latency_mean_us']} µs  std={r['latency_std_us']} µs")
        print(f"  p50={r['latency_p50_us']} µs  p90={r['latency_p90_us']} µs  p99={r['latency_p99_us']} µs")
        print(f"  throughput={r['throughput']} dec/s")
        print()

    # Blossom (moins d'itérations car plus lent)
    bl_dec = qd.BlossomDecoder(c2q_ring)
    ring_results["Blossom"] = benchmark_latency(bl_dec, len(c2q_ring), n_runs=2000)
    r = ring_results["Blossom"]
    print("### Blossom")
    print(f"  mean={r['latency_mean_us']} µs  std={r['latency_std_us']} µs")
    print(f"  p50={r['latency_p50_us']} µs  p90={r['latency_p90_us']} µs  p99={r['latency_p99_us']} µs")
    print(f"  throughput={r['throughput']} dec/s")
    print()

    # LER sur le petit code
    ler_ring = benchmark_ler(c2q_ring, 4, error_rate=0.05, n_shots=5000)
    print("### LER (error_rate=0.05, n_shots=5000)")
    print(f"  UnionFind LER = {ler_ring['uf_ler']}")
    print(f"  Blossom   LER = {ler_ring['bl_ler']}")
    print(f"  Mismatch rate = {ler_ring['mismatch_rate']}")
    print()

    # --- Code de surface d=5 ----------------------------------------------
    print("## Code de surface : distance=5\n")
    c2q_surf, nq_surf = qd.generate_surface_code_checks(5)

    surf_results = {}
    for name, dec in [
        ("UnionFind", qd.UnionFindDecoder(c2q_surf, nq_surf)),
        ("FastUnionFind", qd.FastUnionFindDecoder(c2q_surf, nq_surf)),
    ]:
        surf_results[name] = benchmark_latency(dec, len(c2q_surf), n_runs=10000)
        r = surf_results[name]
        print(f"### {name}")
        print(f"  mean={r['latency_mean_us']} µs  std={r['latency_std_us']} µs")
        print(f"  p50={r['latency_p50_us']} µs  p90={r['latency_p90_us']} µs  p99={r['latency_p99_us']} µs")
        print(f"  throughput={r['throughput']} dec/s")
        print()

    # --- JSON complet ------------------------------------------------------
    full_report = {
        "cpu": cpu,
        "python_version": platform.python_version(),
        "qector_version": qd.__version__,
        "ring": ring_results,
        "surface_d5": surf_results,
        "ler_ring": ler_ring,
    }
    print("---")
    print(json.dumps(full_report, indent=2))


if __name__ == "__main__":
    main()
