#!/usr/bin/env python3
"""QECTOR v3 - Benchmark LER Serieux sur codes de surface toriques.

Benchmark rigoureux du Logical Error Rate (LER) avec :
- Codes de surface toriques d=3, d=5, d=7
- Modeles d'erreur : bit-flip, phase-flip, depolarisation
- Erreurs de mesure simulees (q = p/2)
- Comparaison UnionFind / Blossom / FastUnionFind / BPOSD
- Metriques de latence : p50, p99 (us) et throughput (shots/s)
- Intervalles de confiance Wilson 95%
- Sortie : Markdown + JSON + CSV

Usage:
    python scripts/benchmark_ler_serious.py
    python scripts/benchmark_ler_serious.py --quick   # mode rapide pour tests
    python scripts/benchmark_ler_serious.py --distances 3 5 --models bitflip
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import qector_decoder_v3 as qd


# ---------------------------------------------------------------------------
# Utilitaires mathematiques
# ---------------------------------------------------------------------------

def wilson_score_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Intervalle de confiance Wilson score pour une proportion (clopper-pearson approx)."""
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    denominator = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denominator
    width = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - width), min(1.0, centre + width)


def binomial_std(k: int, n: int) -> float:
    """Écart-type d'une proportion binomiale."""
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1.0 - p) / n)


# ---------------------------------------------------------------------------
# Representation du code de surface torique
# ---------------------------------------------------------------------------

class ToricSurfaceCode:
    """Code de surface torique de distance d."""

    def __init__(self, distance: int):
        self.d = distance
        self.n_qubits = distance * distance
        self.n_x_stab = distance * distance
        self.n_z_stab = distance * distance
        self.n_checks_total = self.n_x_stab + self.n_z_stab

        self.check_to_qubits, _ = qd.generate_surface_code_checks(distance)

        self.x_stabilizers = self.check_to_qubits[:self.n_x_stab]
        self.z_stabilizers = self.check_to_qubits[self.n_x_stab:]

        self.x_logical = [row * distance for row in range(distance)]
        self.z_logical = list(range(distance))

    def compute_syndrome(self, error: np.ndarray, stabilizers: List[List[int]]) -> np.ndarray:
        """Calcule le syndrome pour un sous-ensemble de stabilizers."""
        n_checks = len(stabilizers)
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(stabilizers):
            syndrome[ci] = int(error[qubits].sum()) % 2
        return syndrome

    def check_logical_error_x(self, error: np.ndarray, correction: np.ndarray) -> bool:
        """Verifie si error XOR correction forme un operateur logique X non-trivial."""
        combined = (error + correction) % 2
        parity = int(combined[self.x_logical].sum()) % 2
        return parity == 1

    def check_logical_error_z(self, error: np.ndarray, correction: np.ndarray) -> bool:
        """Verifie si error XOR correction forme un operateur logique Z non-trivial."""
        combined = (error + correction) % 2
        parity = int(combined[self.z_logical].sum()) % 2
        return parity == 1


# ---------------------------------------------------------------------------
# Resultats de benchmark
# ---------------------------------------------------------------------------

@dataclass
class LERResult:
    """Resultat d'un run de benchmark LER."""
    decoder: str
    code_distance: int
    error_model: str
    error_rate: float
    n_shots: int
    n_logical_errors: int
    ler: float
    ler_std: float
    ci_lower: float
    ci_upper: float
    elapsed_time_s: float
    latency_p50_us: float
    latency_p99_us: float
    throughput_sps: float


@dataclass
class BenchmarkReport:
    """Rapport complet de benchmark."""
    timestamp: str
    python_version: str
    qector_version: str
    cpu: str
    results: List[LERResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---------------------------------------------------------------------------
# Benchmarks individuels avec metriques de latence
# ---------------------------------------------------------------------------

def _percentile(arr: List[float], q: float) -> float:
    """Calcule le percentile q (0-100) d'une liste."""
    if not arr:
        return 0.0
    s = sorted(arr)
    k = (len(s) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def benchmark_bitflip(
    code: ToricSurfaceCode,
    decoder,
    p: float,
    n_shots: int,
    seed: int,
    measure_error: bool = False,
) -> LERResult:
    """Benchmark LER pour un modele de bit-flip (erreurs X) via Z-stabilizers."""
    rng = np.random.default_rng(seed)
    q_measure = p / 2.0 if measure_error else 0.0

    n_logical_errors = 0
    latencies = []
    t_global_0 = time.perf_counter()

    for _ in range(n_shots):
        error = (rng.random(code.n_qubits) < p).astype(np.uint8)
        syndrome = code.compute_syndrome(error, code.z_stabilizers)

        if measure_error:
            syndrome ^= (rng.random(len(syndrome)) < q_measure).astype(np.uint8)

        t0 = time.perf_counter()
        correction = decoder.decode(syndrome)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)  # us

        if code.check_logical_error_x(error, correction):
            n_logical_errors += 1

    t_global_1 = time.perf_counter()
    elapsed = t_global_1 - t_global_0
    ler = n_logical_errors / n_shots
    std = binomial_std(n_logical_errors, n_shots)
    ci_lo, ci_hi = wilson_score_interval(n_logical_errors, n_shots)
    p50 = _percentile(latencies, 50)
    p99 = _percentile(latencies, 99)
    throughput = n_shots / elapsed if elapsed > 0 else 0.0

    return LERResult(
        decoder=decoder.__class__.__name__,
        code_distance=code.d,
        error_model="bitflip" + ("+measure" if measure_error else ""),
        error_rate=p,
        n_shots=n_shots,
        n_logical_errors=n_logical_errors,
        ler=round(ler, 6),
        ler_std=round(std, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        elapsed_time_s=round(elapsed, 3),
        latency_p50_us=round(p50, 2),
        latency_p99_us=round(p99, 2),
        throughput_sps=round(throughput, 1),
    )


def benchmark_phaseflip(
    code: ToricSurfaceCode,
    decoder,
    p: float,
    n_shots: int,
    seed: int,
    measure_error: bool = False,
) -> LERResult:
    """Benchmark LER pour un modele de phase-flip (erreurs Z) via X-stabilizers."""
    rng = np.random.default_rng(seed)
    q_measure = p / 2.0 if measure_error else 0.0

    n_logical_errors = 0
    latencies = []
    t_global_0 = time.perf_counter()

    for _ in range(n_shots):
        error = (rng.random(code.n_qubits) < p).astype(np.uint8)
        syndrome = code.compute_syndrome(error, code.x_stabilizers)

        if measure_error:
            syndrome ^= (rng.random(len(syndrome)) < q_measure).astype(np.uint8)

        t0 = time.perf_counter()
        correction = decoder.decode(syndrome)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

        if code.check_logical_error_z(error, correction):
            n_logical_errors += 1

    t_global_1 = time.perf_counter()
    elapsed = t_global_1 - t_global_0
    ler = n_logical_errors / n_shots
    std = binomial_std(n_logical_errors, n_shots)
    ci_lo, ci_hi = wilson_score_interval(n_logical_errors, n_shots)
    p50 = _percentile(latencies, 50)
    p99 = _percentile(latencies, 99)
    throughput = n_shots / elapsed if elapsed > 0 else 0.0

    return LERResult(
        decoder=decoder.__class__.__name__,
        code_distance=code.d,
        error_model="phaseflip" + ("+measure" if measure_error else ""),
        error_rate=p,
        n_shots=n_shots,
        n_logical_errors=n_logical_errors,
        ler=round(ler, 6),
        ler_std=round(std, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        elapsed_time_s=round(elapsed, 3),
        latency_p50_us=round(p50, 2),
        latency_p99_us=round(p99, 2),
        throughput_sps=round(throughput, 1),
    )


def benchmark_depolarizing(
    code: ToricSurfaceCode,
    dec_x,
    dec_z,
    p: float,
    n_shots: int,
    seed: int,
    measure_error: bool = False,
) -> LERResult:
    """Benchmark LER pour le modele de depolarisation."""
    rng = np.random.default_rng(seed)
    p_each = p / 3.0
    q_measure = p / 2.0 if measure_error else 0.0

    n_logical_errors = 0
    latencies = []
    t_global_0 = time.perf_counter()

    for _ in range(n_shots):
        rand = rng.random(code.n_qubits)
        error_x = (rand < p_each).astype(np.uint8)
        error_y = ((rand >= p_each) & (rand < 2 * p_each)).astype(np.uint8)
        error_z = ((rand >= 2 * p_each) & (rand < p)).astype(np.uint8)

        syndrome_x = code.compute_syndrome(error_x | error_y, code.z_stabilizers)
        syndrome_z = code.compute_syndrome(error_z | error_y, code.x_stabilizers)

        if measure_error:
            syndrome_x ^= (rng.random(len(syndrome_x)) < q_measure).astype(np.uint8)
            syndrome_z ^= (rng.random(len(syndrome_z)) < q_measure).astype(np.uint8)

        t0 = time.perf_counter()
        correction_x = dec_x.decode(syndrome_x)
        correction_z = dec_z.decode(syndrome_z)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

        has_logical_error = (
            code.check_logical_error_x(error_x | error_y, correction_x)
            or code.check_logical_error_z(error_z | error_y, correction_z)
        )

        if has_logical_error:
            n_logical_errors += 1

    t_global_1 = time.perf_counter()
    elapsed = t_global_1 - t_global_0
    ler = n_logical_errors / n_shots
    std = binomial_std(n_logical_errors, n_shots)
    ci_lo, ci_hi = wilson_score_interval(n_logical_errors, n_shots)
    p50 = _percentile(latencies, 50)
    p99 = _percentile(latencies, 99)
    throughput = n_shots / elapsed if elapsed > 0 else 0.0

    return LERResult(
        decoder=f"{dec_x.__class__.__name__}+{dec_z.__class__.__name__}",
        code_distance=code.d,
        error_model="depolarizing" + ("+measure" if measure_error else ""),
        error_rate=p,
        n_shots=n_shots,
        n_logical_errors=n_logical_errors,
        ler=round(ler, 6),
        ler_std=round(std, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        elapsed_time_s=round(elapsed, 3),
        latency_p50_us=round(p50, 2),
        latency_p99_us=round(p99, 2),
        throughput_sps=round(throughput, 1),
    )


# ---------------------------------------------------------------------------
# Affichage Markdown
# ---------------------------------------------------------------------------

def print_markdown_table(results: List[LERResult], model: str) -> None:
    """Affiche un tableau Markdown pour un modele d'erreur donne."""
    print(f"\n### {model.upper()}\n")
    print(
        "| d | p | Decodeur | Shots | LER | std(LER) | IC 95% [bas, haut] | p50 us | p99 us | throughput sps | Temps (s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.error_model.startswith(model):
            print(
                f"| {r.code_distance} | {r.error_rate} | {r.decoder} | {r.n_shots} | "
                f"{r.ler:.6f} | {r.ler_std:.6f} | [{r.ci_lower:.6f}, {r.ci_upper:.6f}] | "
                f"{r.latency_p50_us:.2f} | {r.latency_p99_us:.2f} | {r.throughput_sps:.1f} | {r.elapsed_time_s:.2f} |"
            )


def print_markdown_summary(report: BenchmarkReport) -> None:
    """Affiche un resume Markdown complet."""
    print("\n---")
    print("# QECTOR v3 - Benchmark LER Serieux")
    print(f"\n**Date** : {report.timestamp}")
    print(f"**CPU** : {report.cpu}")
    print(f"**Python** : {report.python_version}")
    print(f"**QECTOR** : {report.qector_version}")
    print(f"**Total runs** : {len(report.results)}")

    models = sorted({r.error_model for r in report.results})
    for model in models:
        print_markdown_table(report.results, model)


# ---------------------------------------------------------------------------
# Export CSV / JSON
# ---------------------------------------------------------------------------

def save_csv(results: List[LERResult], path: Path) -> None:
    """Sauvegarde les resultats au format CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "distance", "error_model", "error_rate", "decoder",
            "shots", "logical_errors", "ler", "ler_std",
            "ci_95_lower", "ci_95_upper", "elapsed_s",
            "latency_p50_us", "latency_p99_us", "throughput_sps",
        ])
        for r in results:
            writer.writerow([
                r.code_distance, r.error_model, r.error_rate, r.decoder,
                r.n_shots, r.n_logical_errors, r.ler, r.ler_std,
                r.ci_lower, r.ci_upper, r.elapsed_time_s,
                r.latency_p50_us, r.latency_p99_us, r.throughput_sps,
            ])


def save_json(report: BenchmarkReport, path: Path) -> None:
    """Sauvegarde le rapport complet au format JSON."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.to_json())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark LER serieux QECTOR v3")
    parser.add_argument(
        "--distances", type=int, nargs="+", default=[3, 5, 7],
        help="Distances du code de surface a tester (defaut: 3 5 7)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["bitflip", "phaseflip", "depolarizing"],
        choices=["bitflip", "phaseflip", "depolarizing"],
        help="Modeles d'erreur a tester (defaut: bitflip phaseflip depolarizing)",
    )
    parser.add_argument(
        "--error-rates", type=float, nargs="+",
        default=[0.01, 0.02, 0.05, 0.10, 0.15],
        help="Taux d'erreur physiques p (defaut: 0.01 0.02 0.05 0.10 0.15)",
    )
    parser.add_argument(
        "--shots", type=int, nargs="+", default=[5000, 3000, 2000],
        help="Nombre de shots par distance (defaut: 5000 3000 2000)",
    )
    parser.add_argument(
        "--measure-error", action="store_true",
        help="Active les erreurs de mesure avec q = p/2",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Mode rapide : 1/10 des shots, un seul decodeur, distances 3 et 5",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results",
        help="Repertoire de sortie pour JSON et CSV",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine aleatoire (defaut: 42)",
    )
    args = parser.parse_args()

    # Mode rapide
    if args.quick:
        args.distances = [3, 5]
        args.shots = [s // 10 for s in args.shots[:2]]
        args.models = ["bitflip"]
        print("[QUICK] Mode rapide active (shots reduits, distances 3 et 5, bitflip uniquement)")

    # Verification des arguments
    if len(args.shots) < len(args.distances):
        args.shots = args.shots + [args.shots[-1]] * (len(args.distances) - len(args.shots))

    # Infos systeme
    cpu = platform.processor() or "unknown"
    report = BenchmarkReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        python_version=platform.python_version(),
        qector_version=qd.__version__,
        cpu=cpu,
    )

    print(f"# QECTOR v3 - Benchmark LER Serieux")
    print(f"CPU: {cpu}")
    print(f"Python: {platform.python_version()}")
    print(f"QECTOR: {qd.__version__}")
    print(f"Distances: {args.distances}")
    print(f"Modeles: {args.models}")
    print(f"Erreurs de mesure: {'oui (q=p/2)' if args.measure_error else 'non'}")
    print()

    for d_idx, d in enumerate(args.distances):
        n_shots = args.shots[d_idx]
        print(f"\n## Code de surface torique d={d} ({n_shots} shots)\n")

        code = ToricSurfaceCode(d)

        # Construire les decodeurs pour les sous-codes X et Z
        decoders_x = {
            "UnionFind": qd.UnionFindDecoder(code.x_stabilizers, code.n_qubits),
            "FastUnionFind": qd.FastUnionFindDecoder(code.x_stabilizers, code.n_qubits),
        }
        decoders_z = {
            "UnionFind": qd.UnionFindDecoder(code.z_stabilizers, code.n_qubits),
            "FastUnionFind": qd.FastUnionFindDecoder(code.z_stabilizers, code.n_qubits),
        }

        # Blossom est plus lent ; on l'ajoute sauf en mode quick
        if not args.quick:
            decoders_x["Blossom"] = qd.BlossomDecoder(code.x_stabilizers, code.n_qubits)
            decoders_z["Blossom"] = qd.BlossomDecoder(code.z_stabilizers, code.n_qubits)

        # BP-OSD : ajoute pour tous les modeles
        decoders_x["BPOSD"] = qd.BPOSDDecoder(code.x_stabilizers, code.n_qubits, error_rate=0.1)
        decoders_z["BPOSD"] = qd.BPOSDDecoder(code.z_stabilizers, code.n_qubits, error_rate=0.1)

        for p in args.error_rates:
            # Mettre a jour le taux d'erreur des decodeurs BPOSD
            for name in decoders_x:
                if name == "BPOSD":
                    # Recreer avec le bon taux d'erreur
                    decoders_x[name] = qd.BPOSDDecoder(code.x_stabilizers, code.n_qubits, error_rate=p)
                    decoders_z[name] = qd.BPOSDDecoder(code.z_stabilizers, code.n_qubits, error_rate=p)

            for model in args.models:
                seed = args.seed + d * 100 + int(p * 1000)

                if model == "bitflip":
                    for name, dec in decoders_z.items():
                        res = benchmark_bitflip(
                            code, dec, p, n_shots, seed,
                            measure_error=args.measure_error,
                        )
                        report.results.append(res)
                        print(
                            f"  [d={d}, p={p}, bitflip] {name}: "
                            f"LER={res.ler:.6f} ± {res.ler_std:.6f}  "
                            f"IC=[{res.ci_lower:.6f}, {res.ci_upper:.6f}]  "
                            f"p50={res.latency_p50_us:.2f}us p99={res.latency_p99_us:.2f}us "
                            f"thr={res.throughput_sps:.1f}sps ({res.elapsed_time_s:.2f}s)"
                        )

                elif model == "phaseflip":
                    for name, dec in decoders_x.items():
                        res = benchmark_phaseflip(
                            code, dec, p, n_shots, seed,
                            measure_error=args.measure_error,
                        )
                        report.results.append(res)
                        print(
                            f"  [d={d}, p={p}, phaseflip] {name}: "
                            f"LER={res.ler:.6f} ± {res.ler_std:.6f}  "
                            f"IC=[{res.ci_lower:.6f}, {res.ci_upper:.6f}]  "
                            f"p50={res.latency_p50_us:.2f}us p99={res.latency_p99_us:.2f}us "
                            f"thr={res.throughput_sps:.1f}sps ({res.elapsed_time_s:.2f}s)"
                        )

                elif model == "depolarizing":
                    for name in decoders_x.keys():
                        res = benchmark_depolarizing(
                            code, decoders_x[name], decoders_z[name],
                            p, n_shots, seed,
                            measure_error=args.measure_error,
                        )
                        report.results.append(res)
                        print(
                            f"  [d={d}, p={p}, depolarizing] {name}: "
                            f"LER={res.ler:.6f} ± {res.ler_std:.6f}  "
                            f"IC=[{res.ci_lower:.6f}, {res.ci_upper:.6f}]  "
                            f"p50={res.latency_p50_us:.2f}us p99={res.latency_p99_us:.2f}us "
                            f"thr={res.throughput_sps:.1f}sps ({res.elapsed_time_s:.2f}s)"
                        )

    # Affichage recapitulatif Markdown
    print_markdown_summary(report)

    # Export
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"benchmark_ler_serious_{report.timestamp[:19].replace(':', '-')}.json"
    csv_path = out_dir / f"benchmark_ler_serious_{report.timestamp[:19].replace(':', '-')}.csv"

    save_json(report, json_path)
    save_csv(report.results, csv_path)

    print(f"\n[RESULTS] Resultats sauvegardes :")
    print(f"   JSON -> {json_path.resolve()}")
    print(f"   CSV  -> {csv_path.resolve()}")


if __name__ == "__main__":
    main()
