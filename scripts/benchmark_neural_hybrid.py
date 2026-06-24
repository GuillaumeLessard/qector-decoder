#!/usr/bin/env python3
"""QECTOR v3 - Benchmark LER Neural Hybride.

Pipeline :
  1. Generation de 5000 syndromes pour un code de surface torique d=3 ou d=5.
  2. Calcul des corrections cibles via UnionFindDecoder (teacher).
  3. Entrainement du NeuralPredecoder sur le dataset.
  4. Benchmark LER avec :
     - Neural + UnionFind fallback (si correction invalide)
     - UnionFind pur
     - Blossom pur
     - FastUnionFind pur
  5. Export des resultats JSON + Markdown.

Usage:
    python scripts/benchmark_neural_hybrid.py --distance 3 --shots 5000
    python scripts/benchmark_neural_hybrid.py --distance 5 --shots 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

import qector_decoder_v3 as qd


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def wilson_score_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    denominator = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denominator
    width = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - width), min(1.0, centre + width)


def binomial_std(k: int, n: int) -> float:
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1.0 - p) / n)


def _percentile(arr: List[float], q: float) -> float:
    if not arr:
        return 0.0
    s = sorted(arr)
    k = (len(s) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


# ---------------------------------------------------------------------------
# Code de surface torique (sous-code Z pour bitflip)
# ---------------------------------------------------------------------------

class ToricSurfaceCode:
    def __init__(self, distance: int):
        self.d = distance
        self.n_qubits = distance * distance
        self.n_x_stab = distance * distance
        self.n_z_stab = distance * distance
        self.check_to_qubits, _ = qd.generate_surface_code_checks(distance)
        self.x_stabilizers = self.check_to_qubits[:self.n_x_stab]
        self.z_stabilizers = self.check_to_qubits[self.n_x_stab:]
        self.x_logical = [row * distance for row in range(distance)]
        self.z_logical = list(range(distance))

    def compute_syndrome(self, error: np.ndarray, stabilizers: List[List[int]]) -> np.ndarray:
        n_checks = len(stabilizers)
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(stabilizers):
            syndrome[ci] = int(error[qubits].sum()) % 2
        return syndrome

    def check_logical_error_x(self, error: np.ndarray, correction: np.ndarray) -> bool:
        combined = (error + correction) % 2
        parity = int(combined[self.x_logical].sum()) % 2
        return parity == 1


# ---------------------------------------------------------------------------
# Hybrid Neural Decoder avec fallback UnionFind
# ---------------------------------------------------------------------------

class HybridNeuralDecoder:
    """NeuralPredecoder + UnionFind fallback.

    Si la correction predite par le neural ne verifie pas le syndrome
    (H * correction != syndrome), on fallback sur UnionFind.
    """

    def __init__(self, neural: qd.NeuralPredecoder, fallback: qd.UnionFindDecoder,
                 stabilizers: List[List[int]], n_qubits: int):
        self.neural = neural
        self.fallback = fallback
        self.stabilizers = stabilizers
        self.n_qubits = n_qubits
        self.n_fallbacks = 0
        self.n_calls = 0

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        # Prediction neural
        correction = self.neural.decode(syndrome)

        # Verification : le syndrome de la correction doit etre egal au syndrome d'entree
        syn_check = np.zeros(len(self.stabilizers), dtype=np.uint8)
        for ci, qubits in enumerate(self.stabilizers):
            syn_check[ci] = int(correction[qubits].sum()) % 2

        if not np.array_equal(syn_check, syndrome):
            self.n_fallbacks += 1
            correction = self.fallback.decode(syndrome)

        return correction


# ---------------------------------------------------------------------------
# Dataset generation et entrainement
# ---------------------------------------------------------------------------

def generate_training_dataset(code: ToricSurfaceCode, decoder, n_samples: int,
                                error_rate: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Genere un dataset (syndromes, corrections) via le decodeur teacher."""
    rng = np.random.default_rng(seed)
    n_checks = len(code.z_stabilizers)

    syndromes = np.zeros((n_samples, n_checks), dtype=np.uint8)
    corrections = np.zeros((n_samples, code.n_qubits), dtype=np.uint8)

    for i in range(n_samples):
        error = (rng.random(code.n_qubits) < error_rate).astype(np.uint8)
        syndrome = code.compute_syndrome(error, code.z_stabilizers)
        correction = decoder.decode(syndrome)
        syndromes[i] = syndrome
        corrections[i] = correction

    return syndromes, corrections


def train_neural_decoder(code: ToricSurfaceCode, n_samples: int, error_rate: float,
                         seed: int, n_epochs: int = 50, learning_rate: float = 0.05) -> Tuple[qd.NeuralPredecoder, qd.UnionFindDecoder]:
    """Entraine un NeuralPredecoder sur un dataset genere par UnionFind."""
    teacher = qd.UnionFindDecoder(code.z_stabilizers, code.n_qubits)
    syndromes, corrections = generate_training_dataset(code, teacher, n_samples, error_rate, seed)

    n_input = syndromes.shape[1]
    n_output = code.n_qubits
    neural = qd.NeuralPredecoder(n_input, n_output, n_hidden1=None, n_hidden2=None)

    print(f"  [TRAIN] NeuralPredecoder: n_input={n_input}, n_output={n_output}, "
          f"hidden=({neural.n_hidden1},{neural.n_hidden2})")
    print(f"  [TRAIN] Dataset: {n_samples} echantillons, p={error_rate}, epochs={n_epochs}, lr={learning_rate}")

    t0 = time.perf_counter()
    neural.train(syndromes.flatten(), corrections.flatten(), n_epochs, learning_rate)
    t1 = time.perf_counter()
    print(f"  [TRAIN] Termine en {t1 - t0:.2f}s")

    return neural, teacher


# ---------------------------------------------------------------------------
# Benchmark LER
# ---------------------------------------------------------------------------

@dataclass
class LERResult:
    decoder: str
    code_distance: int
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
    fallback_rate: float = 0.0


@dataclass
class BenchmarkReport:
    timestamp: str
    python_version: str
    qector_version: str
    cpu: str
    distance: int
    n_train_samples: int
    train_error_rate: float
    train_epochs: int
    neural_fallback_rate: float
    results: List[LERResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def benchmark_decoder(code: ToricSurfaceCode, decoder, p: float, n_shots: int,
                      seed: int) -> LERResult:
    """Benchmark LER pour un decodeur sur bitflip (sous-code Z)."""
    rng = np.random.default_rng(seed)
    n_logical_errors = 0
    latencies = []
    t_global_0 = time.perf_counter()

    fallback_rate = 0.0
    if hasattr(decoder, 'n_fallbacks'):
        decoder.n_fallbacks = 0
        decoder.n_calls = 0

    for _ in range(n_shots):
        error = (rng.random(code.n_qubits) < p).astype(np.uint8)
        syndrome = code.compute_syndrome(error, code.z_stabilizers)

        t0 = time.perf_counter()
        correction = decoder.decode(syndrome)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

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

    if hasattr(decoder, 'n_fallbacks') and decoder.n_calls > 0:
        fallback_rate = decoder.n_fallbacks / decoder.n_calls

    return LERResult(
        decoder=decoder.__class__.__name__,
        code_distance=code.d,
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
        fallback_rate=round(fallback_rate, 4),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_csv(results: List[LERResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "distance", "error_rate", "decoder", "shots", "logical_errors",
            "ler", "ler_std", "ci_95_lower", "ci_95_upper", "elapsed_s",
            "latency_p50_us", "latency_p99_us", "throughput_sps", "fallback_rate",
        ])
        for r in results:
            writer.writerow([
                r.code_distance, r.error_rate, r.decoder, r.n_shots,
                r.n_logical_errors, r.ler, r.ler_std, r.ci_lower, r.ci_upper,
                r.elapsed_time_s, r.latency_p50_us, r.latency_p99_us,
                r.throughput_sps, r.fallback_rate,
            ])


def save_json(report: BenchmarkReport, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.to_json())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark LER Neural Hybride QECTOR v3")
    parser.add_argument("--distance", type=int, default=3, choices=[3, 5],
                        help="Distance du code de surface (defaut: 3)")
    parser.add_argument("--train-samples", type=int, default=5000,
                        help="Nombre d'echantillons d'entrainement (defaut: 5000)")
    parser.add_argument("--train-epochs", type=int, default=50,
                        help="Epochs d'entrainement neural (defaut: 50)")
    parser.add_argument("--train-error-rate", type=float, default=0.10,
                        help="Taux d'erreur pour la generation du dataset (defaut: 0.10)")
    parser.add_argument("--test-shots", type=int, default=5000,
                        help="Nombre de shots pour le benchmark LER (defaut: 5000)")
    parser.add_argument(
        "--error-rates", type=float, nargs="+",
        default=[0.01, 0.02, 0.05, 0.10, 0.15],
        help="Taux d'erreur physiques pour le test (defaut: 0.01 0.02 0.05 0.10 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aleatoire (defaut: 42)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Repertoire de sortie")
    args = parser.parse_args()

    d = args.distance
    n_shots = args.test_shots
    print(f"# QECTOR v3 - Benchmark LER Neural Hybride (d={d})")
    print(f"CPU: {platform.processor() or 'unknown'}")
    print(f"Python: {platform.python_version()}")
    print(f"QECTOR: {qd.__version__}")
    print()

    code = ToricSurfaceCode(d)

    # 1. Entrainement du neural
    print(f"## Entrainement du NeuralPredecoder sur {args.train_samples} echantillons\n")
    neural, teacher = train_neural_decoder(
        code, args.train_samples, args.train_error_rate,
        args.seed, args.train_epochs, learning_rate=0.05
    )

    # 2. Construction du decodeur hybride
    hybrid = HybridNeuralDecoder(neural, teacher, code.z_stabilizers, code.n_qubits)

    # 3. Benchmarks LER
    print(f"\n## Benchmark LER ({n_shots} shots, d={d})\n")

    report = BenchmarkReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        python_version=platform.python_version(),
        qector_version=qd.__version__,
        cpu=platform.processor() or "unknown",
        distance=d,
        n_train_samples=args.train_samples,
        train_error_rate=args.train_error_rate,
        train_epochs=args.train_epochs,
        neural_fallback_rate=0.0,
        results=[],
    )

    decoders = {
        "HybridNeural": hybrid,
        "UnionFind": teacher,
        "FastUnionFind": qd.FastUnionFindDecoder(code.z_stabilizers, code.n_qubits),
        "Blossom": qd.BlossomDecoder(code.z_stabilizers, code.n_qubits),
    }

    for p in args.error_rates:
        seed = args.seed + d * 100 + int(p * 1000)
        for name, dec in decoders.items():
            res = benchmark_decoder(code, dec, p, n_shots, seed)
            report.results.append(res)
            if name == "HybridNeural":
                report.neural_fallback_rate = res.fallback_rate
            print(
                f"  [d={d}, p={p}] {name}: "
                f"LER={res.ler:.6f} ± {res.ler_std:.6f}  "
                f"IC=[{res.ci_lower:.6f}, {res.ci_upper:.6f}]  "
                f"p50={res.latency_p50_us:.2f}us p99={res.latency_p99_us:.2f}us "
                f"thr={res.throughput_sps:.1f}sps "
                f"fallback={res.fallback_rate:.2%} ({res.elapsed_time_s:.2f}s)"
            )

    # Affichage recapitulatif Markdown
    print("\n---")
    print("# Recapitulatif Benchmark Neural Hybride")
    print(f"\n**Distance** : {d}")
    print(f"**Entrainement** : {args.train_samples} echantillons, p={args.train_error_rate}, {args.train_epochs} epochs")
    print(f"**Fallback rate neural** : {report.neural_fallback_rate:.2%}")
    print(f"\n### Resultats LER (bitflip)\n")
    print("| p | Decodeur | Shots | LER | std | IC 95% | p50 us | p99 us | thr sps | fallback |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in report.results:
        print(
            f"| {r.error_rate} | {r.decoder} | {r.n_shots} | {r.ler:.6f} | {r.ler_std:.6f} | "
            f"[{r.ci_lower:.6f}, {r.ci_upper:.6f}] | {r.latency_p50_us:.2f} | {r.latency_p99_us:.2f} | "
            f"{r.throughput_sps:.1f} | {r.fallback_rate:.2%} |"
        )

    # Export
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = report.timestamp[:19].replace(":", "-")
    json_path = out_dir / f"benchmark_neural_hybrid_d{d}_{ts}.json"
    csv_path = out_dir / f"benchmark_neural_hybrid_d{d}_{ts}.csv"

    save_json(report, json_path)
    save_csv(report.results, csv_path)

    print(f"\n[RESULTS] Resultats sauvegardes :")
    print(f"   JSON -> {json_path.resolve()}")
    print(f"   CSV  -> {csv_path.resolve()}")


if __name__ == "__main__":
    main()
