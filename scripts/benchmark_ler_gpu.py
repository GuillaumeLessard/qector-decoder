#!/usr/bin/env python3
"""QECTOR v3 - Benchmark LER CUDA/OpenCL vs CPU sur codes de surface toriques.

NOTE: pour la comparaison LER *de référence* contre PyMatching au niveau circuit
(données Stim réelles, intervalles de confiance Wilson, balayage de distance), voir
``scripts/competitive_stim_ler.py`` et ``docs/BENCHMARK_COMPETITIVE.md``. Ce script
reste le banc LER spécifique GPU (CUDA/OpenCL vs CPU).


Benchmark rigoureux comparant le Logical Error Rate (LER) entre :
- CUDABatchDecoder (kernel CUDA natif avec fallback CPU transparent)
- OpenCLBatchDecoder (GPU portable avec fallback CPU transparent)
- CPUBatchDecoder (CPU parallele SIMD + Rayon)
- BatchDecoder (CPU Rayon simple)

Configurations testees :
- Code de surface torique d=5
- Batch sizes : 1024, 4096
- Shots : 3000–5000 par configuration
- Modele d'erreur : bit-flip, phase-flip, depolarizing
- Intervalles de confiance Wilson 95%

Sortie : Markdown + JSON + CSV
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

import qector_decoder_v3 as qd
from benchmark_ler_reporting import (
    BenchmarkReport,
    LERResult,
    binomial_std,
    generate_markdown_report,
    print_markdown_summary,
    save_csv,
    save_json,
    wilson_score_interval,
)


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
# Benchmarks batch LER
# ---------------------------------------------------------------------------

def benchmark_batch_ler(
    code: ToricSurfaceCode,
    decoder,
    decoder_name: str,
    p: float,
    n_shots: int,
    batch_size: int,
    seed: int,
    error_model: str = "bitflip",
    measure_error: bool = False,
) -> LERResult:
    """Benchmark LER en batch avec decodage par lots.

    Genere n_shots erreurs, les regroupe en batchs de taille batch_size,
    decode en batch, puis verifie les erreurs logiques individuellement.
    """
    rng = np.random.default_rng(seed)
    q_measure = p / 2.0 if measure_error else 0.0

    n_logical_errors = 0
    t0 = time.perf_counter()

    # Prepare les stabilizers selon le modele
    if error_model == "bitflip":
        stabilizers = code.z_stabilizers
        check_logical_error = code.check_logical_error_x
    elif error_model == "phaseflip":
        stabilizers = code.x_stabilizers
        check_logical_error = code.check_logical_error_z
    else:
        raise ValueError(f"Modele non supporte pour batch simple: {error_model}")

    n_batches = (n_shots + batch_size - 1) // batch_size
    last_batch_size = n_shots - (n_batches - 1) * batch_size

    for b in range(n_batches):
        current_batch = last_batch_size if b == n_batches - 1 else batch_size

        # Generer les erreurs et syndromes pour ce batch
        errors = np.zeros((current_batch, code.n_qubits), dtype=np.uint8)
        syndromes = np.zeros((current_batch, len(stabilizers)), dtype=np.uint8)

        for i in range(current_batch):
            errors[i] = (rng.random(code.n_qubits) < p).astype(np.uint8)
            syndrome = code.compute_syndrome(errors[i], stabilizers)
            if measure_error:
                syndrome ^= (rng.random(len(syndrome)) < q_measure).astype(np.uint8)
            syndromes[i] = syndrome

        # Decodage en batch (adaptatif selon l'interface du decodeur)
        if hasattr(decoder, 'batch_decode'):
            corrections = decoder.batch_decode(syndromes)
        elif hasattr(decoder, 'parallel_batch_decode'):
            corrections = decoder.parallel_batch_decode(syndromes)
        else:
            raise AttributeError(f"Decoder {decoder_name} has no batch_decode method")

        # Verification des erreurs logiques
        for i in range(current_batch):
            if check_logical_error(errors[i], corrections[i]):
                n_logical_errors += 1

    t1 = time.perf_counter()
    elapsed = t1 - t0
    ler = n_logical_errors / n_shots
    std = binomial_std(n_logical_errors, n_shots)
    ci_lo, ci_hi = wilson_score_interval(n_logical_errors, n_shots)
    throughput = n_shots / elapsed if elapsed > 0 else 0.0

    # Collecte des metriques GPU si disponibles
    gpu_failures = 0
    gpu_degraded = False
    gpu_recoveries = 0
    if hasattr(decoder, 'total_failures'):
        gpu_failures = decoder.total_failures
    if hasattr(decoder, 'is_degraded'):
        gpu_degraded = decoder.is_degraded
    if hasattr(decoder, 'gpu_recoveries'):
        gpu_recoveries = decoder.gpu_recoveries

    return LERResult(
        decoder=decoder_name,
        code_distance=code.d,
        error_model=error_model + ("+measure" if measure_error else ""),
        error_rate=p,
        batch_size=batch_size,
        n_shots=n_shots,
        n_logical_errors=n_logical_errors,
        ler=round(ler, 6),
        ler_std=round(std, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        elapsed_time_s=round(elapsed, 3),
        throughput=round(throughput, 1),
        gpu_failures=gpu_failures,
        gpu_degraded=gpu_degraded,
        gpu_recoveries=gpu_recoveries,
    )


def benchmark_depolarizing_batch(
    code: ToricSurfaceCode,
    dec_x,
    dec_z,
    decoder_name: str,
    p: float,
    n_shots: int,
    batch_size: int,
    seed: int,
    measure_error: bool = False,
) -> LERResult:
    """Benchmark LER depolarizing en batch avec deux decodeurs."""
    rng = np.random.default_rng(seed)
    p_each = p / 3.0
    q_measure = p / 2.0 if measure_error else 0.0

    n_logical_errors = 0
    t0 = time.perf_counter()

    n_batches = (n_shots + batch_size - 1) // batch_size
    last_batch_size = n_shots - (n_batches - 1) * batch_size

    for b in range(n_batches):
        current_batch = last_batch_size if b == n_batches - 1 else batch_size

        errors_x = np.zeros((current_batch, code.n_qubits), dtype=np.uint8)
        errors_y = np.zeros((current_batch, code.n_qubits), dtype=np.uint8)
        errors_z = np.zeros((current_batch, code.n_qubits), dtype=np.uint8)
        syndromes_x = np.zeros((current_batch, len(code.z_stabilizers)), dtype=np.uint8)
        syndromes_z = np.zeros((current_batch, len(code.x_stabilizers)), dtype=np.uint8)

        for i in range(current_batch):
            rand = rng.random(code.n_qubits)
            errors_x[i] = (rand < p_each).astype(np.uint8)
            errors_y[i] = ((rand >= p_each) & (rand < 2 * p_each)).astype(np.uint8)
            errors_z[i] = ((rand >= 2 * p_each) & (rand < p)).astype(np.uint8)

            syndrome_x = code.compute_syndrome(errors_x[i] | errors_y[i], code.z_stabilizers)
            syndrome_z = code.compute_syndrome(errors_z[i] | errors_y[i], code.x_stabilizers)

            if measure_error:
                syndrome_x ^= (rng.random(len(syndrome_x)) < q_measure).astype(np.uint8)
                syndrome_z ^= (rng.random(len(syndrome_z)) < q_measure).astype(np.uint8)

            syndromes_x[i] = syndrome_x
            syndromes_z[i] = syndrome_z

        # Decodage en batch pour depolarizing (adaptatif selon l'interface)
        if hasattr(dec_x, 'batch_decode'):
            corrections_x = dec_x.batch_decode(syndromes_x)
        elif hasattr(dec_x, 'parallel_batch_decode'):
            corrections_x = dec_x.parallel_batch_decode(syndromes_x)
        else:
            raise AttributeError(f"Decoder {decoder_name} has no batch_decode method")

        if hasattr(dec_z, 'batch_decode'):
            corrections_z = dec_z.batch_decode(syndromes_z)
        elif hasattr(dec_z, 'parallel_batch_decode'):
            corrections_z = dec_z.parallel_batch_decode(syndromes_z)
        else:
            raise AttributeError(f"Decoder {decoder_name} has no batch_decode method")

        for i in range(current_batch):
            has_error = (
                code.check_logical_error_x(errors_x[i] | errors_y[i], corrections_x[i])
                or code.check_logical_error_z(errors_z[i] | errors_y[i], corrections_z[i])
            )
            if has_error:
                n_logical_errors += 1

    t1 = time.perf_counter()
    elapsed = t1 - t0
    ler = n_logical_errors / n_shots
    std = binomial_std(n_logical_errors, n_shots)
    ci_lo, ci_hi = wilson_score_interval(n_logical_errors, n_shots)
    throughput = n_shots / elapsed if elapsed > 0 else 0.0

    return LERResult(
        decoder=decoder_name,
        code_distance=code.d,
        error_model="depolarizing" + ("+measure" if measure_error else ""),
        error_rate=p,
        batch_size=batch_size,
        n_shots=n_shots,
        n_logical_errors=n_logical_errors,
        ler=round(ler, 6),
        ler_std=round(std, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        elapsed_time_s=round(elapsed, 3),
        throughput=round(throughput, 1),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark LER GPU vs CPU QECTOR v3")
    parser.add_argument(
        "--distance", type=int, default=5,
        help="Distance du code de surface (defaut: 5)",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1024, 4096],
        help="Tailles de batch a tester (defaut: 1024 4096)",
    )
    parser.add_argument(
        "--shots", type=int, default=3000,
        help="Nombre total de shots (defaut: 3000)",
    )
    parser.add_argument(
        "--error-rates", type=float, nargs="+",
        default=[0.01, 0.05, 0.10],
        help="Taux d'erreur physiques p (defaut: 0.01 0.05 0.10)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["bitflip", "phaseflip", "depolarizing"],
        choices=["bitflip", "phaseflip", "depolarizing"],
        help="Modeles d'erreur a tester (defaut: bitflip phaseflip depolarizing)",
    )
    parser.add_argument(
        "--measure-error", action="store_true",
        help="Active les erreurs de mesure avec q = p/2",
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

    # Infos systeme
    cpu = platform.processor() or "unknown"
    opencl_available = qd.OpenCLBatchDecoder.is_available()
    cuda_available = qd.CUDABatchDecoder.is_available()
    report = BenchmarkReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        python_version=platform.python_version(),
        qector_version=qd.__version__,
        cpu=cpu,
        opencl_available=opencl_available,
        cuda_available=cuda_available,
    )

    print("# QECTOR v3 - Benchmark LER GPU vs CPU")
    print(f"CPU: {cpu}")
    print(f"Python: {platform.python_version()}")
    print(f"QECTOR: {qd.__version__}")
    print(f"GPU OpenCL disponible: {'oui' if opencl_available else 'non'}")
    print(f"GPU CUDA natif disponible: {'oui' if cuda_available else 'non'}")
    print(f"Distance: {args.distance}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Shots: {args.shots}")
    print(f"Modeles: {args.models}")
    print(f"Erreurs de mesure: {'oui (q=p/2)' if args.measure_error else 'non'}")
    print()

    d = args.distance
    code = ToricSurfaceCode(d)

    for batch_size in args.batch_sizes:
        print(f"\n## Batch size = {batch_size}\n")

        for p in args.error_rates:
            for model in args.models:
                seed = args.seed + d * 100 + int(p * 1000) + batch_size

                if model == "bitflip":
                    # Decodeurs pour Z-stabilizers (detection erreurs X)
                    decoders = {}
                    if cuda_available:
                        decoders["GPU_CUDA"] = qd.CUDABatchDecoder(
                            code.z_stabilizers, code.n_qubits
                        )
                    if opencl_available:
                        decoders["GPU_OpenCL"] = qd.OpenCLBatchDecoder(code.z_stabilizers, code.n_qubits)
                    decoders["CPU_SIMD"] = qd.CPUBatchDecoder(code.z_stabilizers, code.n_qubits)
                    decoders["CPU_Rayon"] = qd.BatchDecoder(code.z_stabilizers, code.n_qubits)
                    for dec_name, dec in decoders.items():
                        res = benchmark_batch_ler(
                            code, dec, dec_name, p, args.shots, batch_size, seed,
                            error_model="bitflip", measure_error=args.measure_error,
                        )
                        report.results.append(res)
                        print(
                            f"  [d={d}, p={p}, batch={batch_size}, bitflip] {dec_name}: "
                            f"LER={res.ler:.6f} +/- {res.ler_std:.6f}  "
                            f"IC=[{res.ci_lower:.6f}, {res.ci_upper:.6f}]  "
                            f"{res.elapsed_time_s:.2f}s  "
                            f"{res.throughput:.0f} dec/s"
                        )

                elif model == "phaseflip":
                    # Decodeurs pour X-stabilizers (detection erreurs Z)
                    decoders = {}
                    if cuda_available:
                        decoders["GPU_CUDA"] = qd.CUDABatchDecoder(
                            code.x_stabilizers, code.n_qubits
                        )
                    if opencl_available:
                        decoders["GPU_OpenCL"] = qd.OpenCLBatchDecoder(code.x_stabilizers, code.n_qubits)
                    decoders["CPU_SIMD"] = qd.CPUBatchDecoder(code.x_stabilizers, code.n_qubits)
                    decoders["CPU_Rayon"] = qd.BatchDecoder(code.x_stabilizers, code.n_qubits)
                    for dec_name, dec in decoders.items():
                        res = benchmark_batch_ler(
                            code, dec, dec_name, p, args.shots, batch_size, seed,
                            error_model="phaseflip", measure_error=args.measure_error,
                        )
                        report.results.append(res)
                        print(
                            f"  [d={d}, p={p}, batch={batch_size}, phaseflip] {dec_name}: "
                            f"LER={res.ler:.6f} +/- {res.ler_std:.6f}  "
                            f"IC=[{res.ci_lower:.6f}, {res.ci_upper:.6f}]  "
                            f"{res.elapsed_time_s:.2f}s  "
                            f"{res.throughput:.0f} dec/s"
                        )

                elif model == "depolarizing":
                    for dec_name in ["GPU_CUDA", "GPU_OpenCL", "CPU_SIMD", "CPU_Rayon"]:
                        if dec_name == "GPU_CUDA" and cuda_available:
                            dec_x = qd.CUDABatchDecoder(code.x_stabilizers, code.n_qubits)
                            dec_z = qd.CUDABatchDecoder(code.z_stabilizers, code.n_qubits)
                        elif dec_name == "GPU_OpenCL" and opencl_available:
                            dec_x = qd.OpenCLBatchDecoder(code.x_stabilizers, code.n_qubits)
                            dec_z = qd.OpenCLBatchDecoder(code.z_stabilizers, code.n_qubits)
                        elif dec_name == "CPU_SIMD":
                            dec_x = qd.CPUBatchDecoder(code.x_stabilizers, code.n_qubits)
                            dec_z = qd.CPUBatchDecoder(code.z_stabilizers, code.n_qubits)
                        elif dec_name == "CPU_Rayon":
                            dec_x = qd.BatchDecoder(code.x_stabilizers, code.n_qubits)
                            dec_z = qd.BatchDecoder(code.z_stabilizers, code.n_qubits)
                        else:
                            continue

                        res = benchmark_depolarizing_batch(
                            code, dec_x, dec_z, dec_name, p, args.shots, batch_size, seed,
                            measure_error=args.measure_error,
                        )
                        report.results.append(res)
                        print(
                            f"  [d={d}, p={p}, batch={batch_size}, depolarizing] {dec_name}: "
                            f"LER={res.ler:.6f} +/- {res.ler_std:.6f}  "
                            f"IC=[{res.ci_lower:.6f}, {res.ci_upper:.6f}]  "
                            f"{res.elapsed_time_s:.2f}s  "
                            f"{res.throughput:.0f} dec/s"
                        )

    # Affichage recapitulatif Markdown
    print_markdown_summary(report)

    # Export
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = report.timestamp[:19].replace(":", "-")
    json_path = out_dir / f"benchmark_ler_gpu_{ts}.json"
    csv_path = out_dir / f"benchmark_ler_gpu_{ts}.csv"
    md_path = out_dir / f"BENCHMARK_LER_GPU_{ts}.md"

    save_json(report, json_path)
    save_csv(report.results, csv_path)

    # Generer aussi le markdown statique
    generate_markdown_report(report, md_path)

    print("\n[RESULTS] Resultats sauvegardes :")
    print(f"   JSON -> {json_path.resolve()}")
    print(f"   CSV  -> {csv_path.resolve()}")
    print(f"   MD   -> {md_path.resolve()}")


if __name__ == "__main__":
    main()
