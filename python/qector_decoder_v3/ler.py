"""qector_decoder_v3.ler — logical-error-rate estimation with statistical rigor.

E2 + E6 (K3dev.md): every LER number this module produces carries a shot
count, an error count, a Wilson 95% score interval, the wall-clock decode
seconds and the seed — so figures are defensible science, not marketing
points.  All decoders are constructed **once** per run (S5: warm reuse) and
driven through their batched entry points where available.

Works on any :class:`~qector_decoder_v3.codes.Code` that exposes logicals
(E1/C6) — repetition, ring, rotated surface, toric, heavy-hex, and all CSS
families (bivariate-bicycle, bicycle, hypergraph-product).

Example
-------
>>> from qector_decoder_v3 import codes
>>> from qector_decoder_v3.ler import estimate_ler
>>> r = estimate_ler(codes.rotated_surface_code(5), "blossom", p=0.02, shots=2000, seed=1)
>>> r.ler, r.ci95
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

__all__ = [
    "CIRCUIT_LEVEL",
    "CODE_CAPACITY",
    "NOISE_MODELS",
    "LerResult",
    "NoiseModelMismatch",
    "ThresholdResult",
    "assert_comparable",
    "estimate_ler",
    "estimate_ler_circuit_level",
    "reference_validate",
    "run_competitive_suite",
    "run_memory_experiment",
    "run_threshold_sweep",
    "sample_biased_errors",
    "sinter_threshold_sweep",
    "wilson_ci",
]


def wilson_ci(errors: int, shots: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default ~95%)."""
    if shots <= 0:
        return (0.0, 1.0)
    p = errors / shots
    denom = 1.0 + z * z / shots
    center = (p + z * z / (2.0 * shots)) / denom
    half = z * math.sqrt(p * (1.0 - p) / shots + z * z / (4.0 * shots * shots)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Noise models (A1)
#
# A logical error rate is meaningless without the noise model that produced it,
# and two LERs measured under different models are NOT comparable — not even
# ordinally. Code-capacity LER at a given nominal `p` is lower than circuit-level
# LER at the same `p` by orders of magnitude, because circuit-level noise adds
# measurement error, multiple rounds of syndrome extraction, and error
# propagation through the two-qubit gates.
#
# Every result therefore carries its model, and any function that assembles a
# comparison table refuses to mix them.
# ---------------------------------------------------------------------------
CODE_CAPACITY = "code_capacity"
CIRCUIT_LEVEL = "circuit_level"
NOISE_MODELS = (CODE_CAPACITY, CIRCUIT_LEVEL)


class NoiseModelMismatch(ValueError):
    """Raised when results from different noise models would be compared."""


def assert_comparable(results: Sequence[Any]) -> str:
    """Return the shared noise model of ``results``, or raise.

    ``results`` may be :class:`LerResult` instances or plain dicts carrying a
    ``noise_model`` key. Any mixture raises :class:`NoiseModelMismatch`.
    """
    models: dict[str, list[str]] = {}
    for r in results:
        model = r.noise_model if isinstance(r, LerResult) else r.get("noise_model")
        name = r.decoder if isinstance(r, LerResult) else r.get("decoder", "?")
        if model is None:
            raise NoiseModelMismatch(f"result for {name!r} carries no noise_model tag")
        models.setdefault(model, []).append(str(name))
    if len(models) > 1:
        detail = "; ".join(f"{m}: {', '.join(sorted(set(d)))}" for m, d in sorted(models.items()))
        raise NoiseModelMismatch(
            "refusing to compare logical error rates measured under different noise models "
            f"({detail}). Code-capacity and circuit-level LER are not comparable at the same "
            "nominal p — use estimate_ler_circuit_level() for every decoder in the comparison."
        )
    return next(iter(models))


@dataclass
class LerResult:
    """One logical-error-rate measurement with full statistical context."""

    decoder: str
    code: str
    physical_error_rate: float
    shots: int
    errors: int
    unfaithful: int
    seconds: float
    seed: int
    n_logical_qubits: int = 0
    #: Which noise model produced this number. Never compare across models (A1).
    noise_model: str = CODE_CAPACITY
    #: Rounds of syndrome extraction (circuit-level only; 0 for code-capacity).
    rounds: int = 0

    @property
    def ler(self) -> float:
        return self.errors / self.shots if self.shots else float("nan")

    @property
    def ci95(self) -> tuple[float, float]:
        return wilson_ci(self.errors, self.shots)

    @property
    def decodes_per_s(self) -> float:
        return self.shots / self.seconds if self.seconds > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ler"] = self.ler
        d["ci95_lo"], d["ci95_hi"] = self.ci95
        d["decodes_per_s"] = self.decodes_per_s
        return d


# ---------------------------------------------------------------------------
# Decoder resolution (built once, reused — S5)
# ---------------------------------------------------------------------------
def _build_decoder(decoder: Any, code: Any, error_rate: float):
    """Return ``(name, decode_batch_fn)`` for a decoder spec on ``code``.

    ``decoder`` may be a registered name (``"blossom"``, ``"sparse_blossom"``,
    ``"union_find"``, ``"fast_union_find"``, ``"cpu_batch"``, ``"bp_osd"`` —
    the pure-Python BP-OSD — or ``"rust_bposd"`` for the native BP-OSD) or a
    zero-arg callable returning an object with ``decode``/``batch_decode``.
    """
    c2q = [list(map(int, c)) for c in code.check_to_qubits]
    n = int(code.n_qubits)
    if callable(decoder) and not isinstance(decoder, str):
        inst = decoder()
        name = getattr(inst, "__class__", type(inst)).__name__
        return name, _batch_fn(inst)

    kind = str(decoder).lower()
    if kind in ("blossom", "sparse_blossom", "union_find", "fast_union_find", "cpu_batch"):
        from . import (
            BlossomDecoder,
            CPUBatchDecoder,
            FastUnionFindDecoder,
            SparseBlossomDecoder,
            UnionFindDecoder,
        )

        cls = {
            "blossom": BlossomDecoder,
            "sparse_blossom": SparseBlossomDecoder,
            "union_find": UnionFindDecoder,
            "fast_union_find": FastUnionFindDecoder,
            "cpu_batch": CPUBatchDecoder,
        }[kind]
        return kind, _batch_fn(cls(c2q, n))
    if kind in ("bp_osd", "python_bposd"):
        from .bposd import BpOsdDecoder

        return "bp_osd", _batch_fn(BpOsdDecoder(code.parity_check_matrix(), error_rate=error_rate))
    if kind in ("rust_bposd", "bposd"):
        from . import BPOSDDecoder

        return "rust_bposd", _batch_fn(BPOSDDecoder(c2q, n, error_rate))
    raise ValueError(f"unknown decoder spec: {decoder!r}")


def _batch_fn(inst: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Uniform batched-decode callable; falls back to a per-shot loop."""
    if hasattr(inst, "batch_decode"):
        return lambda S: np.asarray(inst.batch_decode(S), dtype=np.uint8)

    def _loop(S: np.ndarray) -> np.ndarray:
        out = np.zeros((S.shape[0], inst.n_qubits), dtype=np.uint8)
        for i in range(S.shape[0]):
            out[i] = np.asarray(inst.decode(S[i]), dtype=np.uint8)
        return out

    return _loop


# ---------------------------------------------------------------------------
# Monte-Carlo LER estimation (E2)
# ---------------------------------------------------------------------------
def estimate_ler(
    code: Any,
    decoder: Any,
    p: float,
    shots: int,
    seed: int = 0,
    batch_size: int | None = None,
) -> LerResult:
    """Estimate the logical error rate of ``decoder`` on ``code`` at rate ``p``.

    Samples i.i.d. bit-flip errors, decodes the syndromes (one warm decoder
    instance, batched), and counts a logical failure on every shot whose
    residual ``correction XOR error`` carries logical content, measured with
    the code's observable matrix (E1/C6).  Unfaithful corrections (those not
    reproducing the syndrome) are counted separately in
    :attr:`LerResult.unfaithful` — never silently folded into the LER.
    """
    L = code.logicals_matrix() if hasattr(code, "logicals_matrix") else None
    if L is None or L.shape[0] == 0:
        raise ValueError(
            f"code {getattr(code, 'name', '?')!r} exposes no logicals — "
            "logical error rate is undefined for it (see codes module notes)"
        )
    H = code.parity_check_matrix().astype(np.uint8)
    rng = np.random.default_rng(int(seed))
    name, decode_batch = _build_decoder(decoder, code, float(p))
    bs = int(batch_size) if batch_size else int(shots)
    errors = 0
    unfaithful = 0
    seconds = 0.0
    done = 0
    while done < shots:
        take = min(bs, shots - done)
        E = (rng.random((take, code.n_qubits)) < p).astype(np.uint8)
        S = ((E @ H.T) & 1).astype(np.uint8)
        t0 = time.perf_counter()
        C = np.asarray(decode_batch(S), dtype=np.uint8)
        seconds += time.perf_counter() - t0
        if C.shape != E.shape:
            raise RuntimeError(
                f"decoder returned shape {C.shape}, expected {E.shape} — refusing to compute a LER on mismatched output"
            )
        faithful = np.all(((C @ H.T) & 1) == S, axis=1)
        unfaithful += int((~faithful).sum())
        R = (C ^ E).astype(np.uint8)
        flips = (R @ L.T) & 1
        errors += int(np.any(flips, axis=1).sum())
        done += take
    return LerResult(
        decoder=name,
        code=getattr(code, "name", "?"),
        physical_error_rate=float(p),
        shots=int(shots),
        errors=errors,
        unfaithful=unfaithful,
        seconds=seconds,
        seed=int(seed),
        n_logical_qubits=int(L.shape[0]),
    )


# ---------------------------------------------------------------------------
# Circuit-level LER (A1) — the model the QEC field actually benchmarks under
# ---------------------------------------------------------------------------
def _surface_code_circuit(distance: int, p: float, rounds: int | None, basis: str = "x"):
    """Standard rotated-surface memory circuit at uniform circuit-level noise."""
    import stim

    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds if rounds else distance,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def _dem_observable_decoder(kind: str, dem: Any):
    """Return an object with ``decode_batch(dets) -> predicted observables``.

    One resolver for **every** decoder in a comparison — QECTOR's own backends and
    the external references alike — so no decoder can accidentally be measured
    through a different pipeline than the one it is being compared against.
    """
    k = kind.lower().replace("-", "_")
    k = k.removeprefix("qector_")

    if k in ("pymatching", "pm"):
        import pymatching

        return pymatching.Matching.from_detector_error_model(dem)
    if k in ("ldpc_bposd", "ldpc"):
        return _LdpcBpOsdOnDem(dem)

    # QECTOR backends share sinter_compat's resolver, which already returns
    # observable-space predictions for every kind it knows.
    from .sinter_compat import _build_matcher

    return _build_matcher(k, dem)


class _LdpcBpOsdOnDem:
    """External `ldpc` BP-OSD driven off the same DEM (observable-space output)."""

    def __init__(self, dem):
        from ldpc import BpOsdDecoder as LdpcOsd

        from .dem import from_stim

        model = from_stim(dem)
        H = model.check_matrix()
        self._L = model.observables_matrix()
        priors = model.priors()
        self._dec = LdpcOsd(
            H,
            error_channel=list(priors),
            max_iter=30,
            bp_method="product_sum",
            osd_method="OSD_CS",
            osd_order=0,
        )

    def decode_batch(self, shots):
        shots = np.asarray(shots, dtype=np.uint8)
        corr = np.stack([np.asarray(self._dec.decode(s), dtype=np.uint8) for s in shots])
        return ((self._L @ corr.T) & 1).T.astype(np.uint8)


def estimate_ler_circuit_level(
    distance: int,
    decoder: str,
    p: float,
    shots: int,
    rounds: int | None = None,
    seed: int = 0,
    basis: str = "x",
    circuit: Any = None,
) -> LerResult:
    """Circuit-level logical error rate for ``decoder`` on a rotated surface code.

    This is the measurement the QEC field benchmarks under, and the one that must
    be used for **any** cross-decoder comparison. Unlike :func:`estimate_ler`
    (code-capacity: i.i.d. data-qubit flips, one syndrome, no measurement error),
    this samples detectors from a real Stim circuit with gate, reset and
    measurement noise over ``rounds`` rounds of syndrome extraction, and scores
    against the circuit's own logical observables.

    ``decoder`` accepts ``"blossom"``, ``"belief"``, ``"unionfind"``, ``"bposd"``
    (QECTOR backends, with or without a ``qector_`` prefix), ``"pymatching"``, or
    ``"ldpc_bposd"``. Pass ``circuit=`` to score a circuit you built yourself.
    """
    circ = circuit if circuit is not None else _surface_code_circuit(distance, p, rounds, basis)
    dem = circ.detector_error_model(decompose_errors=True)
    sampler = circ.compile_detector_sampler(seed=int(seed))
    det, obs = sampler.sample(shots=shots, separate_observables=True)
    det = np.ascontiguousarray(det.astype(np.uint8))
    obs = np.ascontiguousarray(obs.astype(np.uint8))

    dec = _dem_observable_decoder(decoder, dem)
    t0 = time.perf_counter()
    pred = np.asarray(dec.decode_batch(det), dtype=np.uint8)
    seconds = time.perf_counter() - t0

    pred = pred.reshape(shots, -1)
    obs_r = obs.reshape(shots, -1)
    width = min(pred.shape[1], obs_r.shape[1])
    errors = int(np.any(pred[:, :width] != obs_r[:, :width], axis=1).sum())

    return LerResult(
        decoder=str(decoder),
        code=f"rotated_surface_d{distance}_r{rounds or distance}",
        physical_error_rate=float(p),
        shots=int(shots),
        errors=errors,
        unfaithful=0,  # observable-space scoring: faithfulness is not separable here
        seconds=seconds,
        seed=int(seed),
        n_logical_qubits=int(obs_r.shape[1]),
        noise_model=CIRCUIT_LEVEL,
        rounds=int(rounds or distance),
    )


def reference_validate(
    code,
    decoder_spec: str,
    p: float,
    shots: int,
    seed: int = 0,
) -> dict:
    """E3: validates QECTOR against PyMatching on identical syndromes."""
    import pymatching
    import stim

    d = getattr(code, "distance", 5) or 5
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=shots, separate_observables=True)
    det, obs = det.astype(np.uint8), obs.astype(np.uint8)
    pm = pymatching.Matching.from_detector_error_model(dem)
    ppred = np.asarray(pm.decode_batch(det), dtype=np.uint8)
    pm_err = int(np.any(ppred.reshape(len(det), -1) != obs.reshape(len(det), -1), axis=1).sum())
    from .pymatching_compat import Matching as QMatching

    qm = QMatching.from_detector_error_model(dem)
    qpred = np.asarray(qm.decode_batch(det), dtype=np.uint8)
    q_err = int(np.any(qpred.reshape(len(det), -1) != obs.reshape(len(det), -1), axis=1).sum())
    pm_lo, pm_hi = wilson_ci(pm_err, shots)
    q_lo, q_hi = wilson_ci(q_err, shots)
    overlap = max(pm_lo, q_lo) <= min(pm_hi, q_hi)
    return {
        "decoder": decoder_spec,
        "code": code.name,
        "p": p,
        "shots": shots,
        "pymatching_ler": pm_err / shots,
        "qector_ler": q_err / shots,
        "pymatching_ci95": (pm_lo, pm_hi),
        "qector_ci95": (q_lo, q_hi),
        "ci_overlap": overlap,
        "verdict": "PASS" if overlap else "FAIL",
    }


def run_memory_experiment(
    distance: int,
    rounds_list: list[int],
    p: float,
    decoder: str = "qector_blossom",
    max_shots: int = 40000,
    max_errors: int = 800,
    workers: int = 4,
) -> dict:
    """C3: LER-vs-rounds memory experiment via Sinter."""
    import sinter
    import stim

    from .sinter_compat import qector_sinter_decoders

    tasks = []
    for r in rounds_list:
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=distance,
            rounds=r,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
        )
        tasks.append(sinter.Task(circuit=circ, json_metadata={"d": distance, "rounds": r}))
    custom = qector_sinter_decoders() if decoder.startswith("qector") else {}
    stats = sinter.collect(
        num_workers=workers,
        tasks=tasks,
        decoders=[decoder],
        custom_decoders=custom,
        max_shots=max_shots,
        max_errors=max_errors,
    )
    results = [
        {
            "rounds": st.json_metadata["rounds"],
            "shots": st.shots,
            "errors": st.errors,
            "ler": st.errors / st.shots if st.shots else float("nan"),
            "ci95": wilson_ci(st.errors, st.shots),
            "seconds": st.seconds,
        }
        for st in stats
    ]
    return {"decoder": decoder, "distance": distance, "p": p, "results": results}


def sample_biased_errors(
    rng: np.random.Generator,
    n_qubits: int,
    p_z: float,
    eta: float = 0.5,
    *,
    return_both_sectors: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """C4: sample biased Pauli errors with Z-bias factor ``eta``.

    ``p_z`` is the Z-error probability; ``eta = p_z / (p_z + p_x)`` controls the
    bias (0.5 = unbiased/depolarising-symmetric, 1.0 = pure Z, → 0 = X-dominated).
    Each qubit independently suffers a Z error with probability ``p_z`` and an X
    error with probability ``p_x = p_z * (1 - eta) / eta``.

    Returns the Z-sector error vector by default. Pass
    ``return_both_sectors=True`` to get ``(z_errors, x_errors)``.

    .. note::
       Before v0.7.0 the ``p_x`` expression on the first line of the body was
       written as a bare expression statement, so its value was discarded and
       ``eta`` never reached the sampler — every "biased" result was in fact
       unbiased. Any biased-noise figure produced before v0.7.0 is invalid.
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta!r} (1.0 = pure Z, 0.5 = unbiased)")
    if not 0.0 <= p_z <= 1.0:
        raise ValueError(f"p_z must be in [0, 1], got {p_z!r}")

    p_x = min(p_z * (1.0 - eta) / eta, 1.0)
    z_errors = (rng.random(n_qubits) < p_z).astype(np.uint8)
    if not return_both_sectors:
        return z_errors
    x_errors = (rng.random(n_qubits) < p_x).astype(np.uint8)
    return z_errors, x_errors


def run_competitive_suite(
    p: float = 0.001,
    shots: int = 20000,
    seed: int = 0,
    distances: Sequence[int] = (3, 5, 7),
    rounds: int | None = None,
    decoders: Sequence[str] = ("qector_blossom", "qector_belief", "qector_unionfind", "pymatching"),
    output_json: str | None = None,
) -> list[dict]:
    """Competitive benchmark: every decoder on the **same** circuit-level task.

    All decoders — QECTOR's and the external references — are driven through one
    pipeline: the same Stim circuit, the same detector samples, the same DEM, the
    same observable scoring (:func:`estimate_ler_circuit_level`). The only thing
    that varies between rows is the decoder.

    .. warning::
       Before v0.7.0 this function measured QECTOR with :func:`estimate_ler`
       (**code-capacity**: i.i.d. data-qubit flips, one syndrome, no measurement
       error) while measuring PyMatching on a **circuit-level** Stim circuit with
       ``rounds=d`` — then printed both in one table. Those are different
       experiments and the resulting comparison was meaningless in QECTOR's
       favour. Any competitive figure generated before v0.7.0 must be discarded
       and regenerated with this function.

    Returns a list of result dicts, all tagged ``noise_model="circuit_level"``
    and validated by :func:`assert_comparable` before returning.
    """
    results: list[LerResult] = []
    skipped: list[str] = []

    for d in distances:
        for dec in decoders:
            try:
                results.append(
                    estimate_ler_circuit_level(
                        distance=d,
                        decoder=dec,
                        p=p,
                        shots=shots,
                        rounds=rounds,
                        seed=seed + 17 * d,
                    )
                )
            except ImportError as exc:
                # An absent optional reference (pymatching / ldpc) is skipped, but
                # recorded — a silently missing baseline is how misleading tables
                # get published.
                skipped.append(f"d={d} {dec}: {exc}")

    # Guarantees the table is internally comparable. Raises rather than emitting
    # a mixed-model comparison.
    assert_comparable(results)

    out = [r.to_dict() for r in results]
    for row in out:
        row["skipped_baselines"] = skipped

    if output_json:
        import json

        with open(output_json, "w") as fh:
            json.dump(
                {
                    "noise_model": CIRCUIT_LEVEL,
                    "physical_error_rate": p,
                    "shots": shots,
                    "seed": seed,
                    "distances": list(distances),
                    "skipped_baselines": skipped,
                    "results": out,
                },
                fh,
                indent=2,
            )
    return out


# ---------------------------------------------------------------------------
# Threshold estimation (E6)
# ---------------------------------------------------------------------------
@dataclass
class ThresholdResult:
    """A LER-vs-``p`` grid across code distances, with a crossing estimate."""

    decoder: str
    distances: list[int]
    p_values: list[float]
    results: list[LerResult]
    crossing: float | None
    notes: str = ""

    def grid(self) -> dict[tuple[int, float], LerResult]:
        out: dict[tuple[int, float], LerResult] = {}
        idx = 0
        for d in self.distances:
            for p in self.p_values:
                out[(d, p)] = self.results[idx]
                idx += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder": self.decoder,
            "distances": self.distances,
            "p_values": self.p_values,
            "crossing": self.crossing,
            "notes": self.notes,
            "results": [r.to_dict() for r in self.results],
        }


def _crossing_from_pairs(p_values: list[float], series: dict[float, list[tuple[int, int]]]) -> float | None:
    """First ``p`` where LER no longer decreases *significantly* with distance.

    For each ``p``, ``series[p]`` holds ``(errors, shots)`` per distance in
    ascending order.  The fan-out between the smallest and largest distance
    is significant when ``ler[d_min] - ler[d_max] > 1.96 * SE`` (pooled
    binomial standard error).  At or above threshold the curves meet within
    noise (or invert), so that ``p`` is the crossing.  A naive strict-inequality
    test fails exactly at threshold, where both LERs sit at ~0.5 ± shot noise.
    """
    for p in p_values:
        pairs = series[p]
        (e0, n0), (e1, n1) = pairs[0], pairs[-1]
        if n0 == 0 or n1 == 0:
            return p
        p0, p1 = e0 / n0, e1 / n1
        se = math.sqrt(p0 * (1.0 - p0) / n0 + p1 * (1.0 - p1) / n1)
        if not (p0 - p1 > 1.96 * se):
            return p
    return None


def run_threshold_sweep(
    code_factory: Callable[[int], Any],
    distances: list[int],
    p_values: list[float],
    decoder: Any,
    shots: int,
    seed: int = 0,
) -> ThresholdResult:
    """Code-capacity threshold sweep: LER-vs-``p`` for each distance (E6).

    ``code_factory(distance)`` must return a :class:`Code` with logicals.
    The crossing is the smallest tested ``p`` where increasing distance no
    longer strictly lowers the LER — the classic threshold signature.
    """
    distances = [int(d) for d in distances]
    p_values = [float(p) for p in p_values]
    results: list[LerResult] = []
    for d in distances:
        code = code_factory(d)
        for p in p_values:
            r = estimate_ler(code, decoder, p, shots, seed=int(seed) + 7919 * d + round(p * 1_000_003))
            results.append(r)
    tmp = ThresholdResult(decoder=str(decoder), distances=distances, p_values=p_values, results=results, crossing=None)
    grid = tmp.grid()
    series = {p: [(grid[(d, p)].errors, grid[(d, p)].shots) for d in distances] for p in p_values}
    tmp.crossing = _crossing_from_pairs(p_values, series)
    return tmp


def sinter_threshold_sweep(
    distances: list[int],
    p_values: list[float],
    decoder: str = "qector_blossom",
    max_shots: int = 40000,
    max_errors: int = 800,
    workers: int = 4,
    rounds: int = 0,
) -> dict[str, Any]:
    """Circuit-level threshold sweep through the standard Sinter harness (E6).

    Generates rotated-surface memory-x Stim circuits per ``(d, p)`` and
    collects with ``sinter.collect``; every point carries shots, errors and a
    Wilson 95% interval.  ``decoder`` may be any QECTOR sinter decoder
    (``qector_blossom`` / ``qector_belief`` / ``qector_unionfind``) or a
    built-in sinter decoder such as ``pymatching``.
    """
    import sinter
    import stim

    from .sinter_compat import qector_sinter_decoders

    tasks = []
    for d in distances:
        r = rounds or d
        for p in p_values:
            circ = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=d,
                rounds=r,
                after_clifford_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
            )
            tasks.append(sinter.Task(circuit=circ, json_metadata={"d": d, "p": p}))
    custom = qector_sinter_decoders() if decoder.startswith("qector") else {}
    stats = sinter.collect(
        num_workers=workers,
        tasks=tasks,
        decoders=[decoder],
        custom_decoders=custom,
        max_shots=max_shots,
        max_errors=max_errors,
    )
    results: list[dict[str, Any]] = []
    for st in stats:
        d = st.json_metadata["d"]
        p = st.json_metadata["p"]
        lo, hi = wilson_ci(st.errors, st.shots)
        results.append(
            {
                "d": d,
                "p": p,
                "shots": st.shots,
                "errors": st.errors,
                "ler": st.errors / st.shots if st.shots else float("nan"),
                "ci95": (lo, hi),
                "seconds": st.seconds,
            }
        )
    grid = {(r["d"], r["p"]): r for r in results}
    if all((d, p) in grid for d in distances for p in p_values):
        series = {p: [(grid[(d, p)]["errors"], grid[(d, p)]["shots"]) for d in distances] for p in p_values}
        crossing = _crossing_from_pairs(list(p_values), series)
    else:
        crossing = None
    return {"decoder": decoder, "distances": distances, "p_values": p_values, "results": results, "crossing": crossing}
