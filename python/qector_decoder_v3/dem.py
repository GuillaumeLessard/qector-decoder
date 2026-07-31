"""
qector_decoder_v3.dem - Stim Detector Error Model (DEM) loader.

A correct, dependency-free parser that turns a Stim ``DetectorErrorModel`` (a
``stim.DetectorErrorModel`` object **or** a ``.dem`` text file) into the matching
problem QECTOR decodes:

    * one **column** (fault mechanism) per ``error`` instruction component,
    * one **row** (check) per detector,
    * per-column prior probabilities and matching weights ``log((1-p)/p)``,
    * an **observables matrix** giving which mechanisms flip each logical observable.

This replaces the earlier heuristic in :mod:`qector_decoder_v3.stim_compat`, which
conflated detector indices with qubit indices and produced an incorrect ``H``.
Here, ``H[detector, mechanism] = 1`` iff the error mechanism flips that detector -
exactly the detector graph PyMatching / Stim use.

The parser handles the full flattened DEM grammar - ``error``, ``detector``,
``logical_observable``, ``shift_detectors`` and ``repeat { ... }`` blocks -
without needing Stim installed.  When given a live ``stim.DetectorErrorModel`` it
is flattened first for exactness.

Example
-------
>>> from qector_decoder_v3 import dem
>>> model = dem.load_dem_file("circuit.dem")  # or dem.from_stim(dem_object)
>>> code = model.to_code()  # a codes.Code
>>> decoder = model.make_decoder("sparse_blossom")
>>> correction = decoder.decode(syndrome)
>>> logical_pred = model.predicted_observables(correction)
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DemError",
    "DemModel",
    "estimate_priors_from_detectors",
    "from_stim",
    "load_dem_file",
    "parse_dem",
]


def estimate_priors_from_detectors(
    model: DemModel,
    detection_events: np.ndarray,
    smoothing: float = 1e-8,
) -> np.ndarray:
    """Estimate per-mechanism prior probabilities from empirical detection events.

    Uses per-detector firing rates and pairwise correlations to recover the
    underlying mechanism probabilities.  For weight-1 mechanisms the detector
    firing rate is a direct estimate.  For weight-2 mechanisms the correlated
    (XOR) probability ``P(d1 xor d2)`` is inverted under the independent-error
    model ``P(d xor d) = 2p(1-p)``, giving the closed-form
    ``p = (1 - sqrt(1 - 2*P_xor)) / 2``.  Hyperedges take the maximum firing
    rate among their detectors.

    Parameters
    ----------
    model:
        Parsed DEM whose per-mechanism structure is used to map detector-level
        statistics back to mechanism priors.
    detection_events:
        Binary array of shape ``(n_shots, n_detectors)`` — the observed
        detection events from a Stim sampler.
    smoothing:
        Small constant added to avoid zero / one probabilities
        (clamped to ``[smoothing, 0.5]``).

    Returns
    -------
    np.ndarray
        Per-mechanism prior probabilities, shape ``(num_errors,)``.
    """
    n_shots, n_det = detection_events.shape
    f = detection_events.mean(axis=0, dtype=np.float64)
    priors = np.zeros(model.num_errors, dtype=np.float64)
    for j, e in enumerate(model.errors):
        dets = e.detectors
        if len(dets) == 1:
            priors[j] = f[dets[0]] if dets[0] < n_det else smoothing
        elif len(dets) == 2:
            d1, d2 = dets
            joint = (
                (detection_events[:, d1] & detection_events[:, d2]).mean(dtype=np.float64)
                if d1 < n_det and d2 < n_det
                else 0.0
            )
            # P(d1 xor d2) = f1 + f2 - 2*joint
            p_xor = f[d1] + f[d2] - 2.0 * joint
            p_xor = np.clip(p_xor, 0.0, 0.5)
            priors[j] = (1.0 - np.sqrt(1.0 - 2.0 * p_xor)) / 2.0
        else:
            # Hyperedge: use max firing rate among its detectors
            valid = [f[d] for d in dets if 0 <= d < n_det]
            priors[j] = max(valid) if valid else smoothing
        priors[j] = np.clip(priors[j], smoothing, 0.5)
    return priors


@dataclass
class DemError:
    """One fault mechanism (a DEM ``error`` component, i.e. a column of H)."""

    probability: float
    detectors: tuple[int, ...]
    observables: tuple[int, ...]

    @property
    def weight(self) -> float:
        """Matching weight ``log((1-p)/p)`` (clamped for p in {0, 1})."""
        p = min(max(self.probability, 1e-15), 1.0 - 1e-15)
        return math.log((1.0 - p) / p)

    @property
    def is_graphlike(self) -> bool:
        return len(self.detectors) <= 2


@dataclass
class DemModel:
    """Parsed detector error model.

    Attributes
    ----------
    errors : list[DemError]
        Fault mechanisms (columns of H), in DEM order.
    num_detectors : int
        Number of detectors (rows of H).
    num_observables : int
        Number of logical observables.
    detector_coords : dict[int, tuple[float, ...]]
        Optional detector coordinates (for visualisation / diagnostics).
    """

    errors: list[DemError]
    num_detectors: int
    num_observables: int
    detector_coords: dict = field(default_factory=dict)

    # -- shapes ------------------------------------------------------------
    @property
    def num_errors(self) -> int:
        return len(self.errors)

    @property
    def is_graphlike(self) -> bool:
        """True iff every mechanism flips at most two detectors."""
        return all(e.is_graphlike for e in self.errors)

    # -- graph collapse ----------------------------------------------------
    def collapse_to_graph(self) -> DemModel:
        """Collapse parallel mechanisms into one min-weight edge per detector set.

        A circuit-level DEM (``decompose_errors=True``) has many parallel
        mechanisms between the same pair of detectors - different fault locations
        that flip the same detectors. A matching decoder only ever uses the
        lowest-weight edge between two detectors, so decoding over every raw
        mechanism is wasted work (QECTOR was ~100x slower than PyMatching at
        circuit level for exactly this reason).

        This merges parallel edges the way PyMatching does: probabilities are
        combined under the independent-error rule
        ``p = p1(1-p2) + p2(1-p1)`` (XOR), and the merged edge keeps the
        observable set of its lowest-weight (most likely) member. The result is a
        graphlike :class:`DemModel` with one edge per unique detector signature,
        which decodes orders of magnitude faster at identical logical accuracy.

        Hyperedges (>2 detectors) are passed through unchanged.
        """
        groups: dict = {}
        order: list[tuple] = []
        for e in self.errors:
            sig = e.detectors
            if sig not in groups:
                groups[sig] = []
                order.append(sig)
            groups[sig].append(e)

        merged: list[DemError] = []
        for sig in order:
            members = groups[sig]
            if len(members) == 1:
                merged.append(members[0])
                continue
            # combined probability under repeated XOR (independent errors)
            p = 0.0
            for m in members:
                p = p * (1.0 - m.probability) + m.probability * (1.0 - p)
            best = min(members, key=lambda m: m.weight)  # lowest weight == most likely
            merged.append(DemError(probability=p, detectors=sig, observables=best.observables))

        return DemModel(
            errors=merged,
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
            detector_coords=dict(self.detector_coords),
        )

    # -- matrices ----------------------------------------------------------
    def check_matrix(self) -> np.ndarray:
        """Detector check matrix ``H`` of shape ``(num_detectors, num_errors)``."""
        H = np.zeros((self.num_detectors, self.num_errors), dtype=np.uint8)
        for j, e in enumerate(self.errors):
            for d in e.detectors:
                if 0 <= d < self.num_detectors:
                    H[d, j] ^= np.uint8(1)
        return H

    def observables_matrix(self) -> np.ndarray:
        """Observable matrix of shape ``(num_observables, num_errors)`` (uint8)."""
        L = np.zeros((self.num_observables, self.num_errors), dtype=np.uint8)
        for j, e in enumerate(self.errors):
            for o in e.observables:
                if 0 <= o < self.num_observables:
                    L[o, j] ^= np.uint8(1)
        return L

    def priors(self) -> np.ndarray:
        """Per-mechanism prior probabilities, shape ``(num_errors,)``."""
        return np.array([e.probability for e in self.errors], dtype=np.float64)

    def recalibrate(self, detection_events: np.ndarray) -> DemModel:
        """Return a new ``DemModel`` with priors estimated from empirical data.

        Uses :func:`estimate_priors_from_detectors` to replace every mechanism's
        prior probability with the value inferred from the observed detection
        event statistics.  The returned model has the same detector / observable
        structure but empirically grounded error rates.

        Parameters
        ----------
        detection_events:
            Binary array of shape ``(n_shots, n_detectors)`` from a Stim sampler.

        Returns
        -------
        DemModel
            New model with recalibrated priors.
        """
        priors = estimate_priors_from_detectors(self, detection_events)
        new_errors = [
            DemError(
                probability=float(priors[j]),
                detectors=e.detectors,
                observables=e.observables,
            )
            for j, e in enumerate(self.errors)
        ]
        return DemModel(
            errors=new_errors,
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
            detector_coords=dict(self.detector_coords),
        )

    def weights(self) -> np.ndarray:
        """Per-mechanism matching weights ``log((1-p)/p)``, shape ``(num_errors,)``."""
        return np.array([e.weight for e in self.errors], dtype=np.float64)

    def check_to_qubits(self) -> list[list[int]]:
        """``check_to_qubits`` (per detector, the mechanism indices flipping it)."""
        c2q: list[list[int]] = [[] for _ in range(self.num_detectors)]
        for j, e in enumerate(self.errors):
            for d in e.detectors:
                if 0 <= d < self.num_detectors:
                    c2q[d].append(j)
        return [sorted(set(x)) for x in c2q]

    # -- integration -------------------------------------------------------
    def to_code(self, name: str = "stim_dem"):
        """Return a :class:`qector_decoder_v3.codes.Code` for this model."""
        from .codes import Code

        return Code(
            name=name,
            check_to_qubits=self.check_to_qubits(),
            n_qubits=self.num_errors,
            logicals=None,
            qubit_weights=self.weights(),
            description="Detector error model loaded from a Stim DEM.",
            _meta={"observables_matrix": self.observables_matrix()},
        )

    #: Decoder kinds :meth:`make_decoder` accepts, mapped to their canonical
    #: name. Exposed so callers (and benchmarks) can enumerate what is
    #: constructible from a DEM instead of hard-coding a list that drifts.
    DECODER_KINDS = {
        "union_find": "union_find",
        "uf": "union_find",
        "unionfind": "union_find",
        "fast_union_find": "fast_union_find",
        "fast_uf": "fast_union_find",
        "fastunionfind": "fast_union_find",
        "blossom": "blossom",
        "mwpm": "blossom",
        "sparse_blossom": "sparse_blossom",
        "sparse": "sparse_blossom",
        "bp_osd": "bp_osd",
        "bposd": "bp_osd",
        "bp": "bp_osd",
        "lookup_table": "lookup_table",
        "lookup": "lookup_table",
        "hybrid_cascade": "hybrid_cascade",
        "cascade": "hybrid_cascade",
        "ambiguity_cluster": "ambiguity_cluster",
        "ambig_cluster": "ambiguity_cluster",
        "ambig": "ambiguity_cluster",
        "two_stage": "two_stage",
        "twostage": "two_stage",
    }

    def make_decoder(
        self,
        kind: str = "sparse_blossom",
        *,
        weighted: bool = True,
        check_types: Sequence[bool] | None = None,
    ):
        """Construct a QECTOR decoder over this model's detector graph.

        ``kind`` is any key of :attr:`DECODER_KINDS`: the matching decoders
        (``"union_find"``, ``"fast_union_find"``, ``"blossom"``,
        ``"sparse_blossom"``), ``"bp_osd"``, and the
        ``"lookup_table"`` / ``"hybrid_cascade"`` / ``"ambiguity_cluster"`` /
        ``"two_stage"`` families. Those last four ship in every wheel but were
        previously unreachable from a DEM, which is the entry point real
        circuit-level workloads use.

        Every matching decoder — including the Union-Find variants as of UF-01 —
        receives the DEM's per-mechanism log-likelihood weights
        ``log((1-p)/p)``.  BP-OSD uses the mean prior as its channel probability.

        Pass ``weighted=False`` to get the pre-v0.7.0 topology-only Union-Find,
        which is useful only for reproducing older measurements: an unweighted
        decoder cannot distinguish a ``p = 1e-4`` mechanism from a ``p = 1e-2``
        one, which is precisely what circuit-level noise requires.

        ``check_types`` is required only by ``"two_stage"``, which decodes the X
        and Z sectors separately: it is one bool per detector. A DEM does not
        record the sector, so it cannot be inferred here.
        """
        from . import (
            AmbiguityClusterDecoder,
            BlossomDecoder,
            BPOSDDecoder,
            FastUnionFindDecoder,
            HybridCascadeDecoder,
            LookupTableDecoder,
            SparseBlossomDecoder,
            TwoStageDecoder,
            UnionFindDecoder,
        )

        c2q = self.check_to_qubits()
        nq = self.num_errors
        w = self.weights().tolist()
        uf_w = w if weighted else None
        mean_p = max(float(self.priors().mean()) if self.num_errors else 0.05, 1e-3)

        canonical = self.DECODER_KINDS.get(kind.lower().strip())
        if canonical is None:
            raise ValueError(
                f"unknown decoder kind: {kind!r}; expected one of {sorted(set(self.DECODER_KINDS.values()))}"
            )

        if canonical == "union_find":
            return UnionFindDecoder(c2q, nq, edge_weights=uf_w)
        if canonical == "fast_union_find":
            return FastUnionFindDecoder(c2q, nq, edge_weights=uf_w)
        if canonical == "blossom":
            return BlossomDecoder(c2q, nq, w)
        if canonical == "sparse_blossom":
            return SparseBlossomDecoder(c2q, nq, w)
        if canonical == "bp_osd":
            return BPOSDDecoder(c2q, nq, mean_p)
        if canonical == "lookup_table":
            return LookupTableDecoder(c2q, nq)
        if canonical == "hybrid_cascade":
            # Weighted pre-filter, escalating to an exact solve; the DEM's
            # weights and mean prior are exactly what it needs.
            return HybridCascadeDecoder(c2q, nq, edge_weights=w, error_rate=mean_p)
        if canonical == "ambiguity_cluster":
            return AmbiguityClusterDecoder(c2q, nq, error_rate=mean_p)
        if canonical == "two_stage":
            if check_types is None:
                raise ValueError(
                    "two_stage needs check_types (one bool per detector, X vs Z sector); "
                    "a detector error model does not record the sector, so pass it explicitly: "
                    "make_decoder('two_stage', check_types=[...])"
                )
            types = [bool(t) for t in check_types]
            if len(types) != self.num_detectors:
                raise ValueError(
                    f"check_types has {len(types)} entries, expected {self.num_detectors} (one per detector)"
                )
            return TwoStageDecoder(c2q, types, nq)
        raise AssertionError(f"unhandled decoder kind {canonical!r}")  # pragma: no cover

    def predicted_observables(self, correction: Sequence[int]) -> np.ndarray:
        """Logical observable flips implied by a correction (``L @ c mod 2``)."""
        c = np.asarray(correction, dtype=np.uint8).reshape(-1)
        if c.shape[0] != self.num_errors:
            raise ValueError(f"correction has length {c.shape[0]}, expected {self.num_errors}")
        return (self.observables_matrix() @ c) & 1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DemModel(errors={self.num_errors}, detectors={self.num_detectors}, "
            f"observables={self.num_observables}, graphlike={self.is_graphlike})"
        )


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_ERROR_RE = re.compile(r"^error\(\s*(" + _NUM + r")\s*\)\s*(.*)$")
_DETECTOR_RE = re.compile(r"^detector(?:\(([^)]*)\))?\s*(.*)$")
_OBS_RE = re.compile(r"^logical_observable\s+L(\d+)\s*$")
_SHIFT_RE = re.compile(r"^shift_detectors(?:\(([^)]*)\))?\s+(\d+)\s*$")
_REPEAT_RE = re.compile(r"^repeat\s+(\d+)\s*\{?\s*$")


def parse_dem(text: str) -> DemModel:
    """Parse Stim DEM text into a :class:`DemModel`.

    Supports ``error``, ``detector``, ``logical_observable``, ``shift_detectors``
    and nested ``repeat { ... }`` blocks.  Detector targets are resolved against
    the running ``shift_detectors`` offset, matching Stim semantics.
    """
    tokens = _tokenize(text)
    errors: list[DemError] = []
    coords: dict = {}
    state = {"det_offset": 0, "coord_offset": None, "max_det": -1, "max_obs": -1}
    _exec_block(tokens, 0, len(tokens), state, errors, coords)
    num_detectors = (state["max_det"] if state["max_det"] is not None else -1) + 1
    num_observables = (state["max_obs"] if state["max_obs"] is not None else -1) + 1
    return DemModel(
        errors=errors,
        num_detectors=max(num_detectors, 0),
        num_observables=max(num_observables, 0),
        detector_coords=coords,
    )


def _tokenize(text: str) -> list[str]:
    """Split DEM text into instructions, with ``{`` and ``}`` as standalone tokens."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        token = ""
        for ch in line:
            if ch in "{}":
                if token.strip():
                    out.append(token.strip())
                out.append(ch)
                token = ""
            else:
                token += ch
        if token.strip():
            out.append(token.strip())
    return out


def _exec_block(
    tokens: list[str],
    start: int,
    end: int,
    state: dict,
    errors: list[DemError],
    coords: dict,
) -> int:
    i = start
    while i < end:
        tok = tokens[i]
        if tok == "}":
            return i + 1
        if tok == "{":
            i += 1
            continue

        m = _REPEAT_RE.match(tok)
        if m:
            count = int(m.group(1))
            # find the body bounds (the token after the opening brace .. matching })
            body_start = i + 1
            if body_start < end and tokens[body_start] == "{":
                body_start += 1
            body_end = _matching_brace(tokens, body_start - 1, end)
            for _ in range(count):
                _exec_block(tokens, body_start, body_end, state, errors, coords)
            i = body_end + 1
            continue

        _exec_instruction(tok, state, errors, coords)
        i += 1
    return end


def _matching_brace(tokens: list[str], open_idx: int, end: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_idx``."""
    depth = 0
    i = open_idx
    while i < end:
        if tokens[i] == "{":
            depth += 1
        elif tokens[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise DemParseError("unbalanced repeat braces in DEM")


class DemParseError(ValueError):
    """Raised when DEM text cannot be parsed."""


def _exec_instruction(tok: str, state: dict, errors: list[DemError], coords: dict) -> None:
    if tok.startswith("error("):
        m = _ERROR_RE.match(tok)
        if not m:
            raise DemParseError(f"malformed error instruction: {tok!r}")
        prob = float(m.group(1))
        targets = m.group(2)
        # Split decomposition components on '^'; each is its own mechanism.
        for component in targets.split("^"):
            dets, obs = _parse_targets(component, state["det_offset"])
            if not dets and not obs:
                continue
            for d in dets:
                state["max_det"] = max(state["max_det"], d)
            for o in obs:
                state["max_obs"] = max(state["max_obs"], o)
            errors.append(
                DemError(
                    probability=prob,
                    detectors=tuple(sorted(set(dets))),
                    observables=tuple(sorted(set(obs))),
                )
            )
        return

    if tok.startswith("shift_detectors"):
        m = _SHIFT_RE.match(tok)
        if m:
            state["det_offset"] += int(m.group(2))
            return
        # shift_detectors with only coords (no detector shift)
        if tok.startswith("shift_detectors"):
            return

    if tok.startswith("detector"):
        m = _DETECTOR_RE.match(tok)
        if m:
            coord_str, rest = m.group(1), m.group(2)
            dets, _ = _parse_targets(rest, state["det_offset"])
            for d in dets:
                state["max_det"] = max(state["max_det"], d)
                if coord_str:
                    try:
                        coords[d] = tuple(float(x) for x in coord_str.split(",") if x.strip())
                    except ValueError:
                        pass
        return

    m = _OBS_RE.match(tok)
    if m:
        state["max_obs"] = max(state["max_obs"], int(m.group(1)))
        return

    # detector_coords / tick / unknown directives are ignored safely.


def _parse_targets(text: str, det_offset: int) -> tuple[list[int], list[int]]:
    dets: list[int] = []
    obs: list[int] = []
    for part in text.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if part[0] in "Dd":
            try:
                dets.append(int(part[1:]) + det_offset)
            except ValueError:
                continue
        elif part[0] in "Ll":
            try:
                obs.append(int(part[1:]))
            except ValueError:
                continue
    return dets, obs


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------
def load_dem_file(path: str) -> DemModel:
    """Parse a ``.dem`` file from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return parse_dem(fh.read())


def from_stim(dem: Any) -> DemModel:
    """Build a :class:`DemModel` from a ``stim.DetectorErrorModel`` object.

    The model is flattened first (expanding ``repeat`` and ``shift_detectors``)
    for an exact column-for-column correspondence with Stim, then parsed from
    its text form.  (A native instruction-iteration path was benchmarked and
    measured at 0.60x the speed of the text+regex path — Stim's per-instruction
    ``args_copy``/``targets_copy`` allocations dominate — so the text path is
    kept deliberately.  See K3dev.md S1.)
    """
    if isinstance(dem, str):
        return parse_dem(dem)
    if not hasattr(dem, "num_detectors"):
        raise TypeError(f"expected a stim.DetectorErrorModel (or DEM text), got {type(dem).__name__}")
    try:
        flat = dem.flattened()
    except (AttributeError, RuntimeError):  # pragma: no cover - older Stim
        flat = dem
    model = parse_dem(str(flat))
    # Trust Stim's declared counts when available.
    model.num_detectors = max(model.num_detectors, int(dem.num_detectors))
    model.num_observables = max(model.num_observables, int(dem.num_observables))
    return model
