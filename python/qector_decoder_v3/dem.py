"""
qector_decoder_v3.dem — Stim Detector Error Model (DEM) loader.

A correct, dependency-free parser that turns a Stim ``DetectorErrorModel`` (a
``stim.DetectorErrorModel`` object **or** a ``.dem`` text file) into the matching
problem QECTOR decodes:

    * one **column** (fault mechanism) per ``error`` instruction component,
    * one **row** (check) per detector,
    * per-column prior probabilities and matching weights ``log((1-p)/p)``,
    * an **observables matrix** giving which mechanisms flip each logical observable.

This replaces the earlier heuristic in :mod:`qector_decoder_v3.stim_compat`, which
conflated detector indices with qubit indices and produced an incorrect ``H``.
Here, ``H[detector, mechanism] = 1`` iff the error mechanism flips that detector —
exactly the detector graph PyMatching / Stim use.

The parser handles the full flattened DEM grammar — ``error``, ``detector``,
``logical_observable``, ``shift_detectors`` and ``repeat { ... }`` blocks —
without needing Stim installed.  When given a live ``stim.DetectorErrorModel`` it
is flattened first for exactness.

Example
-------
>>> from qector_decoder_v3 import dem
>>> model = dem.load_dem_file("circuit.dem")     # or dem.from_stim(dem_object)
>>> code = model.to_code()                         # a codes.Code
>>> decoder = model.make_decoder("sparse_blossom")
>>> correction = decoder.decode(syndrome)
>>> logical_pred = model.predicted_observables(correction)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

import numpy as np

__all__ = [
    "DemModel",
    "DemError",
    "parse_dem",
    "load_dem_file",
    "from_stim",
]


@dataclass
class DemError:
    """One fault mechanism (a DEM ``error`` component, i.e. a column of H)."""

    probability: float
    detectors: Tuple[int, ...]
    observables: Tuple[int, ...]

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

    errors: List[DemError]
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
    def collapse_to_graph(self) -> "DemModel":
        """Collapse parallel mechanisms into one min-weight edge per detector set.

        A circuit-level DEM (``decompose_errors=True``) has many parallel
        mechanisms between the same pair of detectors — different fault locations
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
        order: List[tuple] = []
        for e in self.errors:
            sig = e.detectors
            if sig not in groups:
                groups[sig] = []
                order.append(sig)
            groups[sig].append(e)

        merged: List[DemError] = []
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

    def weights(self) -> np.ndarray:
        """Per-mechanism matching weights ``log((1-p)/p)``, shape ``(num_errors,)``."""
        return np.array([e.weight for e in self.errors], dtype=np.float64)

    def check_to_qubits(self) -> List[List[int]]:
        """``check_to_qubits`` (per detector, the mechanism indices flipping it)."""
        c2q: List[List[int]] = [[] for _ in range(self.num_detectors)]
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

    def make_decoder(self, kind: str = "sparse_blossom"):
        """Construct a QECTOR decoder over this model's detector graph.

        ``kind`` is one of ``"union_find"``, ``"fast_union_find"``, ``"blossom"``,
        ``"sparse_blossom"``, ``"bp_osd"``.
        """
        from . import (
            BlossomDecoder,
            BPOSDDecoder,
            FastUnionFindDecoder,
            SparseBlossomDecoder,
            UnionFindDecoder,
        )

        c2q = self.check_to_qubits()
        nq = self.num_errors
        kind = kind.lower()
        if kind in ("union_find", "uf", "unionfind"):
            return UnionFindDecoder(c2q, nq)
        if kind in ("fast_union_find", "fast_uf", "fastunionfind"):
            return FastUnionFindDecoder(c2q, nq)
        if kind in ("blossom", "mwpm"):
            return BlossomDecoder(c2q, nq)
        if kind in ("sparse_blossom", "sparse"):
            return SparseBlossomDecoder(c2q, nq)
        if kind in ("bp_osd", "bposd", "bp"):
            mean_p = float(self.priors().mean()) if self.num_errors else 0.05
            return BPOSDDecoder(c2q, nq, max(mean_p, 1e-3))
        raise ValueError(f"unknown decoder kind: {kind!r}")

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
    errors: List[DemError] = []
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


def _tokenize(text: str) -> List[str]:
    """Split DEM text into instructions, with ``{`` and ``}`` as standalone tokens."""
    out: List[str] = []
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
    tokens: List[str],
    start: int,
    end: int,
    state: dict,
    errors: List[DemError],
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


def _matching_brace(tokens: List[str], open_idx: int, end: int) -> int:
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


def _exec_instruction(tok: str, state: dict, errors: List[DemError], coords: dict) -> None:
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


def _parse_targets(text: str, det_offset: int) -> Tuple[List[int], List[int]]:
    dets: List[int] = []
    obs: List[int] = []
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
    for an exact column-for-column correspondence with Stim.
    """
    if isinstance(dem, str):
        return parse_dem(dem)
    if not hasattr(dem, "num_detectors"):
        raise TypeError(f"expected a stim.DetectorErrorModel (or DEM text), got {type(dem).__name__}")
    try:
        flat = dem.flattened()
    except Exception:  # pragma: no cover - older Stim
        flat = dem
    model = parse_dem(str(flat))
    # Trust Stim's declared counts when available.
    model.num_detectors = max(model.num_detectors, int(dem.num_detectors))
    model.num_observables = max(model.num_observables, int(dem.num_observables))
    return model
