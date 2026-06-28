"""
Regression tests for the 7 issues listed in the Kimi AI audit sheet
(sheet_20260628_182933_generated_by_Kimi_AI.csv).

All 7 findings were reported against v0.5.4 and fixed in v0.5.5.
These tests ensure they can never silently regress.

Issue 1  from_stim_detector_error_model() returns empty checks
Issue 2  stim_decoder_from_dem() returns [] corrections
Issue 3  PyMatching parity 100% mismatch on DEM
Issue 4  Sinter PicklingError
Issue 5  BpOsdDecoder constructor TypeError
Issue 6  decode_with_diagnostics AttributeError
Issue 7  Blossom sub-optimal on repetition code
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import qector_decoder_v3 as qd

# ---------------------------------------------------------------------------
# Optional-dependency markers
# ---------------------------------------------------------------------------
stim = pytest.importorskip("stim", reason="stim not installed")
pymatching = pytest.importorskip("pymatching", reason="pymatching not installed")

try:
    import sinter as _sinter  # noqa: F401

    HAS_SINTER = True
except ImportError:
    HAS_SINTER = False

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
CIRCUIT = stim.Circuit.generated(
    "repetition_code:memory",
    distance=5,
    rounds=3,
    after_clifford_depolarization=0.01,
)
DEM = CIRCUIT.detector_error_model(decompose_errors=True)
_sampler = CIRCUIT.compile_detector_sampler()
SHOTS, _ = _sampler.sample(shots=100, separate_observables=True)


def _build_H(c2q: list, nq: int) -> np.ndarray:
    H = np.zeros((len(c2q), nq), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            if q < nq:
                H[ci, q] = 1
    return H


# ---------------------------------------------------------------------------
# Issue 1 — from_stim_detector_error_model() must not return empty checks
# ---------------------------------------------------------------------------
class TestIssue1StimDemNotEmpty:
    def test_non_empty_check_list(self):
        from qector_decoder_v3.stim_compat import from_stim_detector_error_model

        c2q, nq = from_stim_detector_error_model(DEM)
        assert len(c2q) > 0, "check_to_qubits must not be empty"
        assert nq > 0, "n_qubits must be > 0"

    def test_all_rows_are_lists(self):
        from qector_decoder_v3.stim_compat import from_stim_detector_error_model

        c2q, _ = from_stim_detector_error_model(DEM)
        assert all(isinstance(row, list) for row in c2q)

    def test_no_empty_rows(self):
        from qector_decoder_v3.stim_compat import from_stim_detector_error_model

        c2q, _ = from_stim_detector_error_model(DEM)
        empty = [i for i, row in enumerate(c2q) if len(row) == 0]
        assert not empty, f"rows {empty} are empty"

    def test_accepts_stim_circuit_directly(self):
        """stim_circuit_to_check_matrix must handle a stim.Circuit, not just a DEM."""
        from qector_decoder_v3.stim_compat import stim_circuit_to_check_matrix

        c2q, nq = stim_circuit_to_check_matrix(CIRCUIT)
        assert len(c2q) > 0
        assert nq > 0

    def test_dem_text_string_accepted(self):
        from qector_decoder_v3.stim_compat import from_stim_detector_error_model

        dem_text = str(DEM.flattened())
        c2q, nq = from_stim_detector_error_model(dem_text)
        assert len(c2q) > 0
        assert nq > 0


# ---------------------------------------------------------------------------
# Issue 2 — stim_decoder_from_dem() corrections must be non-empty + faithful
# ---------------------------------------------------------------------------
class TestIssue2StimDecoderCorrections:
    def test_corrections_non_empty(self):
        from qector_decoder_v3.stim_compat import stim_decoder_from_dem

        dec = stim_decoder_from_dem(DEM)
        for i in range(10):
            s = SHOTS[i].astype(np.uint8)
            c = np.asarray(dec.decode(s), dtype=np.uint8).reshape(-1)
            assert len(c) > 0, f"shot {i}: empty correction"

    def test_corrections_syndrome_faithful(self):
        from qector_decoder_v3.stim_compat import (
            from_stim_detector_error_model,
            stim_decoder_from_dem,
        )

        dec = stim_decoder_from_dem(DEM)
        c2q, nq = from_stim_detector_error_model(DEM)
        H = _build_H(c2q, nq)
        wrong = 0
        for i in range(100):
            s = SHOTS[i].astype(np.uint8)
            c = np.asarray(dec.decode(s), dtype=np.uint8).reshape(-1)
            if not np.array_equal((H @ c) & 1, s):
                wrong += 1
        assert wrong == 0, f"{wrong}/100 corrections violate parity"

    def test_correction_length_equals_n_qubits(self):
        from qector_decoder_v3.stim_compat import (
            from_stim_detector_error_model,
            stim_decoder_from_dem,
        )

        dec = stim_decoder_from_dem(DEM)
        _, nq = from_stim_detector_error_model(DEM)
        s = SHOTS[0].astype(np.uint8)
        c = np.asarray(dec.decode(s), dtype=np.uint8).reshape(-1)
        assert len(c) == nq


# ---------------------------------------------------------------------------
# Issue 3 — pymatching_compat.Matching must agree with pymatching on DEM
# ---------------------------------------------------------------------------
class TestIssue3PyMatchingParity:
    def test_matching_class_exists(self):
        from qector_decoder_v3.pymatching_compat import Matching  # noqa: F401

    def test_from_detector_error_model_callable(self):
        from qector_decoder_v3.pymatching_compat import Matching

        m = Matching.from_detector_error_model(DEM)
        assert m is not None

    def test_parity_with_reference_pymatching(self):
        from qector_decoder_v3.pymatching_compat import Matching

        pm_ref = pymatching.Matching.from_detector_error_model(DEM)
        qd_m = Matching.from_detector_error_model(DEM)
        mismatch = 0
        for i in range(100):
            s = SHOTS[i].astype(np.uint8)
            if not np.array_equal(pm_ref.decode(s), qd_m.decode(s)):
                mismatch += 1
        assert mismatch <= 5, f"{mismatch}/100 shots mismatch pymatching (threshold 5)"

    def test_decode_batch_parity(self):
        """decode_batch must also agree with the per-shot path."""
        from qector_decoder_v3.pymatching_compat import Matching

        m = Matching.from_detector_error_model(DEM)
        batch = SHOTS[:20].astype(np.uint8)
        batch_out = m.decode_batch(batch)
        for i in range(20):
            single = m.decode(SHOTS[i].astype(np.uint8))
            assert np.array_equal(single, batch_out[i]), f"shot {i} mismatch"


# ---------------------------------------------------------------------------
# Issue 4 — Sinter decoders must be picklable (no PicklingError)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_SINTER, reason="sinter not installed")
class TestIssue4SinterPickling:
    def test_all_decoders_picklable(self):
        from qector_decoder_v3.sinter_compat import qector_sinter_decoders

        decoders = qector_sinter_decoders()
        assert len(decoders) >= 3
        for name, obj in decoders.items():
            restored = pickle.loads(pickle.dumps(obj))
            assert restored.kind == obj.kind, f"{name}: kind changed after pickle"

    def test_compiled_decoder_works(self):
        from qector_decoder_v3.sinter_compat import QectorSinterDecoder

        for kind in ("blossom", "belief", "unionfind"):
            dec = QectorSinterDecoder(kind)
            compiled = dec.compile_decoder_for_dem(dem=DEM)
            assert compiled.num_detectors == DEM.num_detectors
            assert compiled.num_observables == DEM.num_observables

    def test_decode_shots_bit_packed_shape(self):
        from qector_decoder_v3.sinter_compat import QectorSinterDecoder

        dec = QectorSinterDecoder("blossom")
        compiled = dec.compile_decoder_for_dem(dem=DEM)
        n_shots = 8
        n_det = compiled.num_detectors
        # pack random bit arrays
        raw = np.random.default_rng(0).integers(
            0, 2, size=(n_shots, n_det), dtype=np.uint8
        )
        packed = np.packbits(raw, axis=1, bitorder="little")
        out = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
        assert out.ndim == 2
        assert out.shape[0] == n_shots


# ---------------------------------------------------------------------------
# Issue 5 — BpOsdDecoder constructor must accept H matrix (not check_to_qubits)
# ---------------------------------------------------------------------------
class TestIssue5BpOsdConstructor:
    @pytest.fixture(scope="class")
    @classmethod
    def H_and_shape(cls):
        checks, nq = qd.generate_repetition_code_checks(7)
        H = _build_H(checks, nq)
        return H, nq

    def test_positional_call(self, H_and_shape):
        from qector_decoder_v3.bposd import BpOsdDecoder

        H, nq = H_and_shape
        dec = BpOsdDecoder(H, 0.05)
        s = np.zeros(H.shape[0], dtype=np.uint8)
        c = dec.decode(s)
        assert c.shape == (nq,)

    def test_keyword_call(self, H_and_shape):
        from qector_decoder_v3.bposd import BpOsdDecoder

        H, nq = H_and_shape
        dec = BpOsdDecoder(H=H, error_rate=0.01)
        s = np.zeros(H.shape[0], dtype=np.uint8)
        c = dec.decode(s)
        assert c.shape == (nq,)

    def test_syndrome_faithful(self, H_and_shape):
        from qector_decoder_v3.bposd import BpOsdDecoder

        H, _ = H_and_shape
        dec = BpOsdDecoder(H, 0.05)
        rng = np.random.default_rng(0)
        for _ in range(30):
            e = (rng.random(H.shape[1]) < 0.05).astype(np.uint8)
            s = (H @ e) & 1
            c = np.asarray(dec.decode(s.astype(np.uint8)), dtype=np.uint8)
            assert np.array_equal((H @ c) & 1, s), "BpOsdDecoder parity broken"

    def test_rejects_1d_input(self):
        from qector_decoder_v3.bposd import BpOsdDecoder

        with pytest.raises((ValueError, TypeError)):
            # 1-D array is not a valid 2-D H matrix
            BpOsdDecoder(np.array([0, 1, 1, 0], dtype=np.uint8), 0.05)

    def test_batch_decode(self, H_and_shape):
        from qector_decoder_v3.bposd import BpOsdDecoder

        H, nq = H_and_shape
        dec = BpOsdDecoder(H, 0.05)
        batch = np.zeros((5, H.shape[0]), dtype=np.uint8)
        out = dec.batch_decode(batch)
        assert out.shape == (5, nq)


# ---------------------------------------------------------------------------
# Issue 6 — decode_with_diagnostics must accept a Code object (not raw decoder)
# ---------------------------------------------------------------------------
class TestIssue6DecodeWithDiagnostics:
    def test_blossom_kind(self):
        from qector_decoder_v3 import codes
        from qector_decoder_v3.result import decode_with_diagnostics

        code = codes.repetition_code(7)
        s = np.zeros(code.n_checks, dtype=np.uint8)
        r = decode_with_diagnostics(code, s, kind="blossom")
        assert r.syndrome_valid is True
        assert r.backend == "blossom"

    def test_union_find_kind(self):
        from qector_decoder_v3 import codes
        from qector_decoder_v3.result import decode_with_diagnostics

        code = codes.repetition_code(7)
        s = np.zeros(code.n_checks, dtype=np.uint8)
        r = decode_with_diagnostics(code, s, kind="union_find")
        assert r.syndrome_valid is True
        assert r.backend == "union_find"

    def test_prebuilt_decoder_accepted(self):
        from qector_decoder_v3 import codes
        from qector_decoder_v3.result import decode_with_diagnostics

        code = codes.repetition_code(7)
        pre = qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)
        s = np.zeros(code.n_checks, dtype=np.uint8)
        r = decode_with_diagnostics(code, s, decoder=pre)
        assert r.syndrome_valid is True

    def test_non_trivial_syndrome_faithful(self):
        from qector_decoder_v3 import codes
        from qector_decoder_v3.result import decode_with_diagnostics

        code = codes.repetition_code(9)
        rng = np.random.default_rng(42)
        for kind in ("blossom", "union_find", "sparse_blossom"):
            for _ in range(10):
                e = (rng.random(code.n_qubits) < 0.05).astype(np.uint8)
                s = code.syndrome(e)
                r = decode_with_diagnostics(code, s, kind=kind)
                assert r.syndrome_valid, f"{kind}: syndrome not valid"

    def test_result_fields_populated(self):
        from qector_decoder_v3 import codes
        from qector_decoder_v3.result import decode_with_diagnostics

        code = codes.repetition_code(7)
        s = np.zeros(code.n_checks, dtype=np.uint8)
        r = decode_with_diagnostics(code, s, kind="blossom")
        assert r.correction.shape == (code.n_qubits,)
        assert r.n_qubits == code.n_qubits
        assert r.n_checks == code.n_checks
        assert r.decode_seconds is not None and r.decode_seconds >= 0
        assert r.weight is not None


# ---------------------------------------------------------------------------
# Issue 7 — BlossomDecoder must be weight-optimal vs UnionFindDecoder
# ---------------------------------------------------------------------------
class TestIssue7BlossomOptimality:
    @pytest.fixture(scope="class")
    @classmethod
    def decoders_and_H(cls):
        checks, nq = qd.generate_repetition_code_checks(9)
        H = _build_H(checks, nq)
        return (
            qd.BlossomDecoder(checks, nq),
            qd.UnionFindDecoder(checks, nq),
            H,
            nq,
        )

    def test_both_syndrome_faithful(self, decoders_and_H):
        bl, uf, H, nq = decoders_and_H
        rng = np.random.default_rng(42)
        for _ in range(100):
            e = (rng.random(nq) < 0.05).astype(np.uint8)
            s = (H @ e) & 1
            s8 = s.astype(np.uint8)
            assert np.array_equal((H @ np.asarray(bl.decode(s8), np.uint8)) & 1, s)
            assert np.array_equal((H @ np.asarray(uf.decode(s8), np.uint8)) & 1, s)

    def test_blossom_weight_leq_unionfind(self, decoders_and_H):
        """Blossom weight must be <= UF weight on >=80% of shots."""
        bl, uf, H, nq = decoders_and_H
        rng = np.random.default_rng(42)
        n_trials = 200
        bl_wins = 0
        for _ in range(n_trials):
            e = (rng.random(nq) < 0.05).astype(np.uint8)
            s = (H @ e) & 1
            s8 = s.astype(np.uint8)
            cb = np.asarray(bl.decode(s8), np.uint8)
            cu = np.asarray(uf.decode(s8), np.uint8)
            if int(cb.sum()) <= int(cu.sum()):
                bl_wins += 1
        pct = bl_wins / n_trials * 100
        assert pct >= 80, f"Blossom only {pct:.1f}% optimal (need >=80%)"

    def test_blossom_weight_never_worse_on_trivial_syndrome(self, decoders_and_H):
        """Zero syndrome -> zero correction for both."""
        bl, uf, H, nq = decoders_and_H
        s = np.zeros(H.shape[0], dtype=np.uint8)
        cb = np.asarray(bl.decode(s), np.uint8)
        cu = np.asarray(uf.decode(s), np.uint8)
        assert int(cb.sum()) == 0
        assert int(cu.sum()) == 0
