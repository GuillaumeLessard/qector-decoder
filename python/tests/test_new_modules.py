"""Tests for new QECTOR v3 modules: FastUF, Blossom, SlidingWindow, ecosystem."""

import numpy as np
import pytest

import qector_decoder_v3 as qd


class TestFastUnionFindDecoder:
    """Tests for the SIMD-accelerated zero-allocation Fast decoder."""

    def test_decode_matches_uf(self):
        checks, n_qubits = qd.generate_ring_code_checks(5)
        fast_dec = qd.FastUnionFindDecoder(checks, n_qubits)
        uf_dec = qd.UnionFindDecoder(checks, n_qubits)
        for _ in range(50):
            syndrome = np.random.randint(0, 2, size=(len(checks),), dtype=np.uint8)
            fast_corr = fast_dec.decode(syndrome)
            uf_corr = uf_dec.decode(syndrome)
            assert np.array_equal(fast_corr, uf_corr)

    def test_batch_decode_matches_uf(self):
        checks, n_qubits = qd.generate_ring_code_checks(5)
        fast_dec = qd.FastUnionFindDecoder(checks, n_qubits)
        uf_dec = qd.UnionFindDecoder(checks, n_qubits)
        syndromes = np.random.randint(0, 2, size=(20, len(checks)), dtype=np.uint8)
        fast_corr = fast_dec.batch_decode(syndromes)
        uf_corr = uf_dec.batch_decode(syndromes)
        assert np.array_equal(fast_corr, uf_corr)

    def test_high_weight_check(self):
        # A weight-4 check with syndrome [1] must get an ODD-parity correction so
        # that H @ correction == syndrome. The previous assertion (`corr == 1`)
        # encoded an over-flip bug: flipping all 4 qubits yields even parity (0),
        # which does NOT reproduce syndrome [1]. The corrected decoder is
        # syndrome-faithful.
        checks = [[0, 1, 2, 3]]
        n_qubits = 4
        dec = qd.FastUnionFindDecoder(checks, n_qubits)
        syndrome = np.array([1], dtype=np.uint8)
        corr = dec.decode(syndrome)
        parity = int(np.bitwise_xor.reduce(corr)) if corr.size else 0
        assert parity == 1, f"correction must reproduce syndrome [1], got {corr}"

    def test_empty_syndrome(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        dec = qd.FastUnionFindDecoder(checks, n_qubits)
        syndrome = np.zeros(len(checks), dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert np.all(corr == 0)

    def test_invalid_shape(self):
        checks, n_qubits = qd.generate_ring_code_checks(5)
        dec = qd.FastUnionFindDecoder(checks, n_qubits)
        with pytest.raises(Exception):
            dec.decode(np.zeros(5, dtype=np.uint8))

    def test_surface_code(self):
        checks, n_qubits = qd.generate_surface_code_checks(5)
        dec = qd.FastUnionFindDecoder(checks, n_qubits)
        syndrome = np.random.randint(0, 2, size=(len(checks),), dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert corr.shape == (n_qubits,)
        assert corr.dtype == np.uint8


class TestBlossomDecoder:
    """Tests for the MWPM Blossom decoder."""

    def test_empty_syndrome(self):
        checks = [[0, 1], [1, 2], [2, 3]]
        dec = qd.BlossomDecoder(checks, 4)
        syndrome = np.zeros(len(checks), dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert np.all(corr == 0)

    def test_single_pair(self):
        checks = [[0, 1], [1, 2], [2, 3]]
        dec = qd.BlossomDecoder(checks, 4)
        syndrome = np.array([1, 0, 1], dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert corr.shape == (4,)

    def test_weighted_decode(self):
        checks = [[0, 1], [1, 2], [2, 3]]
        n_qubits = 4
        weights = [1.0, 10.0, 1.0]
        dec = qd.BlossomDecoder(checks, n_qubits, weights)
        syndrome = np.array([1, 0, 1], dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert corr.shape == (n_qubits,)

    def test_surface_code_blossom(self):
        checks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        dec = qd.BlossomDecoder(checks, 8)
        syndrome = np.array([1, 0], dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert corr.shape == (8,)

    def test_batch_decode_shape(self):
        checks = [[0, 1], [1, 2], [2, 3]]
        dec = qd.BlossomDecoder(checks, 4)
        syndromes = np.random.randint(0, 2, size=(5, len(checks)), dtype=np.uint8)
        corr = dec.batch_decode(syndromes)
        assert corr.shape == (5, 4)

    def test_edges_property(self):
        checks = [[0, 1], [1, 2]]
        dec = qd.BlossomDecoder(checks, 3)
        edges = dec.edges
        assert isinstance(edges, list)

    def test_invalid_shape(self):
        checks = [[0, 1], [1, 2], [2, 3]]
        dec = qd.BlossomDecoder(checks, 4)
        with pytest.raises(Exception):
            dec.decode(np.zeros(5, dtype=np.uint8))


class TestSlidingWindowDecoder:
    """Tests for the sliding-window decoder with exponential decay."""

    def test_empty_window(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        dec = qd.SlidingWindowDecoder(checks, n_qubits, window_size=5, decay_factor=0.8)
        syndrome = np.array([1, 0, 0, 0], dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert corr.shape == (n_qubits,)

    def test_multi_round_decay(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        dec = qd.SlidingWindowDecoder(checks, n_qubits, window_size=3, decay_factor=0.5)
        r1 = np.array([1, 0, 0, 0], dtype=np.uint8)
        r2 = np.array([0, 1, 0, 0], dtype=np.uint8)
        c1 = dec.update(r1)
        c2 = dec.update(r2)
        assert c2.shape == (n_qubits,)

    def test_flush_resets(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        dec = qd.SlidingWindowDecoder(checks, n_qubits, window_size=3, decay_factor=0.8)
        dec.update(np.array([1, 0, 0, 0], dtype=np.uint8))
        dec.flush()
        c = dec.update(np.zeros(len(checks), dtype=np.uint8))
        assert np.all(c == 0)

    def test_window_size_property(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        dec = qd.SlidingWindowDecoder(checks, n_qubits, window_size=5, decay_factor=0.8)
        assert dec.window_size == 5
        assert dec.decay_factor == 0.8
        assert dec.current_round == 0
        dec.update(np.zeros(len(checks), dtype=np.uint8))
        assert dec.current_round == 1

    def test_invalid_decay_factor(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        with pytest.raises(Exception):
            qd.SlidingWindowDecoder(checks, n_qubits, window_size=3, decay_factor=1.5)

    def test_invalid_window_size(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        with pytest.raises(Exception):
            qd.SlidingWindowDecoder(checks, n_qubits, window_size=0, decay_factor=0.8)

    def test_surface_code_window(self):
        checks, n_qubits = qd.generate_surface_code_checks(5)
        dec = qd.SlidingWindowDecoder(checks, n_qubits, window_size=5, decay_factor=0.9)
        syndrome = np.random.randint(0, 2, size=(len(checks),), dtype=np.uint8)
        corr = dec.update(syndrome)
        assert corr.shape == (n_qubits,)


class TestQiskitPlugin:
    """Tests for the optional Qiskit integration."""

    def test_dict_raw_mode(self):
        raw = {"counts": {"0x0": 400, "0x3": 100}}
        out = qd.qiskit_plugin.decode_qiskit_result(raw, code_distance=3)
        assert "correction" in out
        assert "syndrome" in out
        assert "metadata" in out
        assert out["metadata"]["code_distance"] == 3

    def test_empty_counts(self):
        raw = {"counts": {}}
        out = qd.qiskit_plugin.decode_qiskit_result(raw, code_distance=3)
        assert out["correction"].shape[0] == 0

    def test_create_decoder_factory(self):
        decoder = qd.qiskit_plugin.create_qiskit_decoder(code_distance=3)
        assert hasattr(decoder, "_inner_decoder")
        raw = {"counts": {"0x0": 10}}
        out = decoder(raw)
        assert "correction" in out

    def test_hex_to_syndrome(self):
        raw = {"counts": {"0x3": 1}}
        out = qd.qiskit_plugin.decode_qiskit_result(raw, code_distance=3)
        assert out["metadata"]["shots"] == 1


class TestStimCompat:
    """Tests for the Stim compatibility layer."""

    def test_to_stim_decoder(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        decoder = qd.stim_compat.to_stim_decoder(checks, n_qubits)
        syndrome = np.array([1, 0, 1, 0], dtype=np.uint8)
        corr = decoder.decode(syndrome)
        assert corr.shape == (n_qubits,)

    def test_stim_decoder_with_batch(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        decoder = qd.stim_compat.to_stim_decoder(checks, n_qubits, use_batch=True)
        syndrome = np.array([1, 0, 1, 0], dtype=np.uint8)
        corr = decoder.decode(syndrome)
        assert corr.shape == (n_qubits,)

    def test_stim_decoder_repr(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        decoder = qd.stim_compat.to_stim_decoder(checks, n_qubits)
        assert "QECTORStimDecoder" in repr(decoder)

    def test_from_stim_dem_raises(self):
        # If stim is installed, passing None raises TypeError.
        # If stim is not installed, it raises ImportError.
        with pytest.raises((ImportError, TypeError)):
            qd.stim_compat.from_stim_detector_error_model(None)

    def test_stim_decoder_from_dem_raises(self):
        with pytest.raises((ImportError, TypeError)):
            qd.stim_compat.stim_decoder_from_dem(None)


class TestRestApi:
    """Tests for the REST API (FastAPI/Flask)."""

    def test_create_app(self):
        app = qd.rest_api.create_app()
        assert app is not None

    def test_fastapi_routes(self):
        # Only valid if FastAPI is installed
        if qd.rest_api._FRAMEWORK != "fastapi":
            pytest.skip("FastAPI not installed")
        from fastapi.testclient import TestClient
        app = qd.rest_api.create_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        resp = client.get("/version")
        assert resp.status_code == 200
        assert "version" in resp.json()

        payload = {
            "check_to_qubits": [[0, 1], [1, 2]],
            "syndrome": [1, 0],
        }
        resp = client.post("/decode", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "correction" in data
        assert data["n_qubits"] == 3

    def test_rest_api_routes(self):
        """Test the active REST API framework routes (FastAPI or Flask)."""
        app = qd.rest_api.create_app()
        if qd.rest_api._FRAMEWORK == "fastapi":
            from fastapi.testclient import TestClient
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

            resp = client.get("/version")
            assert resp.status_code == 200
            assert "version" in resp.json()

            resp = client.post("/decode", json={
                "check_to_qubits": [[0, 1], [1, 2]],
                "syndrome": [1, 0],
            })
            assert resp.status_code == 200
            assert "correction" in resp.json()
        elif qd.rest_api._FRAMEWORK == "flask":
            with app.test_client() as client:
                resp = client.get("/health")
                assert resp.status_code == 200
                assert resp.json["status"] == "ok"

                resp = client.get("/version")
                assert resp.status_code == 200
                assert "version" in resp.json

                resp = client.post("/decode", json={
                    "check_to_qubits": [[0, 1], [1, 2]],
                    "syndrome": [1, 0],
                })
                assert resp.status_code == 200
                assert "correction" in resp.json
        else:
            pytest.skip(
                f"REST API framework is '{qd.rest_api._FRAMEWORK}', expected fastapi or flask"
            )

    def test_decode_error_empty_checks(self):
        if qd.rest_api._FRAMEWORK == "fastapi":
            from fastapi.testclient import TestClient
            app = qd.rest_api.create_app()
            client = TestClient(app)
            resp = client.post("/decode", json={"check_to_qubits": [], "syndrome": []})
            assert resp.status_code == 400
        else:
            app = qd.rest_api.create_app()
            with app.test_client() as client:
                resp = client.post("/decode", json={"check_to_qubits": [], "syndrome": []})
                assert resp.status_code == 400

    def test_app_global_instance(self):
        assert qd.rest_api.app is not None
