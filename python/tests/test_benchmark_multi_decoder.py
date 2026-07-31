"""Tests for multi-decoder benchmarking via benchmark_decoder."""

import numpy as np
import pytest
from qector_decoder_v3 import codes
from qector_decoder_v3.benchmarking import benchmark_decoder


@pytest.fixture
def surf5():
    return codes.rotated_surface_code(5)


ALL_KINDS = [
    "union_find",
    "fast_union_find",
    "blossom",
    "sparse_blossom",
    "cpu_batch",
    "bp_osd",
    "lookup_table",
    "batch",
    "cascade",
    "auto",
]


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_benchmark_all_decoders_run(kind, surf5):
    report = benchmark_decoder(kind, surf5, n_trials=10, warmup=2, measure_memory=False)
    assert "decoder" in report
    assert "latency_us" in report
    assert report["decoder"] == kind


def test_benchmark_with_decoder_type_param(surf5):
    report = benchmark_decoder("blossom", surf5, n_trials=10, warmup=2, measure_memory=False, decoder_type="custom")
    assert report["decoder_type"] == "custom"
