"""Section 9 (latency): the syndrome-faithfulness GATE.

``benchmark_decoder`` must report ``syndrome_faithful is True`` for every
CPU decoder kind across two code families. This is the correctness gate that
backs every reported latency/throughput number: if a benchmarked decode were
wrong, the benchmark is meaningless. No wall-clock thresholds.
"""
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import benchmarking as bm, codes

KINDS = ["blossom", "sparse_blossom", "union_find", "fast_union_find", "cpu_batch"]


def _codes():
    return [
        ("rotated_surface_d5", codes.rotated_surface_code(5)),
        ("repetition_d11", codes.repetition_code(11)),
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("code_name,code", _codes())
def test_benchmark_syndrome_faithful_gate(kind, code_name, code):
    result = bm.benchmark_decoder(kind, code, n_trials=500, warmup=60)
    assert result["syndrome_faithful"] is True, (
        f"{kind} on {code_name} produced an unfaithful decode during benchmarking"
    )
    # Sanity: the benchmark actually ran trials, so the gate is meaningful.
    assert result["latency_us"]["n"] > 0
    assert result["decoder"] == kind
