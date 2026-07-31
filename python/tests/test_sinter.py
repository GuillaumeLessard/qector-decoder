"""Tests for qector_decoder_v3.sinter_compat - Sinter integration.

Runs a small real ``sinter.collect`` with QECTOR decoders. The key assertions:
the decoders are accepted by Sinter, bit-packing is correct (sane LERs, not ~0.5),
and belief-matching is competitive with PyMatching through the standard harness.
"""

import pytest

stim = pytest.importorskip("stim")
sinter = pytest.importorskip("sinter")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3.sinter_compat import qector_sinter_decoders


def _tasks():
    tasks = []
    for d in (3, 5):
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=d,
            rounds=d,
            after_clifford_depolarization=0.005,
            before_measure_flip_probability=0.005,
            after_reset_flip_probability=0.005,
        )
        tasks.append(sinter.Task(circuit=circ, json_metadata={"d": d}))
    return tasks


@pytest.mark.timeout(600)
def test_sinter_collect_runs_and_is_sane():
    """
    Runtime: ~160 s (A10-02 landed: cached BlossomDecoder + set_edge_weights
    removed the 627x per-shot decoder-construction overhead). The remaining
    cost is per-shot BP in Python (~190 shots/s for `qector_belief`).

        qector_blossom   57 021 shots/s
        qector_belief       190 shots/s   <- BP-bound, not decoder-construction-bound

    With 8 (task, decoder) combinations at up to 4000 shots each, the test
    completes in minutes. The explicit 600 s budget keeps the suite-wide
    `--timeout=300` tight for every other test.
    """
    results = sinter.collect(
        num_workers=2,
        tasks=_tasks(),
        decoders=["qector_blossom", "qector_belief", "qector_bposd", "pymatching"],
        custom_decoders=qector_sinter_decoders(),
        max_shots=4000,
        max_errors=4000,
    )
    assert len(results) == 8  # 2 distances x 4 decoders
    by = {(r.json_metadata["d"], r.decoder): r for r in results}
    for (d, dec), r in by.items():
        ler = r.errors / r.shots if r.shots else 1.0
        # bit-packing correct => LER far below random (0.5); below ~5% at p=0.005
        assert ler < 0.05, f"{dec} d={d} LER={ler} (bad bit-packing?)"

    # belief-matching competitive with PyMatching at d=5 through Sinter
    pm5 = by[(5, "pymatching")]
    bm5 = by[(5, "qector_belief")]
    pm_ler = pm5.errors / pm5.shots
    bm_ler = bm5.errors / bm5.shots
    assert bm_ler <= pm_ler * 1.5 + 0.002, (bm_ler, pm_ler)
