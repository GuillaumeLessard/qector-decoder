"""Workbench: environment snapshot is complete and reproducible."""
from qector_decoder_v3.workbench import Workbench


def test_environment_snapshot_has_provenance_fields():
    wb = Workbench()
    env = wb.environment_snapshot()
    for key in ("python_version", "platform", "qector_decoder_v3_version",
                "git_commit", "cuda_available", "opencl_available", "backends"):
        assert key in env, key
    assert isinstance(env["backends"], dict)
    assert env["backends"]["cpu"] is True


def test_environment_snapshot_git_commit_is_string():
    wb = Workbench()
    env = wb.environment_snapshot()
    # either a real 40-char hash or the explicit "unknown" sentinel
    gc = env["git_commit"]
    assert isinstance(gc, str) and (gc == "unknown" or len(gc) >= 7)


def test_environment_snapshot_embedded_in_artifact():
    wb = Workbench()
    art = wb.run_benchmark({"code": "repetition", "distances": [5],
                            "decoders": ["blossom"], "trials": 100})
    assert "environment" in art
    assert art["environment"]["git_commit"] == wb.environment_snapshot()["git_commit"]
