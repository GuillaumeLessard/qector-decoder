"""Workbench: job cancellation (queued and running)."""

import time

from qector_decoder_v3.workbench import Workbench


def test_cancel_queued_job_is_instant():
    """A job still in the queue cancels instantly and never runs."""
    wb = Workbench()
    # Keep the worker busy with a real first job.
    busy = wb.submit_job(
        {
            "code": "rotated_surface",
            "distances": [3, 5, 7],
            "decoders": ["blossom", "sparse_blossom"],
            "trials": 400,
        }
    )
    queued = wb.submit_job(
        {
            "code": "rotated_surface",
            "distances": [3],
            "decoders": ["blossom"],
            "trials": 100,
        }
    )
    status = wb.cancel_job(queued)
    assert status == "cancelled"
    assert wb.get_job(queued)["status"] == "cancelled"
    # the busy job still finishes fine
    assert wb.wait(busy, timeout=30)["status"] == "completed"
    assert wb.job_artifact(queued) is None
    wb.shutdown()


def test_cancel_running_job_stops_early():
    """A running job stops at the next unit boundary, leaving partial progress."""
    wb = Workbench()
    jid = wb.submit_job(
        {
            "code": "rotated_surface",
            "distances": [3, 5, 7, 9],
            "decoders": ["blossom", "sparse_blossom"],
            "trials": 300,
            "throttle_seconds": 0.3,
        }
    )
    # wait until it is actually running
    for _ in range(500):
        if wb.get_job(jid)["status"] == "running":
            break
        time.sleep(0.01)
    time.sleep(0.2)
    wb.cancel_job(jid)
    final = wb.wait(jid, timeout=30)
    assert final["status"] == "cancelled"
    assert final["units_done"] < final["units_total"]
    wb.shutdown()
