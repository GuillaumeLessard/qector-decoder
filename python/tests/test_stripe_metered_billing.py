"""Tests for the metered-billing functions (record_shots, flush_metered_usage)."""

import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3.stripe_integration import flush_metered_usage


def test_record_decode_shots_increments():
    qd.record_shots(100)
    assert qd.get_accumulated_shots() >= 100


def test_flush_metered_usage_reports_missing_key_clearly():
    """Unconfigured flush must fail loudly and legibly, not obscurely.

    This test used to assert a dict unconditionally and so crashed with
    ``RuntimeError: Stripe API key cannot be empty`` on any machine without
    ``STRIPE_SECRET_KEY`` — which is every CI runner and every developer box
    that has not opted into billing (T7-01 is still blocked on key
    permissions). Both environments are now asserted explicitly.
    """
    import os

    configured = bool(os.environ.get("STRIPE_SECRET_KEY"))
    if configured:
        result = flush_metered_usage(customer_id="cus_test_flush_no_crash")
        assert isinstance(result, dict)
        return
    with pytest.raises(RuntimeError) as exc:
        flush_metered_usage(customer_id="cus_test_flush_no_crash")
    assert "api key" in str(exc.value).lower(), (
        f"error message should name the missing key, got: {exc.value}"
    )


def test_flush_metered_usage_rejects_empty_customer():
    with pytest.raises(ValueError):
        flush_metered_usage(customer_id="")
    with pytest.raises(ValueError):
        flush_metered_usage(customer_id="   ")
