"""Python-to-Rust license bridge: the pure-Python fallback paths.

These cover `set_license_key` / `get_license_info` in `qector_decoder_v3.__init__`
for the case where the compiled Rust core is unavailable.

Two things this file has to get right, and previously did not:

* **The fallback must be forced.** `get_license_info()` prefers the native
  `py_get_license_info`, which reads the Rust `LicenseManager`. That manager only
  accepts a signed `QECT-{TIER}-{payload}.{signature}` token, so on a machine
  with the core built these tests saw Community for every key and the
  `QECT-PRO-` / `QECT-ENT-` prefix branch was never reached. Each test pins
  `_native_module` to `None`.
* **`max_distance` is an int.** The native core returns an int; the fallback
  used to return a string, so the assertions here encoded the inconsistency
  instead of catching it.

Environment mutation goes through `monkeypatch` so no test leaks
`QECTOR_LICENSE_KEY` into the rest of the session.
"""

from __future__ import annotations

import os

import pytest
import qector_decoder_v3 as qd

TIER_CAPS = {"Community": 7, "Pro": 19, "Enterprise": 63}


@pytest.fixture
def no_native(monkeypatch):
    """Force the pure-Python fallback these tests are written against."""
    monkeypatch.setattr(qd, "_native_module", None)
    monkeypatch.delenv("QECTOR_LICENSE_KEY", raising=False)


def test_set_license_key_updates_env(no_native, monkeypatch):
    """A key the verifier accepts is installed in the environment.

    Runs against the pure-Python fallback (`no_native`) because
    ``QECT-PRO-test123`` is an unsigned placeholder: with the compiled core
    present it is correctly rejected, which is what
    :func:`test_set_license_key_rejects_invalid_key` covers.
    """
    monkeypatch.delenv("QECTOR_LICENSE_KEY", raising=False)
    qd.set_license_key("QECT-PRO-test123")
    assert os.environ.get("QECTOR_LICENSE_KEY") == "QECT-PRO-test123"


def test_set_license_key_rejects_invalid_key(monkeypatch):
    """An unverifiable key raises instead of being silently accepted.

    This used to swallow the native verifier's rejection and set
    ``QECTOR_LICENSE_KEY`` anyway, so a typo'd, expired, or revoked key looked
    like it had been applied and only showed up later as an unexplained
    Community-tier distance cap.
    """
    if getattr(qd, "_native_module", None) is None:
        pytest.skip("needs the compiled core's Ed25519 verifier")
    monkeypatch.delenv("QECTOR_LICENSE_KEY", raising=False)
    with pytest.raises(ValueError, match="rejected"):
        qd.set_license_key("QECT-PRO-not-a-real-signed-token")
    assert os.environ.get("QECTOR_LICENSE_KEY") is None, (
        "a rejected key must not be left in the environment"
    )


def test_get_license_info_returns_dict(no_native, monkeypatch):
    monkeypatch.setenv("QECTOR_LICENSE_KEY", "QECT-PRO-test456")
    info = qd.get_license_info()
    assert isinstance(info, dict)
    for field in ("tier", "max_distance", "customer_id"):
        assert field in info, f"missing field {field!r} in {info}"


def test_get_license_info_community_default(no_native):
    info = qd.get_license_info()
    assert info["tier"] == "Community"
    assert info["max_distance"] == TIER_CAPS["Community"]


def test_get_license_info_pro(no_native, monkeypatch):
    monkeypatch.setenv("QECTOR_LICENSE_KEY", "QECT-PRO-abc123")
    info = qd.get_license_info()
    assert info["tier"] == "Pro"
    assert info["max_distance"] == TIER_CAPS["Pro"]


def test_get_license_info_enterprise(no_native, monkeypatch):
    monkeypatch.setenv("QECTOR_LICENSE_KEY", "QECT-ENT-xyz789")
    info = qd.get_license_info()
    assert info["tier"] == "Enterprise"
    assert info["max_distance"] == TIER_CAPS["Enterprise"]


@pytest.mark.parametrize("tier", ["Community", "Pro", "Enterprise"])
def test_max_distance_is_always_an_int(no_native, monkeypatch, tier):
    """The fallback must not disagree with the native core on type."""
    prefix = {"Community": "", "Pro": "QECT-PRO-x", "Enterprise": "QECT-ENT-x"}[tier]
    if prefix:
        monkeypatch.setenv("QECTOR_LICENSE_KEY", prefix)
    info = qd.get_license_info()
    assert isinstance(info["max_distance"], int), (
        f"{tier}: max_distance is {type(info['max_distance']).__name__}, expected int"
    )


def test_native_path_also_reports_int_max_distance():
    """Whatever source answers, the contract is the same."""
    info = qd.get_license_info()
    assert isinstance(info["max_distance"], int)
