"""Authentication coverage for every REST route that is not a health probe."""

from __future__ import annotations

import pytest
from qector_decoder_v3 import rest_api


@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    """Keep request-rate state from one API client out of another test."""
    with rest_api._rate_lock:
        rest_api._rate_buckets.clear()
    yield
    with rest_api._rate_lock:
        rest_api._rate_buckets.clear()


def _client_for(app):
    if rest_api._FRAMEWORK == "fastapi":
        from fastapi.testclient import TestClient

        return TestClient(app), False
    if rest_api._FRAMEWORK == "flask":
        return app.test_client(), True
    pytest.skip("no supported REST framework is installed")


def _get(client, flask: bool, path: str, **kwargs):
    return client.get(path, **kwargs) if not flask else client.get(path, **kwargs)


def _post(client, flask: bool, path: str, **kwargs):
    return client.post(path, **kwargs) if not flask else client.post(path, **kwargs)


def test_license_activation_requires_bearer_token(monkeypatch):
    """An unauthenticated caller must not be able to mutate global licensing."""
    monkeypatch.setenv("QECTOR_API_KEY", "test-api-key")
    client, flask = _client_for(rest_api.create_app())

    response = _post(client, flask, "/api/license/activate", json={"key": "attacker-key"})

    assert response.status_code == 401


def test_non_health_routes_are_authenticated(monkeypatch):
    """Licence inspection and version metadata share the middleware policy."""
    monkeypatch.setenv("QECTOR_API_KEY", "test-api-key")
    client, flask = _client_for(rest_api.create_app())

    assert _get(client, flask, "/health").status_code == 200
    assert _get(client, flask, "/api/license/info").status_code == 401
    assert _get(client, flask, "/version").status_code == 401
    authorised = _get(
        client,
        flask,
        "/api/license/info",
        headers={"Authorization": "Bearer test-api-key"},
    )
    assert authorised.status_code == 200
