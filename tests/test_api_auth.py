"""Config-driven API authentication with real HTTP dependencies and tenant state."""

import importlib
import json
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bananabread import config
from bananabread.cache import UserScopedCache
from bananabread.tenancy import TenantStore


@pytest.mark.parametrize(
    ("settings", "flags", "disabled"),
    [
        (None, [], False),
        ({}, [], False),
        ({"disable_api_key_auth": False}, [], False),
        ({"disable_api_key_auth": True}, [], True),
        ({"disable-api-key-auth": True}, [], True),
        ({"disable_api_key_auth": None}, [], False),
        ({"disable_api_key_auth": False}, ["--disable-api-key-auth"], True),
    ],
)
def test_auth_config_and_cli(monkeypatch, tmp_path, settings, flags, disabled):
    path = tmp_path / "config.json"
    if settings is not None:
        path.write_text(json.dumps(settings))
    monkeypatch.setattr(sys, "argv", ["bananabread", "--config", str(path), *flags])
    assert config.parse_args().disable_api_key_auth is disabled


@pytest.mark.parametrize("value", ["false", "true", 0, 1, [], {}])
def test_auth_config_rejects_non_boolean_values(monkeypatch, tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"disable_api_key_auth": value}))
    monkeypatch.setattr(sys, "argv", ["bananabread", "--config", str(path)])
    with pytest.raises(ValueError, match="disable_api_key_auth must be a boolean"):
        config.parse_args()


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    # Keep startup's generated key and JWT files out of the working checkout.
    monkeypatch.chdir(tmp_path)
    api = importlib.import_module("bananabread.api")
    api.tenant_store.close()
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({
        "management_key": "management-key",
        "users": {"alice": {"api_key": "alice-key"}},
        "cache": {"scope": "per_user"},
    }))
    store = TenantStore(str(path), usage_save_interval=0)
    monkeypatch.setattr(api, "tenant_store", store)
    monkeypatch.setattr(api, "api_keys", {})
    monkeypatch.setattr(api.args, "disable_api_key_auth", False)
    monkeypatch.setattr(api.args, "quant", "standard")
    monkeypatch.setattr(api, "embedding_cache", UserScopedCache(1024 * 1024))
    monkeypatch.setattr(api, "rerank_cache", UserScopedCache(1024 * 1024))
    monkeypatch.setattr(api, "log_embedding_result", lambda *a, **kw: None)

    class Model:
        tokenizer = None
        calls = 0

        def encode(self, inputs, **kwargs):
            self.calls += 1
            return np.tile([1.0, 0.0], (len(inputs), 1))

        def rank(self, query, documents, **kwargs):
            return [{"index": 0, "relevance_score": 1.0}]

    model = Model()
    monkeypatch.setattr(api.models_manager, "embedding_model", model)
    monkeypatch.setattr(api.models_manager, "embedding_model_pool", None)
    monkeypatch.setattr(api.models_manager, "embedding_model_name", "test/auth")
    monkeypatch.setattr(api.models_manager, "rerank_model", model)
    monkeypatch.setattr(api.models_manager, "rerank_model_pool", None)
    monkeypatch.setattr(api.models_manager, "initialize_models", lambda: None)
    monkeypatch.setattr(api, "cleanup_resources", lambda: None)
    api.load_api_keys()
    try:
        with TestClient(api.app, follow_redirects=False) as client:
            yield api, client, store, model
    finally:
        store.close()


USER_ROUTES = [
    ("GET", "/v1/health", None),
    ("GET", "/v1/models", None),
    ("GET", "/api/tags", None),
    ("POST", "/api/show", {"model": "test/auth"}),
    ("POST", "/v1/embeddings", {"input": "hello"}),
    ("POST", "/v0/embeddings", {"model": "test/auth", "input": "hello"}),
    ("POST", "/api/v0/embeddings", {"model": "test/auth", "input": "hello"}),
    ("POST", "/api/embeddings", {"model": "test/auth", "prompt": "hello"}),
    ("POST", "/api/embed", {"model": "test/auth", "input": "hello"}),
    ("POST", "/embedding", {"content": "hello"}),
    ("POST", "/v1/rerank", {"query": "hello", "documents": ["world"]}),
]


@pytest.mark.parametrize(("method", "route", "body"), USER_ROUTES)
def test_authentication_is_required_by_default(api_client, method, route, body):
    api, client, store, model = api_client
    assert "anonymous" not in store.data["users"]
    for headers in ({}, {"Authorization": "Basic ignored"}, {"Authorization": "Bearer invalid"}):
        assert client.request(method, route, json=body, headers=headers).status_code == 401
    response = client.request(method, route, json=body, headers={"Authorization": "Bearer alice-key"})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(("method", "route", "body"), USER_ROUTES)
def test_disabled_auth_accepts_missing_and_placeholder_keys(
    api_client, monkeypatch, method, route, body
):
    api, client, store, model = api_client
    monkeypatch.setattr(api.args, "disable_api_key_auth", True)
    api.load_api_keys()
    for headers in ({}, {"Authorization": "Basic ignored"}, {"Authorization": "Bearer placeholder"}):
        response = client.request(method, route, json=body, headers=headers)
        assert response.status_code == 200, response.text


@pytest.mark.parametrize("route", ["/v0/embeddings", "/api/v0/embeddings"])
@pytest.mark.parametrize("inputs", ["hello world", ["hello", "world"]])
def test_lmstudio_response_contract_and_shared_cache(api_client, route, inputs):
    api, client, store, model = api_client
    payload = {"model": "client-model-alias", "input": inputs}
    headers = {"Authorization": "Bearer alice-key"}
    baseline = client.post("/v1/embeddings", json=payload, headers=headers)
    inference_calls = model.calls
    response = client.post(route, json=payload, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == baseline.json()
    count = 1 if isinstance(inputs, str) else len(inputs)
    assert response.json() == {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [1.0, 0.0], "index": index}
            for index in range(count)
        ],
        "model": "test/auth",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    assert inference_calls > 0
    assert model.calls == inference_calls
    assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 4


@pytest.mark.parametrize("route", ["/v0/embeddings", "/api/v0/embeddings"])
def test_lmstudio_requests_enforce_shared_token_limits(api_client, route):
    api, client, store, model = api_client
    store.update_config(default_limits={"daily": 1})
    headers = {"Authorization": "Bearer alice-key"}
    payload = {"model": "test/auth", "input": "hello"}
    assert client.post(route, json=payload, headers=headers).status_code == 200
    assert client.post(route, json=payload, headers=headers).status_code == 429


def test_anonymous_requests_share_usage_and_cache(api_client, monkeypatch):
    api, client, store, model = api_client
    monkeypatch.setattr(api.args, "disable_api_key_auth", True)
    api.load_api_keys()
    for headers in ({}, {"Authorization": "Bearer alice-key"}):
        response = client.post("/v1/embeddings", json={"input": "hello"}, headers=headers)
        assert response.status_code == 200, response.text
    assert model.calls == 1
    users = store.snapshot()["users"]
    assert users["anonymous"]["usage"]["daily"]["tokens"] == 2
    assert users["alice"]["usage"]["daily"]["tokens"] == 0


@pytest.mark.parametrize("disabled", [False, True])
def test_embedding_trailing_slash_redirect_preserves_post(api_client, monkeypatch, disabled):
    api, client, store, model = api_client
    monkeypatch.setattr(api.args, "disable_api_key_auth", disabled)
    api.load_api_keys()
    headers = {} if disabled else {"Authorization": "Bearer alice-key"}
    response = client.post("/v1/embeddings/", json={"input": "hello"}, headers=headers)
    assert response.status_code == 307, response.text
    assert response.headers["location"] == "http://testserver/v1/embeddings"
    response = client.post(
        "/v1/embeddings/", json={"input": "hello"}, headers=headers, follow_redirects=True
    )
    assert response.status_code == 200, response.text
    assert response.request.method == "POST"


def test_options_handler_does_not_mask_unknown_paths_or_allowed_methods(api_client):
    api, client, store, model = api_client
    response = client.get("/v1/embeddings")
    assert response.status_code == 405
    assert "POST" in response.headers["allow"]
    assert client.post("/embed", json={"content": "hello"}).status_code == 404
    assert client.post("/v1/v1/embeddings", json={"input": "hello"}).status_code == 404


@pytest.mark.parametrize("path", ["/v1/embeddings", "/v1/embeddings/", "/unknown"])
def test_options_requests_work_without_auth_or_origin(api_client, path):
    api, client, store, model = api_client
    response = client.options(path)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"]


@pytest.mark.parametrize("method, expected", [("POST", 200), ("TRACE", 400)])
def test_cors_preflight_is_validated_without_auth(api_client, method, expected):
    api, client, store, model = api_client
    response = client.options("/v1/embeddings", headers={
        "Origin": "https://client.example",
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == expected
    assert response.headers["access-control-allow-origin"] == "*"


def test_existing_anonymous_user_limits_survive_restart(api_client, monkeypatch):
    api, client, store, model = api_client
    anonymous = store.create_user("anonymous", limits={"daily": 1})
    monkeypatch.setattr(api.args, "disable_api_key_auth", True)
    api.load_api_keys()
    assert client.post("/v1/embeddings", json={"input": "hello"}).status_code == 200
    api.load_api_keys()
    assert store.data["users"]["anonymous"]["api_key"] == anonymous["api_key"]
    assert client.post("/v1/embeddings", json={"input": "hello"}).status_code == 429
    monkeypatch.setattr(api.args, "disable_api_key_auth", False)
    assert client.get("/v1/health").status_code == 401


@pytest.mark.parametrize(("method", "route", "body"), [
    ("GET", "/v1/management/config", None),
    ("POST", "/v1/management/users", {"username": "bob"}),
    ("POST", "/v1/models/download", {"path": "test/model"}),
    ("POST", "/v1/classify", {"input": "hello"}),
    ("GET", "/v1/memory", None),
])
def test_management_auth_is_required_when_api_auth_is_disabled(
    api_client, monkeypatch, method, route, body
):
    api, client, store, model = api_client
    monkeypatch.setattr(api.args, "disable_api_key_auth", True)
    api.load_api_keys()
    assert client.request(method, route, json=body).status_code == 401
    response = client.request(method, route, json=body, headers={"Authorization": "Bearer alice-key"})
    assert response.status_code == 403
    assert client.get("/management").status_code == 302
    response = client.get("/v1/management/config", headers={"Authorization": "Bearer management-key"})
    assert response.status_code == 200
    assert client.post("/management/auth", json={"key": "management-key"}).status_code == 200
    assert client.get("/v1/management/config").status_code == 200
