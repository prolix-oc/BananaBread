"""Config-driven API authentication with real HTTP dependencies and tenant state."""

import importlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.parametrize(("route", "body"), [
    (route, body) for method, route, body in USER_ROUTES if method == "POST" and route != "/api/show"
])
def test_tokenizer_contention_does_not_block_http(api_client, monkeypatch, route, body):
    api, client, store, model = api_client
    encoding = threading.Event()
    release_encode = threading.Event()
    counting_waits = threading.Event()
    lock = threading.RLock()

    class ObservedLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                counting_waits.set()
                assert lock.acquire(timeout=5), "tokenizer lock was never released"
            return self

        def __exit__(self, *exc):
            lock.release()

    original_encode = model.encode

    def slow_encode(inputs, **kwargs):
        if inputs == ["held"]:
            encoding.set()
            assert release_encode.wait(5), "test did not release inference"
        return original_encode(inputs, **kwargs)

    monkeypatch.setattr(model, "tokenizer_lock", ObservedLock(), raising=False)
    monkeypatch.setattr(model, "encode", slow_encode)
    headers = {"Authorization": "Bearer alice-key"}
    with ThreadPoolExecutor(max_workers=3) as clients:
        first = clients.submit(client.post, "/v1/embeddings", json={"input": "held"}, headers=headers)
        try:
            assert encoding.wait(2), "first request did not reach inference"
            retry = clients.submit(client.post, route, json=body, headers=headers)
            assert counting_waits.wait(2), "second request did not attempt token accounting"
            health = clients.submit(client.get, "/v1/health", headers=headers)
            # This must finish while inference still owns the tokenizer lock.
            assert health.result(timeout=1).status_code == 200
        finally:
            release_encode.set()
        assert first.result(timeout=2).status_code == 200
        response = retry.result(timeout=2)
        assert response.status_code == 200, response.text


@pytest.mark.parametrize("inputs", ["hello", {"text": "hello"}])
def test_tenant_accounting_does_not_block_http(api_client, monkeypatch, inputs):
    api, client, store, model = api_client
    accounting = threading.Event()
    release_accounting = threading.Event()
    original_consume = store.check_and_consume

    def slow_consume(username, tokens):
        accounting.set()
        assert release_accounting.wait(5), "test did not release accounting"
        return original_consume(username, tokens)

    monkeypatch.setattr(store, "check_and_consume", slow_consume)
    headers = {"Authorization": "Bearer alice-key"}
    with ThreadPoolExecutor(max_workers=2) as clients:
        request = clients.submit(client.post, "/v1/embeddings", json={"input": inputs}, headers=headers)
        try:
            assert accounting.wait(2)
            health = clients.submit(client.get, "/v1/health", headers=headers)
            assert health.result(timeout=1).status_code == 200
        finally:
            release_accounting.set()
        response = request.result(timeout=2)
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["prompt_tokens"] == 1


def test_concurrent_accounting_preserves_token_limit(api_client, monkeypatch):
    api, client, store, model = api_client
    store.update_config(default_limits={"daily": 1})
    both_counted = threading.Barrier(2)
    original_count = api.count_usage_tokens

    def count_together(*args, **kwargs):
        tokens = original_count(*args, **kwargs)
        both_counted.wait(timeout=3)
        return tokens

    monkeypatch.setattr(api, "count_usage_tokens", count_together)
    headers = {"Authorization": "Bearer alice-key"}
    with ThreadPoolExecutor(max_workers=2) as clients:
        requests = [
            clients.submit(client.post, "/v1/embeddings", json={"input": "hello"}, headers=headers)
            for _ in range(2)
        ]
        assert sorted(request.result(timeout=5).status_code for request in requests) == [200, 429]
    assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 1
    assert model.calls == 1


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


@pytest.mark.parametrize("route,field", [
    ("/v1/embeddings", "input"), ("/v0/embeddings", "input"),
    ("/api/v0/embeddings", "input"), ("/api/embed", "input"),
    ("/api/embeddings", "prompt"), ("/embedding", "content"),
    ("/v1/rerank", "documents"),
])
def test_overload_rejected_before_accounting_on_every_user_inference_route(api_client, monkeypatch, route, field):
    from bananabread.scheduling import InferenceAdmission

    api, client, store, model = api_client
    admission = InferenceAdmission(1, 1)
    monkeypatch.setattr(api, "inference_admission", admission)
    payload = {field: ["hello"], "query": "hello"} if field == "documents" else {field: "hello"}
    with admission.slot("alice"):
        response = client.post(route, json=payload, headers={"Authorization": "Bearer alice-key"})
    assert response.status_code == 429, response.text
    assert response.headers["Retry-After"] == "1"
    assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 0
    assert model.calls == 0


@pytest.mark.parametrize("route,field", [
    ("/v1/embeddings", "input"), ("/api/embed", "input"), ("/embedding", "content"),
])
def test_batch_encoding_usage_and_quota_are_content_sensitive(api_client, monkeypatch, route, field):
    from transformers import BatchEncoding

    api, client, store, model = api_client
    # The real BatchEncoding container caused the 23-token production defect.
    def tokenizer(texts, **kwargs):
        if kwargs.get("return_tensors"):
            return BatchEncoding({"input_ids": np.zeros((len(texts), max(map(len, texts))))})
        return BatchEncoding({
            "input_ids": [list(range(len(text.split()))) for text in texts],
            "attention_mask": [[1] * len(text.split()) for text in texts],
        })

    monkeypatch.setattr(model, "tokenizer", tokenizer)
    # Avoid making the tokenizer stub double as the batch planner.
    monkeypatch.setattr(api, "_padded_input_length", lambda *args: 1)
    store.update_config(default_limits={"daily": 5})
    headers = {"Authorization": "Bearer alice-key"}
    for text in ("one", "one two three"):
        response = client.post(route, json={field: text, "model": "test/auth"}, headers=headers)
        assert response.status_code == 200, response.text
        if route == "/v1/embeddings":
            assert response.json()["usage"]["prompt_tokens"] == len(text.split())
        elif route == "/api/embed":
            assert response.json()["prompt_eval_count"] == len(text.split())
    denied = client.post(route, json={field: "one two", "model": "test/auth"}, headers=headers)
    assert denied.status_code == 429
    assert denied.json()["detail"]["requested"] == 2
    assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 4


def test_simultaneous_http_requests_share_batches_without_mixing_tenant_results(api_client, monkeypatch):
    api, client, store, model = api_client
    store.create_user("bob")
    store.set_user_api_key("bob", "bob-key")
    monkeypatch.setattr(api.app.state.embedding_scheduler, "wait_s", 0.05)
    # A measurable model lets the scheduler combine the independent requests.
    monkeypatch.setattr(model, "preprocess", lambda inputs: {"input_ids": np.zeros((1, 1))}, raising=False)
    encoded = []

    def encode(inputs, **kwargs):
        encoded.append(list(inputs))
        return np.array([[float(text), 1.0] for text in inputs])

    monkeypatch.setattr(model, "encode", encode)
    barrier = threading.Barrier(8)

    def request(i):
        username = "alice" if i % 2 else "bob"
        barrier.wait(timeout=3)
        response = client.post("/v1/embeddings", json={"input": str(i)},
                               headers={"Authorization": f"Bearer {username}-key"})
        assert response.status_code == 200, response.text
        return response.json()["data"][0]["embedding"]

    with ThreadPoolExecutor(max_workers=8) as clients:
        results = list(clients.map(request, range(1, 9)))
    assert results == [[float(i), 1.0] for i in range(1, 9)]
    assert any(len(batch) > 1 for batch in encoded)
    assert sum(map(len, encoded)) == 8
    users = store.snapshot()["users"]
    assert users["alice"]["usage"]["daily"]["tokens"] == 4
    assert users["bob"]["usage"]["daily"]["tokens"] == 4
    assert api.inference_admission.stats()["active_requests"] == 0


@pytest.mark.parametrize("route,field,value", [
    ("/v1/embeddings", "input", "hello"),
    ("/v1/embeddings", "input", {"text": "hello"}),
    ("/v0/embeddings", "input", "hello"),
    ("/api/v0/embeddings", "input", "hello"),
    ("/api/embed", "input", "hello"),
    ("/api/embeddings", "prompt", "hello"),
    ("/embedding", "content", "hello"),
    ("/v1/rerank", "documents", ["hello"]),
])
def test_token_work_overflow_is_not_charged_on_any_user_route(api_client, monkeypatch, route, field, value):
    from bananabread.scheduling import InferenceAdmission

    api, client, store, model = api_client
    admission = InferenceAdmission(64, 8, max_tokens=4, per_user_tokens=4)
    monkeypatch.setattr(api, "inference_admission", admission)
    payload = {field: value, "model": "test/auth", "query": "hello"}
    with admission.slot("busy") as busy:
        busy.reserve_tokens(4)
        response = client.post(route, json=payload, headers={"Authorization": "Bearer alice-key"})
        assert response.status_code == 429, response.text
        assert int(response.headers["Retry-After"]) >= 1
        assert admission.stats()["active_tokens"] == 4
        assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 0
    response = client.post(route, json=payload, headers={"Authorization": "Bearer alice-key"})
    assert response.status_code == 200, response.text
    assert admission.stats()["active_tokens"] == 0
    assert admission.stats()["active_requests"] == 0


def test_scheduled_encoding_does_not_repeat_batch_planning(api_client, monkeypatch):
    api, client, store, model = api_client
    planned = []

    def preprocess(inputs):
        planned.extend(inputs)
        return {"input_ids": np.zeros((1, 1))}

    monkeypatch.setattr(model, "preprocess", preprocess, raising=False)
    inputs = [f"document {i}" for i in range(20)]
    response = client.post("/v1/embeddings", json={"input": inputs},
                           headers={"Authorization": "Bearer alice-key"})
    assert response.status_code == 200, response.text
    assert planned == inputs
    assert model.calls == 3


@pytest.mark.parametrize("global_budget,user_budget", [(1, 100), (100, 1)])
def test_parallel_token_reservations_enforce_budget_before_charging(api_client, monkeypatch, global_budget, user_budget):
    from concurrent.futures import as_completed
    from bananabread.scheduling import InferenceAdmission

    api, client, store, model = api_client
    admission = InferenceAdmission(64, 8, max_tokens=global_budget, per_user_tokens=user_budget)
    monkeypatch.setattr(api, "inference_admission", admission)
    barrier = threading.Barrier(2)
    release = threading.Event()
    original_count = api.count_usage_tokens
    original_encode = model.encode

    def count(*args, **kwargs):
        tokens = original_count(*args, **kwargs)
        barrier.wait(timeout=3)
        return tokens

    def encode(*args, **kwargs):
        assert release.wait(3)
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(api, "count_usage_tokens", count)
    monkeypatch.setattr(model, "encode", encode)
    with ThreadPoolExecutor(max_workers=2) as clients:
        futures = [clients.submit(client.post, "/v1/embeddings", json={"input": "hello"},
                                  headers={"Authorization": "Bearer alice-key"}) for _ in range(2)]
        try:
            first = next(as_completed(futures, timeout=2)).result()
            assert first.status_code == 429, first.text
            assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 1
        finally:
            release.set()
        assert sorted(f.result(timeout=2).status_code for f in futures) == [200, 429]
    assert admission.stats()["active_tokens"] == 0


@pytest.mark.parametrize("input_value", ["hello", {"text": "hello"}])
def test_optional_admission_wait_keeps_health_live_and_charges_only_after_capacity(api_client, monkeypatch, input_value):
    from bananabread.scheduling import InferenceAdmission

    api, client, store, model = api_client
    admission = InferenceAdmission(64, 8, max_tokens=1)
    monkeypatch.setattr(api, "inference_admission", admission)
    monkeypatch.setattr(api.args, "inference_admission_wait_ms", 1000)
    waiting = threading.Event()
    original_wait = admission.wait_for_tokens

    async def wait(*args):
        waiting.set()
        return await original_wait(*args)

    monkeypatch.setattr(admission, "wait_for_tokens", wait)
    headers = {"Authorization": "Bearer alice-key"}
    with ThreadPoolExecutor(max_workers=1) as clients:
        with admission.slot("busy") as busy:
            busy.reserve_tokens(1)
            response = clients.submit(client.post, "/v1/embeddings", json={"input": input_value}, headers=headers)
            assert waiting.wait(2)
            assert client.get("/v1/health", headers=headers).status_code == 200
            assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 0
        result = response.result(timeout=2)
        assert result.status_code == 200, result.text
    assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 1
    assert admission.stats()["active_tokens"] == 0
    assert admission.stats()["waiting_requests"] == 0


@pytest.mark.parametrize("route,field,value", [
    ("/v1/embeddings", "input", "hello"),
    ("/v1/embeddings", "input", {"text": "hello"}),
    ("/api/embed", "input", "hello"),
    ("/embedding", "content", "hello"),
    ("/v1/rerank", "documents", ["hello"]),
])
def test_optional_admission_deadline_returns_429_without_quota_charge(api_client, monkeypatch, route, field, value):
    from bananabread.scheduling import InferenceAdmission

    api, client, store, model = api_client
    admission = InferenceAdmission(64, 8, max_tokens=1)
    monkeypatch.setattr(api, "inference_admission", admission)
    monkeypatch.setattr(api.args, "inference_admission_wait_ms", 10)
    with admission.slot("busy") as busy:
        busy.reserve_tokens(1)
        response = client.post(route, json={field: value, "model": "test/auth", "query": "hello"},
                               headers={"Authorization": "Bearer alice-key"})
        assert response.status_code == 429, response.text
        assert int(response.headers["Retry-After"]) >= 1
        assert store.snapshot()["users"]["alice"]["usage"]["daily"]["tokens"] == 0
        assert admission.stats()["active_requests"] == 1
    assert admission.stats()["active_tokens"] == 0
    assert admission.stats()["waiting_requests"] == 0
    assert model.calls == 0


def test_running_rerank_and_embedding_share_work_capacity_and_release_it(api_client, monkeypatch):
    from bananabread.scheduling import InferenceAdmission

    api, client, store, model = api_client
    admission = InferenceAdmission(64, 8, max_tokens=2)
    monkeypatch.setattr(api, "inference_admission", admission)
    store.create_user("bob")
    store.set_user_api_key("bob", "bob-key")
    started, release = threading.Event(), threading.Event()
    original_rank = model.rank

    def rank(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return original_rank(*args, **kwargs)

    monkeypatch.setattr(model, "rank", rank)
    with ThreadPoolExecutor(max_workers=1) as clients:
        future = clients.submit(client.post, "/v1/rerank", json={"query": "hello", "documents": ["world"]},
                                headers={"Authorization": "Bearer alice-key"})
        try:
            assert started.wait(2)
            assert admission.stats()["active_tokens"] == 2
            response = client.post("/v1/embeddings", json={"input": "hello"},
                                   headers={"Authorization": "Bearer bob-key"})
            assert response.status_code == 429
            assert store.snapshot()["users"]["bob"]["usage"]["daily"]["tokens"] == 0
        finally:
            release.set()
        assert future.result(timeout=2).status_code == 200
    assert admission.stats()["active_tokens"] == 0
    assert admission.stats()["estimated_requests_per_second"] is not None
