import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from bananabread.cache import LimitedCache, UserScopedCache
from bananabread.tenancy import TenantStore, count_text_tokens


def test_migrates_flat_api_key_file(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"user": "user-key"}))

    store = TenantStore(str(path))
    store.load()

    assert store.data["management_key"] is None
    assert store.data["users"]["user"]["api_key"] == "user-key"
    assert store.authenticate_user("user-key") == "user"


def test_management_key_is_separate_from_user_keys(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(
        json.dumps(
            {
                "management_key": "management-key",
                "users": {"user": {"api_key": "user-key"}},
            }
        )
    )
    store = TenantStore(str(path))
    store.load()

    store.authenticate_management("management-key")
    with pytest.raises(HTTPException) as exc:
        store.authenticate_management("user-key")

    assert exc.value.status_code == 403


def test_create_user_generates_key_and_applies_tier_limits(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(
        json.dumps(
            {
                "management_key": "management-key",
                "tiers": {"free": {"daily": 10, "weekly": 50}},
                "users": {},
            }
        )
    )
    store = TenantStore(str(path))
    store.load()

    created = store.create_user("alice", tier="free")

    assert created["username"] == "alice"
    assert created["api_key"]
    assert created["limits"] == {"daily": 10, "weekly": 50}
    assert store.authenticate_user(created["api_key"]) == "alice"


def test_user_limits_override_tier_limits(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(
        json.dumps(
            {
                "tiers": {"free": {"daily": 10, "weekly": 50}},
                "users": {"alice": {"api_key": "alice-key", "tier": "free", "limits": {"daily": 3}}},
            }
        )
    )
    store = TenantStore(str(path))
    store.load()

    store.check_and_consume("alice", 3)
    with pytest.raises(HTTPException) as exc:
        store.check_and_consume("alice", 1)

    assert exc.value.status_code == 429
    assert exc.value.detail["window"] == "daily"


def test_usage_resets_after_interval(tmp_path):
    reset_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    path = tmp_path / "api_keys.json"
    path.write_text(
        json.dumps(
            {
                "users": {
                    "alice": {
                        "api_key": "alice-key",
                        "limits": {"daily": 5, "weekly": 10},
                        "usage": {
                            "daily": {"tokens": 5, "reset_at": reset_at},
                            "weekly": {"tokens": 5, "reset_at": reset_at},
                        },
                    }
                }
            }
        )
    )
    store = TenantStore(str(path))
    store.load()

    usage = store.check_and_consume("alice", 2)

    assert usage["usage"]["daily"]["tokens"] == 2
    assert usage["usage"]["weekly"]["tokens"] == 2


def test_count_text_tokens_uses_embedding_tokenizer():
    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2], [1, 2, 3]]}

    assert count_text_tokens(["hello", "world"], FakeTokenizer()) == 5


def test_update_config_sets_management_defaults_and_tiers(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"management_key": "old-key", "users": {"alice": {"api_key": "alice-key"}}}))
    store = TenantStore(str(path))
    store.load()

    snapshot = store.update_config(
        management_key="new-key",
        default_limits={"daily": 100, "weekly": None},
        tiers={"pro": {"daily": 1000, "weekly": 5000}},
    )

    assert snapshot["management_key_set"] is True
    assert snapshot["default_limits"] == {"daily": 100, "weekly": None}
    assert snapshot["tiers"] == {"pro": {"daily": 1000, "weekly": 5000}}
    store.authenticate_management("new-key")


def test_update_config_rejects_removing_assigned_tier(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(
        json.dumps(
            {
                "tiers": {"free": {"daily": 10}},
                "users": {"alice": {"api_key": "alice-key", "tier": "free"}},
            }
        )
    )
    store = TenantStore(str(path))
    store.load()

    with pytest.raises(HTTPException) as exc:
        store.update_config(tiers={})

    assert exc.value.status_code == 400
    assert "free" in exc.value.detail


# ----- Cache Tests -----


def test_limited_cache_evicts_oldest():
    cache = LimitedCache(600)
    cache["a"] = "x" * 250
    cache["b"] = "x" * 250
    assert len(cache) == 2
    cache["c"] = "x" * 250
    assert len(cache) == 2
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache


def test_user_scoped_cache_global_mode_shares_entries():
    cache = UserScopedCache(1000)
    cache.set("alice", "k1", "v1")
    assert cache.get("alice", "k1") == "v1"
    assert cache.get("bob", "k1") == "v1"


def test_user_scoped_cache_per_user_mode_isolates_entries():
    cache = UserScopedCache(1000)
    cache.configure(scope="per_user")
    cache.set("alice", "k1", "v1")
    assert cache.get("alice", "k1") == "v1"
    assert cache.get("bob", "k1") is None


def test_user_scoped_cache_per_user_mode_respects_limits():
    cache = UserScopedCache(10000)
    cache.configure(
        scope="per_user",
        default_limit_bytes=2000,
        user_limits={"alice": 500},
    )
    cache.set("alice", "k1", "x" * 250)
    assert cache.get("alice", "k1") == "x" * 250
    cache.set("alice", "k2", "y" * 250)
    assert cache.get("alice", "k1") is None
    assert cache.get("alice", "k2") == "y" * 250


def test_user_scoped_cache_configure_updates_limits():
    cache = UserScopedCache(1000)
    cache.set("alice", "k1", "x" * 40)
    assert cache.get("alice", "k1") == "x" * 40
    cache.configure(scope="global", default_limit_bytes=20)
    assert cache.get("alice", "k1") is None


def test_limited_cache_stats():
    cache = LimitedCache(1000)
    cache["a"] = "x" * 100
    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["limit"] == 1000
    assert stats["current_size"] > 0


def test_user_scoped_cache_total_size():
    cache = UserScopedCache(1000)
    cache.configure(scope="per_user")
    cache.set("alice", "k1", "x" * 40)
    cache.set("bob", "k2", "y" * 40)
    assert cache.total_size() > 0


def test_tenant_store_cache_config_validation(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"users": {"alice": {"api_key": "alice-key"}}}))
    store = TenantStore(str(path))
    store.load()

    with pytest.raises(HTTPException) as exc:
        store.update_config(cache={"scope": "invalid"})
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        store.update_config(cache={"users": {"unknown_user": {"embedding_mb": 10}}})
    assert exc.value.status_code == 400
    assert "unknown_user" in exc.value.detail


def test_tenant_store_cache_config_roundtrip(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"users": {"alice": {"api_key": "alice-key"}}}))
    store = TenantStore(str(path))
    store.load()

    snapshot = store.update_config(
        cache={
            "scope": "per_user",
            "default_embedding_mb": 64,
            "default_rerank_mb": 32,
            "users": {"alice": {"embedding_mb": 128, "rerank_mb": None}},
        }
    )

    assert snapshot["cache"]["scope"] == "per_user"
    assert snapshot["cache"]["default_embedding_mb"] == 64
    assert snapshot["cache"]["default_rerank_mb"] == 32
    assert snapshot["cache"]["users"]["alice"]["embedding_mb"] == 128
    assert snapshot["cache"]["users"]["alice"]["rerank_mb"] is None


def test_set_user_api_key_updates_key(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"users": {"alice": {"api_key": "old-key"}}}))
    store = TenantStore(str(path))
    store.load()

    store.set_user_api_key("alice", "new-key")
    assert store.authenticate_user("new-key") == "alice"

    with pytest.raises(HTTPException) as exc:
        store.authenticate_user("old-key")
    assert exc.value.status_code == 401


def test_set_user_api_key_rejects_unknown_user(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"users": {}}))
    store = TenantStore(str(path))
    store.load()

    with pytest.raises(HTTPException) as exc:
        store.set_user_api_key("bob", "some-key")
    assert exc.value.status_code == 404


def test_fresh_file_creates_default_user_with_null_management_key(tmp_path):
    path = tmp_path / "api_keys.json"
    store = TenantStore(str(path))
    assert not store.path.exists()
    store.load()

    assert "user" in store.data["users"]
    assert store.data["users"]["user"]["api_key"]
    assert store.data["management_key"] is None
