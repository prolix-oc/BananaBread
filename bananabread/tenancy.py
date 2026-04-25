import json
import os
import secrets
import threading
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException


DEFAULT_LIMITS = {"daily": None, "weekly": None}
DEFAULT_CACHE_CONFIG = {
    "scope": "global",
    "default_embedding_mb": None,
    "default_rerank_mb": None,
    "users": {},
}


class TenantStore:
    def __init__(self, path: str = "./api_keys.json"):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {}
        self.key_to_user: dict[str, str] = {}

    def load(self) -> None:
        with self.lock:
            if self.path.exists():
                with self.path.open("r") as f:
                    raw = json.load(f)
            else:
                raw = {"user": secrets.token_hex(16)}

            self.data = self._normalize(raw)
            self._rebuild_key_index()
            self._save_locked()

    def _normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        if "users" in raw or "management_key" in raw:
            data = deepcopy(raw)
            data.setdefault("management_key", None)
            data.setdefault("default_limits", deepcopy(DEFAULT_LIMITS))
            data.setdefault("cache", deepcopy(DEFAULT_CACHE_CONFIG))
            data.setdefault("tiers", {})
            data.setdefault("users", {})
            data["cache"] = self._clean_cache_config(data["cache"], validate_users=False)
            for username, record in list(data["users"].items()):
                if isinstance(record, str) or record is None:
                    data["users"][username] = {"api_key": record or secrets.token_hex(16)}
                self._normalize_user(data["users"][username])
            return data

        users = {}
        for username, api_key in raw.items():
            users[username] = {"api_key": api_key or secrets.token_hex(16)}
            self._normalize_user(users[username])
        if not users:
            users["user"] = {"api_key": secrets.token_hex(16)}
            self._normalize_user(users["user"])
        return {
            "management_key": None,
            "default_limits": deepcopy(DEFAULT_LIMITS),
            "cache": deepcopy(DEFAULT_CACHE_CONFIG),
            "tiers": {},
            "users": users,
        }

    def _normalize_user(self, record: dict[str, Any]) -> None:
        record.setdefault("api_key", secrets.token_hex(16))
        record.setdefault("tier", None)
        record.setdefault("limits", {})
        record.setdefault("usage", {})
        self._ensure_usage_windows(record, now=datetime.now(timezone.utc))

    def _rebuild_key_index(self) -> None:
        self.key_to_user = {}
        for username, record in self.data.get("users", {}).items():
            api_key = record.get("api_key")
            if api_key:
                self.key_to_user[api_key] = username

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w") as f:
            json.dump(self.data, f, indent=4)
        os.replace(tmp_path, self.path)

    def authenticate_user(self, api_key: str) -> str:
        with self.lock:
            username = self.key_to_user.get(api_key)
        if not username:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return username

    def authenticate_management(self, api_key: str) -> None:
        with self.lock:
            management_key = self.data.get("management_key")
        if not management_key or api_key != management_key:
            raise HTTPException(status_code=403, detail="Management key required")

    def set_user_api_key(self, username: str, api_key: str) -> None:
        with self.lock:
            users = self.data.get("users", {})
            if username not in users:
                raise HTTPException(status_code=404, detail=f"User not found: {username}")
            users[username]["api_key"] = api_key
            self._rebuild_key_index()
            self._save_locked()

    def regenerate_user_api_key(self, username: str) -> str:
        with self.lock:
            users = self.data.get("users", {})
            if username not in users:
                raise HTTPException(status_code=404, detail=f"User not found: {username}")
            new_key = secrets.token_hex(24)
            users[username]["api_key"] = new_key
            self._rebuild_key_index()
            self._save_locked()
            return new_key

    def bulk_regenerate_user_api_keys(self, usernames: list[str]) -> dict[str, Any]:
        results = []
        errors = []
        with self.lock:
            users = self.data.get("users", {})
            for username in usernames:
                username = username.strip()
                if not username:
                    continue
                if username not in users:
                    errors.append({"username": username, "error": "User not found"})
                    continue
                new_key = secrets.token_hex(24)
                users[username]["api_key"] = new_key
                results.append({"username": username, "api_key": new_key})
            self._rebuild_key_index()
            self._save_locked()
        return {"regenerated": results, "errors": errors}

    def create_user(
        self,
        username: str,
        tier: str | None = None,
        limits: dict[str, int | None] | None = None,
    ) -> dict[str, Any]:
        username = username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="Username must be provided")

        with self.lock:
            if username in self.data["users"]:
                raise HTTPException(status_code=409, detail="User already exists")
            if tier and tier not in self.data.get("tiers", {}):
                raise HTTPException(status_code=400, detail=f"Unknown tier: {tier}")

            record = {
                "api_key": secrets.token_hex(24),
                "tier": tier,
                "limits": self._clean_limits(limits or {}),
                "usage": {},
            }
            self._normalize_user(record)
            self.data["users"][username] = record
            self._rebuild_key_index()
            self._save_locked()
            return {"username": username, "api_key": record["api_key"], "tier": tier, "limits": self.effective_limits_locked(record)}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            users = {}
            for username, record in self.data.get("users", {}).items():
                users[username] = {
                    "api_key_preview": self._preview_key(record.get("api_key")),
                    "tier": record.get("tier"),
                    "limits": self._clean_limits(record.get("limits", {})),
                    "effective_limits": self.effective_limits_locked(record),
                    "usage": deepcopy(record.get("usage", {})),
                }
            return {
                "management_key_set": bool(self.data.get("management_key")),
                "default_limits": self._clean_limits(self.data.get("default_limits", {})),
                "cache": deepcopy(self.data.get("cache", DEFAULT_CACHE_CONFIG)),
                "tiers": deepcopy(self.data.get("tiers", {})),
                "users": users,
            }

    def update_config(
        self,
        management_key: str | None = None,
        default_limits: dict[str, int | None] | None = None,
        tiers: dict[str, dict[str, int | None]] | None = None,
        cache: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if management_key is not None:
                management_key = management_key.strip()
                if not management_key:
                    raise HTTPException(status_code=400, detail="Management key cannot be blank")
                self.data["management_key"] = management_key

            if default_limits is not None:
                self.data["default_limits"] = self._clean_limits(default_limits)

            if cache is not None:
                self.data["cache"] = self._clean_cache_config(cache, validate_users=True)

            if tiers is not None:
                cleaned_tiers = {}
                for tier_name, tier_limits in tiers.items():
                    tier_name = tier_name.strip()
                    if not tier_name:
                        raise HTTPException(status_code=400, detail="Tier names cannot be blank")
                    cleaned_tiers[tier_name] = self._clean_limits(tier_limits or {})

                assigned_tiers = {
                    record.get("tier")
                    for record in self.data.get("users", {}).values()
                    if record.get("tier")
                }
                missing_tiers = sorted(assigned_tiers - set(cleaned_tiers))
                if missing_tiers:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot remove tiers assigned to users: {', '.join(missing_tiers)}",
                    )
                self.data["tiers"] = cleaned_tiers

            self._save_locked()
            return self.snapshot_locked()

    def snapshot_locked(self) -> dict[str, Any]:
        users = {}
        for username, record in self.data.get("users", {}).items():
            users[username] = {
                "api_key_preview": self._preview_key(record.get("api_key")),
                "tier": record.get("tier"),
                "limits": self._clean_limits(record.get("limits", {})),
                "effective_limits": self.effective_limits_locked(record),
                "usage": deepcopy(record.get("usage", {})),
            }
        return {
            "management_key_set": bool(self.data.get("management_key")),
            "default_limits": self._clean_limits(self.data.get("default_limits", {})),
            "cache": deepcopy(self.data.get("cache", DEFAULT_CACHE_CONFIG)),
            "tiers": deepcopy(self.data.get("tiers", {})),
            "users": users,
        }

    def cache_config(self) -> dict[str, Any]:
        with self.lock:
            return deepcopy(self.data.get("cache", DEFAULT_CACHE_CONFIG))

    def check_and_consume(self, username: str, tokens: int) -> dict[str, Any]:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")

        now = datetime.now(timezone.utc)
        with self.lock:
            record = self.data["users"].get(username)
            if not record:
                raise HTTPException(status_code=401, detail="Unauthorized")

            self._ensure_usage_windows(record, now=now)
            limits = self.effective_limits_locked(record)
            usage = record["usage"]

            for window in ("daily", "weekly"):
                limit = limits.get(window)
                if limit is not None and usage[window]["tokens"] + tokens > limit:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "Token limit exceeded",
                            "window": window,
                            "limit": limit,
                            "used": usage[window]["tokens"],
                            "requested": tokens,
                            "reset_at": usage[window]["reset_at"],
                        },
                    )

            usage["daily"]["tokens"] += tokens
            usage["weekly"]["tokens"] += tokens
            self._save_locked()
            return {
                "tokens": tokens,
                "limits": limits,
                "usage": deepcopy(usage),
            }

    def effective_limits_locked(self, record: dict[str, Any]) -> dict[str, int | None]:
        limits = deepcopy(DEFAULT_LIMITS)
        limits.update(self._clean_limits(self.data.get("default_limits", {})))
        tier = record.get("tier")
        if tier:
            limits.update(self._clean_limits(self.data.get("tiers", {}).get(tier, {})))
        limits.update(self._clean_limits(record.get("limits", {})))
        return limits

    def _clean_limits(self, limits: dict[str, Any]) -> dict[str, int | None]:
        cleaned = {}
        for window in ("daily", "weekly"):
            if window not in limits:
                continue
            value = limits[window]
            if value is None:
                cleaned[window] = None
            else:
                value = int(value)
                if value < 0:
                    raise HTTPException(status_code=400, detail=f"{window} limit must be non-negative")
                cleaned[window] = value
        return cleaned

    def _clean_cache_config(self, cache: dict[str, Any], validate_users: bool) -> dict[str, Any]:
        scope = cache.get("scope", DEFAULT_CACHE_CONFIG["scope"])
        if scope not in {"global", "per_user"}:
            raise HTTPException(status_code=400, detail="Cache scope must be 'global' or 'per_user'")

        users = {}
        known_users = set(self.data.get("users", {})) if validate_users else None
        for username, limits in (cache.get("users") or {}).items():
            username = username.strip()
            if not username:
                raise HTTPException(status_code=400, detail="Cache user names cannot be blank")
            if known_users is not None and username not in known_users:
                raise HTTPException(status_code=400, detail=f"Unknown cache user: {username}")
            users[username] = self._clean_cache_limits(limits or {})

        return {
            "scope": scope,
            "default_embedding_mb": self._clean_cache_limit(cache.get("default_embedding_mb")),
            "default_rerank_mb": self._clean_cache_limit(cache.get("default_rerank_mb")),
            "users": users,
        }

    def _clean_cache_limits(self, limits: dict[str, Any]) -> dict[str, int | None]:
        return {
            "embedding_mb": self._clean_cache_limit(limits.get("embedding_mb")),
            "rerank_mb": self._clean_cache_limit(limits.get("rerank_mb")),
        }

    def _clean_cache_limit(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        value = int(value)
        if value < 0:
            raise HTTPException(status_code=400, detail="Cache limits must be non-negative")
        return value

    def _ensure_usage_windows(self, record: dict[str, Any], now: datetime) -> None:
        usage = record.setdefault("usage", {})
        windows = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
        }
        for name, delta in windows.items():
            window = usage.get(name) or {}
            reset_at = self._parse_datetime(window.get("reset_at"))
            if reset_at is None or now >= reset_at:
                usage[name] = {"tokens": 0, "reset_at": (now + delta).isoformat()}
            else:
                window.setdefault("tokens", 0)
                window["reset_at"] = reset_at.isoformat()
                usage[name] = window

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _preview_key(self, api_key: str | None) -> str | None:
        if not api_key:
            return None
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return f"{api_key[:4]}...{api_key[-4:]}"


def count_text_tokens(texts: list[str], tokenizer: Any) -> int:
    if not texts:
        return 0
    if tokenizer is None:
        return sum(len(text.split()) for text in texts)

    try:
        encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        input_ids = encoded.get("input_ids", encoded) if isinstance(encoded, dict) else encoded
        return sum(_token_count(item) for item in input_ids)
    except Exception:
        pass

    try:
        encoded = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
        return sum(len(item) for item in encoded)
    except Exception:
        return sum(len(text.split()) for text in texts)


def _token_count(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    return len(value)
