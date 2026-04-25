import os
import base64
import json
import time
import hmac
import hashlib
from pathlib import Path
from fastapi import Request, HTTPException

JWT_SECRET_FILE = "./.jwt_secret"
JWT_EXPIRATION_SECONDS = 86400  # 24 hours


def get_or_create_jwt_secret() -> str:
    path = Path(JWT_SECRET_FILE)
    if path.exists():
        return path.read_text().strip()
    secret = base64.urlsafe_b64encode(os.urandom(32)).decode()
    path.write_text(secret)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return secret


JWT_SECRET = get_or_create_jwt_secret()


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def create_jwt(payload: dict) -> str:
    header = _base64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload_copy = {**payload, "iat": int(time.time())}
    if "exp" not in payload_copy:
        payload_copy["exp"] = payload_copy["iat"] + JWT_EXPIRATION_SECONDS

    payload_b64 = _base64url_encode(
        json.dumps(payload_copy, separators=(",", ":")).encode()
    )
    message = f"{header}.{payload_b64}".encode()
    signature = hmac.new(JWT_SECRET.encode(), message, hashlib.sha256).digest()
    sig_b64 = _base64url_encode(signature)
    return f"{header}.{payload_b64}.{sig_b64}"


def verify_jwt(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")

    header_b64, payload_b64, signature_b64 = parts
    message = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(JWT_SECRET.encode(), message, hashlib.sha256).digest()
    expected_sig_b64 = _base64url_encode(expected_sig)

    if not hmac.compare_digest(signature_b64, expected_sig_b64):
        raise ValueError("Invalid JWT signature")

    payload = json.loads(_base64url_decode(payload_b64))
    exp = payload.get("exp")
    if exp and int(time.time()) > exp:
        raise ValueError("JWT expired")

    return payload


def set_auth_cookie(response, token: str) -> None:
    response.set_cookie(
        key="bananabread_auth",
        value=token,
        httponly=True,
        secure=False,  # Set True in production with HTTPS
        samesite="lax",
        max_age=JWT_EXPIRATION_SECONDS,
        path="/",
    )


def clear_auth_cookie(response) -> None:
    response.delete_cookie(key="bananabread_auth", path="/")


def require_auth_cookie(request: Request) -> dict:
    token = request.cookies.get("bananabread_auth")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return verify_jwt(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
