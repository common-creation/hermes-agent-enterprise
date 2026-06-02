"""Signed workspace UI tokens.

The token format is intentionally stdlib-only:
base64url(json payload) + "." + base64url(HMAC-SHA256(payload)).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def sign_workspace_token(payload: dict[str, Any], secret: str, ttl_seconds: int = 3600) -> str:
    token_payload = dict(payload)
    token_payload["exp"] = int(time.time()) + ttl_seconds
    encoded_payload = _b64encode(
        json.dumps(token_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        secret.encode("utf-8"),
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{encoded_payload}.{_b64encode(signature)}"


def verify_workspace_token(token: str, secret: str, now: int | None = None) -> dict[str, Any]:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("invalid token format") from exc

    expected = _b64encode(
        hmac.new(
            secret.encode("utf-8"),
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
    )
    if not hmac.compare_digest(expected, encoded_signature):
        raise ValueError("invalid token signature")

    payload = json.loads(_b64decode(encoded_payload))
    current = int(time.time()) if now is None else now
    if int(payload.get("exp", 0)) < current:
        raise ValueError("token expired")
    return payload
