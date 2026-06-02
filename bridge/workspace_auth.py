"""Signed workspace UI tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


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
