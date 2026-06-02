"""S3-backed workspace file operations for the settings UI."""

from __future__ import annotations

import time
from typing import Any

MAX_TEXT_BYTES = 256 * 1024
VERSION_MARKER = ".workspace-version"
ROOT_FILES = {"SOUL.md", "USER.md", "MEMORY.md", "config.yaml"}
ROOT_DIRS = ("memories/", "skills/", "cron/")
DENIED_PREFIXES = ("logs/", "cache/", "sessions/")
DENIED_SUFFIXES = (
    ".db",
    ".db-journal",
    ".db-wal",
    ".db-shm",
    ".pyc",
    ".pyo",
    ".sock",
)


def validate_workspace_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")

    if not normalized:
        raise ValueError("path is required")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        raise ValueError("path traversal is not allowed")
    if normalized == ".." or any(part in ("", ".", "..") for part in normalized.split("/")):
        raise ValueError("invalid path")
    if normalized.startswith(".") or "/." in normalized:
        raise ValueError("hidden files are not editable")
    if normalized.startswith(DENIED_PREFIXES) or normalized.endswith(DENIED_SUFFIXES):
        raise ValueError("path is not editable")
    if normalized in ROOT_FILES or normalized.startswith(ROOT_DIRS):
        return normalized
    raise ValueError("path is outside the editable workspace")


def _workspace_prefix(workspace_key: str) -> str:
    return f"{workspace_key}/.hermes/"


def _object_key(workspace_key: str, path: str) -> str:
    return f"{_workspace_prefix(workspace_key)}{validate_workspace_path(path)}"


def _version_key(workspace_key: str) -> str:
    return f"{_workspace_prefix(workspace_key)}{VERSION_MARKER}"


def _touch_version(s3: Any, bucket: str, workspace_key: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=_version_key(workspace_key),
        Body=str(int(time.time() * 1000)).encode("ascii"),
        ContentType="text/plain; charset=utf-8",
    )


def list_workspace_files(s3: Any, bucket: str, workspace_key: str) -> list[dict[str, Any]]:
    prefix = _workspace_prefix(workspace_key)
    paginator = s3.get_paginator("list_objects_v2")
    files: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            path = key[len(prefix):]
            if not path or path == VERSION_MARKER:
                continue
            try:
                validate_workspace_path(path)
            except ValueError:
                continue
            files.append({
                "path": path,
                "size": int(obj.get("Size", 0)),
                "updatedAt": obj.get("LastModified").isoformat() if obj.get("LastModified") else "",
            })
    return sorted(files, key=lambda item: item["path"])


def get_workspace_file(s3: Any, bucket: str, workspace_key: str, path: str) -> dict[str, Any]:
    key = _object_key(workspace_key, path)
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    if len(body) > MAX_TEXT_BYTES:
        raise ValueError("file is too large")
    try:
        content = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("file is not UTF-8 text") from exc
    return {"path": validate_workspace_path(path), "content": content}


def put_workspace_file(
    s3: Any,
    bucket: str,
    workspace_key: str,
    path: str,
    content: str,
) -> dict[str, Any]:
    normalized = validate_workspace_path(path)
    raw = content.encode("utf-8")
    if len(raw) > MAX_TEXT_BYTES:
        raise ValueError("file is too large")
    if "\x00" in content:
        raise ValueError("binary content is not allowed")
    s3.put_object(
        Bucket=bucket,
        Key=f"{_workspace_prefix(workspace_key)}{normalized}",
        Body=raw,
        ContentType="text/plain; charset=utf-8",
    )
    _touch_version(s3, bucket, workspace_key)
    return {"path": normalized, "size": len(raw)}


def delete_workspace_file(s3: Any, bucket: str, workspace_key: str, path: str) -> dict[str, Any]:
    normalized = validate_workspace_path(path)
    s3.delete_object(Bucket=bucket, Key=f"{_workspace_prefix(workspace_key)}{normalized}")
    _touch_version(s3, bucket, workspace_key)
    return {"path": normalized, "deleted": True}
