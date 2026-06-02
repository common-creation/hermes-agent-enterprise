"""Tests for the Router Lambda."""

from __future__ import annotations

import json
import os
import sys
import time
import hmac
import hashlib
import urllib.parse
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda", "router"))

# Patch boto3 before importing the module.
mock_dynamodb_resource = MagicMock()
mock_table = MagicMock()
mock_dynamodb_resource.Table.return_value = mock_table


@pytest.fixture(autouse=True)
def _setup_env():
    """Set required environment variables."""
    with patch.dict(os.environ, {
        "AGENTCORE_RUNTIME_ARN": "arn:aws:bedrock:us-east-1:123456789:agent-runtime/hermes",
        "AGENTCORE_QUALIFIER": "production",
        "IDENTITY_TABLE": "hermes-identity",
        "S3_BUCKET": "hermes-user-files",
        "WORKSPACE_UI_SIGNING_KEY": "test-signing-key",
    }):
        yield


# --------------------------------------------------------------------------
# Tests — helper functions
# --------------------------------------------------------------------------

def test_build_session_id():
    """Session IDs must be >= 33 characters."""
    # Import after env is set.
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import _build_session_id

    session_id = _build_session_id("user_abc123", "telegram")
    assert len(session_id) >= 33
    assert "user_abc123" in session_id
    assert "telegram" in session_id


def test_split_message():
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import _split_message

    # Short message — single chunk.
    chunks = _split_message("hello", max_len=4096)
    assert chunks == ["hello"]

    # Long message — should split.
    long_msg = "x" * 5000
    chunks = _split_message(long_msg, max_len=4096)
    assert len(chunks) == 2
    assert len(chunks[0]) <= 4096
    assert "".join(chunks) == long_msg


def test_split_message_on_newline():
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import _split_message

    # Should prefer splitting on newlines.
    msg = "line1\n" + "x" * 4090 + "\nline3"
    chunks = _split_message(msg, max_len=4096)
    assert len(chunks) >= 2
    assert chunks[0].endswith("line1")  # Split at the first newline within limit.


def test_parse_body_json():
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import _parse_body

    event = {"body": '{"key": "value"}', "isBase64Encoded": False}
    result = _parse_body(event)
    assert result == {"key": "value"}


def test_parse_body_base64():
    import base64
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import _parse_body

    encoded = base64.b64encode(b'{"key": "value"}').decode()
    event = {"body": encoded, "isBase64Encoded": True}
    result = _parse_body(event)
    assert result == {"key": "value"}


# --------------------------------------------------------------------------
# Tests — handler routing
# --------------------------------------------------------------------------

def test_health_endpoint():
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import handler

    event = {
        "rawPath": "/health",
        "requestContext": {"http": {"method": "GET"}},
    }
    result = handler(event, None)
    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["status"] == "healthy"


def test_unknown_path():
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import handler

    event = {
        "rawPath": "/unknown",
        "requestContext": {"http": {"method": "GET"}},
    }
    result = handler(event, None)
    assert result["statusCode"] == 404


def test_telegram_empty_update():
    """Telegram updates without a message should be ignored."""
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import handler

    event = {
        "rawPath": "/webhook/telegram",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({"update_id": 123}),
        "isBase64Encoded": False,
    }
    result = handler(event, None)
    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["status"] == "ignored"


def test_slack_url_verification():
    """Slack URL verification challenge should be echoed back."""
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import handler

    event = {
        "rawPath": "/webhook/slack",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({
            "type": "url_verification",
            "challenge": "test_challenge_123",
        }),
        "isBase64Encoded": False,
    }
    result = handler(event, None)
    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["challenge"] == "test_challenge_123"


def test_slack_workspace_resolver_dm_mpim_public_private():
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from index import _resolve_slack_workspace

    assert _resolve_slack_workspace("T1", "D1", "U1", "im") == {
        "workspaceKey": "slack/T1/users/U1",
        "workspaceType": "slack-dm",
    }
    assert _resolve_slack_workspace("T1", "G1", "U1", "mpim") == {
        "workspaceKey": "slack/T1/mpim/G1",
        "workspaceType": "slack-mpim",
    }
    assert _resolve_slack_workspace("T1", "C1", "U1", "channel") == {
        "workspaceKey": "slack/T1/channels/public-shared",
        "workspaceType": "slack-public-shared",
    }
    assert _resolve_slack_workspace("T1", "G2", "U1", "group") == {
        "workspaceKey": "slack/T1/private/G2",
        "workspaceType": "slack-private-channel",
    }


def test_workspace_token_roundtrip_and_tamper_rejected():
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from workspace_auth import sign_workspace_token, verify_workspace_token

    token = sign_workspace_token(
        {"workspaceKey": "slack/T1/users/U1", "scope": ["workspace:read"]},
        "secret",
        ttl_seconds=60,
    )
    assert verify_workspace_token(token, "secret")["workspaceKey"] == "slack/T1/users/U1"

    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(ValueError):
        verify_workspace_token(tampered, "secret")


def test_workspace_token_expiry_rejected():
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"):
        from workspace_auth import sign_workspace_token, verify_workspace_token

    token = sign_workspace_token({"workspaceKey": "w", "scope": []}, "secret", ttl_seconds=1)
    with pytest.raises(ValueError):
        verify_workspace_token(token, "secret", now=int(time.time()) + 5)


def test_setting_ui_slash_command_bypasses_agentcore():
    body = urllib.parse.urlencode({
        "team_id": "T1",
        "channel_id": "D1",
        "user_id": "U1",
        "command": "/setting-ui",
    })
    timestamp = str(int(time.time()))
    signature = "v0=" + hmac.new(
        b"signing-secret",
        f"v0:{timestamp}:{body}".encode(),
        hashlib.sha256,
    ).hexdigest()
    event = {
        "rawPath": "/slack/commands/setting-ui",
        "headers": {
            "host": "example.com",
            "x-slack-request-timestamp": timestamp,
            "x-slack-signature": signature,
        },
        "requestContext": {"http": {"method": "POST"}},
        "body": body,
        "isBase64Encoded": False,
    }
    with patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("boto3.client"), \
         patch("index._get_secret", return_value="signing-secret"), \
         patch("index._is_allowed", return_value=True), \
         patch("index._fetch_slack_channel_type", return_value="im"), \
         patch("index._invoke_agentcore") as invoke_agentcore:
        from index import handler

        result = handler(event, None)

    assert result["statusCode"] == 200
    response = json.loads(result["body"])
    assert response["response_type"] == "ephemeral"
    assert "/ui?token=" in response["text"]
    invoke_agentcore.assert_not_called()
