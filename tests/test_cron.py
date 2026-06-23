"""Tests for the Cron Lambda."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_cron_module():
    path = Path(__file__).resolve().parents[1] / "lambda" / "cron" / "index.py"
    spec = importlib.util.spec_from_file_location("cron_index", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workspace_session_id_is_scoped_by_workspace():
    cron = _load_cron_module()

    dm_session = cron._build_cron_session_id(
        "user_abc",
        "daily",
        "slack/T1/users/U1",
    )
    mpim_session = cron._build_cron_session_id(
        "user_abc",
        "daily",
        "slack/T1/mpim/G1",
    )

    assert len(dm_session) >= 33
    assert dm_session != mpim_session
    assert dm_session.startswith("workspace:")


def test_handler_passes_workspace_to_agentcore():
    with patch.dict(os.environ, {
        "AGENTCORE_RUNTIME_ARN": "arn:aws:bedrock:us-east-1:123456789:agent-runtime/hermes",
        "AGENTCORE_QUALIFIER": "production",
        "S3_BUCKET": "hermes-user-files",
    }):
        cron = _load_cron_module()

    payload_body = MagicMock()
    payload_body.read.return_value = b'{"response":"ok"}'
    client = MagicMock()
    client.invoke_agent_runtime.return_value = {"payload": payload_body}

    with patch.object(cron, "_agentcore", return_value=client):
        result = cron.handler({
            "jobId": "daily",
            "userId": "user_abc",
            "prompt": "summarize",
            "workspaceKey": "slack/T1/users/U1",
            "workspaceType": "slack-dm",
        }, None)

    assert result["status"] == "ok"
    call = client.invoke_agent_runtime.call_args.kwargs
    assert call["runtimeUserId"].startswith("cron:workspace:")
    assert '"workspaceKey": "slack/T1/users/U1"' in call["payload"]
    assert '"s3Bucket": "hermes-user-files"' in call["payload"]
