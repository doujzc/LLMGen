from __future__ import annotations

from pathlib import Path

import pytest

from llmgen.clawhub_dataset import (
    ApiConfig,
    ChatBatchClient,
    DatasetBuildError,
    load_api_config,
    parse_json_object,
)


def test_plain_api_key_uses_environment_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "provider_key.txt"
    path.write_text("secret-key\n", encoding="utf-8")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.test/")
    config = load_api_config(path, model="example-model")
    assert config.base_url == "https://api.example.test"
    assert config.api_key == "secret-key"
    assert config.model == "example-model"


def test_plain_api_key_requires_environment_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "provider_key.txt"
    path.write_text("secret-key\n", encoding="utf-8")
    monkeypatch.delenv("API_BASE_URL", raising=False)
    with pytest.raises(DatasetBuildError, match="base_url"):
        load_api_config(path, model="example-model")


def test_empty_api_config_uses_environment_endpoint_and_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "provider.conf"
    path.write_text("# supplied by pipeline environment\n", encoding="utf-8")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    config = load_api_config(path, model="example-model")

    assert config.base_url == "https://api.example.test/v1"
    assert config.api_key == "environment-secret"


def test_chat_batch_client_accepts_pipeline_timeout_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMGEN_CHAT_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLMGEN_CHAT_MAX_RETRIES", "2")
    client = ChatBatchClient(ApiConfig("https://example.test", "key", "model"))
    assert client.timeout == 12.5
    assert client.max_retries == 2


def test_json_object_parser_normalizes_top_level_object_array() -> None:
    assert parse_json_object('[{"query_id":"q1"},{"query_id":"q2"}]') == {
        "items": [{"query_id": "q1"}, {"query_id": "q2"}]
    }


def test_json_object_parser_rejects_scalar_array() -> None:
    with pytest.raises(DatasetBuildError, match="array of objects"):
        parse_json_object("[1, 2]")
