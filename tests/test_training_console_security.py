from __future__ import annotations

from pathlib import Path
import os

import pytest

from training_console.config import ConfigResolver, ConfigValidationError
from training_console.store import snapshot_runtime_environment


REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_env() -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
    }


def test_runtime_snapshot_filters_sensitive_prefixed_names() -> None:
    environment = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NCCL_DEBUG": "INFO",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_API_KEY": "cuda-secret",
        "CUDA_apiKey": "camel-secret",
        "NCCL_AUTH_TOKEN": "nccl-secret",
        "NCCL_PASSWORD_FILE": "/tmp/password",
        "NVIDIA_CLIENT_SECRET": "nvidia-secret",
        "NVIDIA_SIGNING_KEY": "signing-secret",
        "NVIDIA_ACCESSTOKEN": "compact-access-token",
        "CUDA_CLIENTSECRET": "compact-client-secret",
        "NCCL_AUTHTOKEN": "compact-auth-token",
        "CUDA_TOKENVALUE": "compact-token-value",
        "NCCL_SECRETVALUE": "compact-secret-value",
        "NVIDIA_SESSIONTOKEN": "session-token",
        "CUDA_IDTOKEN": "id-token",
        "NCCL_WEBHOOKTOKEN": "webhook-token",
        "NVIDIA_WEBHOOKSECRET": "webhook-secret",
        "CUDA_SHAREDSECRET": "shared-secret",
        "NCCL_MASTERKEY": "master-key",
        "NVIDIA_AUTHKEY": "auth-key",
        "CUDA_AUTHORIZATIONTOKEN": "authorization-token",
        "NCCL_SASTOKEN": "sas-token",
    }

    captured = snapshot_runtime_environment(environment)

    assert captured == {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NCCL_DEBUG": "INFO",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "TOKENIZERS_PARALLELISM": "false",
    }
    assert not {
        "cuda-secret",
        "camel-secret",
        "nccl-secret",
        "/tmp/password",
        "nvidia-secret",
        "signing-secret",
        "compact-access-token",
        "compact-client-secret",
        "compact-auth-token",
        "compact-token-value",
        "compact-secret-value",
        "session-token",
        "id-token",
        "webhook-token",
        "webhook-secret",
        "shared-secret",
        "master-key",
        "auth-key",
        "authorization-token",
        "sas-token",
    } & set(captured.values())


@pytest.mark.parametrize(
    "url",
    [
        "https://user@embedding.example/v1",
        "https://user:password@embedding.example/v1",
        "https://embedding.example/v1?api_key=secret",
        "https://embedding.example/v1?api%5Fkey=secret",
        "https://embedding.example/v1?accessToken=secret",
        "https://embedding.example/v1?client_secret=secret",
        "https://embedding.example/v1?password=secret",
        "https://embedding.example/v1?X-Amz-Credential=secret",
        "https://embedding.example/v1?X-Amz-Signature=secret",
        "https://embedding.example/v1?accesstoken=secret",
        "https://embedding.example/v1?clientsecret=secret",
        "https://embedding.example/v1?authtoken=secret",
        "https://embedding.example/v1?sessiontokenvalue=secret",
        "https://embedding.example/v1?secretvalue=secret",
        "https://embedding.example/v1?sessiontoken=secret",
        "https://embedding.example/v1?idtoken=secret",
        "https://embedding.example/v1?webhooktoken=secret",
        "https://embedding.example/v1?webhooksecret=secret",
        "https://embedding.example/v1?sharedsecret=secret",
        "https://embedding.example/v1?masterkey=secret",
        "https://embedding.example/v1?authkey=secret",
        "https://embedding.example/v1?authorizationtoken=secret",
        "https://embedding.example/v1?sastoken=secret",
    ],
)
def test_url_configuration_rejects_embedded_credentials(url: str) -> None:
    resolver = ConfigResolver(REPO_ROOT, inherited_env=_clean_env())

    with pytest.raises(ConfigValidationError) as caught:
        resolver.validate(
            "clawhub",
            "full",
            {"EMBEDDING_BASE_URL": url},
        )

    assert any(
        error["field"] == "EMBEDDING_BASE_URL"
        for error in caught.value.errors
    )


def test_url_configuration_allows_non_sensitive_query_parameters() -> None:
    resolver = ConfigResolver(REPO_ROOT, inherited_env=_clean_env())
    url = (
        "https://embedding.example/v1"
        "?api-version=2026-07-01&deployment=qwen3"
    )

    validated = resolver.validate(
        "clawhub",
        "full",
        {"EMBEDDING_BASE_URL": url},
    )

    assert validated["resolved"]["EMBEDDING_BASE_URL"] == url
