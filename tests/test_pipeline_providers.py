"""Contracts for Provider-to-subprocess adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llmgen.pipeline import providers
from llmgen.pipeline.ledger import JsonlShardLedger
from llmgen.pipeline.stages import legacy


@dataclass
class _Config:
    values: dict[str, Any]

    def require(self, path: str) -> Any:
        return self.values[path]


@dataclass
class _Context:
    root: Path
    config: _Config
    spec: Any = field(default_factory=lambda: SimpleNamespace(name="generate-queries"))
    progress: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attempt_dir(self) -> Path:
        return self.root / "attempt"

    @property
    def stage_dir(self) -> Path:
        return self.root / "stage"

    def update_progress(self, **progress: Any) -> None:
        self.progress.append(progress)


def _context(tmp_path: Path) -> _Context:
    return _Context(
        tmp_path,
        _Config(
            {
                "providers.generation": {
                    "base_url": "https://provider.example/v1",
                    "api_key_env": "PIPELINE_TEST_PROVIDER_KEY",
                    "api_config": "",
                    "concurrency": 3,
                    "timeout_seconds": 12.5,
                    "max_retries": 4,
                },
                "checkpointing.llm_batch_records": 7,
                "checkpointing.embedding_batch_records": 11,
            }
        ),
    )


def test_legacy_provider_aliases_preserve_the_public_adapter_contract() -> None:
    assert legacy._provider is providers.provider_config
    assert legacy._provider_api_config is providers.provider_api_config
    assert legacy._provider_environment is providers.provider_environment
    assert legacy._ledger_outputs is providers.ledger_outputs
    assert legacy._workers is providers.workers


def test_provider_environment_and_secret_free_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path)
    monkeypatch.setenv("PIPELINE_TEST_PROVIDER_KEY", "not-persisted")

    marker = Path(providers.provider_api_config(context, "generation"))
    environment = providers.provider_environment(
        context,
        "generation",
        operation="generate alignment/round 1",
    )

    assert marker.read_text(encoding="utf-8") == (
        "# endpoint and credential are supplied through the child environment\n"
    )
    assert marker.stat().st_mode & 0o777 == 0o600
    assert environment == {
        "API_BASE_URL": "https://provider.example/v1",
        "OPENAI_API_KEY": "not-persisted",
        "LLMGEN_CHAT_TIMEOUT_SECONDS": "12.5",
        "LLMGEN_CHAT_MAX_RETRIES": "4",
        "LLMGEN_LLM_LEDGER_ROOT": str(
            context.stage_dir / "ledger" / "generation" / "generate-alignment-round-1"
        ),
        "LLMGEN_LLM_LEDGER_NAMESPACE": (
            "generate-queries:generation:generate-alignment-round-1"
        ),
        "LLMGEN_LLM_LEDGER_BATCH_RECORDS": "7",
    }
    assert providers.workers(providers.provider_config(context, "generation")) == "3"


def test_ledger_outputs_preserves_artifact_name_stats_and_progress(tmp_path: Path) -> None:
    context = _context(tmp_path)
    root = context.stage_dir / "ledger" / "generation" / "generate-alignment"
    JsonlShardLedger(root, batch_size=7).initialize()

    artifacts, progress = providers.ledger_outputs(context, "generation")

    assert len(artifacts) == 1
    assert artifacts[0].logical_name == "ledger.generate-queries.generation"
    assert artifacts[0].path == context.stage_dir / "ledger" / "generation"
    assert artifacts[0].artifact_schema == "provider_ledger/v1"
    assert artifacts[0].metadata == {"ledgers": progress["generation"]}
    assert context.progress == [{"provider_ledgers": progress}]


def test_provider_environment_rejects_an_empty_sanitized_operation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provider ledger operation"):
        providers.provider_environment(_context(tmp_path), "generation", operation="///")
