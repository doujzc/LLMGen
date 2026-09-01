"""Provider-to-subprocess adaptation and durable Provider-ledger reporting.

This module deliberately does not implement a second Provider client.  The
existing data and embedding scripts remain the algorithm boundary; these
helpers translate the typed Pipeline configuration into their secret-safe
arguments and environment.  Keeping this boundary independent of concrete
Stage modules lets data-generation stages share it without import cycles.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from .io import atomic_write_text
from .ledger import JsonlShardLedger

if TYPE_CHECKING:
    from .stages.base import ArtifactOutput


class _PipelineConfig(Protocol):
    """Subset of resolved config used by Provider adapters."""

    def require(self, path: str) -> Any:
        """Return one validated configuration value."""


class ProviderStageContext(Protocol):
    """Minimal Stage runtime contract needed for Provider adaptation."""

    config: _PipelineConfig
    attempt_dir: Path
    stage_dir: Path
    spec: Any

    def update_progress(self, **progress: Any) -> None:
        """Persist structured Stage progress."""


def provider_config(context: ProviderStageContext, name: str) -> dict[str, Any]:
    """Return one validated Provider configuration mapping."""

    value = context.config.require(f"providers.{name}")
    assert isinstance(value, dict)
    return value


def provider_api_config(context: ProviderStageContext, name: str) -> str:
    """Return the legacy API-config path without persisting credentials.

    Existing scripts require an API config *path*.  When no path is configured,
    an attempt-owned marker file preserves that calling convention while the
    endpoint and secret are supplied only through the child environment.
    """

    provider = provider_config(context, name)
    path = str(provider.get("api_config") or "").strip()
    if path:
        return str(Path(path).expanduser())

    if not str(provider.get("base_url") or "").strip():
        raise ValueError(
            f"providers.{name} requires base_url when api_config is not set"
        )
    key_environment = str(provider.get("api_key_env") or "").strip()
    if not key_environment:
        raise ValueError(
            f"providers.{name} requires api_key_env when api_config is not set"
        )
    if not os.environ.get(key_environment):
        raise ValueError(
            f"providers.{name}.api_key_env references unset environment "
            f"variable {key_environment}"
        )
    marker = context.attempt_dir / "provider" / f"{name}.conf"
    if not marker.exists():
        atomic_write_text(
            marker,
            "# endpoint and credential are supplied through the child environment\n",
            mode=0o600,
        )
    return str(marker)


def provider_environment(
    context: ProviderStageContext,
    name: str,
    *,
    operation: str,
) -> dict[str, str]:
    """Build the child environment and immutable-ledger location for one call."""

    provider = provider_config(context, name)
    environment: dict[str, str] = {}
    base_url = str(provider.get("base_url") or "").strip()
    if base_url:
        environment["API_BASE_URL"] = base_url
    key_environment = str(provider.get("api_key_env") or "").strip()
    if key_environment:
        secret = os.environ.get(key_environment)
        if secret:
            environment["OPENAI_API_KEY"] = secret
    timeout = provider.get("timeout_seconds")
    if timeout is not None:
        environment["LLMGEN_CHAT_TIMEOUT_SECONDS"] = str(float(timeout))
    retries = provider.get("max_retries")
    if retries is not None:
        environment["LLMGEN_CHAT_MAX_RETRIES"] = str(int(retries))
    safe_operation = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in operation
    ).strip("-")
    if not safe_operation:
        raise ValueError("provider ledger operation must be non-empty")
    environment["LLMGEN_LLM_LEDGER_ROOT"] = str(
        context.stage_dir / "ledger" / name / safe_operation
    )
    environment["LLMGEN_LLM_LEDGER_NAMESPACE"] = (
        f"{context.spec.name}:{name}:{safe_operation}"
    )
    environment["LLMGEN_LLM_LEDGER_BATCH_RECORDS"] = str(
        int(context.config.require("checkpointing.llm_batch_records"))
    )
    return environment


def ledger_outputs(
    context: ProviderStageContext,
    *provider_names: str,
) -> tuple[tuple[ArtifactOutput, ...], dict[str, Any]]:
    """Verify Provider ledger manifests and expose their existing artifacts.

    ``ArtifactOutput`` is imported only when this function executes.  That
    avoids importing the ``stages`` package while the latter is importing its
    adapters, so this shared module has no import-time dependency on concrete
    Stage modules.
    """

    # Runtime import avoids a providers -> stages package initialization cycle.
    from .stages.base import ArtifactOutput

    artifacts: list[ArtifactOutput] = []
    progress: dict[str, Any] = {}
    for provider_name in provider_names:
        root = context.stage_dir / "ledger" / provider_name
        if not root.is_dir():
            continue
        ledgers: dict[str, Any] = {}
        for manifest in sorted(root.rglob("manifest.json")):
            ledger_root = manifest.parent
            relative = ledger_root.relative_to(root).as_posix()
            batch_field = (
                "embedding_batch_records"
                if provider_name == "embedding"
                else "llm_batch_records"
            )
            verification = JsonlShardLedger(
                ledger_root,
                batch_size=int(context.config.require(f"checkpointing.{batch_field}")),
            ).verify()
            ledgers[relative] = verification["stats"]
        if not ledgers:
            raise ValueError(
                f"Provider ledger directory contains no verified ledger: {root}"
            )
        logical_provider = provider_name.replace("-", "_")
        artifacts.append(
            ArtifactOutput(
                f"ledger.{context.spec.name}.{logical_provider}",
                root,
                "provider_ledger/v1",
                metadata={"ledgers": ledgers},
            )
        )
        progress[provider_name] = ledgers
    if progress:
        context.update_progress(provider_ledgers=progress)
    return tuple(artifacts), progress


def workers(provider: Mapping[str, Any]) -> str:
    """Return the legacy scripts' decimal worker-count argument."""

    return str(int(provider.get("concurrency") or provider.get("workers") or 1))


__all__ = [
    "ProviderStageContext",
    "ledger_outputs",
    "provider_api_config",
    "provider_config",
    "provider_environment",
    "workers",
]
