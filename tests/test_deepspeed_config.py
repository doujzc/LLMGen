from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from llmgen.router import RouterDataError
from scripts import train_router
from scripts.train_router import (
    SUPPORTED_DEEPSPEED_VERSION,
    _gradient_checkpointing_kwargs,
    _read_deepspeed_config,
    _require_supported_deepspeed_version,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "offloads_parameters"),
    (
        ("deepspeed_zero3.json", False),
        ("deepspeed_zero3_offload.json", True),
    ),
)
def test_deepspeed_configs_are_zero3_and_trainer_synchronized(
    name, offloads_parameters
):
    path, payload, stage = _read_deepspeed_config(ROOT / "configs" / name)
    zero = payload["zero_optimization"]

    assert path.is_absolute()
    assert stage == 3
    assert zero["stage3_gather_16bit_weights_on_model_save"] is True
    assert (zero.get("offload_param", {}).get("device") == "cpu") is offloads_parameters
    assert payload["bf16"]["enabled"] == "auto"
    assert payload["fp16"]["enabled"] == "auto"
    assert payload["train_batch_size"] == "auto"
    assert payload["train_micro_batch_size_per_gpu"] == "auto"
    assert payload["gradient_accumulation_steps"] == "auto"


def test_deepspeed_config_requires_a_valid_zero_stage(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"zero_optimization": {"stage": "3"}}))

    with pytest.raises(RouterDataError, match="zero_optimization.stage"):
        _read_deepspeed_config(path)


def test_deepspeed_version_guard_accepts_pinned_version(monkeypatch):
    monkeypatch.setattr(
        train_router.importlib.metadata,
        "version",
        lambda package: SUPPORTED_DEEPSPEED_VERSION,
    )

    assert _require_supported_deepspeed_version() == SUPPORTED_DEEPSPEED_VERSION


def test_deepspeed_version_guard_explains_modules_to_save_regression(monkeypatch):
    monkeypatch.setattr(
        train_router.importlib.metadata, "version", lambda package: "0.19.1"
    )

    with pytest.raises(RouterDataError, match="PEFT modules_to_save") as exc_info:
        _require_supported_deepspeed_version()

    assert f"deepspeed=={SUPPORTED_DEEPSPEED_VERSION}" in str(exc_info.value)


@pytest.mark.parametrize(
    ("enabled", "deepspeed", "mode", "expected"),
    (
        (False, "config.json", "auto", None),
        (True, "config.json", "auto", {"use_reentrant": True}),
        (True, None, "auto", {"use_reentrant": False}),
        (True, "config.json", "non-reentrant", {"use_reentrant": False}),
        (True, None, "reentrant", {"use_reentrant": True}),
    ),
)
def test_gradient_checkpointing_mode(enabled, deepspeed, mode, expected):
    args = Namespace(
        gradient_checkpointing=enabled,
        gradient_checkpointing_mode=mode,
        deepspeed=deepspeed,
    )

    assert _gradient_checkpointing_kwargs(args) == expected
