from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmgen.router import RouterDataError
from scripts.train_router import _read_deepspeed_config


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
