from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from llmgen.direct_router import CURRENT_CONVERSATION_TEMPLATE
from llmgen.router import RouterDataError, read_jsonl, write_jsonl
from scripts import train_router
from scripts.top1.export_sft_data import export_sft_jsonl


ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return [100 + ord(character) for character in text]


def _row(target: str = "Ecommerce") -> dict:
    return {
        "id": {"ignored": True},
        "messages": [
            {"role": "system", "content": "untrusted source prompt"},
            {"role": "user", "content": "推荐一款耳机"},
            {"role": "assistant", "content": "预算是多少？"},
            {"role": "user", "content": "500 元以内"},
        ],
        "target_candidate_name": target,
        "scenario_family": "ignored",
    }


def test_export_writes_standard_messages_only_jsonl(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    destination = tmp_path / "train.sft.jsonl"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("route exactly one candidate", encoding="utf-8")
    write_jsonl(source, [_row(), {"query": "你好", "target": "ChitChat"}])

    report = export_sft_jsonl(
        input_jsonl=source,
        output_jsonl=destination,
        candidate_registry=ROOT / "configs/top1_candidates.json",
        system_prompt_file=prompt,
    )
    rows = read_jsonl(destination)

    assert report["rows"] == 2
    assert report["candidate_counts"] == {"Ecommerce": 1, "ChitChat": 1}
    assert report["token_length_fitted"] is False
    assert report["conversation_template"] == CURRENT_CONVERSATION_TEMPLATE
    assert [set(row) for row in rows] == [{"messages"}, {"messages"}]
    assert rows[0]["messages"][0] == {
        "role": "system",
        "content": "route exactly one candidate",
    }
    assert rows[0]["messages"][-1] == {
        "role": "assistant",
        "content": "Ecommerce",
    }
    assert rows[1]["messages"][-1]["content"] == "ChitChat"

    user_content = rows[0]["messages"][1]["content"]
    assert user_content.startswith("<conversation_json>")
    assert '"history":[{"role":"user","content":"推荐一款耳机"}' in user_content
    assert '"current_user_request":"500 元以内"' in user_content
    assert "<contextualize>" in user_content
    assert user_content.endswith("仅输出候选名称：")
    assert "untrusted source prompt" not in user_content


def test_export_rejects_unknown_candidate_with_source_location(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    destination = tmp_path / "train.sft.jsonl"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("route", encoding="utf-8")
    write_jsonl(source, [_row("NotRegistered")])

    with pytest.raises(RouterDataError, match=r"train\.jsonl:1: unknown target"):
        export_sft_jsonl(
            input_jsonl=source,
            output_jsonl=destination,
            candidate_registry=ROOT / "configs/top1_candidates.json",
            system_prompt_file=prompt,
        )


def test_tokenizer_aware_export_applies_router_length_fitting(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    destination = tmp_path / "train.sft.jsonl"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("route", encoding="utf-8")
    row = _row()
    row["messages"][1]["content"] = "very old context " * 30
    write_jsonl(source, [row])

    report = export_sft_jsonl(
        input_jsonl=source,
        output_jsonl=destination,
        candidate_registry=ROOT / "configs/top1_candidates.json",
        system_prompt_file=prompt,
        tokenizer=CharacterTokenizer(),
        max_length=512,
    )
    user_content = read_jsonl(destination)[0]["messages"][1]["content"]

    assert report["token_length_fitted"] is True
    assert report["max_length"] == 512
    assert "500 元以内" in user_content
    assert "very old context" not in user_content


def test_export_shell_uses_user_paths_without_loading_a_tokenizer(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    run_dir = tmp_path / "run"
    destination = run_dir / "router" / "retrieval" / "sft_input.jsonl"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("route", encoding="utf-8")
    write_jsonl(source, [_row()])
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON": sys.executable,
            "TOP1_TRAIN_DATA": str(source),
            "TOP1_RUN_DIR": str(run_dir),
            "TOP1_CANDIDATE_REGISTRY": str(
                (ROOT / "configs/top1_candidates.json").resolve()
            ),
            "TOP1_SYSTEM_PROMPT": str(prompt),
        }
    )

    completed = subprocess.run(
        ["bash", "scripts/top1/export_sft.sh"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout[completed.stdout.index("{") :])["rows"] == 1
    assert read_jsonl(destination)[0]["messages"][-1]["content"] == "Ecommerce"


def test_direct_training_automatically_dumps_sft_input(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    write_jsonl(source, [_row()])
    destination = tmp_path / "router" / "retrieval" / "sft_input.jsonl"

    class StopAfterExport(Exception):
        pass

    class Trainer:
        def __init__(self, **kwargs):
            del kwargs
            self.args = SimpleNamespace(world_size=1)
            self.accelerator = SimpleNamespace(wait_for_everyone=lambda: None)

        def is_world_process_zero(self):
            return True

        def train(self, **kwargs):
            del kwargs
            assert destination.is_file()
            raise StopAfterExport

    args = Namespace(
        seed=42,
        local_rank=-1,
        routing_mode="candidate_name_top1",
        num_levels=None,
        max_length=1024,
        output_dir=str(tmp_path / "router"),
    )

    with pytest.raises(StopAfterExport):
        train_router._run_phase(
            phase="retrieval",
            train_path=str(source),
            validation_path=None,
            system_prompt="route",
            epochs=1.0,
            learning_rate=1e-5,
            resume_from_checkpoint=None,
            args=args,
            torch=torch,
            transformers=SimpleNamespace(Trainer=Trainer),
            tokenizer=CharacterTokenizer(),
            model=object(),
            token_ids={},
            candidate_names=("Ecommerce", "StockQuery"),
            training_args=object(),
        )

    rows = read_jsonl(destination)
    assert list(rows[0]) == ["messages"]
    assert rows[0]["messages"][-1] == {
        "role": "assistant",
        "content": "Ecommerce",
    }
