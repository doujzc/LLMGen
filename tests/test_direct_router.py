from __future__ import annotations

from argparse import Namespace
import json
import sys

import pytest

from scripts import train_router
from llmgen.direct_router import (
    CURRENT_CONVERSATION_TEMPLATE,
    LEGACY_CONVERSATION_TEMPLATE,
    CandidateNameTokenTrie,
    build_conversation_user_prompt,
    candidate_token_sequences,
    encode_candidate_name_example,
    fit_candidate_router_prompt,
    load_candidate_registry,
    normalize_conversation_messages,
)
from llmgen.router import RouterDataError


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return [100 + ord(character) for character in text]


def test_top1_registry_has_two_real_and_five_virtual_candidates() -> None:
    routes = load_candidate_registry("configs/top1_candidates.json")

    assert [route.name for route in routes] == [
        "StockAdvice",
        "StockOther",
        "StockQuery",
        "ProductOther",
        "Ecommerce",
        "ChitChat",
        "NoAvailable",
    ]
    assert [route.name for route in routes if route.virtual] == [
        "StockAdvice",
        "StockOther",
        "ProductOther",
        "ChitChat",
        "NoAvailable",
    ]


def test_conversation_normalization_drops_source_system_and_preserves_latest_user() -> None:
    messages = normalize_conversation_messages(
        [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "x" * 20},
            {"role": "user", "content": "current request"},
        ],
        max_history_messages=3,
        max_history_chars=30,
        max_assistant_history_chars=4,
    )

    assert messages[-1] == {"role": "user", "content": "current request"}
    assert all(message["role"] != "system" for message in messages)
    assert sum(len(message["content"]) for message in messages) <= 30


def test_conversation_must_end_with_user() -> None:
    with pytest.raises(RouterDataError, match="final non-system"):
        normalize_conversation_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )


def test_current_template_requests_latent_standalone_rewrite() -> None:
    prompt = build_conversation_user_prompt(
        [
            {"role": "user", "content": "推荐耳机"},
            {"role": "assistant", "content": "预算是多少？"},
            {"role": "user", "content": "500 元以内"},
        ]
    )

    assert "<contextualize>" in prompt
    assert "还原为可独立理解的请求" in prompt
    assert "若当前请求完整或已切换目标，忽略 history" in prompt
    assert prompt.endswith("仅输出候选名称：")


def test_legacy_template_reproduces_original_prompt() -> None:
    prompt = build_conversation_user_prompt(
        [{"role": "user", "content": "查询贵州茅台"}],
        conversation_template=LEGACY_CONVERSATION_TEMPLATE,
    )

    assert prompt == (
        '<conversation_json>{"current_user_request":"查询贵州茅台"}'
        "</conversation_json>\n输出候选名称："
    )
    assert CURRENT_CONVERSATION_TEMPLATE not in prompt


def test_unknown_conversation_template_is_rejected() -> None:
    with pytest.raises(RouterDataError, match="unsupported conversation template"):
        build_conversation_user_prompt(
            [{"role": "user", "content": "hello"}],
            conversation_template="unknown-v9",
        )


def test_prompt_fitting_drops_old_history_before_latest_request() -> None:
    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "user", "content": "very old " * 30},
        {"role": "assistant", "content": "old answer " * 30},
        {"role": "user", "content": "最新请求"},
    ]

    prompt, kept = fit_candidate_router_prompt(
        tokenizer,
        messages,
        "route",
        max_prompt_tokens=420,
    )

    assert kept == ({"role": "user", "content": "最新请求"},)
    assert "最新请求" in prompt
    assert "very old" not in prompt


def test_direct_target_loss_covers_only_candidate_name_and_eos() -> None:
    tokenizer = CharacterTokenizer()
    row = {
        "messages": [
            {"role": "user", "content": "查一下贵州茅台股价"},
        ],
        "target_candidate_name": "StockQuery",
    }

    encoded = encode_candidate_name_example(
        tokenizer,
        row,
        candidate_names=("StockQuery", "Ecommerce"),
        max_length=512,
        system_prompt="route",
    )

    target = tokenizer.encode("StockQuery") + [tokenizer.eos_token_id]
    assert encoded["labels"][-len(target) :] == target
    assert all(value == -100 for value in encoded["labels"][: -len(target)])
    assert encoded["input_ids"][-len(target) :] == target


def test_candidate_name_trie_supports_variable_length_shared_prefixes() -> None:
    trie = CandidateNameTokenTrie(
        {
            "StockAdvice": (10, 20),
            "StockOther": (10, 21),
            "Ecommerce": (11,),
        },
        eos_token_id=2,
    )

    assert trie.allowed_next(()) == (10, 11)
    assert trie.allowed_next((10,)) == (20, 21)
    assert trie.allowed_next((11,)) == (2,)
    assert trie.resolve((10, 21)) == "StockOther"
    assert trie.max_name_tokens == 2


def test_tokenizer_cannot_alias_two_candidate_names() -> None:
    class AliasedTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False, **kwargs):
            del text, add_special_tokens, kwargs
            return [10]

    with pytest.raises(RouterDataError, match="share one token sequence"):
        candidate_token_sequences(AliasedTokenizer(), ("A", "B"))


def test_registry_rejects_virtual_candidate_with_backend_intent(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "routing_mode": "candidate_name_top1",
                "candidates": [
                    {
                        "name": "Bad",
                        "candidate_id": "bad",
                        "intent_label": "Backend",
                        "virtual": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RouterDataError, match="exactly when non-virtual"):
        load_candidate_registry(path)


def test_training_accepts_prompt_without_candidate_names(tmp_path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("只选择最匹配的候选名称。", encoding="utf-8")

    def stop_before_model_loading(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("reached training stack")

    monkeypatch.setattr(train_router, "_load_training_stack", stop_before_model_loading)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_router.py",
            "--model-name-or-path",
            "unused-model",
            "--routing-mode",
            "candidate_name_top1",
            "--candidate-registry",
            "configs/top1_candidates.json",
            "--retrieval-system-prompt-file",
            str(prompt_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--stage",
            "retrieval",
        ],
    )

    with pytest.raises(RuntimeError, match="reached training stack"):
        train_router.main()


def test_direct_training_phase_allows_partial_candidate_coverage(
    tmp_path, monkeypatch
) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "查一下股票价格"}],
                "target_candidate_name": "StockQuery",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class StopAfterValidation(Exception):
        pass

    def stop_before_dataset(*args, **kwargs):
        del args, kwargs
        raise StopAfterValidation

    args = type(
        "Args",
        (),
        {
            "seed": 42,
            "local_rank": -1,
            "routing_mode": "candidate_name_top1",
            "num_levels": None,
            "max_length": 128,
        },
    )()
    monkeypatch.setattr(train_router, "_dataset_class", stop_before_dataset)

    with pytest.raises(StopAfterValidation):
        train_router._run_phase(
            phase="retrieval",
            train_path=str(train_path),
            validation_path=None,
            system_prompt="route",
            epochs=1.0,
            learning_rate=1e-5,
            resume_from_checkpoint=None,
            args=args,
            torch=object(),
            transformers=object(),
            tokenizer=object(),
            model=object(),
            token_ids={},
            candidate_names=("StockQuery", "Ecommerce"),
        )


def test_training_arguments_use_memory_safe_loss_only_evaluation(tmp_path) -> None:
    captured = {}

    class TrainingArguments:
        def __init__(
            self,
            output_dir,
            per_device_eval_batch_size,
            eval_accumulation_steps,
            prediction_loss_only,
            evaluation_strategy,
        ):
            captured.update(locals())

    transformers = type(
        "Transformers", (), {"TrainingArguments": TrainingArguments}
    )()
    args = Namespace(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=5,
        save_steps=25,
        eval_steps=25,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_mode="auto",
        dataloader_num_workers=4,
        deepspeed="configs/deepspeed_zero3.json",
        local_rank=-1,
        seed=42,
    )

    train_router._build_training_arguments(
        phase="retrieval",
        has_validation=True,
        epochs=3.0,
        learning_rate=1e-5,
        resume_from_checkpoint=None,
        args=args,
        transformers=transformers,
    )

    assert captured["per_device_eval_batch_size"] == 1
    assert captured["eval_accumulation_steps"] == 1
    assert captured["prediction_loss_only"] is True
    assert captured["evaluation_strategy"] == "steps"
