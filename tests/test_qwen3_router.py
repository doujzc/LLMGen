from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from scripts.train_router import _load_training_stack


def test_qwen3_lora_preserves_resized_tied_embeddings(monkeypatch):
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    base_model = transformers.AutoModelForCausalLM.from_config(config)

    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"
        additional_special_tokens = []

        def __len__(self):
            return 34

        def add_special_tokens(self, payload):
            self.additional_special_tokens = payload["additional_special_tokens"]

        def encode(self, token, add_special_tokens=False):
            del add_special_tokens
            return {"<SK_L1_0>": [32], "<SK_L2_0>": [33]}[token]

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: base_model,
    )
    captured = {}
    original_get_peft_model = peft.get_peft_model

    def capture(model, lora_config):
        captured["config"] = lora_config
        captured["modules_to_save"] = set(lora_config.modules_to_save)
        captured["ensure_weight_tying"] = getattr(
            lora_config, "ensure_weight_tying", None
        )
        return original_get_peft_model(model, lora_config)

    monkeypatch.setattr(peft, "get_peft_model", capture)
    args = SimpleNamespace(
        adapter_name_or_path=None,
        model_name_or_path="Qwen/Qwen3-1.7B",
        trust_remote_code=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        lora=True,
        lora_modules_to_save="auto",
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
    )

    _, _, _, model, token_ids = _load_training_stack(
        args, ("<SK_L1_0>", "<SK_L2_0>")
    )

    assert token_ids == {"<SK_L1_0>": 32, "<SK_L2_0>": 33}
    assert any("lora_" in name for name, _ in model.named_parameters())
    assert captured["modules_to_save"] == {"embed_tokens", "lm_head"}
    if "ensure_weight_tying" in inspect.signature(peft.LoraConfig).parameters:
        assert captured["ensure_weight_tying"] is True


def test_incremental_lora_can_leave_existing_token_embeddings_frozen(monkeypatch):
    config = transformers.Qwen3Config(
        vocab_size=34,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    base_model = transformers.AutoModelForCausalLM.from_config(config)

    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"
        additional_special_tokens = ["<SK_L1_0>", "<SK_L2_0>"]

        def __len__(self):
            return 34

        def add_special_tokens(self, payload):
            self.additional_special_tokens = payload["additional_special_tokens"]

        def encode(self, token, add_special_tokens=False):
            del add_special_tokens
            return {"<SK_L1_0>": [32], "<SK_L2_0>": [33]}[token]

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: base_model,
    )
    captured = {}
    original_get_peft_model = peft.get_peft_model

    def capture(model, lora_config):
        captured["modules_to_save"] = lora_config.modules_to_save
        return original_get_peft_model(model, lora_config)

    monkeypatch.setattr(peft, "get_peft_model", capture)
    args = SimpleNamespace(
        adapter_name_or_path=None,
        model_name_or_path="trained-router",
        trust_remote_code=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        lora=True,
        lora_modules_to_save="none",
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
    )

    _, _, _, model, _ = _load_training_stack(
        args, ("<SK_L1_0>", "<SK_L2_0>")
    )

    assert captured["modules_to_save"] is None
    assert any("lora_" in name for name, _ in model.named_parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "embed_tokens" in name or "lm_head" in name
    )


def test_direct_candidate_name_lora_does_not_resize_or_save_embeddings(monkeypatch):
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    base_model = transformers.AutoModelForCausalLM.from_config(config)

    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __len__(self):
            return 32

        def add_special_tokens(self, payload):
            del payload
            raise AssertionError("direct routing must not add virtual tokens")

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: base_model,
    )
    monkeypatch.setattr(
        base_model,
        "resize_token_embeddings",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct routing must not resize embeddings")
        ),
    )
    captured = {}
    original_get_peft_model = peft.get_peft_model

    def capture(model, lora_config):
        captured["modules_to_save"] = lora_config.modules_to_save
        return original_get_peft_model(model, lora_config)

    monkeypatch.setattr(peft, "get_peft_model", capture)
    args = SimpleNamespace(
        adapter_name_or_path=None,
        model_name_or_path="Qwen/Qwen3-1.7B",
        trust_remote_code=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        lora=True,
        lora_modules_to_save="auto",
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
    )

    _, _, _, model, token_ids = _load_training_stack(args, ())

    assert token_ids == {}
    assert captured["modules_to_save"] is None
    assert any("lora_" in name for name, _ in model.named_parameters())
