from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

from scripts import merge_router_adapter


def test_merge_router_adapter_writes_full_portable_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    for name in (
        "router_manifest.json",
        "skill_decode_map.json",
        "virtual_tokens.txt",
    ):
        (adapter / name).write_text(name, encoding="utf-8")
    output = tmp_path / "merged"
    calls: dict[str, object] = {}

    class FakeBase:
        def resize_token_embeddings(self, size):
            calls["vocab_size"] = size

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls["base_model"] = model
            calls["model_kwargs"] = kwargs
            return FakeBase()

    class FakeMerged:
        def save_pretrained(self, destination, *, safe_serialization):
            assert safe_serialization is True
            destination = Path(destination)
            (destination / "config.json").write_text("{}", encoding="utf-8")
            (destination / "model.safetensors").write_bytes(b"full-weights")

    class FakePeft:
        def merge_and_unload(self, *, safe_merge):
            assert safe_merge is True
            return FakeMerged()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, source):
            calls["adapter"] = source
            return FakePeft()

    class FakeTokenizer:
        def __len__(self):
            return 37

        def save_pretrained(self, destination):
            (Path(destination) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["tokenizer_source"] = str(source)
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(PeftModel=FakePeftModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_router_adapter.py",
            "--base-model",
            "base/model",
            "--adapter",
            str(adapter),
            "--output-dir",
            str(output),
        ],
    )

    merge_router_adapter.main()

    assert calls["base_model"] == "base/model"
    assert calls["adapter"] == str(adapter.resolve())
    assert calls["tokenizer_source"] == str(adapter.resolve())
    assert calls["vocab_size"] == 37
    assert (output / "model.safetensors").read_bytes() == b"full-weights"
    assert (output / "skill_decode_map.json").is_file()
    assert not (output / "adapter_config.json").exists()
