from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from llmgen.top1 import Top1DataError, read_jsonl, write_jsonl
from scripts import train_top1


def _training_args(root: Path) -> Namespace:
    train_data = root / "train.jsonl"
    prompt = root / "prompt.md"
    write_jsonl(
        train_data,
        [
            {
                "messages": [{"role": "user", "content": "查询股价"}],
                "target_candidate_name": "StockQuery",
            }
        ],
    )
    prompt.write_text("route", encoding="utf-8")
    return Namespace(
        model_name_or_path="Qwen/Qwen3-1.7B",
        train_data=str(train_data),
        validation_data=None,
        candidate_registry="configs/top1_candidates.json",
        system_prompt_file=str(prompt),
        output_dir=str(root / "output"),
        experiment_name="test-top1",
        run_id="run-001",
        max_length=1024,
        epochs=3.0,
        learning_rate=1e-5,
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
        dataloader_num_workers=4,
        seed=42,
        precision="bf16",
        deepspeed=None,
        local_rank=-1,
        gradient_checkpointing=True,
        gradient_checkpointing_mode="auto",
        trust_remote_code=False,
        resume_from_checkpoint=None,
        finetune_mode="full",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
    )


class Top1TrainingTests(unittest.TestCase):
    def test_training_arguments_keep_evaluation_loss_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
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

            args = _training_args(Path(temporary))
            train_top1._build_training_arguments(
                args,
                SimpleNamespace(TrainingArguments=TrainingArguments),
                has_validation=True,
                resume_from_checkpoint=None,
            )

            self.assertEqual(captured["per_device_eval_batch_size"], 1)
            self.assertEqual(captured["eval_accumulation_steps"], 1)
            self.assertIs(captured["prediction_loss_only"], True)
            self.assertEqual(captured["evaluation_strategy"], "steps")
            self.assertTrue(captured["output_dir"].endswith("output/checkpoints"))

    def test_full_training_never_resizes_token_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _training_args(Path(temporary))

            class Tokenizer:
                eos_token_id = 1
                pad_token_id = None
                eos_token = "<eos>"
                pad_token = None

            class Model:
                config = SimpleNamespace(use_cache=True)

                def resize_token_embeddings(self, *_args, **_kwargs):
                    raise AssertionError("Top1 training must not resize token embeddings")

            tokenizer = Tokenizer()
            model = Model()
            transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=lambda *args, **kwargs: tokenizer
                ),
                AutoModelForCausalLM=SimpleNamespace(
                    from_pretrained=lambda *args, **kwargs: model
                ),
            )
            torch = SimpleNamespace(bfloat16="bf16", float16="fp16")

            loaded_tokenizer, loaded_model = train_top1._load_model_and_tokenizer(
                args,
                torch,
                transformers,
            )

            self.assertIs(loaded_tokenizer, tokenizer)
            self.assertEqual(loaded_tokenizer.pad_token, tokenizer.eos_token)
            self.assertIs(loaded_model, model)
            self.assertIs(model.config.use_cache, False)

    def test_deepspeed_config_requires_numeric_zero_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "deepspeed.json"
            invalid.write_text(
                json.dumps({"zero_optimization": {"stage": "3"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Top1DataError, "zero_optimization.stage"):
                train_top1._read_deepspeed_config(invalid)

    def test_bundle_contains_only_top1_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _training_args(Path(temporary))
            output = Path(args.output_dir)
            model_output = output / "final" / "model"
            model_output.mkdir(parents=True)
            prepared = output / "prepared" / "train.sft.jsonl"
            write_jsonl(prepared, [{"messages": []}])

            class Tokenizer:
                def save_pretrained(self, destination):
                    Path(destination, "tokenizer.json").write_text(
                        "{}",
                        encoding="utf-8",
                    )

            model = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
            train_top1._write_bundle(
                args=args,
                output_dir=model_output,
                tokenizer=Tokenizer(),
                model=model,
                candidate_names=("StockQuery", "Ecommerce"),
                system_prompt="route",
                train_report={
                    "rows": 1,
                    "multi_turn_rows": 0,
                    "candidate_counts": {"StockQuery": 1},
                },
                validation_report=None,
                world_size=1,
                deepspeed_metadata=None,
                training_run_id="run-001",
                prepared_train_path=prepared,
            )

            manifest = json.loads((model_output / "router_manifest.json").read_text())
            self.assertEqual(manifest["routing_mode"], "candidate_name_top1")
            self.assertEqual(manifest["target"], "candidate_name_tokens_plus_eos")
            self.assertEqual(
                manifest["conversation"]["template"],
                "standalone_request_v2",
            )
            self.assertEqual(
                manifest["training"]["effective_global_batch_size"],
                16,
            )
            registry = json.loads((model_output / "candidate_registry.json").read_text())
            self.assertEqual(
                registry["candidates"],
                ["StockQuery", "Ecommerce"],
            )
            self.assertEqual(
                read_jsonl(prepared),
                [{"messages": []}],
            )


if __name__ == "__main__":
    unittest.main()
