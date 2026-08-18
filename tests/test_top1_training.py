from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from llmgen.evaluation import BackendDecisionPolicy, load_backend_decision_policy
from llmgen.top1 import Top1DataError, read_jsonl, write_jsonl
from scripts import evaluate_top1, train_top1
from test_top1_core import CharacterTokenizer


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
        memorization_data=None,
        memorization_steps=0,
        validation_data=None,
        candidate_registry="configs/top1_candidates.json",
        decision_policy="configs/top1_decision_policy.json",
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
                    num_train_epochs,
                    train_sampling_strategy=None,
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
            self.assertEqual(captured["num_train_epochs"], 3.0)
            self.assertTrue(captured["output_dir"].endswith("output/checkpoints"))

            train_top1._build_training_arguments(
                args,
                SimpleNamespace(TrainingArguments=TrainingArguments),
                has_validation=True,
                resume_from_checkpoint=None,
                staged_training=True,
            )
            self.assertEqual(captured["num_train_epochs"], 1.0)
            self.assertEqual(captured["train_sampling_strategy"], "sequential")

    def test_memorization_schedule_precedes_main_training(self) -> None:
        memorization = [
            {"input_ids": [100 + index], "attention_mask": [1], "labels": [1]}
            for index in range(3)
        ]
        main = [
            {"input_ids": [200 + index], "attention_mask": [1], "labels": [1]}
            for index in range(2)
        ]

        scheduled, metadata = train_top1._build_training_schedule(
            memorization,
            main,
            memorization_steps=3,
            main_epochs=2.5,
            effective_global_batch_size=4,
            seed=42,
        )
        scheduled_again, metadata_again = train_top1._build_training_schedule(
            memorization,
            main,
            memorization_steps=3,
            main_epochs=2.5,
            effective_global_batch_size=4,
            seed=42,
        )

        self.assertEqual(len(scheduled), 17)
        self.assertTrue(all(row in memorization for row in scheduled[:12]))
        self.assertTrue(all(row in main for row in scheduled[12:]))
        self.assertEqual(metadata["memorization"]["optimizer_steps"], 3)
        self.assertEqual(metadata["main"]["scheduled_samples"], 5)
        self.assertEqual(metadata["total"]["expected_optimizer_steps"], 5)
        self.assertEqual(scheduled, scheduled_again)
        self.assertEqual(metadata["order_sha256"], metadata_again["order_sha256"])

    def test_history_uses_main_epoch_and_preserves_trainer_epoch(self) -> None:
        history = train_top1._annotate_history_stages(
            [
                {"step": 2, "epoch": 0.25, "loss": 1.0},
                {"step": 5, "epoch": 0.625, "loss": 0.5},
                {"step": 8, "epoch": 1.0, "loss": 0.25},
            ],
            memorization_steps=2,
            main_epochs=3.0,
            total_steps=8,
        )

        self.assertEqual(history[0]["stage"], "memorization")
        self.assertEqual(history[0]["epoch"], 0.0)
        self.assertEqual(history[0]["trainer_epoch"], 0.25)
        self.assertEqual(history[1]["stage"], "main")
        self.assertEqual(history[1]["epoch"], 1.5)
        self.assertEqual(history[2]["epoch"], 3.0)

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

            tokenizer = Tokenizer()
            model = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
            decision_policy = BackendDecisionPolicy(
                candidate_to_backend={
                    "StockQuery": "StockQuery",
                    "EcommerceProduct": "NoAvailable",
                },
                backend_labels=("StockQuery", "NoAvailable"),
                fallback_backend_label="NoAvailable",
            )
            train_top1._write_bundle(
                args=args,
                output_dir=model_output,
                tokenizer=tokenizer,
                model=model,
                candidate_names=("StockQuery", "EcommerceProduct"),
                candidate_tokens={
                    "StockQuery": (1, 2),
                    "EcommerceProduct": (3, 4),
                },
                decision_policy=decision_policy,
                system_prompt="route",
                train_report={
                    "rows": 1,
                    "multi_turn_rows": 0,
                    "candidate_counts": {"StockQuery": 1},
                },
                memorization_report=None,
                validation_report=None,
                world_size=1,
                deepspeed_metadata=None,
                training_run_id="run-001",
                prepared_train_path=prepared,
                prepared_memorization_path=None,
                training_schedule=None,
                transformers_version="5.5.4",
            )

            manifest = json.loads((model_output / "router_manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(manifest["routing_mode"], "candidate_name_top1")
            self.assertEqual(manifest["target"], "candidate_name_tokens_plus_eos")
            self.assertEqual(
                manifest["conversation"]["template"],
                "routing_envelope_markdown_v1",
            )
            self.assertEqual(
                manifest["inference"]["decision_rule"],
                "selected_route_threshold_v1",
            )
            self.assertEqual(
                manifest["training"]["effective_global_batch_size"],
                16,
            )
            registry = json.loads((model_output / "candidate_registry.json").read_text())
            self.assertEqual(
                registry["candidates"],
                ["StockQuery", "EcommerceProduct"],
            )
            bundled_policy = load_backend_decision_policy(
                model_output / "decision_policy.json",
                ("StockQuery", "EcommerceProduct"),
            )
            self.assertEqual(
                bundled_policy.candidate_to_backend["EcommerceProduct"],
                "NoAvailable",
            )
            self.assertEqual(
                read_jsonl(prepared),
                [{"messages": []}],
            )
            contract = evaluate_top1._load_router_contract(model_output)
            evaluate_top1._verify_base_model_dependency(
                contract,
                model_dir=model_output,
            )
            contract_args = Namespace(
                max_length=None,
                trust_remote_code=None,
            )
            evaluate_top1._apply_router_contract(
                contract_args,
                contract,
                registry_path=model_output / "candidate_registry.json",
                prompt_path=model_output / "router_system_prompt.md",
                candidate_names=("StockQuery", "EcommerceProduct"),
            )
            self.assertEqual(contract_args.max_length, 1024)
            self.assertFalse(contract_args.trust_remote_code)
            evaluate_top1._validate_loaded_tokenizer(
                contract,
                tokenizer=tokenizer,
                candidate_names=("StockQuery", "EcommerceProduct"),
                candidate_tokens={
                    "StockQuery": (1, 2),
                    "EcommerceProduct": (3, 4),
                },
                transformers_version="5.5.4",
            )
            legacy_manifest = {
                **manifest,
                "schema_version": 3,
                "inference": {
                    "scoring_rule": "candidate_path_sum_logprob",
                    "decision_rule": "backend_group_threshold_v1",
                    "include_eos": True,
                },
                "backend_decision_policy": {
                    **manifest["backend_decision_policy"],
                    "decision_rule": "backend_group_threshold_v1",
                },
            }
            (model_output / "router_manifest.json").write_text(
                json.dumps(legacy_manifest),
                encoding="utf-8",
            )
            self.assertEqual(
                evaluate_top1._load_router_contract(model_output)["schema_version"],
                3,
            )
            legacy_schema2 = {
                key: value
                for key, value in manifest.items()
                if key != "backend_decision_policy"
            }
            legacy_schema2.update(
                schema_version=2,
                inference={
                    "decision_rule": "candidate_path_sum_logprob",
                    "include_eos": True,
                },
            )
            (model_output / "router_manifest.json").write_text(
                json.dumps(legacy_schema2),
                encoding="utf-8",
            )
            self.assertEqual(
                evaluate_top1._load_router_contract(model_output)["schema_version"],
                2,
            )
            (model_output / "router_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            tokenizer.chat_template = "changed"
            with self.assertRaisesRegex(Top1DataError, "differs from training"):
                evaluate_top1._validate_loaded_tokenizer(
                    contract,
                    tokenizer=tokenizer,
                    candidate_names=("StockQuery", "EcommerceProduct"),
                    candidate_tokens={
                        "StockQuery": (1, 2),
                        "EcommerceProduct": (3, 4),
                    },
                    transformers_version="5.5.4",
                )
            del tokenizer.chat_template

            contract_args.max_length = 512
            with self.assertRaisesRegex(Top1DataError, "must equal"):
                evaluate_top1._apply_router_contract(
                    contract_args,
                    contract,
                    registry_path=model_output / "candidate_registry.json",
                    prompt_path=model_output / "router_system_prompt.md",
                    candidate_names=("StockQuery", "EcommerceProduct"),
                )

            (model_output / "router_system_prompt.md").write_text(
                "different prompt\n",
                encoding="utf-8",
            )
            contract_args.max_length = None
            with self.assertRaisesRegex(Top1DataError, "system prompt differs"):
                evaluate_top1._apply_router_contract(
                    contract_args,
                    contract,
                    registry_path=model_output / "candidate_registry.json",
                    prompt_path=model_output / "router_system_prompt.md",
                    candidate_names=("StockQuery", "EcommerceProduct"),
                )

    def test_lora_local_base_model_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_model = root / "base"
            base_model.mkdir()
            weights = base_model / "model.safetensors"
            weights.write_bytes(b"base weights")
            args = SimpleNamespace(
                finetune_mode="lora",
                model_name_or_path=str(base_model),
            )
            model = SimpleNamespace(config=SimpleNamespace(_commit_hash=None))
            dependency = train_top1._base_model_dependency(args, model)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            manifest = {
                "finetune_mode": "lora",
                "base_model_dependency": dependency,
            }

            evaluate_top1._verify_base_model_dependency(
                manifest,
                model_dir=adapter,
            )
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(Top1DataError, "changed"):
                evaluate_top1._verify_base_model_dependency(
                    manifest,
                    model_dir=adapter,
                )

    def test_training_and_evaluation_prepare_identical_prompt_tokens(self) -> None:
        tokenizer = CharacterTokenizer()
        candidate_tokens = train_top1.candidate_token_sequences(
            tokenizer,
            ("A", "MuchLongerCandidateName"),
        )
        row = {
            "messages": [
                {"role": "user", "content": "历史" * 100},
                {"role": "assistant", "content": "回复" * 100},
                {"role": "user", "content": "当前" * 100},
            ],
            "target_candidate_name": "A",
        }
        trained = train_top1.prepare_example(
            tokenizer,
            row,
            candidate_tokens=candidate_tokens,
            max_length=420,
            system_prompt="route",
        )
        evaluated = evaluate_top1._prepare_prompts(
            [row],
            tokenizer=tokenizer,
            candidate_names=("A", "MuchLongerCandidateName"),
            candidate_tokens=candidate_tokens,
            system_prompt="route",
            max_length=420,
            history_ablation=False,
        )[0]

        training_prompt = trained.encoded["input_ids"][: -len(candidate_tokens["A"]) - 1]
        self.assertEqual(training_prompt, evaluated["prompt_ids"])


if __name__ == "__main__":
    unittest.main()
