from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from llmgen.evaluation import (
    BackendDecisionPolicy,
    aggregate_predictions,
    candidate_confidence,
    load_backend_decision_policy,
    prediction_from_generation,
)
from llmgen.top1 import CandidateNameTokenTrie, Top1DataError
from scripts import evaluate_top1


def _policy() -> BackendDecisionPolicy:
    return BackendDecisionPolicy(
        candidate_to_backend={"A": "A", "B": "Fallback"},
        backend_labels=("A", "Fallback"),
        fallback_backend_label="Fallback",
    )


def _prediction(
    *,
    row_index: int,
    predicted: str,
    target: str,
    path_logprob: float,
    threshold: float | None,
    messages: int = 1,
    history_ablation=None,
):
    return prediction_from_generation(
        row_index=row_index,
        candidate_names=("A", "B"),
        generated_candidate_name=predicted,
        path_logprob=path_logprob,
        path_tokens=2,
        target_candidate_name=target,
        diagnostics={"original_message_count": messages},
        decision_policy=_policy(),
        route_threshold=threshold,
        decoding={"mode": "greedy", "num_beams": 1},
        history_ablation=history_ablation,
    )


class EvaluationTests(unittest.TestCase):
    def test_multi_device_memory_limits_are_parsed_and_json_normalized(self) -> None:
        parsed = evaluate_top1._parse_max_memory(
            ("0=22GiB,1=22GiB", "cpu=64GiB"),
        )

        self.assertEqual(parsed, {0: "22GiB", 1: "22GiB", "cpu": "64GiB"})
        self.assertEqual(
            evaluate_top1._json_max_memory(parsed),
            {"0": "22GiB", "1": "22GiB", "cpu": "64GiB"},
        )
        with self.assertRaisesRegex(Top1DataError, "duplicate max_memory device"):
            evaluate_top1._parse_max_memory(("0=22GiB", "0=20GiB"))

    def test_evaluation_rejects_torchrun_replication(self) -> None:
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(Top1DataError, "one process"):
                evaluate_top1._validate_single_process()

    def test_full_model_loading_forwards_accelerate_device_map(self) -> None:
        class AutoModel:
            call = None

            @classmethod
            def from_pretrained(cls, reference, **kwargs):
                cls.call = (reference, kwargs)
                return "loaded-model"

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            loaded = evaluate_top1._load_model(
                model_dir=model_dir,
                transformers=SimpleNamespace(AutoModelForCausalLM=AutoModel),
                dtype="bf16",
                trust_remote_code=False,
                router_contract={},
                device_map="auto",
                max_memory={0: "22GiB", 1: "22GiB"},
            )

        self.assertEqual(loaded, "loaded-model")
        self.assertEqual(AutoModel.call[0], str(model_dir))
        self.assertEqual(AutoModel.call[1]["device_map"], "auto")
        self.assertEqual(
            AutoModel.call[1]["max_memory"],
            {0: "22GiB", 1: "22GiB"},
        )
        self.assertTrue(AutoModel.call[1]["low_cpu_mem_usage"])

    def test_lora_loading_shards_base_and_adapter(self) -> None:
        base_model = SimpleNamespace(hf_device_map={"model": 0})

        class AutoModel:
            call = None

            @classmethod
            def from_pretrained(cls, reference, **kwargs):
                cls.call = (reference, kwargs)
                return base_model

        class PeftModel:
            call = None

            @classmethod
            def from_pretrained(cls, model, reference, **kwargs):
                cls.call = (model, reference, kwargs)
                return "loaded-adapter"

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            with mock.patch.dict(
                sys.modules,
                {"peft": SimpleNamespace(PeftModel=PeftModel)},
            ):
                loaded = evaluate_top1._load_model(
                    model_dir=model_dir,
                    transformers=SimpleNamespace(AutoModelForCausalLM=AutoModel),
                    dtype="bf16",
                    trust_remote_code=False,
                    router_contract={
                        "base_model_dependency": {
                            "kind": "hub_revision",
                            "reference": "base-model",
                            "revision": "revision-1",
                        }
                    },
                    device_map="balanced",
                    max_memory={0: "20GiB", 1: "20GiB"},
                )

        self.assertEqual(loaded, "loaded-adapter")
        self.assertEqual(AutoModel.call[0], "base-model")
        self.assertEqual(AutoModel.call[1]["revision"], "revision-1")
        self.assertEqual(AutoModel.call[1]["device_map"], "balanced")
        self.assertIs(PeftModel.call[0], base_model)
        self.assertEqual(PeftModel.call[2]["device_map"], "balanced")
        self.assertEqual(
            PeftModel.call[2]["max_memory"],
            {0: "20GiB", 1: "20GiB"},
        )

    def test_dispatched_model_uses_embedding_device_and_reports_layout(self) -> None:
        input_device = SimpleNamespace(type="cuda")
        model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(
                weight=SimpleNamespace(device=input_device)
            ),
            hf_device_map={"model.embed_tokens": 0, "model.layers.0": 1},
        )
        torch = SimpleNamespace(device=lambda value: value)

        self.assertIs(
            evaluate_top1._dispatched_input_device(model, torch),
            input_device,
        )
        self.assertEqual(
            evaluate_top1._resolved_device_map(model),
            {"model.embed_tokens": "cuda:0", "model.layers.0": "cuda:1"},
        )

    def test_constrained_generation_returns_name_and_normalized_path_score(self) -> None:
        class Tensor:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, index):
                if isinstance(index, tuple):
                    row, column = index
                    values = self.values[row]
                    return Tensor(values[column])
                return Tensor(self.values[index])

            def tolist(self):
                return self.values

            def sum(self):
                return Tensor(sum(self.values))

            def item(self):
                return self.values

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        class Torch:
            long = "long"

            @staticmethod
            def tensor(values, **_kwargs):
                return Tensor(values)

            @staticmethod
            def inference_mode():
                return InferenceMode()

        class LogitsProcessor:
            pass

        transformers = SimpleNamespace(
            LogitsProcessor=LogitsProcessor,
            LogitsProcessorList=list,
        )

        class Model:
            generate_kwargs = None

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                sequences = [
                    [*prompt, 10, 2]
                    for prompt in kwargs["input_ids"].tolist()
                ]
                return SimpleNamespace(
                    sequences=Tensor(sequences),
                    scores=(object(), object()),
                    beam_indices=None,
                )

            def compute_transition_scores(self, *_args, **kwargs):
                self.normalize_logits = kwargs["normalize_logits"]
                return Tensor([[-0.1, -0.2]])

        model = Model()
        result = evaluate_top1._generate_prepared(
            ({"row_index": 3, "prompt_ids": [90, 91]},),
            prompt_key="prompt_ids",
            model=model,
            tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0),
            trie=CandidateNameTokenTrie({"A": (10,)}, eos_token_id=2),
            torch=Torch(),
            transformers=transformers,
            device="cpu",
            decoding_mode="greedy",
            num_beams=4,
            route_threshold=0.5,
        )

        self.assertEqual(result[3]["candidate_name"], "A")
        self.assertAlmostEqual(result[3]["path_logprob"], -0.3)
        self.assertEqual(result[3]["path_tokens"], 2)
        self.assertTrue(model.normalize_logits)
        self.assertEqual(model.generate_kwargs["num_return_sequences"], 1)
        self.assertTrue(model.generate_kwargs["renormalize_logits"])

    def test_candidate_trie_enforces_complete_legal_names(self) -> None:
        trie = CandidateNameTokenTrie(
            {"A": (10,), "AB": (10, 11), "C": (12,)},
            eos_token_id=2,
        )

        self.assertEqual(trie.allowed_next(()), (10, 12))
        self.assertEqual(trie.allowed_next((10,)), (11, 2))
        self.assertEqual(trie.resolve((10, 11)), "AB")
        self.assertEqual(trie.max_name_tokens, 2)

    def test_selected_route_threshold_abstains_and_preserves_raw_candidate(self) -> None:
        record = _prediction(
            row_index=0,
            predicted="A",
            target="A",
            path_logprob=-0.6,
            threshold=0.6,
        )

        self.assertEqual(record["predicted_candidate_name"], "A")
        self.assertTrue(record["correct"])
        self.assertAlmostEqual(record["candidate_confidence"], math.exp(-0.6))
        self.assertTrue(record["backend_decision"]["threshold_triggered"])
        self.assertEqual(
            record["backend_decision"]["predicted_backend_label"],
            "Fallback",
        )
        self.assertEqual(record["backend_decision"]["status"], "abstained")

    def test_threshold_never_changes_a_raw_fallback_candidate(self) -> None:
        record = _prediction(
            row_index=0,
            predicted="B",
            target="B",
            path_logprob=-4.0,
            threshold=0.99,
        )

        self.assertFalse(record["backend_decision"]["threshold_triggered"])
        self.assertEqual(
            record["backend_decision"]["predicted_backend_label"],
            "Fallback",
        )
        self.assertEqual(record["backend_decision"]["status"], "no_route")

    def test_metrics_separate_raw_candidate_and_thresholded_backend(self) -> None:
        first = _prediction(
            row_index=0,
            predicted="A",
            target="A",
            path_logprob=-0.1,
            threshold=0.6,
        )
        second = _prediction(
            row_index=1,
            predicted="A",
            target="B",
            path_logprob=-1.0,
            threshold=0.6,
            messages=3,
            history_ablation={
                "candidate_name": "B",
                "path_logprob": -0.2,
                "path_tokens": 2,
            },
        )

        metrics = aggregate_predictions((first, second), ("A", "B"), _policy())

        self.assertEqual(metrics["top1_accuracy"], 0.5)
        self.assertEqual(metrics["backend"]["accuracy"], 1.0)
        self.assertEqual(metrics["backend"]["available_oos"]["oos_recall"], 1.0)
        self.assertEqual(metrics["routing_policy"]["threshold_triggered_examples"], 1)
        self.assertEqual(metrics["conversation_strata"]["multi_turn"]["rows"], 1)
        self.assertEqual(metrics["history_ablation"]["history_hurt"], 1)
        self.assertIsNotNone(metrics["calibration"]["expected_calibration_error"])

    def test_candidate_confidence_handles_negative_infinity(self) -> None:
        self.assertEqual(candidate_confidence(-math.inf), 0.0)
        self.assertEqual(candidate_confidence(0.0), 1.0)

    def test_repository_policy_encodes_operational_oos_mapping(self) -> None:
        candidates = (
            "StockAdvice",
            "StockOther",
            "StockQuery",
            "ProductGeneral",
            "ProductEcommerce",
            "ChitChat",
            "NoAvailable",
        )
        policy = load_backend_decision_policy(
            "configs/top1_decision_policy.json",
            candidates,
        )

        self.assertEqual(
            policy.available_backend_labels,
            ("StockQuery", "Ecommerce"),
        )
        self.assertEqual(
            policy.candidate_to_backend["ProductEcommerce"],
            "Ecommerce",
        )
        self.assertEqual(
            policy.candidate_to_backend["ProductGeneral"],
            "NoAvailable",
        )
        self.assertEqual(policy.candidate_to_backend["StockAdvice"], "NoAvailable")
        self.assertEqual(policy.candidate_to_backend["ChitChat"], "NoAvailable")

    def test_schema1_policy_is_normalized_for_new_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decision_policy.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "routing_mode": "candidate_name_top1",
                        "decision_rule": "backend_group_threshold_v1",
                        "backend_labels": ["A", "Fallback"],
                        "fallback_backend_label": "Fallback",
                        "candidate_to_backend": {"A": "A", "B": "Fallback"},
                        "available_threshold": 0.5,
                        "temperature": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            policy = load_backend_decision_policy(path, ("A", "B"))

            self.assertEqual(policy.payload()["schema_version"], 2)
            self.assertEqual(policy.candidate_to_backend["B"], "Fallback")


if __name__ == "__main__":
    unittest.main()
