"""Offline provider E2E coverage for the default generic pipeline adapters."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Any, Mapping

import pytest

from llmgen.pipeline.config import load_pipeline_config
from llmgen.pipeline.io import atomic_write_json, atomic_write_jsonl
from llmgen.pipeline.ledger import JsonlShardLedger
from llmgen.pipeline.runner import PipelineRunnerError, create_pipeline_run
from llmgen.pipeline.stages import StageContext
from llmgen.pipeline.stages.base import StageExecutionError
from llmgen.pipeline.stages import common, evaluate as evaluate_stage
from llmgen.pipeline.stages import finalize_dataset as finalize_dataset_stage
from llmgen.pipeline.stages import legacy
from llmgen.router_bundle import dump_router_decoder_artifacts


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "router_pipeline.yaml"


def _build_local_tiny_router_base(path: Path) -> None:
    """Create a complete offline CausalLM used by the real Runner smoke test."""

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "Ċ": 4,
        "System": 5,
        "User": 6,
        "Assistant": 7,
        ":": 8,
        "route": 9,
        "calendar": 10,
        "skill": 11,
    }
    raw = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    path.mkdir()
    tokenizer.save_pretrained(path)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    ).save_pretrained(path)


def _prompt_payload(prompt: str, marker: str) -> list[dict[str, Any]]:
    value = prompt.rsplit(marker, 1)[1].strip()
    payload = json.loads(value)
    return payload if isinstance(payload, list) else [payload]


class _MockProvider:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.chat_calls = 0
        self.embedding_calls = 0

    def response(self, prompt: str) -> dict[str, Any]:
        if "每个 skill 必须输出" in prompt:
            rows = _prompt_payload(prompt, "输入：")
            return {
                "items": [
                    self._profile(row)
                    for row in rows
                ]
            }
        if "输入 workflows：" in prompt:
            workflows = _prompt_payload(prompt, "输入 workflows：")
            return {
                "items": [
                    self._multiskill_queries(workflow, index)
                    for index, workflow in enumerate(workflows)
                ]
            }
        if "严格的数据质检员" in prompt:
            rows = _prompt_payload(prompt, "待审核：")
            return {
                "items": [
                    {
                        "query_id": row["query_id"],
                        "scores": {
                            "mobile_style": 5,
                            "complexity": 5,
                            "target_necessity": 5,
                            "coherence": 5,
                            "specificity": 5,
                        },
                        "missing_skill_ids": [],
                        "redundant_skill_ids": [],
                        "unsafe": False,
                        "pass": True,
                        "issues": [],
                    }
                    for row in rows
                ]
            }
        if "每项正好" in prompt and "single-skill" not in prompt:
            rows = _prompt_payload(prompt, "输入：")
            return {
                "items": [
                    {
                        "skill_id": row["skill_id"],
                        "variants": [
                            {
                                "query": (
                                    "请在明天上午整理我的待办事项并设置日程提醒"
                                    if index == 0
                                    else "请在周五下班前整理会议要点并保存行动清单"
                                ),
                                "evidence": (
                                    "整理我的待办事项并设置日程提醒"
                                    if index == 0
                                    else "整理会议要点并保存行动清单"
                                ),
                            }
                        ],
                    }
                    for index, row in enumerate(rows)
                ]
            }
        if "alignment review" in prompt or "target_relevance" in prompt:
            rows = _prompt_payload(prompt, "输入：")
            return {
                "items": [
                    {
                        "query_id": row["query_id"],
                        "scores": {
                            "mobile_style": 5,
                            "target_relevance": 5,
                            "specificity": 4,
                            "coherence": 5,
                        },
                        "missing": False,
                        "extra_capability_needed": False,
                        "requirement_satisfied": True,
                        "unsafe": False,
                        "pass": True,
                        "issues": [],
                    }
                    for row in rows
                ]
            }
        raise AssertionError(f"unexpected mock-provider prompt: {prompt[:120]!r}")

    @staticmethod
    def _profile(row: Mapping[str, Any]) -> dict[str, Any]:
        calendar = "日程" in str(row.get("name") or "")
        return {
            "skill_id": row["skill_id"],
            "domain": "productivity_planning" if calendar else "documents_office",
            "roles": ["schedule"] if calendar else ["store"],
            "capability_zh": (
                "为用户整理待办事项并安排日程提醒"
                if calendar
                else "为用户整理会议记录并保存行动清单"
            ),
            "aliases": [row.get("name") or "日程助手"],
            "capability_facets": (
                ["整理待办并安排提醒"]
                if calendar
                else ["整理会议记录并保存清单"]
            ),
            "trigger_phrases": (
                ["安排日程", "待办提醒"]
                if calendar
                else ["整理会议记录", "保存行动清单"]
            ),
            "negative_boundaries": [],
            "routing_mode": "atomic",
            "mobile_fit": "high",
            "unsafe_action": False,
        }

    @staticmethod
    def _multiskill_queries(
        workflow: Mapping[str, Any], index: int
    ) -> dict[str, Any]:
        """Build schema-valid explicit/implicit variants for any two targets."""

        target_ids = [str(value) for value in workflow["required_target_ids"]]
        assert len(target_ids) == 2
        day = ("周一", "周二", "周三", "周四")[index % 4]
        explicit_first = f"把{day}下午三点的客户回访安排进日程并提醒我"
        explicit_second = "把会议要点整理成待办清单保存好"
        implicit_first = f"{day}下午三点客户回访前"
        implicit_second = "把相关待办一次整理清楚"
        return {
            "workflow_id": workflow["workflow_id"],
            "variants": [
                {
                    "intent_mode": "explicit",
                    "query": (
                        f"明天开完项目会后，{explicit_second}，再{explicit_first}，"
                        "别遗漏需要带给客户的资料。"
                    ),
                    "evidence": {
                        target_ids[0]: explicit_first,
                        target_ids[1]: explicit_second,
                    },
                    "implicit_skill_ids": [],
                    "implicit_rationales": {},
                },
                {
                    "intent_mode": "implicit",
                    "query": (
                        f"明天{implicit_first}，请{implicit_second}，"
                        "让我能按时带齐项目资料去沟通。"
                    ),
                    "evidence": {
                        target_ids[0]: implicit_first,
                        target_ids[1]: implicit_second,
                    },
                    "implicit_skill_ids": [target_ids[0]],
                    "implicit_rationales": {
                        target_ids[0]: "客户回访前的明确时间约束要求提前安排提醒。"
                    },
                },
            ],
        }


@pytest.fixture
def mock_provider():
    provider = _MockProvider()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request = json.loads(body)
            if self.path == "/v1/chat/completions":
                prompt = request["messages"][-1]["content"]
                with provider.lock:
                    provider.chat_calls += 1
                payload = {
                    "id": f"chat-{provider.chat_calls}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(provider.response(prompt)),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            elif self.path == "/v1/embeddings":
                values = request["input"]
                if isinstance(values, str):
                    values = [values]
                with provider.lock:
                    provider.embedding_calls += 1
                payload = {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": index, "embedding": [0.1, 0.2, 0.3]}
                        for index, _ in enumerate(values)
                    ],
                    "model": "mock-embedding",
                    "usage": {"prompt_tokens": len(values), "total_tokens": len(values)},
                }
            else:
                self.send_error(404)
                return
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield provider, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()


def _write_candidates(path: Path) -> None:
    atomic_write_jsonl(
        path,
        [
            {
                "id": "calendar",
                "name": "日程助手",
                "description": "整理待办事项并创建带提醒的日历日程。",
            }
        ],
    )


def _write_multiskill_candidates(path: Path) -> None:
    atomic_write_jsonl(
        path,
        [
            {
                "id": "calendar",
                "name": "日程助手",
                "description": "整理待办事项并创建带提醒的日历日程。",
            },
            {
                "id": "meeting-notes",
                "name": "会议笔记",
                "description": "提取会议要点并保存成可执行的待办清单。",
            },
        ],
    )


def _real_tiny_pipeline_config(
    *,
    candidates: Path,
    output: Path,
    base_model: Path,
    provider_base_url: str,
    candidate_count: int,
):
    """Resolve a minimal CPU configuration without weakening real Stage work."""

    python = Path(sys.executable)
    return load_pipeline_config(
        CONFIG,
        candidates=candidates,
        output=output,
        environment={
            "PIPELINE_PYTHON": str(python),
            "GENERATION_API_KEY": "mock-key",
            "REVIEW_API_KEY": "mock-key",
            "EMBEDDING_API_KEY": "mock-key",
            "ROUTER_BASE_MODEL": str(base_model),
        },
        overrides=(
            f"runtime.python={python}",
            "runtime.device=cpu",
            "runtime.num_devices=1",
            "runtime.deepspeed=none",
            "runtime.dataloader_workers=0",
            f"router.base_model={base_model}",
            "router.finetune_mode=full",
            "router.precision=fp32",
            "router.max_length=48",
            "router.per_device_train_batch_size=1",
            "router.per_device_eval_batch_size=1",
            "router.gradient_accumulation_steps=1",
            "router.gradient_checkpointing=false",
            "router.logging_steps=1",
            "router.validation_fraction=0",
            "router.memorization.epochs=1",
            "router.memorization.learning_rate=0.001",
            "router.alignment.enabled=true",
            "router.alignment.epochs=1",
            "router.alignment.learning_rate=0.001",
            "router.retrieval.epochs=1",
            "router.retrieval.learning_rate=0.001",
            "router.retrieval.alignment_replay_fraction=0",
            "router.retrieval.memorization_replay_fraction=0",
            "input.single_candidate_policy=alignment_only",
            f"providers.generation.base_url={provider_base_url}",
            f"providers.review.base_url={provider_base_url}",
            f"providers.embedding.base_url={provider_base_url}",
            "providers.generation.concurrency=1",
            "providers.review.concurrency=1",
            f"providers.embedding.batch_size={candidate_count}",
            "data_generation.workflows_per_skill=2",
            "data_generation.explicit_variants=2",
            "data_generation.implicit_variants=1",
            "data_generation.alignment_queries_per_skill=1",
            "data_generation.alignment_backfill_rounds=0",
            "data_generation.final_alignment_backfill_rounds=0",
            "data_generation.max_backfill_rounds=0",
            "data_generation.retrieval_positives_per_skill=1",
            "data_generation.split.train=1",
            "data_generation.split.validation=0",
            "data_generation.split.test=0",
            "code.mode=manual",
            "code.num_levels=1",
            f"code.branching_factors=[{candidate_count}]",
            "code.spare_capacity_ratio=1",
            "code.rq_layers=[4]",
            "code.embedding_dim=2",
            "code.sk_epsilons=[0]",
            "code.epochs=1",
            f"code.batch_size={candidate_count}",
            "code.learning_rate=0.001",
            "code.scheduler=constant",
            "code.warmup_ratio=0",
            "code.graph_lambda=0",
            "code.amp_dtype=none",
            "code.assignment=balanced_hierarchical",
            "code.max_bucket_size=1",
            "checkpointing.training_save_steps=1",
            "checkpointing.training_eval_steps=1",
            "checkpointing.keep_last=2",
            "evaluation.query_split=train",
            "evaluation.cutoffs=[1]",
            "evaluation.top_k=1",
            "evaluation.max_code_paths=1",
            "evaluation.batch_size=1",
            "evaluation.dtype=float32",
            "evaluation.require_format_valid_rate=1",
            "evaluation.require_candidate_coverage=1",
            "export.smoke_test=true",
            "export.require_all_gates=true",
            "export.allow_failed_gates=false",
        ),
    )


def _stub_router_command(context: StageContext, command: str) -> None:
    paths = legacy._paths(context)
    if command == "train-tokenizer":
        paths["stage1"].mkdir(parents=True, exist_ok=True)
        (paths["stage1"] / "best.pt").write_bytes(b"deterministic-codebook")
        return
    if command == "export-codes":
        candidates = [
            json.loads(line)
            for line in context.artifact("candidates.normalized").read_text().splitlines()
            if line.strip()
        ]
        plan = json.loads(paths["code_plan"].read_text(encoding="utf-8"))
        factors = [int(value) for value in plan["branching_factors"]]
        codes = []
        buckets: dict[str, list[str]] = {}
        for candidate_index, candidate in enumerate(candidates):
            indices = [candidate_index % factor for factor in factors]
            tokens = [
                f"<SK_L{level}_{index}>"
                for level, index in enumerate(indices, start=1)
            ]
            skill_id = str(candidate["skill_id"])
            codes.append({"skill_id": skill_id, "indices": indices, "tokens": tokens})
            buckets["/".join(map(str, indices))] = [skill_id]
        paths["index"].mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(
            paths["index"] / "train_codes.jsonl",
            codes,
        )
        atomic_write_json(
            paths["index"] / "train_registry.json",
            {
                "schema_version": 1,
                "split": "train",
                "num_levels": len(factors),
                "branching_factors": factors,
                "token_format": "<SK_L{level}_{index}>",
                "buckets": buckets,
            },
        )
        virtual_tokens = [
            f"<SK_L{level}_{index}>"
            for level, factor in enumerate(factors, start=1)
            for index in range(factor)
        ]
        (paths["index"] / "virtual_tokens.txt").write_text(
            "\n".join(virtual_tokens) + "\n", encoding="utf-8"
        )
        atomic_write_json(
            paths["index"] / "manifest.json",
            {"num_levels": 1, "checkpoint_sha256": "mock-codebook"},
        )
        return
    if command == "train-memorization":
        # Training adapters publish their model beneath the current attempt's
        # output directory; the runner then atomically promotes that tree.
        model = _write_model(context, name="memorization")
        (model / "training.complete").write_text("mock\n", encoding="utf-8")
        return
    raise AssertionError(f"unexpected stubbed router command: {command}")


def _write_model(context: StageContext, *, name: str = "retrieval") -> Path:
    paths = legacy._paths(context)
    model = context.output_dir / name
    model.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model / "config.json", {"model_type": "mock", "vocab_size": 1})
    atomic_write_json(model / "tokenizer_config.json", {"model_type": "mock"})
    atomic_write_json(model / "tokenizer.json", {"version": "1.0"})
    (model / "model.safetensors").write_bytes(b"deterministic-model")
    atomic_write_json(model / "router_manifest.json", {"phase": "retrieval"})
    dump_router_decoder_artifacts(
        output_dir=model,
        catalog_path=paths["processed"] / "catalog_train.jsonl",
        codes_path=paths["index"] / "train_codes.jsonl",
        registry_path=paths["index"] / "train_registry.json",
        virtual_tokens_path=paths["index"] / "virtual_tokens.txt",
        training_data_path=paths["router_data"] / "retrieval_train.jsonl",
        supervision_phase="retrieval",
    )
    return model


def _write_evaluation(context: StageContext) -> None:
    """Write deterministic evaluator output accepted by the real audit path."""

    paths = legacy._paths(context)
    evaluation = paths["evaluation"]
    code = json.loads(
        (paths["index"] / "train_codes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    evaluation.mkdir(parents=True, exist_ok=True)
    atomic_write_json(evaluation / "metrics.json", {"metrics": {"recall@1": 1.0}})
    atomic_write_jsonl(
        evaluation / "predictions.jsonl",
        [
            {
                "query_id": "mock-evaluation",
                "paths": [
                    {
                        "skill_ids": [code["skill_id"]],
                        "code_tokens": code["tokens"],
                    }
                ],
            }
        ],
    )


def test_default_pipeline_mock_provider_e2e_recovers_ledgers_without_reissuing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_provider: tuple[_MockProvider, str],
) -> None:
    provider, base_url = mock_provider
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates)
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    atomic_write_json(base_model / "config.json", {"model_type": "mock"})
    # Preserve the active virtual-environment launcher. Resolving the symlink can
    # select uv's bare base interpreter, which does not contain this project's
    # installed dependencies.
    python = Path(sys.executable)
    config = load_pipeline_config(
        CONFIG,
        candidates=candidates,
        output=tmp_path / "run",
        environment={
            "PIPELINE_PYTHON": str(python),
            "GENERATION_API_KEY": "mock-key",
            "REVIEW_API_KEY": "mock-key",
            "EMBEDDING_API_KEY": "mock-key",
        },
        overrides=(
            f"runtime.python={python}",
            f"router.base_model={base_model}",
            "runtime.device=cpu",
            "runtime.num_devices=1",
            "runtime.deepspeed=none",
            f"providers.generation.base_url={base_url}",
            f"providers.review.base_url={base_url}",
            f"providers.embedding.base_url={base_url}",
            "providers.generation.concurrency=1",
            "providers.review.concurrency=1",
            "providers.embedding.batch_size=1",
            "input.single_candidate_policy=alignment_only",
            "data_generation.alignment_queries_per_skill=1",
            "data_generation.alignment_backfill_rounds=0",
            "data_generation.final_alignment_backfill_rounds=0",
            "data_generation.max_backfill_rounds=0",
            "data_generation.retrieval_positives_per_skill=1",
            "router.alignment.enabled=false",
            "router.alignment.epochs=0",
            "router.retrieval.alignment_replay_fraction=0",
            "router.retrieval.memorization_replay_fraction=0",
            "export.smoke_test=false",
        ),
    )
    monkeypatch.setenv("GENERATION_API_KEY", "mock-key")
    monkeypatch.setenv("REVIEW_API_KEY", "mock-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "mock-key")

    original_router = common.router_pipeline
    original_command = StageContext.run_command
    fail_once = {"value": True}

    def router_command(context: StageContext, command: str, **kwargs: object) -> None:
        if command in {"prepare", "build-router-data"}:
            original_router(context, command, **kwargs)
        elif command == "evaluate":
            _write_evaluation(context)
        else:
            _stub_router_command(context, command)

    def run_command(self: StageContext, argv, *, environment=None, label=None) -> None:
        if label == "profile-candidates" and fail_once["value"]:
            original_command(self, argv, environment=environment, label=label)
            fail_once["value"] = False
            raise StageExecutionError("simulated interruption after provider ledger commit")
        if label == "legacy-router-train-retrieval":
            _write_model(self)
            return
        if label == "materialize-router-bundle":
            return
        return original_command(self, argv, environment=environment, label=label)

    monkeypatch.setattr(common, "router_pipeline", router_command)
    monkeypatch.setattr(finalize_dataset_stage, "router_pipeline", router_command)
    monkeypatch.setattr(evaluate_stage, "router_pipeline", router_command)
    monkeypatch.setattr(StageContext, "run_command", run_command)
    runner = create_pipeline_run(config, repo_root=ROOT)

    with pytest.raises(PipelineRunnerError, match="simulated interruption"):
        runner.run(to_stage="enrich")
    assert provider.chat_calls == 1

    runner.run(from_stage="enrich")

    assert provider.chat_calls == 3  # profile + alignment generation + alignment review
    assert provider.embedding_calls == 1
    profile_ledger = JsonlShardLedger(
        runner.state.stage_dir("enrich") / "ledger" / "generation" / "profile-candidates",
        batch_size=20,
    )
    profile_stats = profile_ledger.verify()["stats"]
    assert profile_stats["requests"]["unique"] == 1
    assert profile_stats["responses"]["success_unique"] == 1
    assert runner.state.read_stage("enrich")["attempt"] == 2
    assert (runner.run_dir / "export" / "model" / "model.safetensors").is_file()
    assert (runner.run_dir / "export" / "report" / "quality_gates.json").is_file()
    assert (runner.state.stage_dir("finalize-dataset") / "output" / "dataset" / "qrels_alignment.jsonl").is_file()
    assert (runner.state.stage_dir("build-sft") / "output" / "router_data" / "retrieval_train.jsonl").is_file()


def test_finalize_dataset_reuses_committed_embedding_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_provider: tuple[_MockProvider, str],
) -> None:
    """A failed Stage attempt must not resend a committed embedding batch."""

    provider, base_url = mock_provider
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates)
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    atomic_write_json(base_model / "config.json", {"model_type": "mock"})
    python = Path(sys.executable)
    config = load_pipeline_config(
        CONFIG,
        candidates=candidates,
        output=tmp_path / "run",
        environment={
            "PIPELINE_PYTHON": str(python),
            "GENERATION_API_KEY": "mock-key",
            "REVIEW_API_KEY": "mock-key",
            "EMBEDDING_API_KEY": "mock-key",
        },
        overrides=(
            f"runtime.python={python}",
            f"router.base_model={base_model}",
            "runtime.device=cpu",
            "runtime.num_devices=1",
            "runtime.deepspeed=none",
            f"providers.generation.base_url={base_url}",
            f"providers.review.base_url={base_url}",
            f"providers.embedding.base_url={base_url}",
            "providers.generation.concurrency=1",
            "providers.review.concurrency=1",
            "providers.embedding.batch_size=1",
            "input.single_candidate_policy=alignment_only",
            "data_generation.alignment_queries_per_skill=1",
            "data_generation.alignment_backfill_rounds=0",
            "data_generation.final_alignment_backfill_rounds=0",
            "data_generation.max_backfill_rounds=0",
            "data_generation.retrieval_positives_per_skill=1",
        ),
    )
    monkeypatch.setenv("GENERATION_API_KEY", "mock-key")
    monkeypatch.setenv("REVIEW_API_KEY", "mock-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "mock-key")

    original_command = StageContext.run_command
    fail_once = {"value": True}

    def run_command(self: StageContext, argv, *, environment=None, label=None) -> None:
        original_command(self, argv, environment=environment, label=label)
        if label == "generic-router-prepare" and fail_once["value"]:
            fail_once["value"] = False
            raise StageExecutionError(
                "simulated interruption after embedding ledger commit"
            )

    monkeypatch.setattr(StageContext, "run_command", run_command)
    runner = create_pipeline_run(config, repo_root=ROOT)

    with pytest.raises(PipelineRunnerError, match="embedding ledger commit"):
        runner.run(to_stage="finalize-dataset")
    assert provider.embedding_calls == 1

    runner.run(from_stage="finalize-dataset", to_stage="finalize-dataset")

    assert provider.embedding_calls == 1
    assert runner.state.read_stage("finalize-dataset")["attempt"] == 2
    ledger = JsonlShardLedger(
        runner.state.stage_dir("finalize-dataset")
        / "ledger"
        / "embedding"
        / "candidate-catalog",
        batch_size=100,
    )
    assert ledger.verify()["stats"]["embeddings"]["success_unique"] == 1
    assert (
        runner.state.stage_dir("finalize-dataset")
        / "output"
        / "embeddings"
        / "train.npy"
    ).is_file()


def test_default_pipeline_mock_provider_e2e_runs_multiskill_retrieval_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_provider: tuple[_MockProvider, str],
) -> None:
    """Exercise the non-alignment-only DAG with real Provider data adapters."""

    provider, base_url = mock_provider
    candidates = tmp_path / "candidates.jsonl"
    _write_multiskill_candidates(candidates)
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    atomic_write_json(base_model / "config.json", {"model_type": "mock"})
    python = Path(sys.executable)
    config = load_pipeline_config(
        CONFIG,
        candidates=candidates,
        output=tmp_path / "run",
        environment={
            "PIPELINE_PYTHON": str(python),
            "GENERATION_API_KEY": "mock-key",
            "REVIEW_API_KEY": "mock-key",
            "EMBEDDING_API_KEY": "mock-key",
        },
        overrides=(
            f"runtime.python={python}",
            f"router.base_model={base_model}",
            "runtime.device=cpu",
            "runtime.num_devices=1",
            "runtime.deepspeed=none",
            f"providers.generation.base_url={base_url}",
            f"providers.review.base_url={base_url}",
            f"providers.embedding.base_url={base_url}",
            "providers.generation.concurrency=1",
            "providers.review.concurrency=1",
            "providers.embedding.batch_size=2",
            "data_generation.workflows_per_skill=2",
            "data_generation.explicit_variants=2",
            "data_generation.implicit_variants=1",
            "data_generation.retrieval_positives_per_skill=1",
            "data_generation.alignment_queries_per_skill=1",
            "data_generation.alignment_backfill_rounds=0",
            "data_generation.final_alignment_backfill_rounds=0",
            "data_generation.max_backfill_rounds=0",
            "data_generation.query_batch_size=10",
            "data_generation.review_batch_size=10",
            "router.alignment.enabled=true",
            "router.alignment.epochs=1",
            "router.retrieval.alignment_replay_fraction=0",
            "router.retrieval.memorization_replay_fraction=0",
            "export.smoke_test=false",
        ),
    )
    monkeypatch.setenv("GENERATION_API_KEY", "mock-key")
    monkeypatch.setenv("REVIEW_API_KEY", "mock-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "mock-key")

    original_router = common.router_pipeline
    original_command = StageContext.run_command
    handoffs: dict[str, Path] = {}

    def router_command(context: StageContext, command: str, **kwargs: object) -> None:
        if command in {"prepare", "build-router-data"}:
            original_router(context, command, **kwargs)
        elif command == "evaluate":
            _write_evaluation(context)
        else:
            _stub_router_command(context, command)

    def run_command(self: StageContext, argv, *, environment=None, label=None) -> None:
        if label == "legacy-router-train-alignment":
            assert environment is not None
            handoffs["alignment_input"] = Path(
                environment["ROUTER_MEMORIZATION_MODEL_DIR"]
            )
            _write_model(self, name="retrieval_alignment")
            return
        if label == "legacy-router-train-retrieval":
            assert environment is not None
            handoffs["retrieval_input"] = Path(
                environment["ROUTER_RETRIEVAL_INIT_DIR"]
            )
            _write_model(self)
            return
        if label == "materialize-router-bundle":
            return
        return original_command(self, argv, environment=environment, label=label)

    monkeypatch.setattr(common, "router_pipeline", router_command)
    monkeypatch.setattr(finalize_dataset_stage, "router_pipeline", router_command)
    monkeypatch.setattr(evaluate_stage, "router_pipeline", router_command)
    monkeypatch.setattr(StageContext, "run_command", run_command)
    runner = create_pipeline_run(config, repo_root=ROOT)
    runner.run()

    dataset = runner.state.stage_dir("finalize-dataset") / "output" / "dataset"
    assert len((dataset / "skills.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    qrels = [
        json.loads(line)
        for line in (dataset / "qrels_train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in qrels:
        by_query.setdefault(str(row["query_id"]), []).append(row)
    assert by_query and all(
        [row["position"] for row in rows] == [0, 1]
        and {row["skill_id"] for row in rows} == {"calendar", "meeting-notes"}
        for rows in by_query.values()
    )
    covered = {row["skill_id"] for row in qrels}
    assert covered == {"calendar", "meeting-notes"}
    assert provider.chat_calls >= 5  # profile, alignment/multiskill generate + review
    assert provider.embedding_calls == 1

    memorization = runner.state.stage_dir("train-memorization") / "output" / "memorization"
    alignment = runner.state.stage_dir("train-alignment") / "output" / "retrieval_alignment"
    retrieval = runner.state.stage_dir("train-retrieval") / "output" / "retrieval"
    assert handoffs == {
        "alignment_input": memorization,
        "retrieval_input": alignment,
    }
    assert all((model / "skill_decode_map.json").is_file() for model in (memorization, alignment, retrieval))

    exported = runner.run_dir / "export" / "model"
    report = runner.run_dir / "export" / "report"
    router_manifest = json.loads((exported / "router_manifest.json").read_text())
    lineage = router_manifest["pipeline_lineage"]
    candidate_input = json.loads(
        (runner.run_dir / "config" / "candidate_input.json").read_text()
    )
    assert lineage["candidate_input_sha256"] == candidate_input["sha256"]
    assert {"dataset.manifest", "model.retrieval", "evaluation.metrics"} <= set(
        lineage["artifacts"]
    )
    model_files = json.loads((report / "model_files.json").read_text())
    recorded_files = {row["path"] for row in model_files["files"]}
    assert {"model.safetensors", "router_manifest.json", "skill_decode_map.json"} <= recorded_files
    actual_files = {
        path.relative_to(exported).as_posix()
        for path in exported.rglob("*")
        if path.is_file()
    }
    assert recorded_files == actual_files
    assert (exported / "model.safetensors").is_file()
    assert (report / "quality_gates.json").is_file()


def test_default_pipeline_runs_real_tiny_model_from_candidates_to_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_provider: tuple[_MockProvider, str],
) -> None:
    """Run the default DAG with real codebook/training/eval/export commands."""

    _provider, base_url = mock_provider
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates)
    base_model = tmp_path / "base-model"
    _build_local_tiny_router_base(base_model)
    python = Path(sys.executable)
    for name, value in {
        "GENERATION_API_KEY": "mock-key",
        "REVIEW_API_KEY": "mock-key",
        "EMBEDDING_API_KEY": "mock-key",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",
    }.items():
        monkeypatch.setenv(name, value)

    config = load_pipeline_config(
        CONFIG,
        candidates=candidates,
        output=tmp_path / "run",
        environment={
            "PIPELINE_PYTHON": str(python),
            "GENERATION_API_KEY": "mock-key",
            "REVIEW_API_KEY": "mock-key",
            "EMBEDDING_API_KEY": "mock-key",
            "ROUTER_BASE_MODEL": str(base_model),
        },
        overrides=(
            f"runtime.python={python}",
            "runtime.device=cpu",
            "runtime.num_devices=1",
            "runtime.deepspeed=none",
            "runtime.dataloader_workers=0",
            f"router.base_model={base_model}",
            "router.finetune_mode=full",
            "router.precision=fp32",
            "router.max_length=48",
            "router.per_device_train_batch_size=1",
            "router.per_device_eval_batch_size=1",
            "router.gradient_accumulation_steps=1",
            "router.gradient_checkpointing=false",
            "router.logging_steps=1",
            "router.validation_fraction=0",
            "router.memorization.epochs=1",
            "router.memorization.learning_rate=0.001",
            "router.alignment.enabled=true",
            "router.alignment.epochs=1",
            "router.alignment.learning_rate=0.001",
            "router.retrieval.epochs=1",
            "router.retrieval.alignment_replay_fraction=0",
            "router.retrieval.memorization_replay_fraction=0",
            "input.single_candidate_policy=alignment_only",
            f"providers.generation.base_url={base_url}",
            f"providers.review.base_url={base_url}",
            f"providers.embedding.base_url={base_url}",
            "providers.generation.concurrency=1",
            "providers.review.concurrency=1",
            "providers.embedding.batch_size=1",
            "data_generation.alignment_queries_per_skill=1",
            "data_generation.alignment_backfill_rounds=0",
            "data_generation.final_alignment_backfill_rounds=0",
            "data_generation.max_backfill_rounds=0",
            "data_generation.retrieval_positives_per_skill=1",
            "data_generation.split.train=1",
            "data_generation.split.validation=0",
            "data_generation.split.test=0",
            "code.mode=manual",
            "code.num_levels=1",
            "code.branching_factors=[1]",
            "code.rq_layers=[4]",
            "code.embedding_dim=2",
            "code.sk_epsilons=[0]",
            "code.epochs=1",
            "code.batch_size=1",
            "code.learning_rate=0.001",
            "code.scheduler=constant",
            "code.warmup_ratio=0",
            "code.graph_lambda=0",
            "code.amp_dtype=none",
            "code.assignment=nearest",
            "code.max_bucket_size=1",
            "checkpointing.training_save_steps=1",
            "checkpointing.training_eval_steps=1",
            "checkpointing.keep_last=2",
            "evaluation.query_split=train",
            "evaluation.cutoffs=[1]",
            "evaluation.top_k=1",
            "evaluation.max_code_paths=1",
            "evaluation.batch_size=1",
            "evaluation.dtype=float32",
            "evaluation.require_format_valid_rate=1",
            "evaluation.require_candidate_coverage=1",
            "export.smoke_test=true",
            "export.require_all_gates=true",
            "export.allow_failed_gates=false",
        ),
    )

    original_command = StageContext.run_command
    interrupt_once = {"value": True}

    def interrupt_after_checkpoint(
        self: StageContext,
        argv,
        *,
        environment=None,
        label=None,
    ) -> None:
        original_command(self, argv, environment=environment, label=label)
        if label == "generic-router-train-memorization" and interrupt_once["value"]:
            interrupt_once["value"] = False
            raise StageExecutionError(
                "simulated interruption after router checkpoint commit"
            )

    monkeypatch.setattr(StageContext, "run_command", interrupt_after_checkpoint)
    runner = create_pipeline_run(config, repo_root=ROOT)
    with pytest.raises(PipelineRunnerError, match="router checkpoint commit"):
        runner.run()
    assert runner.state.read_stage("train-memorization")["attempt"] == 1

    executions = runner.run(from_stage="train-memorization")

    assert all(execution.action == "executed" for execution in executions)
    processed_manifest_path = runner.registry.resolve("processed.manifest")
    processed_manifest = json.loads(
        processed_manifest_path.read_text(encoding="utf-8")
    )
    manifest_references = [
        processed_manifest["source"]["path"],
        processed_manifest["source"]["manifest"],
        processed_manifest["graph"]["path"],
        *(
            path
            for split in processed_manifest["splits"].values()
            for path in split["files"].values()
        ),
    ]
    assert all(not Path(reference).is_absolute() for reference in manifest_references)
    assert all(
        (processed_manifest_path.parent / reference).resolve().exists()
        for reference in manifest_references
    )
    memorization_state = runner.state.read_stage("train-memorization")
    assert memorization_state["attempt"] == 2
    assert memorization_state["progress"]["checkpoint_resume"]["selected"][
        "global_step"
    ] == 1
    exported = runner.run_dir / "export" / "model"
    report = runner.run_dir / "export" / "report"
    assert (exported / "model.safetensors").is_file()
    assert (exported / "skill_decode_map.json").is_file()
    assert (report / "run_summary.json").is_file()
    quality = json.loads((report / "quality_gates.json").read_text())
    assert quality["passed"] is True
    assert quality["model_load_smoke_test"] == "passed"
    assert quality["deployment_qualified"] is True

    import service
    import service_910b
    import service_openai

    for serving_module in (service, service_openai, service_910b):
        serving_module._validate_full_model_bundle(exported)
        bundle = serving_module._load_candidate_bundle(exported)
        assert tuple(bundle.skills) == ("calendar",)


def test_default_pipeline_runs_real_multiskill_retrieval_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_provider: tuple[_MockProvider, str],
) -> None:
    """Prove the non-passthrough retrieval curriculum on a local CausalLM."""

    _provider, base_url = mock_provider
    candidates = tmp_path / "candidates.jsonl"
    _write_multiskill_candidates(candidates)
    base_model = tmp_path / "base-model"
    _build_local_tiny_router_base(base_model)
    for name, value in {
        "GENERATION_API_KEY": "mock-key",
        "REVIEW_API_KEY": "mock-key",
        "EMBEDDING_API_KEY": "mock-key",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",
    }.items():
        monkeypatch.setenv(name, value)
    config = _real_tiny_pipeline_config(
        candidates=candidates,
        output=tmp_path / "run",
        base_model=base_model,
        provider_base_url=base_url,
        candidate_count=2,
    )

    runner = create_pipeline_run(config, repo_root=ROOT)
    executions = runner.run()

    assert all(execution.action == "executed" for execution in executions)
    retrieval_record = runner.registry.get("model.retrieval")
    assert retrieval_record.producer == "train-retrieval"
    assert retrieval_record.metadata.get("passthrough") is not True
    retrieval = runner.registry.resolve("model.retrieval")
    assert retrieval.name == "retrieval"
    checkpoint_sidecars = sorted(
        retrieval.glob("checkpoint-*/pipeline_lineage.json")
    )
    assert checkpoint_sidecars
    checkpoint_steps = [
        json.loads(sidecar.read_text(encoding="utf-8"))["global_step"]
        for sidecar in checkpoint_sidecars
    ]
    assert max(checkpoint_steps) > 0
    exported = runner.registry.resolve("export.model")
    decode_map = json.loads((exported / "skill_decode_map.json").read_text())
    assert set(decode_map["skills"]) == {"calendar", "meeting-notes"}
    assert decode_map["num_levels"] == 1
    assert len(decode_map["virtual_tokens"]) == 2
    quality = json.loads(
        (runner.run_dir / "export" / "report" / "quality_gates.json").read_text()
    )
    assert quality["deployment_qualified"] is True
