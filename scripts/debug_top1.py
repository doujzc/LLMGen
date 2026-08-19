#!/usr/bin/env python3
"""Launch the read-only Top1 development and diagnostics console."""

import argparse
import gc
import math
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from llmgen.evaluation import (
    candidate_confidence,
    load_backend_decision_policy,
    prediction_from_generation,
)
from llmgen.experiment import load_and_verify_model_artifact
from llmgen.inspection import (
    compare_evaluation_runs,
    discover_evaluation_runs,
    discover_training_runs,
    load_evaluation_run,
    load_training_run,
)
from llmgen.top1 import (
    CandidateNameTokenTrie,
    Top1DataError,
    candidate_token_sequences,
    load_candidate_names,
)

try:
    from scripts import evaluate_top1 as evaluator
except ImportError:  # Direct execution places scripts/ at the front of sys.path.
    import evaluate_top1 as evaluator  # type: ignore[no-redef]


EVALUATION_COLUMNS = (
    "created_at",
    "state",
    "evaluation_id",
    "suite_id",
    "model_id",
    "dataset",
    "decoding",
    "route_threshold",
    "rows",
    "backend_accuracy",
    "raw_candidate_accuracy",
    "unsafe_oos_accept_rate",
    "run_dir",
)
TRAINING_COLUMNS = (
    "created_at",
    "state",
    "experiment_name",
    "run_id",
    "model_id",
    "best_eval_loss",
    "best_checkpoint",
    "run_dir",
)
CASE_COLUMNS = (
    "row_index",
    "target",
    "predicted",
    "candidate_correct",
    "target_backend",
    "predicted_backend",
    "backend_correct",
    "route_status",
    "confidence",
    "message_count",
    "history_dropped",
    "current_user_truncated",
    "history_changed_prediction",
    "last_user",
)
COMPARE_COLUMNS = (
    "row_index",
    "target",
    "first_candidate",
    "second_candidate",
    "first_backend",
    "second_backend",
    "first_correct",
    "second_correct",
    "change",
    "last_user",
)


class Top1DebugRuntime:
    """One verified model kept in memory for interactive, artifact-free inference."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_artifact: Mapping[str, Any],
        router_contract: Mapping[str, Any],
        system_prompt: str,
        candidate_names: Sequence[str],
        candidate_tokens: Mapping[str, Sequence[int]],
        decision_policy: Any,
        tokenizer: Any,
        trie: CandidateNameTokenTrie,
        model: Any,
        torch_module: Any,
        transformers_module: Any,
        device: Any,
        resolved_precision: str,
        max_length: int,
    ) -> None:
        self.model_dir = model_dir
        self.model_artifact = dict(model_artifact)
        self.router_contract = dict(router_contract)
        self.system_prompt = system_prompt
        self.candidate_names = tuple(candidate_names)
        self.candidate_tokens = {
            name: tuple(map(int, candidate_tokens[name]))
            for name in self.candidate_names
        }
        self.decision_policy = decision_policy
        self.tokenizer = tokenizer
        self.trie = trie
        self.model = model
        self.torch = torch_module
        self.transformers = transformers_module
        self.device = device
        self.resolved_precision = resolved_precision
        self.max_length = max_length

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        device: str,
        precision: str,
    ) -> "Top1DebugRuntime":
        """Verify and load exactly the model bundle accepted by batch evaluation."""

        resolved_model_dir = Path(model_dir).expanduser().resolve()
        router_contract = evaluator._load_router_contract(resolved_model_dir)
        registry_path = evaluator._resolve_bundle_file(
            None,
            resolved_model_dir,
            "candidate_registry.json",
            label="candidate registry",
        )
        prompt_path = evaluator._resolve_bundle_file(
            None,
            resolved_model_dir,
            "router_system_prompt.md",
            label="system prompt",
        )
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise Top1DataError("system prompt file is empty")
        candidate_names = load_candidate_names(registry_path)
        decision_policy_path = evaluator._resolve_decision_policy(
            None,
            resolved_model_dir,
        )
        decision_policy = load_backend_decision_policy(
            decision_policy_path,
            candidate_names,
        )
        evaluator._validate_bundled_decision_policy(
            router_contract,
            model_dir=resolved_model_dir,
            decision_policy_path=decision_policy_path,
        )
        arguments = SimpleNamespace(
            max_length=None,
            trust_remote_code=None,
            device=device,
            precision=precision,
        )
        evaluator._apply_router_contract(
            arguments,
            router_contract,
            registry_path=registry_path,
            prompt_path=prompt_path,
            candidate_names=candidate_names,
        )
        evaluator._verify_base_model_dependency(
            router_contract,
            model_dir=resolved_model_dir,
        )
        model_artifact = load_and_verify_model_artifact(
            resolved_model_dir,
            verify_files=True,
        )
        torch_module, transformers_module, _ = evaluator._import_dependencies()
        resolved_device, dtype, resolved_precision = evaluator._device_and_dtype(
            arguments,
            torch_module,
        )
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(
            str(resolved_model_dir),
            trust_remote_code=arguments.trust_remote_code,
        )
        if tokenizer.eos_token_id is None:
            raise Top1DataError("model tokenizer must define an EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        candidate_tokens = candidate_token_sequences(tokenizer, candidate_names)
        trie = CandidateNameTokenTrie(
            candidate_tokens,
            eos_token_id=int(tokenizer.eos_token_id),
        )
        evaluator._validate_loaded_tokenizer(
            router_contract,
            tokenizer=tokenizer,
            candidate_names=candidate_names,
            candidate_tokens=candidate_tokens,
            transformers_version=str(transformers_module.__version__),
        )
        model = evaluator._load_model(
            model_dir=resolved_model_dir,
            transformers=transformers_module,
            dtype=dtype,
            trust_remote_code=arguments.trust_remote_code,
            router_contract=router_contract,
        ).to(resolved_device)
        model.eval()
        return cls(
            model_dir=resolved_model_dir,
            model_artifact=model_artifact,
            router_contract=router_contract,
            system_prompt=system_prompt,
            candidate_names=candidate_names,
            candidate_tokens=candidate_tokens,
            decision_policy=decision_policy,
            tokenizer=tokenizer,
            trie=trie,
            model=model,
            torch_module=torch_module,
            transformers_module=transformers_module,
            device=resolved_device,
            resolved_precision=resolved_precision,
            max_length=int(arguments.max_length),
        )

    def predict(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        expected_candidate: str | None,
        route_threshold: float | None,
        decoding_mode: str,
        num_beams: int,
        history_ablation: bool,
        detailed_scores: bool,
    ) -> dict[str, Any]:
        """Run one case in memory without creating an evaluation run."""

        _validate_decoding(decoding_mode, num_beams)
        threshold = _normalize_threshold(route_threshold)
        row: dict[str, Any] = {"messages": list(messages)}
        if expected_candidate:
            row["target_candidate_name"] = expected_candidate
        prepared = evaluator._prepare_prompts(
            (row,),
            tokenizer=self.tokenizer,
            candidate_names=self.candidate_names,
            candidate_tokens=self.candidate_tokens,
            system_prompt=self.system_prompt,
            max_length=self.max_length,
            history_ablation=history_ablation,
        )
        generated = evaluator._generate_prepared(
            prepared,
            prompt_key="prompt_ids",
            model=self.model,
            tokenizer=self.tokenizer,
            trie=self.trie,
            torch=self.torch,
            transformers=self.transformers,
            device=self.device,
            decoding_mode=decoding_mode,
            num_beams=num_beams,
            route_threshold=threshold,
        )[0]
        ablation = (
            evaluator._generate_prepared(
                prepared,
                prompt_key="history_ablation_prompt_ids",
                model=self.model,
                tokenizer=self.tokenizer,
                trie=self.trie,
                torch=self.torch,
                transformers=self.transformers,
                device=self.device,
                decoding_mode=decoding_mode,
                num_beams=num_beams,
                route_threshold=threshold,
            ).get(0)
            if history_ablation
            else None
        )
        decoding = {
            "mode": decoding_mode,
            "num_beams": 1 if decoding_mode == "greedy" else num_beams,
            "num_return_sequences": 1,
            "scope": "candidate_name_top1",
        }
        record = prediction_from_generation(
            row_index=0,
            candidate_names=self.candidate_names,
            generated_candidate_name=generated["candidate_name"],
            path_logprob=generated["path_logprob"],
            path_tokens=generated["path_tokens"],
            target_candidate_name=expected_candidate,
            diagnostics=prepared[0]["diagnostics"],
            decision_policy=self.decision_policy,
            route_threshold=threshold,
            decoding=decoding,
            history_ablation=ablation,
        )
        scores = (
            self._candidate_path_scores(
                prepared[0]["prompt_ids"],
                generated_candidate_name=generated["candidate_name"],
            )
            if detailed_scores
            else []
        )
        return {
            "record": record,
            "candidate_scores": scores,
            "score_margin": _score_margin(scores),
            "prompt_text": prepared[0]["prompt_text"],
            "fitted_messages": prepared[0]["fitted_messages"],
            "system_prompt": self.system_prompt,
            "contract": {
                "model_id": self.model_artifact.get("model_id"),
                "model_dir": str(self.model_dir),
                "max_length": self.max_length,
                "device": str(self.device),
                "precision": self.resolved_precision,
                "candidate_names": list(self.candidate_names),
                "prompt_contract_verified": True,
                "artifact_files_verified": True,
            },
        }

    def _candidate_path_scores(
        self,
        prompt_ids: Sequence[int],
        *,
        generated_candidate_name: str,
    ) -> list[dict[str, Any]]:
        """Score every complete candidate under the constrained token grammar."""

        paths = {
            name: [*self.candidate_tokens[name], self.trie.eos_token_id]
            for name in self.candidate_names
        }
        sequences = [[*map(int, prompt_ids), *paths[name]] for name in self.candidate_names]
        width = max(map(len, sequences))
        pad_token_id = int(self.tokenizer.pad_token_id)
        padded = [
            [*sequence, *([pad_token_id] * (width - len(sequence)))]
            for sequence in sequences
        ]
        attention = [
            [1] * len(sequence) + [0] * (width - len(sequence))
            for sequence in sequences
        ]
        with self.torch.inference_mode():
            outputs = self.model(
                input_ids=self.torch.tensor(
                    padded,
                    dtype=self.torch.long,
                    device=self.device,
                ),
                attention_mask=self.torch.tensor(
                    attention,
                    dtype=self.torch.long,
                    device=self.device,
                ),
                use_cache=False,
                return_dict=True,
            )
        prompt_length = len(prompt_ids)
        result = []
        for row_index, name in enumerate(self.candidate_names):
            prefix: list[int] = []
            path_logprob = 0.0
            for path_index, token_id in enumerate(paths[name]):
                allowed = self.trie.allowed_next(prefix)
                if token_id not in allowed:
                    raise RuntimeError(f"candidate path left the legal trie: {name}")
                values = outputs.logits[
                    row_index,
                    prompt_length + path_index - 1,
                    list(allowed),
                ].float()
                log_probabilities = self.torch.log_softmax(values, dim=-1)
                selected_index = allowed.index(token_id)
                path_logprob += float(log_probabilities[selected_index].item())
                if token_id != self.trie.eos_token_id:
                    prefix.append(token_id)
            result.append(
                {
                    "candidate": name,
                    "backend": self.decision_policy.candidate_to_backend[name],
                    "path_logprob": path_logprob,
                    "confidence": candidate_confidence(path_logprob),
                    "generated": name == generated_candidate_name,
                }
            )
        ordered = sorted(result, key=lambda row: float(row["path_logprob"]), reverse=True)
        for rank, row in enumerate(ordered, start=1):
            row["rank"] = rank
        return ordered


class RuntimeCache:
    """Serialize access to one lazily loaded debug runtime."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runtime: Top1DebugRuntime | None = None
        self._key: tuple[str, str, str] | None = None

    def load(
        self,
        model_dir: str,
        *,
        device: str,
        precision: str,
    ) -> Top1DebugRuntime:
        """Load or reuse a model for the requested immutable bundle and device."""

        if not model_dir.strip():
            raise Top1DataError("model directory is required")
        key = (str(Path(model_dir).expanduser().resolve()), device, precision)
        with self._lock:
            if self._runtime is not None and self._key == key:
                return self._runtime
            self._release_locked()
            runtime = Top1DebugRuntime.load(
                key[0],
                device=device,
                precision=precision,
            )
            self._runtime = runtime
            self._key = key
            return runtime

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run one prediction while preventing concurrent model mutation."""

        with self._lock:
            if self._runtime is None:
                raise Top1DataError("load a model before running a case")
            return self._runtime.predict(*args, **kwargs)

    def _release_locked(self) -> None:
        if self._runtime is None:
            return
        torch_module = self._runtime.torch
        self._runtime = None
        self._key = None
        gc.collect()
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the intentionally small read-only server configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--training-root", default="runs/top1")
    parser.add_argument("--evaluation-root", default="runs/evaluations/top1")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--inbrowser",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "values") and hasattr(value.values, "tolist"):
        value = value.values.tolist()
    if not isinstance(value, list):
        raise Top1DataError("dialogue must be a two-column table")
    messages = []
    for row_index, row in enumerate(value, start=1):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise Top1DataError(f"dialogue row {row_index} must contain role and content")
        role = str(row[0] or "").strip()
        content = str(row[1] or "").strip()
        if not role and not content:
            continue
        if not role or not content:
            raise Top1DataError(
                f"dialogue row {row_index} must contain both role and content"
            )
        messages.append({"role": role, "content": content})
    if not messages:
        raise Top1DataError("dialogue cannot be empty")
    return messages


def _normalize_threshold(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Top1DataError("route threshold must be a number")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise Top1DataError("route threshold must be between 0 and 1")
    return threshold


def _validate_decoding(mode: str, num_beams: int) -> None:
    if mode not in evaluator.DECODING_MODES:
        raise Top1DataError(f"unsupported decoding mode: {mode}")
    if mode == "beam_search" and num_beams < 2:
        raise Top1DataError("beam_search requires num_beams >= 2")


def _score_margin(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if len(scores) < 2:
        return None
    first, second = scores[:2]
    return {
        "first": first.get("candidate"),
        "second": second.get("candidate"),
        "logprob_margin": float(first["path_logprob"])
        - float(second["path_logprob"]),
        "confidence_margin": float(first["confidence"])
        - float(second["confidence"]),
    }


def _case_markdown(result: Mapping[str, Any]) -> str:
    record = result["record"]
    backend = record["backend_decision"]
    expected = record.get("target_candidate_name") or "未填写"
    confidence = record.get("candidate_confidence")
    confidence_text = f"{float(confidence):.6f}" if confidence is not None else "不可用"
    margin = result.get("score_margin")
    margin_text = (
        f"；详细评分 Top1/Top2 logprob margin = {float(margin['logprob_margin']):.6f}"
        if isinstance(margin, Mapping)
        else ""
    )
    return (
        "### 推理结果\n\n"
        f"- 原始候选：`{record['predicted_candidate_name']}`\n"
        f"- 后端结果：`{backend['predicted_backend_label']}`\n"
        f"- 路由状态：`{backend['status']}`\n"
        f"- 期望候选：`{expected}`\n"
        f"- 候选置信度：`{confidence_text}`{margin_text}\n"
    )


def _rows_frame(pandas: Any, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]):
    return pandas.DataFrame(
        [{column: row.get(column) for column in columns} for row in rows],
        columns=list(columns),
    )


def build_app(
    *,
    default_model_dir: str,
    training_root: str,
    evaluation_root: str,
    device: str,
    precision: str,
) -> Any:
    """Build the UI without loading a model or writing repository artifacts."""

    try:
        import gradio as gr
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Top1 debug UI requires the optional dependency: install -e '.[debug]'"
        ) from exc

    repository = Path(__file__).resolve().parents[1]
    default_candidates = load_candidate_names(repository / "configs/top1_candidates.json")
    cache = RuntimeCache()

    def refresh_evaluations(root: str):
        rows = discover_evaluation_runs(root)
        choices = [row["run_dir"] for row in rows if row.get("state") != "INVALID"]
        selected = choices[0] if choices else None
        return (
            _rows_frame(pd, rows, EVALUATION_COLUMNS),
            rows,
            gr.Dropdown(choices=choices, value=selected),
            gr.Dropdown(choices=choices, value=selected),
            gr.Dropdown(choices=choices, value=choices[1] if len(choices) > 1 else selected),
        )

    def refresh_training(root: str):
        rows = discover_training_runs(root)
        choices = [row["run_dir"] for row in rows if row.get("state") != "INVALID"]
        return (
            _rows_frame(pd, rows, TRAINING_COLUMNS),
            rows,
            gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        )

    def load_model(model_dir: str) -> str:
        try:
            runtime = cache.load(model_dir, device=device, precision=precision)
            return (
                "✅ 已验证并加载模型（只驻留内存）  \n"
                f"`{runtime.model_artifact.get('model_id')}` · "
                f"`{runtime.device}` · `{runtime.resolved_precision}`"
            )
        except (Exception, SystemExit) as exc:
            return f"❌ `{type(exc).__name__}`：{exc}"

    def run_case(
        model_dir: str,
        dialogue: Any,
        expected_candidate: str | None,
        route_threshold: Any,
        decoding_mode: str,
        num_beams: int,
        history_ablation: bool,
        detailed_scores: bool,
    ):
        try:
            runtime = cache.load(model_dir, device=device, precision=precision)
            result = cache.predict(
                _normalize_messages(dialogue),
                expected_candidate=expected_candidate or None,
                route_threshold=route_threshold,
                decoding_mode=decoding_mode,
                num_beams=int(num_beams),
                history_ablation=history_ablation,
                detailed_scores=detailed_scores,
            )
            fitted = [
                [row.get("role"), row.get("content")]
                for row in result["fitted_messages"]
            ]
            return (
                _case_markdown(result),
                _rows_frame(
                    pd,
                    result["candidate_scores"],
                    (
                        "rank",
                        "candidate",
                        "backend",
                        "path_logprob",
                        "confidence",
                        "generated",
                    ),
                ),
                result["prompt_text"],
                fitted,
                {
                    "diagnostics": result["record"]["diagnostics"],
                    "contract": result["contract"],
                    "score_margin": result["score_margin"],
                },
                result["record"],
                result["system_prompt"],
            )
        except (Exception, SystemExit) as exc:
            return (
                f"❌ `{type(exc).__name__}`：{exc}",
                _rows_frame(pd, [], ("rank", "candidate", "backend", "path_logprob", "confidence", "generated")),
                "",
                [],
                {},
                {},
                "",
            )

    def load_evaluation(
        run_dir: str,
        target: str | None,
        predicted: str | None,
        errors_only: bool,
    ):
        try:
            detail = load_evaluation_run(
                run_dir,
                target_candidate=target or None,
                predicted_candidate=predicted or None,
                errors_only=errors_only,
            )
            dataset = detail["dataset_status"]
            state = dataset.get("state")
            message = (
                f"数据集：`{state}`；匹配 {detail['matching_rows']} 条，"
                f"显示 {detail['displayed_rows']} 条（上限 500）。"
            )
            return (
                message,
                detail["summary"],
                detail["metrics"],
                _rows_frame(pd, detail["cases"], CASE_COLUMNS),
                detail["cases"],
                _matrix_frame(pd, detail["metrics"].get("confusion_matrix")),
                _matrix_frame(
                    pd,
                    _mapping(detail["metrics"].get("backend")).get(
                        "confusion_matrix"
                    ),
                ),
            )
        except (Exception, SystemExit) as exc:
            return (
                f"❌ `{type(exc).__name__}`：{exc}",
                {},
                {},
                _rows_frame(pd, [], CASE_COLUMNS),
                [],
                pd.DataFrame(),
                pd.DataFrame(),
            )

    def select_case(cases: Sequence[Mapping[str, Any]], evt: gr.SelectData):
        if not cases or evt.index is None:
            return "", {}
        row_index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
        if not isinstance(row_index, int) or not 0 <= row_index < len(cases):
            return "", {}
        return cases[row_index].get("dialogue", ""), cases[row_index].get(
            "prediction_record", {}
        )

    def select_run(rows: Sequence[Mapping[str, Any]], evt: gr.SelectData):
        if not rows or evt.index is None:
            return None
        row_index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
        if not isinstance(row_index, int) or not 0 <= row_index < len(rows):
            return None
        return rows[row_index].get("run_dir")

    def compare_runs(first: str, second: str):
        try:
            result = compare_evaluation_runs(first, second)
            scope = (
                "数据哈希一致，已进行逐 Case 对比。"
                if result["same_dataset"]
                else "数据哈希不同，仅比较聚合指标。"
            )
            return (
                scope,
                _rows_frame(pd, result["aggregate"], ("metric", "first", "second", "delta")),
                _rows_frame(pd, result["case_changes"], COMPARE_COLUMNS),
            )
        except (Exception, SystemExit) as exc:
            return (
                f"❌ `{type(exc).__name__}`：{exc}",
                _rows_frame(pd, [], ("metric", "first", "second", "delta")),
                _rows_frame(pd, [], COMPARE_COLUMNS),
            )

    def load_training(run_dir: str):
        try:
            detail = load_training_run(run_dir)
            loss = _rows_frame(pd, detail["loss_rows"], ("step", "epoch", "stage", "metric", "value"))
            optimization = _rows_frame(pd, detail["optimization_rows"], ("step", "stage", "metric", "value"))
            learning_rate = optimization[optimization["metric"] == "learning_rate"]
            grad_norm = optimization[optimization["metric"] == "grad_norm"]
            return (
                detail["status"],
                detail["summary"],
                loss,
                learning_rate,
                grad_norm,
                detail["manifest"],
                detail["event_tail"],
            )
        except (Exception, SystemExit) as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)}
            empty_loss = pd.DataFrame(
                columns=["step", "epoch", "stage", "metric", "value"]
            )
            empty_metric = pd.DataFrame(
                columns=["step", "stage", "metric", "value"]
            )
            return error, {}, empty_loss, empty_metric, empty_metric, {}, []

    initial_evaluations = discover_evaluation_runs(evaluation_root)
    initial_training = discover_training_runs(training_root)
    evaluation_choices = [
        row["run_dir"] for row in initial_evaluations if row.get("state") != "INVALID"
    ]
    training_choices = [
        row["run_dir"] for row in initial_training if row.get("state") != "INVALID"
    ]

    with gr.Blocks(title="Top1 Debug Console", analytics_enabled=False) as app:
        gr.Markdown(
            "# Top1 Debug Console\n"
            "**只读模式**：不启动训练或批量评测，不修改数据、Prompt、模型和 `runs/`。"
        )
        with gr.Tab("Case 调试"):
            with gr.Row():
                model_dir = gr.Textbox(
                    value=default_model_dir,
                    label="模型目录（final/model）",
                    scale=5,
                )
                load_model_button = gr.Button("验证并加载", variant="secondary", scale=1)
            model_status = gr.Markdown("尚未加载模型。")
            dialogue = gr.Dataframe(
                value=[["user", ""]],
                headers=["role", "content"],
                datatype=["str", "str"],
                type="array",
                row_count=(1, "dynamic"),
                column_count=(2, "fixed"),
                label="对话（最后一条非 system 消息必须是 user）",
            )
            with gr.Row():
                expected_candidate = gr.Dropdown(
                    choices=[None, *default_candidates],
                    value=None,
                    label="期望候选（可选）",
                )
                route_threshold = gr.Number(
                    value=None,
                    minimum=0,
                    maximum=1,
                    label="Route threshold（可选）",
                )
                decoding_mode = gr.Radio(
                    choices=list(evaluator.DECODING_MODES),
                    value="greedy",
                    label="解码",
                )
                num_beams = gr.Slider(2, 7, value=4, step=1, label="Beam 数")
            with gr.Row():
                history_ablation = gr.Checkbox(
                    value=False,
                    label="运行历史消融（额外一次生成）",
                )
                detailed_scores = gr.Checkbox(
                    value=False,
                    label="分析全部候选（额外一次前向）",
                )
                run_case_button = gr.Button("运行 Case", variant="primary")
            case_result = gr.Markdown()
            score_table = gr.Dataframe(
                headers=["rank", "candidate", "backend", "path_logprob", "confidence", "generated"],
                interactive=False,
                label="候选路径诊断（按需）",
            )
            with gr.Accordion("Prompt 与合约检查", open=False):
                prompt_text = gr.Code(label="最终模型输入", language=None)
                fitted_messages = gr.Dataframe(
                    headers=["role", "content"],
                    interactive=False,
                    label="裁剪后消息",
                )
                prompt_diagnostics = gr.JSON(label="Token 与合约诊断")
                system_prompt = gr.Code(label="Checkpoint 内置 System Prompt", language="markdown")
            with gr.Accordion("原始推理记录", open=False):
                raw_prediction = gr.JSON()

            load_model_button.click(load_model, inputs=[model_dir], outputs=[model_status])
            run_case_button.click(
                run_case,
                inputs=[
                    model_dir,
                    dialogue,
                    expected_candidate,
                    route_threshold,
                    decoding_mode,
                    num_beams,
                    history_ablation,
                    detailed_scores,
                ],
                outputs=[
                    case_result,
                    score_table,
                    prompt_text,
                    fitted_messages,
                    prompt_diagnostics,
                    raw_prediction,
                    system_prompt,
                ],
            )

        with gr.Tab("Evaluation 浏览"):
            evaluation_rows_state = gr.State(initial_evaluations)
            evaluation_cases_state = gr.State([])
            with gr.Row():
                evaluation_root_input = gr.Textbox(
                    value=str(Path(evaluation_root).expanduser()),
                    label="Evaluation 根目录",
                    scale=5,
                )
                refresh_evaluation_button = gr.Button("手动刷新", scale=1)
            evaluation_table = gr.Dataframe(
                value=_rows_frame(pd, initial_evaluations, EVALUATION_COLUMNS),
                interactive=False,
                label="Evaluation Runs",
            )
            evaluation_run = gr.Dropdown(
                choices=evaluation_choices,
                value=evaluation_choices[0] if evaluation_choices else None,
                label="查看 Run",
            )
            with gr.Row():
                target_filter = gr.Dropdown(
                    choices=[None, *default_candidates],
                    value=None,
                    label="Target 过滤",
                )
                predicted_filter = gr.Dropdown(
                    choices=[None, *default_candidates],
                    value=None,
                    label="Prediction 过滤",
                )
                errors_only = gr.Checkbox(value=True, label="仅候选错误")
                load_evaluation_button = gr.Button("读取", variant="primary")
            evaluation_status = gr.Markdown()
            with gr.Row():
                evaluation_summary = gr.JSON(label="Summary")
                evaluation_metrics = gr.JSON(label="Metrics")
            with gr.Row():
                candidate_confusion = gr.Dataframe(
                    interactive=False,
                    label="候选混淆矩阵",
                )
                backend_confusion = gr.Dataframe(
                    interactive=False,
                    label="后端混淆矩阵",
                )
            evaluation_cases = gr.Dataframe(
                interactive=False,
                label="Cases（点击一行查看完整对话）",
            )
            with gr.Row():
                selected_dialogue = gr.Code(label="对话", language=None)
                selected_prediction = gr.JSON(label="Prediction Record")
            gr.Markdown("### 两次 Evaluation 对比")
            with gr.Row():
                compare_first = gr.Dropdown(
                    choices=evaluation_choices,
                    value=evaluation_choices[0] if evaluation_choices else None,
                    label="Run A",
                )
                compare_second = gr.Dropdown(
                    choices=evaluation_choices,
                    value=evaluation_choices[1] if len(evaluation_choices) > 1 else (evaluation_choices[0] if evaluation_choices else None),
                    label="Run B",
                )
                compare_button = gr.Button("对比")
            compare_status = gr.Markdown()
            compare_metrics = gr.Dataframe(interactive=False, label="指标差异（B - A）")
            compare_cases = gr.Dataframe(interactive=False, label="预测变化")

            refresh_evaluation_button.click(
                refresh_evaluations,
                inputs=[evaluation_root_input],
                outputs=[
                    evaluation_table,
                    evaluation_rows_state,
                    evaluation_run,
                    compare_first,
                    compare_second,
                ],
            )
            evaluation_table.select(
                select_run,
                inputs=[evaluation_rows_state],
                outputs=[evaluation_run],
            )
            load_evaluation_button.click(
                load_evaluation,
                inputs=[evaluation_run, target_filter, predicted_filter, errors_only],
                outputs=[
                    evaluation_status,
                    evaluation_summary,
                    evaluation_metrics,
                    evaluation_cases,
                    evaluation_cases_state,
                    candidate_confusion,
                    backend_confusion,
                ],
            )
            evaluation_cases.select(
                select_case,
                inputs=[evaluation_cases_state],
                outputs=[selected_dialogue, selected_prediction],
            )
            compare_button.click(
                compare_runs,
                inputs=[compare_first, compare_second],
                outputs=[compare_status, compare_metrics, compare_cases],
            )

        with gr.Tab("Training 浏览"):
            training_rows_state = gr.State(initial_training)
            with gr.Row():
                training_root_input = gr.Textbox(
                    value=str(Path(training_root).expanduser()),
                    label="Training 根目录",
                    scale=5,
                )
                refresh_training_button = gr.Button("手动刷新", scale=1)
            training_table = gr.Dataframe(
                value=_rows_frame(pd, initial_training, TRAINING_COLUMNS),
                interactive=False,
                label="Training Runs",
            )
            with gr.Row():
                training_run = gr.Dropdown(
                    choices=training_choices,
                    value=training_choices[0] if training_choices else None,
                    label="查看 Run",
                    scale=5,
                )
                load_training_button = gr.Button("读取", variant="primary", scale=1)
            with gr.Row():
                training_status = gr.JSON(label="Status")
                training_summary = gr.JSON(label="Summary")
            loss_plot = gr.LinePlot(
                value=pd.DataFrame(columns=["step", "metric", "value"]),
                x="step",
                y="value",
                color="metric",
                title="Train / Eval Loss",
            )
            with gr.Row():
                learning_rate_plot = gr.LinePlot(
                    value=pd.DataFrame(columns=["step", "value"]),
                    x="step",
                    y="value",
                    title="Learning Rate",
                )
                grad_norm_plot = gr.LinePlot(
                    value=pd.DataFrame(columns=["step", "value"]),
                    x="step",
                    y="value",
                    title="Gradient Norm",
                )
            with gr.Accordion("Manifest 与最近事件", open=False):
                training_manifest = gr.JSON(label="Run Manifest")
                training_events = gr.JSON(label="最近 100 条事件")

            refresh_training_button.click(
                refresh_training,
                inputs=[training_root_input],
                outputs=[training_table, training_rows_state, training_run],
            )
            training_table.select(
                select_run,
                inputs=[training_rows_state],
                outputs=[training_run],
            )
            load_training_button.click(
                load_training,
                inputs=[training_run],
                outputs=[
                    training_status,
                    training_summary,
                    loss_plot,
                    learning_rate_plot,
                    grad_norm_plot,
                    training_manifest,
                    training_events,
                ],
            )

        gr.Markdown(
            "界面没有保存、修改、训练或批量评测操作；刷新只重新读取磁盘上的现有产物。"
        )
    return app


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _matrix_frame(pandas: Any, value: Any):
    matrix = _mapping(value)
    columns = sorted(
        {
            str(column)
            for row in matrix.values()
            if isinstance(row, Mapping)
            for column in row
        }
    )
    rows = []
    for target, values in matrix.items():
        normalized = _mapping(values)
        rows.append(
            {"target": target, **{column: normalized.get(column, 0) for column in columns}}
        )
    return pandas.DataFrame(rows, columns=["target", *columns])


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the local-only read-only console."""

    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    app = build_app(
        default_model_dir=args.model_dir,
        training_root=args.training_root,
        evaluation_root=args.evaluation_root,
        device=args.device,
        precision=args.precision,
    )
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=args.inbrowser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
