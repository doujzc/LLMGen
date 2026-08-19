"""Optional confusion-aware margin objective for Top1 candidate training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .top1 import Top1DataError


MARGIN_LOSS_ALGORITHM = "candidate_trie_branch_margin_v1"
MARGIN_LOSS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MarginLossConfig:
    """Validated candidate-confusion margin-loss configuration."""

    loss_weight: float
    logit_margin: float
    priority_matrix: Mapping[str, Mapping[str, int]]

    def payload(self, candidate_names: Sequence[str]) -> dict[str, Any]:
        """Return a canonical JSON-serializable configuration payload."""

        return {
            "schema_version": MARGIN_LOSS_SCHEMA_VERSION,
            "algorithm": MARGIN_LOSS_ALGORITHM,
            "loss_weight": self.loss_weight,
            "logit_margin": self.logit_margin,
            "normalization": "priority_weighted_pair_mean",
            "candidate_order": list(candidate_names),
            "priority_matrix": {
                target: {
                    competitor: int(self.priority_matrix[target][competitor])
                    for competitor in candidate_names
                }
                for target in candidate_names
            },
        }


@dataclass(frozen=True)
class CandidateMarginBranch:
    """One directed confusion pair at its first candidate-token divergence."""

    target_index: int
    competitor_index: int
    target_name: str
    competitor_name: str
    divergence_offset: int
    target_token_id: int
    competitor_token_id: int
    priority: int


@dataclass(frozen=True)
class MarginLossBatch:
    """Margin objective and content-free diagnostics for one model batch."""

    loss: Any
    weighted_violations: Any
    total_priority: int


def load_margin_loss_config(
    path: str | Path,
    candidate_names: Sequence[str],
) -> MarginLossConfig:
    """Load one complete directed confusion-priority matrix."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Top1DataError(f"invalid margin-loss config: {source}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError("margin-loss config must be a JSON object")
    if payload.get("schema_version") != MARGIN_LOSS_SCHEMA_VERSION:
        raise Top1DataError(
            "margin-loss config schema_version must be "
            f"{MARGIN_LOSS_SCHEMA_VERSION}"
        )
    if payload.get("algorithm") != MARGIN_LOSS_ALGORITHM:
        raise Top1DataError(
            f"margin-loss algorithm must be {MARGIN_LOSS_ALGORITHM!r}"
        )
    names = tuple(candidate_names)
    if payload.get("candidate_order") != list(names):
        raise Top1DataError(
            "margin-loss candidate_order must exactly match the candidate registry"
        )

    def positive_number(field: str) -> float:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Top1DataError(f"margin-loss {field} must be a finite positive number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise Top1DataError(f"margin-loss {field} must be a finite positive number")
        return normalized

    loss_weight = positive_number("loss_weight")
    logit_margin = positive_number("logit_margin")
    if payload.get("normalization") != "priority_weighted_pair_mean":
        raise Top1DataError(
            "margin-loss normalization must be 'priority_weighted_pair_mean'"
        )
    raw_matrix = payload.get("priority_matrix")
    if not isinstance(raw_matrix, dict) or set(raw_matrix) != set(names):
        raise Top1DataError(
            "margin-loss priority_matrix must cover exactly the candidate registry"
        )
    matrix: dict[str, dict[str, int]] = {}
    positive_pairs = 0
    for target in names:
        raw_row = raw_matrix[target]
        if not isinstance(raw_row, dict) or set(raw_row) != set(names):
            raise Top1DataError(
                f"margin-loss row {target!r} must cover exactly the candidate registry"
            )
        row: dict[str, int] = {}
        for competitor in names:
            priority = raw_row[competitor]
            if (
                isinstance(priority, bool)
                or not isinstance(priority, int)
                or not 0 <= priority <= 3
            ):
                raise Top1DataError(
                    "margin-loss priorities must be integers from 0 to 3"
                )
            if target == competitor and priority != 0:
                raise Top1DataError("margin-loss matrix diagonal must be zero")
            row[competitor] = priority
            positive_pairs += int(target != competitor and priority > 0)
        matrix[target] = row
    if not positive_pairs:
        raise Top1DataError("margin-loss matrix must contain a positive confusion pair")
    return MarginLossConfig(
        loss_weight=loss_weight,
        logit_margin=logit_margin,
        priority_matrix=matrix,
    )


def candidate_margin_branches(
    candidate_names: Sequence[str],
    candidate_tokens: Mapping[str, Sequence[int]],
    *,
    eos_token_id: int,
    config: MarginLossConfig,
) -> dict[int, tuple[CandidateMarginBranch, ...]]:
    """Map directed confusion pairs to their first Trie decision."""

    names = tuple(candidate_names)
    paths = {
        name: (*map(int, candidate_tokens[name]), int(eos_token_id))
        for name in names
    }
    branches: dict[int, tuple[CandidateMarginBranch, ...]] = {}
    for target_index, target in enumerate(names):
        target_path = paths[target]
        target_branches = []
        for competitor_index, competitor in enumerate(names):
            priority = int(config.priority_matrix[target][competitor])
            if priority <= 0:
                continue
            competitor_path = paths[competitor]
            divergence = next(
                (
                    offset
                    for offset, (target_token, competitor_token) in enumerate(
                        zip(target_path, competitor_path)
                    )
                    if target_token != competitor_token
                ),
                None,
            )
            if divergence is None:
                raise Top1DataError(
                    f"candidate paths do not diverge: {target!r}, {competitor!r}"
                )
            target_branches.append(
                CandidateMarginBranch(
                    target_index=target_index,
                    competitor_index=competitor_index,
                    target_name=target,
                    competitor_name=competitor,
                    divergence_offset=divergence,
                    target_token_id=int(target_path[divergence]),
                    competitor_token_id=int(competitor_path[divergence]),
                    priority=priority,
                )
            )
        branches[target_index] = tuple(target_branches)
    return branches


def margin_target_lookup(
    candidate_names: Sequence[str],
    candidate_tokens: Mapping[str, Sequence[int]],
    *,
    eos_token_id: int,
) -> dict[tuple[int, ...], int]:
    """Index the exact supervised candidate paths used by SFT labels."""

    return {
        (*map(int, candidate_tokens[name]), int(eos_token_id)): index
        for index, name in enumerate(candidate_names)
    }


def candidate_margin_loss(
    torch: Any,
    *,
    logits: Any,
    labels: Any,
    target_indices: Any,
    branches: Mapping[int, Sequence[CandidateMarginBranch]],
    logit_margin: float,
) -> MarginLossBatch:
    """Compute a priority-weighted hinge loss at candidate Trie branches."""

    if target_indices.ndim != 1 or target_indices.shape[0] != labels.shape[0]:
        raise Top1DataError("margin target indices do not match the batch")
    target_starts = labels.ne(-100).to(dtype=torch.long).argmax(dim=1)
    weighted_losses = []
    weighted_violations = []
    total_priority = 0
    for target_index, target_branches in branches.items():
        sample_indices = torch.nonzero(
            target_indices.eq(target_index),
            as_tuple=False,
        ).flatten()
        sample_count = int(sample_indices.numel())
        if not sample_count:
            continue
        starts = target_starts[sample_indices]
        for branch in target_branches:
            decision_positions = starts + branch.divergence_offset - 1
            decision_logits = logits[sample_indices, decision_positions].float()
            logit_gap = (
                decision_logits[:, branch.target_token_id]
                - decision_logits[:, branch.competitor_token_id]
            )
            weighted_losses.append(
                torch.relu(float(logit_margin) - logit_gap) * branch.priority
            )
            weighted_violations.append(
                (logit_gap.detach() < float(logit_margin)).float()
                * branch.priority
            )
            total_priority += sample_count * branch.priority
    if not weighted_losses:
        zero = logits[..., :1].sum() * 0.0
        return MarginLossBatch(zero, zero.detach(), 0)
    return MarginLossBatch(
        loss=torch.cat(weighted_losses).sum() / total_priority,
        weighted_violations=torch.cat(weighted_violations).sum(),
        total_priority=total_priority,
    )


def _output_logits(outputs: Any) -> Any:
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    if logits is None and isinstance(outputs, tuple) and len(outputs) > 1:
        logits = outputs[1]
    if logits is None:
        raise Top1DataError("causal-LM output does not contain logits for margin loss")
    return logits


def margin_trainer_class(transformers: Any, torch: Any) -> type[Any]:
    """Build a Trainer subclass compatible with the installed Transformers."""

    class CandidateMarginTrainer(transformers.Trainer):
        """Add the configured Trie-branch margin to the native SFT loss."""

        def __init__(
            self,
            *args: Any,
            margin_loss_config: MarginLossConfig,
            margin_branches: Mapping[int, Sequence[CandidateMarginBranch]],
            **kwargs: Any,
        ) -> None:
            self.margin_loss_config = margin_loss_config
            self.margin_branches = margin_branches
            self._margin_statistics: dict[str, dict[str, Any]] = {
                phase: {
                    "weighted_loss": None,
                    "weighted_violations": None,
                    "total_priority": 0,
                }
                for phase in ("train", "eval")
            }
            super().__init__(*args, **kwargs)

        def _record_margin_statistics(
            self,
            phase: str,
            batch: MarginLossBatch,
        ) -> None:
            if batch.total_priority <= 0:
                return
            statistics = self._margin_statistics[phase]
            weighted_loss = batch.loss.detach() * batch.total_priority
            weighted_violations = batch.weighted_violations.detach()
            statistics["weighted_loss"] = (
                weighted_loss
                if statistics["weighted_loss"] is None
                else statistics["weighted_loss"] + weighted_loss
            )
            statistics["weighted_violations"] = (
                weighted_violations
                if statistics["weighted_violations"] is None
                else statistics["weighted_violations"] + weighted_violations
            )
            statistics["total_priority"] += batch.total_priority

        def _append_margin_logs(self, logs: dict[str, float]) -> None:
            phase = None
            prefix = ""
            if "eval_loss" in logs:
                phase, prefix = "eval", "eval_"
            elif "final_loss" in logs:
                phase, prefix = "eval", "final_"
            elif "loss" in logs:
                phase = "train"
            if phase is None:
                return
            statistics = self._margin_statistics[phase]
            total_priority = int(statistics["total_priority"])
            if total_priority <= 0:
                return
            totals = torch.stack(
                (
                    statistics["weighted_loss"].float(),
                    statistics["weighted_violations"].float(),
                    statistics["weighted_loss"].new_tensor(
                        float(total_priority)
                    ),
                )
            )
            if int(getattr(self.args, "world_size", 1)) > 1:
                totals = self.accelerator.reduce(totals, reduction="sum")
            weighted_loss, weighted_violations, global_priority = (
                float(value) for value in totals.tolist()
            )
            logs[f"{prefix}margin_loss"] = weighted_loss / global_priority
            logs[f"{prefix}margin_violation_rate"] = (
                weighted_violations / global_priority
            )
            self._margin_statistics[phase] = {
                "weighted_loss": None,
                "weighted_violations": None,
                "total_priority": 0,
            }

        def log(
            self,
            logs: dict[str, float],
            start_time: float | None = None,
        ) -> None:
            self._append_margin_logs(logs)
            super().log(logs, start_time=start_time)

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            model_inputs = dict(inputs)
            target_indices = model_inputs.pop("margin_target_index", None)
            labels = model_inputs.get("labels")
            if target_indices is None or labels is None:
                raise Top1DataError(
                    "margin-loss batches require labels and margin_target_index"
                )
            sft_loss, outputs = super().compute_loss(
                model,
                model_inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            margin_batch = candidate_margin_loss(
                torch,
                logits=_output_logits(outputs),
                labels=labels,
                target_indices=target_indices,
                branches=self.margin_branches,
                logit_margin=self.margin_loss_config.logit_margin,
            )
            training = bool(getattr(model, "training", False))
            self._record_margin_statistics(
                "train" if training else "eval",
                margin_batch,
            )
            margin_scale = 1.0
            if (
                training
                and bool(getattr(self, "model_accepts_loss_kwargs", False))
                and num_items_in_batch is not None
            ):
                accumulation_steps = max(
                    1,
                    int(
                        getattr(
                            self,
                            "current_gradient_accumulation_steps",
                            self.args.gradient_accumulation_steps,
                        )
                    ),
                )
                margin_scale /= accumulation_steps
            loss = (
                sft_loss
                + self.margin_loss_config.loss_weight
                * margin_batch.loss
                * margin_scale
            )
            return (loss, outputs) if return_outputs else loss

    return CandidateMarginTrainer
