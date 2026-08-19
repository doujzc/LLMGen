"""Controlled multi-turn synthesis primitives for Top1 training data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from llmgen.top1 import Top1DataError, normalize_messages, read_jsonl


SYNTHESIS_VERSION = "top1_controlled_multiturn_v1"
DIRECTNESS_AUDIT_VERSION = 3
OBSERVED_PHENOMENA = (
    "intent_change",
    "progressive_reveal",
    "contextual_follow_up",
    "clarification_revision",
    "assistant_distractor",
    "rambling",
    "other",
)
QUALITY_FIELDS = (
    "coherent",
    "natural",
    "current_target_identifiable",
    "policy_boundary_respected",
    "no_candidate_name_leakage",
)
NON_SWITCH_WEIGHTS = (
    ("progressive_reveal", 18),
    ("contextual_follow_up", 14),
    ("clarification_revision", 10),
    ("assistant_distractor", 8),
    ("rambling", 4),
)

PHENOMENON_INSTRUCTIONS = {
    "intent_change": (
        "前面的用户轮次保持 source_candidate 目标；最后一轮直接提出 target_candidate "
        "的新需求。最后一轮必须整条消息都只表达新需求：不得回应或评价上一轮，不得致谢、确认，"
        "也不得用连接语宣布或暗示切换、取消、顺带提问或换题。直接从新需求的对象、动作或约束说起。"
    ),
    "progressive_reveal": (
        "所有用户轮次属于同一 target_candidate。用户逐轮补充约束或细节；最后一轮仍应包含足够的"
        "对象和动作信息，使当前目标在上下文中稳定可判。"
    ),
    "contextual_follow_up": (
        "所有用户轮次属于同一 target_candidate。最后一轮必须使用自然的省略、指代、确认或短追问，"
        "需要结合紧邻历史才能完整理解；不能变成无法恢复对象的残缺输入。"
    ),
    "clarification_revision": (
        "所有用户轮次属于同一 target_candidate。最后一轮自然纠正或修订先前的对象属性、条件或"
        "期望结果，但不改变候选类别。"
    ),
    "assistant_distractor": (
        "用户目标始终属于 target_candidate；某一条 assistant 消息出现合理的误解、无关建议或错误"
        "侧重点，最后一轮用户直接澄清真实目标。不要让干扰内容改变最终目标。"
    ),
    "rambling": (
        "用户在保持 target_candidate 目标的同时加入生活化背景或无关细节；最后一轮回到清晰可判的"
        "当前请求，冗余信息不能成为另一个待执行任务。"
    ),
}


@dataclass(frozen=True)
class DialogueBlueprint:
    """A deterministic semantic plan rendered by an LLM later."""

    scenario_id: str
    phenomenon: str
    target_candidate_name: str
    source_candidate_name: str | None
    user_turn_count: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON representation."""

        return {
            "scenario_id": self.scenario_id,
            "phenomenon": self.phenomenon,
            "target_candidate_name": self.target_candidate_name,
            "source_candidate_name": self.source_candidate_name,
            "user_turn_count": self.user_turn_count,
            "seed": self.seed,
            "turn_plan": _turn_plan(self),
        }


@dataclass(frozen=True)
class ModelCall:
    """One parsed OpenAI-compatible chat completion."""

    content: str
    usage: Mapping[str, Any]
    finish_reason: str | None
    elapsed_seconds: float


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _turn_plan(blueprint: DialogueBlueprint) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for turn in range(1, blueprint.user_turn_count + 1):
        candidate = blueprint.target_candidate_name
        behavior = blueprint.phenomenon
        if blueprint.phenomenon == "intent_change" and turn < blueprint.user_turn_count:
            candidate = str(blueprint.source_candidate_name)
            behavior = "source_context"
        elif blueprint.phenomenon == "intent_change":
            behavior = "direct_intent_change"
        plan.append(
            {
                "user_turn": turn,
                "candidate_name": candidate,
                "behavior": behavior,
            }
        )
    return plan


def build_dialogue_blueprints(
    candidate_names: Sequence[str],
    *,
    target_count: int = 800,
    intent_change_per_pair: int = 10,
    seed: int = 20260818,
) -> list[DialogueBlueprint]:
    """Build a balanced, plan-first set of controlled dialogue scenarios."""

    names = tuple(candidate_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise Top1DataError("synthesis requires at least two unique candidates")
    if intent_change_per_pair < 0:
        raise Top1DataError("intent_change_per_pair cannot be negative")
    pair_rows = len(names) * (len(names) - 1) * intent_change_per_pair
    if target_count < pair_rows:
        raise Top1DataError(
            f"target_count={target_count} is smaller than the {pair_rows} required intent-change rows"
        )

    rng = random.Random(seed)
    plans: list[DialogueBlueprint] = []
    for source in names:
        for target in names:
            if source == target:
                continue
            for index in range(1, intent_change_per_pair + 1):
                plans.append(
                    DialogueBlueprint(
                        scenario_id=(
                            f"{SYNTHESIS_VERSION}_intent_change_"
                            f"{_slug(source)}_to_{_slug(target)}_{index:03d}"
                        ),
                        phenomenon="intent_change",
                        target_candidate_name=target,
                        source_candidate_name=source,
                        user_turn_count=rng.choice((3, 4, 5)),
                        seed=rng.randrange(2**31),
                    )
                )

    weighted_cycle = [
        phenomenon
        for phenomenon, weight in NON_SWITCH_WEIGHTS
        for _ in range(weight)
    ]
    per_key_count: Counter[tuple[str, str]] = Counter()
    remaining = target_count - len(plans)
    for offset in range(remaining):
        target = names[offset % len(names)]
        candidate_sequence_index = offset // len(names)
        phenomenon = weighted_cycle[candidate_sequence_index % len(weighted_cycle)]
        key = (phenomenon, target)
        per_key_count[key] += 1
        plans.append(
            DialogueBlueprint(
                scenario_id=(
                    f"{SYNTHESIS_VERSION}_{phenomenon}_"
                    f"{_slug(target)}_{per_key_count[key]:03d}"
                ),
                phenomenon=phenomenon,
                target_candidate_name=target,
                source_candidate_name=None,
                user_turn_count=rng.choice((3, 4, 5)),
                seed=rng.randrange(2**31),
            )
        )

    if len({plan.scenario_id for plan in plans}) != len(plans):
        raise Top1DataError("duplicate synthesis scenario IDs")
    rng.shuffle(plans)
    return plans


def load_taxonomy_descriptions(
    path: str | Path,
    candidate_names: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Load concise and extended definitions from the reviewed LabelDesc data."""

    names = tuple(candidate_names)
    result = {name: {} for name in names}
    for row in read_jsonl(path):
        candidate = row.get("target_candidate_name")
        description_type = row.get("description_type")
        messages = row.get("messages")
        if candidate not in result or description_type not in {
            "concise_definition",
            "extended_definition",
        }:
            continue
        normalized = normalize_messages(messages)
        if len(normalized) != 1:
            raise Top1DataError("LabelDesc definitions must be single-turn")
        result[str(candidate)][str(description_type)] = normalized[0]["content"]

    missing = [
        f"{candidate}:{description_type}"
        for candidate in names
        for description_type in ("concise_definition", "extended_definition")
        if description_type not in result[candidate]
    ]
    if missing:
        raise Top1DataError("missing synthesis taxonomy definitions: " + ", ".join(missing))
    return result


def taxonomy_prompt(descriptions: Mapping[str, Mapping[str, str]]) -> str:
    """Render the closed candidate taxonomy for generators and blind judges."""

    sections: list[str] = []
    for candidate, values in descriptions.items():
        sections.extend(
            (
                f"### {candidate}",
                str(values["concise_definition"]),
                str(values["extended_definition"]),
            )
        )
    return "\n\n".join(sections)


def generation_messages(
    blueprints: Sequence[DialogueBlueprint],
    taxonomy: str,
) -> list[dict[str, str]]:
    """Build one structured batch request for the language realizer."""

    plans = []
    for blueprint in blueprints:
        payload = blueprint.to_dict()
        payload["phenomenon_instruction"] = PHENOMENON_INSTRUCTIONS[blueprint.phenomenon]
        plans.append(payload)
    system = f"""你是中文多轮对话数据生成器。你只负责把给定的结构化计划实现成自然对话，不得改变计划中的类别和轮次。

候选定义：
{taxonomy}

统一要求：
- 每个样本严格由 user 和 assistant 交替组成，第一条和最后一条都是 user。
- user 消息数必须等于计划中的 user_turn_count；不要加入 system 或 tool 消息。
- 当前分类目标永远是最后一条 user 消息。历史要真实、有帮助，但不得泄漏英文候选名。
- 对话使用简洁自然的中文，主题、实体、表达方式和 assistant 回复应有变化，避免批量模板腔。
- EcommerceProduct 生成京东、淘宝等平台通常可推荐或购买的普通商品在购买前的搜索、品牌或型号选择、比较、价格优惠、性能评价、适用性判断、推荐或购买需求；即使已经给出具体型号或没有点名平台也属于它。药品、整车、房屋、服务和软件不属于它。
- GeneralProduct 不得生成普通电商商品的购买前比较、价格、优惠、性能评价或适用性判断；它只处理非普通电商消费对象，以及已有商品的使用、故障、售后和订单事务。
- StockQuery 只查询已经存在的股票或证券公开行情事实；未来预测、标的推荐、买卖和仓位决策属于 StockAdvice。
- 输出必须是一个 JSON 对象，不要输出 Markdown 或解释。"""
    user = {
        "plans": plans,
        "output_schema": {
            "samples": [
                {
                    "scenario_id": "与输入完全一致",
                    "messages": [
                        {"role": "user|assistant", "content": "自然中文消息"}
                    ],
                    "scenario_summary": "不含候选英文名的一句场景摘要",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def judgment_messages(
    samples: Sequence[Mapping[str, Any]],
    taxonomy: str,
) -> list[dict[str, str]]:
    """Build a blind classification and quality-review request."""

    phenomenon_rules = "\n".join(
        (
            "- intent_change：较早用户目标属于另一个候选，最后一轮直接提出不同候选的新目标。",
            "- progressive_reveal：同一目标逐轮补充信息，最后目标不依赖纠错或意图切换。",
            "- contextual_follow_up：最后一轮有省略、指代、确认或短追问，必须结合历史理解。",
            "- clarification_revision：最后一轮修正先前条件或对象属性，但候选类别不变。",
            "- assistant_distractor：assistant 曾误解或引入无关方向，最后用户澄清原目标。",
            "- rambling：存在明显生活化冗余背景，但没有形成第二个待执行目标。",
            "- other：不符合以上任一种。",
        )
    )
    system = f"""你是独立的多轮 Top1 数据审计员。你看不到生成计划，必须仅根据对话盲判最后一条 user 消息。

候选定义：
{taxonomy}

判别规则：当前消息可独立理解时以当前消息为准；存在省略、指代、确认、修订时只使用必要历史；当前目标变化时选择新目标。只允许七个英文候选名。

现象定义：
{phenomenon_rules}

质量字段必须是布尔值。intent_change_is_direct 仅在观察到 intent_change 时表示最后一轮是否直接表达新需求、没有用宣布换题或取消旧任务的元话语；其它现象填写 true。observed_source_candidate_name 仅在 intent_change 时填写较早用户目标，否则为 null。
只输出 JSON 对象，不要输出 Markdown。"""
    user = {
        "samples": [
            {
                "scenario_id": str(sample["scenario_id"]),
                "messages": sample["messages"],
            }
            for sample in samples
        ],
        "output_schema": {
            "judgments": [
                {
                    "scenario_id": "与输入完全一致",
                    "predicted_candidate_name": "七个候选之一",
                    "observed_phenomenon": "七种现象之一",
                    "observed_source_candidate_name": "候选名或null",
                    "intent_change_is_direct": True,
                    "quality": {field: True for field in QUALITY_FIELDS},
                    "issues": ["问题；没有则空数组"],
                }
            ]
        },
        "output_requirement": "每个输入 scenario_id 恰好输出一次，不得遗漏、重复或增加其它 ID。",
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def directness_messages(
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a strict, label-blind audit for direct IntentChange final turns."""

    system = """你是 IntentChange 最后一轮表达纯净度的极严格审计员。只判断表达方式，不判断候选标签。

唯一合格条件：最后一条 user 消息从新需求本身开始，整条消息只陈述新需求及其背景、对象、动作、条件或问题。普通请求礼貌词（如“请”“麻烦”）可以属于新需求。

以下任一情况都必须判为不合格，即使后面的新需求非常清楚：
1. 回应、确认、致谢或评价上一轮回答；
2. 提及结束、取消、放弃、搁置或改变原任务；
3. 使用承接或切换话题的元话语引出新需求；
4. 把新需求说成“顺便”附带提出的事情，而不是直接提出。

不合格示例：“好的，我回头试试。对了，我想……”“谢谢。帮我……”“明白了，那查一下……”“顺便帮我……”“换个话题……”“那个不用了，帮我……”。合格示例：“我想买台轻薄本，预算七千，主要用于写代码。”“查一下宁德时代今天的收盘价和涨跌幅。”

审计完整对话只为定位最后一轮。不要输出或推断候选名。每个输入 scenario_id 恰好输出一次，只输出 JSON 对象。"""
    user = {
        "samples": [
            {
                "scenario_id": str(sample["scenario_id"]),
                "messages": sample["messages"],
            }
            for sample in samples
        ],
        "output_schema": {
            "audits": [
                {
                    "scenario_id": "与输入完全一致",
                    "contains_only_new_request": True,
                    "references_previous_exchange": False,
                    "uses_transition_or_acknowledgment": False,
                    "direct_final_request": True,
                    "has_switch_meta_language": False,
                    "reason": "一句简短理由",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating an accidental fenced wrapper."""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise Top1DataError("model response contains no JSON object")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise Top1DataError(f"invalid model JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError("model response must be a JSON object")
    return payload


def parse_generated_samples(
    content: str,
    assigned: Mapping[str, DialogueBlueprint],
    candidate_names: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse and structurally validate one generation batch."""

    errors = {scenario_id: [] for scenario_id in assigned}
    try:
        payload = parse_json_object(content)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in assigned}
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        return {}, {scenario_id: ["response.samples must be a list"] for scenario_id in assigned}

    parsed: dict[str, dict[str, Any]] = {}
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, Mapping):
            continue
        scenario_id = raw_sample.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in assigned:
            continue
        if scenario_id in parsed:
            errors[scenario_id].append("duplicate scenario in generation response")
            continue
        messages = raw_sample.get("messages")
        sample_errors = validate_generated_messages(
            messages,
            assigned[scenario_id],
            candidate_names,
        )
        if sample_errors:
            errors[scenario_id].extend(sample_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            "messages": [dict(message) for message in messages],
            "scenario_summary": str(raw_sample.get("scenario_summary", "")).strip(),
        }
    for scenario_id in assigned:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append("scenario missing from generation response")
    return parsed, errors


def validate_generated_messages(
    messages: Any,
    blueprint: DialogueBlueprint,
    candidate_names: Sequence[str],
) -> list[str]:
    """Apply label-independent structural quality gates."""

    issues: list[str] = []
    if not isinstance(messages, list):
        return ["messages must be a list"]
    expected_message_count = blueprint.user_turn_count * 2 - 1
    if len(messages) != expected_message_count:
        issues.append(
            f"expected {expected_message_count} alternating messages, got {len(messages)}"
        )
    total_characters = 0
    previous_content: str | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            issues.append(f"messages[{index}] must be an object")
            continue
        expected_role = "user" if index % 2 == 0 else "assistant"
        role = message.get("role")
        content = message.get("content")
        if role != expected_role:
            issues.append(f"messages[{index}] role must be {expected_role}")
        if not isinstance(content, str) or not content.strip():
            issues.append(f"messages[{index}] content must be non-empty")
            continue
        clean_content = content.strip()
        total_characters += len(clean_content)
        if len(clean_content) > 320:
            issues.append(f"messages[{index}] exceeds 320 characters")
        if previous_content == clean_content:
            issues.append("adjacent messages cannot be identical")
        previous_content = clean_content
        for candidate in candidate_names:
            if candidate in clean_content:
                issues.append(f"messages[{index}] leaks a candidate name")
                break
    if total_characters > 2_000:
        issues.append("conversation exceeds 2,000 characters")
    return issues


def parse_judgments(
    content: str,
    assigned_ids: Iterable[str],
    candidate_names: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse one blind-judge response with strict structured fields."""

    ids = tuple(assigned_ids)
    errors = {scenario_id: [] for scenario_id in ids}
    try:
        payload = parse_json_object(content)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in ids}
    raw_judgments = payload.get("judgments")
    if not isinstance(raw_judgments, list):
        return {}, {scenario_id: ["response.judgments must be a list"] for scenario_id in ids}

    allowed_ids = set(ids)
    allowed_candidates = set(candidate_names)
    parsed: dict[str, dict[str, Any]] = {}
    for raw in raw_judgments:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = raw.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in allowed_ids:
            continue
        if scenario_id in parsed:
            parsed.pop(scenario_id)
            errors[scenario_id].append("duplicate scenario in judgment response")
            continue
        item_errors: list[str] = []
        predicted = raw.get("predicted_candidate_name")
        source = raw.get("observed_source_candidate_name")
        phenomenon = raw.get("observed_phenomenon")
        quality = raw.get("quality")
        if predicted not in allowed_candidates:
            item_errors.append("invalid predicted candidate")
        if source is not None and source not in allowed_candidates:
            item_errors.append("invalid observed source candidate")
        if phenomenon not in OBSERVED_PHENOMENA:
            item_errors.append("invalid observed phenomenon")
        if not isinstance(raw.get("intent_change_is_direct"), bool):
            item_errors.append("intent_change_is_direct must be boolean")
        if not isinstance(quality, Mapping):
            item_errors.append("quality must be an object")
        else:
            for field in QUALITY_FIELDS:
                if not isinstance(quality.get(field), bool):
                    item_errors.append(f"quality.{field} must be boolean")
        issues = raw.get("issues", [])
        if not isinstance(issues, list) or any(not isinstance(value, str) for value in issues):
            item_errors.append("issues must be a string list")
        if item_errors:
            errors[scenario_id].extend(item_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            "predicted_candidate_name": predicted,
            "observed_phenomenon": phenomenon,
            "observed_source_candidate_name": source,
            "intent_change_is_direct": raw["intent_change_is_direct"],
            "quality": {field: bool(quality[field]) for field in QUALITY_FIELDS},
            "issues": list(issues),
        }
    for scenario_id in ids:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append("scenario missing from judgment response")
    return parsed, errors


def parse_directness_audits(
    content: str,
    assigned_ids: Iterable[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Parse strict IntentChange directness audit output."""

    ids = tuple(assigned_ids)
    errors = {scenario_id: [] for scenario_id in ids}
    try:
        payload = parse_json_object(content)
    except Top1DataError as exc:
        return {}, {scenario_id: [str(exc)] for scenario_id in ids}
    raw_audits = payload.get("audits")
    if not isinstance(raw_audits, list):
        return {}, {scenario_id: ["response.audits must be a list"] for scenario_id in ids}
    allowed_ids = set(ids)
    parsed: dict[str, dict[str, Any]] = {}
    for raw in raw_audits:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = raw.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id not in allowed_ids:
            continue
        if scenario_id in parsed:
            parsed.pop(scenario_id)
            errors[scenario_id].append("duplicate scenario in directness response")
            continue
        only_new = raw.get("contains_only_new_request")
        previous = raw.get("references_previous_exchange")
        transition = raw.get("uses_transition_or_acknowledgment")
        direct = raw.get("direct_final_request")
        meta = raw.get("has_switch_meta_language")
        reason = raw.get("reason")
        item_errors: list[str] = []
        if not isinstance(only_new, bool):
            item_errors.append("contains_only_new_request must be boolean")
        if not isinstance(previous, bool):
            item_errors.append("references_previous_exchange must be boolean")
        if not isinstance(transition, bool):
            item_errors.append("uses_transition_or_acknowledgment must be boolean")
        if not isinstance(direct, bool):
            item_errors.append("direct_final_request must be boolean")
        if not isinstance(meta, bool):
            item_errors.append("has_switch_meta_language must be boolean")
        if not isinstance(reason, str):
            item_errors.append("reason must be a string")
        if item_errors:
            errors[scenario_id].extend(item_errors)
            continue
        parsed[scenario_id] = {
            "scenario_id": scenario_id,
            "contains_only_new_request": only_new,
            "references_previous_exchange": previous,
            "uses_transition_or_acknowledgment": transition,
            "direct_final_request": direct,
            "has_switch_meta_language": meta,
            "reason": reason.strip(),
        }
    for scenario_id in ids:
        if scenario_id not in parsed and not errors[scenario_id]:
            errors[scenario_id].append("scenario missing from directness response")
    return parsed, errors


def combine_directness_audits(
    audits_by_model: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine independent directness judgments with reject-on-disagreement gates."""

    if len(audits_by_model) < 2:
        raise Top1DataError("at least two directness audits are required")
    items = list(audits_by_model.items())
    return {
        "contains_only_new_request": all(
            audit["contains_only_new_request"] is True for _, audit in items
        ),
        "references_previous_exchange": any(
            audit["references_previous_exchange"] is True for _, audit in items
        ),
        "uses_transition_or_acknowledgment": any(
            audit["uses_transition_or_acknowledgment"] is True
            for _, audit in items
        ),
        "direct_final_request": all(
            audit["direct_final_request"] is True for _, audit in items
        ),
        "has_switch_meta_language": any(
            audit["has_switch_meta_language"] is True for _, audit in items
        ),
        "reason": " | ".join(
            f"{model}: {audit['reason']}" for model, audit in items
        ),
        "model_audits": [
            {"model": model, "audit": dict(audit)} for model, audit in items
        ],
    }


def acceptance_reasons(
    blueprint: DialogueBlueprint,
    labeler: Mapping[str, Any] | None,
    reviewer: Mapping[str, Any] | None,
    directness: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return structured reasons preventing one sample from being accepted."""

    reasons: list[str] = []
    for role, judgment in (("labeler", labeler), ("reviewer", reviewer)):
        if judgment is None:
            reasons.append(f"missing_{role}_judgment")
            continue
        if judgment.get("predicted_candidate_name") != blueprint.target_candidate_name:
            reasons.append(f"{role}_label_mismatch")
        quality = judgment.get("quality")
        if not isinstance(quality, Mapping):
            reasons.append(f"{role}_quality_missing")
        else:
            for field in QUALITY_FIELDS:
                if quality.get(field) is not True:
                    reasons.append(f"{role}_quality_{field}")

    if reviewer is not None:
        if reviewer.get("observed_phenomenon") != blueprint.phenomenon:
            reasons.append("reviewer_phenomenon_mismatch")
        if blueprint.phenomenon == "intent_change":
            if reviewer.get("observed_source_candidate_name") != blueprint.source_candidate_name:
                reasons.append("reviewer_source_candidate_mismatch")
            if reviewer.get("intent_change_is_direct") is not True:
                reasons.append("intent_change_not_direct")
    if blueprint.phenomenon == "intent_change":
        if directness is None:
            reasons.append("missing_directness_audit")
        else:
            if directness.get("contains_only_new_request") is not True:
                reasons.append("directness_not_only_new_request")
            if directness.get("references_previous_exchange") is not False:
                reasons.append("directness_references_previous_exchange")
            if directness.get("uses_transition_or_acknowledgment") is not False:
                reasons.append("directness_transition_or_acknowledgment")
            if directness.get("direct_final_request") is not True:
                reasons.append("directness_final_request_failed")
            if directness.get("has_switch_meta_language") is not False:
                reasons.append("directness_switch_meta_language")
    return reasons


def load_api_credentials(path: str | Path) -> tuple[str, str]:
    """Read base URL and API key without exposing either through CLI arguments."""

    values: dict[str, str] = {}
    for raw_line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        key, separator, value = raw_line.partition(":")
        if not separator or not value.strip():
            raise Top1DataError("credentials file must use key:value lines")
        values[key.strip()] = value.strip()
    base_url = values.get("base_url", "").rstrip("/")
    api_key = values.get("api_key", "")
    if not base_url or not api_key:
        raise Top1DataError("credentials file must define base_url and api_key")
    return base_url, api_key


class OpenAICompatibleClient:
    """Small retrying JSON client that keeps synthesis dependency-free."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 180.0,
        request_attempts: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.request_attempts = request_attempts

    def chat_json(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ModelCall:
        """Call `/chat/completions` and return the assistant JSON text."""

        payload = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.request_attempts + 1):
            started = time.monotonic()
            try:
                def curl_quote(value: str) -> str:
                    return value.replace("\\", "\\\\").replace('"', '\\"')

                with tempfile.NamedTemporaryFile(
                    prefix="top1-synthesis-",
                    suffix=".json",
                ) as body_file:
                    body_file.write(encoded)
                    body_file.flush()
                    config = "\n".join(
                        (
                            "silent",
                            "show-error",
                            "fail",
                            f"max-time = {self.timeout_seconds}",
                            f'url = "{curl_quote(f"{self.base_url}/chat/completions")}"',
                            'request = "POST"',
                            f'header = "Authorization: Bearer {curl_quote(self._api_key)}"',
                            'header = "Content-Type: application/json"',
                            f'data-binary = "@{curl_quote(body_file.name)}"',
                        )
                    )
                    completed = subprocess.run(
                        ("curl", "--config", "-"),
                        input=config.encode("utf-8"),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.timeout_seconds + 10,
                        check=False,
                    )
                if completed.returncode != 0:
                    detail = completed.stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(
                        f"curl model request failed with exit {completed.returncode}: {detail[-300:]}"
                    )
                raw_payload = json.loads(completed.stdout.decode("utf-8"))
                choice = raw_payload["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise Top1DataError("model returned empty completion content")
                usage = raw_payload.get("usage")
                return ModelCall(
                    content=content,
                    usage=dict(usage) if isinstance(usage, Mapping) else {},
                    finish_reason=choice.get("finish_reason"),
                    elapsed_seconds=time.monotonic() - started,
                )
            except (
                subprocess.TimeoutExpired,
                RuntimeError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                Top1DataError,
            ) as exc:
                last_error = exc
            if attempt < self.request_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"model request failed after {self.request_attempts} attempts: {last_error}")


def content_sha256(value: str) -> str:
    """Hash prompt or endpoint text for reproducible manifests."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
