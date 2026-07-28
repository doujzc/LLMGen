"""Resolve and validate training-console configuration without importing training code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlsplit


DATASETS = {
    "clawhub": {
        "label": "clawhub · 1,000 Skills",
        "config": "configs/clawhub.env",
        "profile": "clawhub-full-4gpu",
    },
    "light": {
        "label": "light · 301 Skills",
        "config": "configs/light.env",
        "profile": "light-full-4gpu",
    },
}

PIPELINE_COMMANDS = (
    "full",
    "prepare",
    "train-tokenizer",
    "export-codes",
    "build-router-data",
    "train-memorization",
    "train-retrieval",
    "evaluate",
    "diagnose",
    "diagnose-memorization",
    "export-web",
)

STAGES = (
    {
        "id": "base",
        "number": "",
        "label": "基础配置",
        "description": "全局与资源设置",
    },
    {
        "id": "embedding",
        "number": "01",
        "label": "数据与 Embedding",
        "description": "数据源与向量化",
    },
    {
        "id": "tokenizer",
        "number": "02",
        "label": "层级 Tokenizer",
        "description": "分层编码与词表构建",
    },
    {
        "id": "code",
        "number": "03",
        "label": "Code 导出与质量门禁",
        "description": "Code 导出与校验",
    },
    {
        "id": "router_data",
        "number": "04",
        "label": "Router 数据",
        "description": "训练样本与标签构建",
    },
    {
        "id": "memorization",
        "number": "05",
        "label": "Memorization",
        "description": "记忆阶段训练",
    },
    {
        "id": "alignment",
        "number": "06a",
        "label": "Alignment",
        "description": "单 Skill 对齐训练",
    },
    {
        "id": "retrieval",
        "number": "06b",
        "label": "Retrieval",
        "description": "多 Skill 检索训练",
    },
    {
        "id": "evaluation",
        "number": "07",
        "label": "评估",
        "description": "闭集评估与报告",
    },
)


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    stage: str
    section: str
    kind: str = "text"
    help: str = ""
    source: str = "configs/skillret.env"
    options: tuple[str, ...] = ()
    advanced: bool = False
    required: bool = False
    placeholder: str = ""
    derived_from: str = ""
    derived_suffix: str = ""

    def payload(self, dataset: str) -> dict[str, Any]:
        result = asdict(self)
        result["options"] = list(self.options)
        if result["source"] == "dataset":
            result["source"] = DATASETS[dataset]["config"]
        return result


def _field(
    key: str,
    label: str,
    stage: str,
    section: str,
    *,
    kind: str = "text",
    help: str = "",
    source: str = "configs/skillret.env",
    options: Iterable[str] = (),
    advanced: bool = False,
    required: bool = False,
    placeholder: str = "",
    derived_from: str = "",
    derived_suffix: str = "",
) -> FieldSpec:
    return FieldSpec(
        key=key,
        label=label,
        stage=stage,
        section=section,
        kind=kind,
        help=help,
        source=source,
        options=tuple(options),
        advanced=advanced,
        required=required,
        placeholder=placeholder,
        derived_from=derived_from,
        derived_suffix=derived_suffix,
    )


FIELDS = (
    _field(
        "DATASET",
        "数据集",
        "base",
        "运行身份",
        kind="select",
        options=DATASETS,
        source="控制台",
        required=True,
    ),
    _field(
        "PIPELINE_COMMAND",
        "执行范围",
        "base",
        "运行身份",
        kind="select",
        options=PIPELINE_COMMANDS,
        source="控制台",
        required=True,
    ),
    _field(
        "RUN_DIR",
        "运行目录",
        "base",
        "运行身份",
        kind="path",
        source="dataset",
        help="整个训练流程的工作目录；各阶段产物目录默认随它联动。",
        required=True,
    ),
    _field(
        "PYTHON",
        "Python 可执行文件",
        "base",
        "运行环境",
        kind="path",
        help="默认优先使用仓库 .venv，否则使用当前环境的 python。",
        advanced=True,
    ),
    _field(
        "DEVICE",
        "训练设备",
        "base",
        "运行环境",
        kind="text",
        required=True,
    ),
    _field(
        "CUDA_VISIBLE_DEVICES",
        "可见 GPU",
        "base",
        "运行环境",
        kind="gpu_csv",
        help="逗号分隔，例如 0,1,2,3。",
        required=True,
    ),
    _field(
        "SKIP_PREPARE",
        "完整流程跳过预处理",
        "base",
        "运行环境",
        kind="bool",
        help="仅在 processed/ 和 embeddings/ 已经匹配当前数据时启用。",
    ),
    _field(
        "DATASET_DIR",
        "数据集目录",
        "embedding",
        "数据路径",
        kind="path",
        source="dataset",
        required=True,
    ),
    _field(
        "PROCESSED_DIR",
        "预处理目录",
        "embedding",
        "数据路径",
        kind="path",
        source="RUN_DIR",
        help="默认跟随 $RUN_DIR/processed；手动修改可单独覆盖。",
        required=True,
        derived_from="RUN_DIR",
        derived_suffix="processed",
    ),
    _field(
        "EMBEDDING_DIR",
        "Embedding 目录",
        "embedding",
        "数据路径",
        kind="path",
        source="RUN_DIR",
        help="默认跟随 $RUN_DIR/embeddings；手动修改可单独覆盖。",
        required=True,
        derived_from="RUN_DIR",
        derived_suffix="embeddings",
    ),
    _field(
        "EMBEDDING_PROVIDER",
        "Embedding Provider",
        "embedding",
        "Embedding 接口",
        kind="select",
        options=("openai", "sentence-transformers"),
    ),
    _field(
        "EMBEDDING_MODEL",
        "Embedding 模型",
        "embedding",
        "Embedding 接口",
        required=True,
    ),
    _field(
        "EMBEDDING_BASE_URL",
        "OpenAI-compatible Base URL",
        "embedding",
        "Embedding 接口",
        kind="url",
    ),
    _field(
        "EMBEDDING_BATCH_SIZE",
        "请求 Batch Size",
        "embedding",
        "Embedding 请求",
        kind="int",
    ),
    _field(
        "EMBEDDING_DIMENSIONS",
        "Embedding 维度",
        "embedding",
        "Embedding 请求",
        kind="optional_int",
        advanced=True,
    ),
    _field(
        "EMBEDDING_TIMEOUT",
        "请求超时（秒）",
        "embedding",
        "Embedding 请求",
        kind="int",
        advanced=True,
    ),
    _field(
        "EMBEDDING_MAX_RETRIES",
        "最大重试次数",
        "embedding",
        "Embedding 请求",
        kind="int",
        advanced=True,
    ),
    _field(
        "EMBEDDING_MAX_BATCH_CHARS",
        "单批最大字符数",
        "embedding",
        "Embedding 请求",
        kind="optional_int",
        source="configs/closedset.env",
        advanced=True,
    ),
    _field(
        "EMBEDDING_MAX_SKILL_CHARS",
        "单 Skill 最大字符数",
        "embedding",
        "Embedding 请求",
        kind="optional_int",
        source="configs/closedset.env",
        advanced=True,
    ),
    _field(
        "STAGE1_DIR",
        "Stage 1 输出目录",
        "tokenizer",
        "产物路径",
        kind="path",
        source="RUN_DIR",
        help="默认跟随 $RUN_DIR/stage1；手动修改可单独覆盖。",
        required=True,
        derived_from="RUN_DIR",
        derived_suffix="stage1",
    ),
    _field(
        "NUM_LEVELS",
        "Code 层数",
        "tokenizer",
        "层级结构",
        kind="int",
        source="configs/closedset.env",
    ),
    _field(
        "BRANCHING_FACTORS",
        "各层分支数",
        "tokenizer",
        "层级结构",
        kind="number_list",
        source="dataset",
        help="空格分隔；数量必须等于 Code 层数。",
    ),
    _field(
        "SK_EPSILONS",
        "Sinkhorn Epsilon",
        "tokenizer",
        "层级结构",
        kind="number_list",
        source="configs/closedset.env",
    ),
    _field(
        "RQ_LAYERS",
        "RQ-VAE 隐层",
        "tokenizer",
        "网络结构",
        kind="number_list",
        source="dataset",
    ),
    _field(
        "TOKENIZER_E_DIM",
        "Code Embedding 维度",
        "tokenizer",
        "网络结构",
        kind="int",
    ),
    _field(
        "TOKENIZER_BETA",
        "Commitment Beta",
        "tokenizer",
        "优化参数",
        kind="float",
        source="configs/closedset.env",
    ),
    _field(
        "TOKENIZER_EPOCHS",
        "训练 Epochs",
        "tokenizer",
        "优化参数",
        kind="int",
        source="dataset",
    ),
    _field(
        "TOKENIZER_BATCH_SIZE",
        "Batch Size",
        "tokenizer",
        "优化参数",
        kind="int",
        source="dataset",
    ),
    _field(
        "TOKENIZER_LR",
        "学习率",
        "tokenizer",
        "优化参数",
        kind="float",
        source="configs/closedset.env",
    ),
    _field(
        "TOKENIZER_SCHEDULER",
        "Scheduler",
        "tokenizer",
        "优化参数",
        kind="select",
        source="configs/closedset.env",
        options=("constant", "cosine", "linear"),
    ),
    _field(
        "TOKENIZER_WARMUP_RATIO",
        "Warmup Ratio",
        "tokenizer",
        "优化参数",
        kind="float",
        source="configs/closedset.env",
    ),
    _field(
        "TOKENIZER_EVAL_EVERY",
        "每 N Epoch 评估",
        "tokenizer",
        "优化参数",
        kind="int",
        source="configs/closedset.env",
    ),
    _field(
        "TOKENIZER_GRAPH_LAMBDA",
        "协作图损失权重",
        "tokenizer",
        "优化参数",
        kind="float",
        advanced=True,
    ),
    _field(
        "TOKENIZER_AMP_DTYPE",
        "AMP 精度",
        "tokenizer",
        "优化参数",
        kind="select",
        options=("bf16", "fp16", "none"),
        advanced=True,
    ),
    _field(
        "TOKENIZER_RESUME",
        "恢复 Checkpoint",
        "tokenizer",
        "恢复",
        kind="path",
        advanced=True,
    ),
    _field(
        "CODEBOOK_VERSION",
        "Codebook 版本",
        "tokenizer",
        "产物标识",
        source="dataset",
    ),
    _field(
        "INDEX_DIR",
        "索引输出目录",
        "code",
        "产物路径",
        kind="path",
        source="RUN_DIR",
        help="默认跟随 $RUN_DIR/index；手动修改可单独覆盖。",
        required=True,
        derived_from="RUN_DIR",
        derived_suffix="index",
    ),
    _field(
        "CODE_SPLITS",
        "导出 Splits",
        "code",
        "Code 导出",
        source="configs/closedset.env",
    ),
    _field(
        "CODE_EXPORT_BATCH_SIZE",
        "导出 Batch Size",
        "code",
        "Code 导出",
        kind="int",
    ),
    _field(
        "CODE_ASSIGNMENT_MODE",
        "分配方式",
        "code",
        "Code 分配",
        kind="select",
        options=("sinkhorn", "nearest", "balanced_hierarchical"),
        source="configs/closedset.env",
    ),
    _field(
        "CODE_ASSIGNMENT_EXACT_GROUP_SIZE",
        "精确分配分组大小",
        "code",
        "Code 分配",
        kind="int",
        source="dataset",
        advanced=True,
    ),
    _field(
        "CODE_QUALITY_GATE_SPLIT",
        "质量门禁 Split",
        "code",
        "质量门禁",
        source="configs/closedset.env",
    ),
    _field(
        "CODE_MAX_COLLISION_RATE",
        "最大碰撞率",
        "code",
        "质量门禁",
        kind="float",
        source="dataset",
    ),
    _field(
        "CODE_MAX_RAW_COLLISION_RATE",
        "最大原始碰撞率",
        "code",
        "质量门禁",
        kind="float",
        source="dataset",
    ),
    _field(
        "CODE_MAX_BUCKET_SIZE",
        "最大碰撞桶大小",
        "code",
        "质量门禁",
        kind="int",
        source="configs/closedset.env",
    ),
    _field(
        "CODE_MIN_LEVEL_UTILIZATION",
        "最小层利用率",
        "code",
        "质量门禁",
        kind="float",
        source="dataset",
    ),
    _field(
        "CODE_MIN_NORMALIZED_ENTROPY",
        "最小归一化熵",
        "code",
        "质量门禁",
        kind="float",
        source="dataset",
    ),
    _field(
        "CODE_MIN_RAW_LEVEL_UTILIZATION",
        "各层最小原始利用率",
        "code",
        "质量门禁",
        kind="number_list",
        source="dataset",
    ),
    _field(
        "CODE_MIN_RAW_NORMALIZED_ENTROPY",
        "最小原始归一化熵",
        "code",
        "质量门禁",
        kind="float",
        source="dataset",
    ),
    _field(
        "ROUTER_DATA_DIR",
        "Router 数据目录",
        "router_data",
        "产物路径",
        kind="path",
        source="RUN_DIR",
        help="默认跟随 $RUN_DIR/router_data；手动修改可单独覆盖。",
        required=True,
        derived_from="RUN_DIR",
        derived_suffix="router_data",
    ),
    _field(
        "MEMORIZATION_VALIDATION_FRACTION",
        "Memorization 验证比例",
        "router_data",
        "数据拆分",
        kind="float",
    ),
    _field(
        "ROUTER_VALIDATION_FRACTION",
        "Retrieval 验证比例",
        "router_data",
        "数据拆分",
        kind="float",
        source="dataset",
    ),
    _field(
        "ROUTER_DATA_SEED",
        "数据随机种子",
        "router_data",
        "数据拆分",
        kind="int",
    ),
    _field(
        "ROUTER_OUTPUT_DIR",
        "Router 输出目录",
        "memorization",
        "模型与产物",
        kind="path",
        source="RUN_DIR",
        help="默认跟随 $RUN_DIR/router；手动修改可单独覆盖。",
        required=True,
        derived_from="RUN_DIR",
        derived_suffix="router",
    ),
    _field(
        "ROUTER_MODEL",
        "Router 模型",
        "memorization",
        "模型与产物",
        required=True,
    ),
    _field(
        "ROUTER_FINETUNE_MODE",
        "微调方式",
        "memorization",
        "模型与产物",
        kind="select",
        options=("full", "lora"),
        source="configs/closedset.env",
    ),
    _field(
        "ROUTER_MEMORIZATION_EPOCHS",
        "Memorization Epochs",
        "memorization",
        "阶段超参数",
        kind="int",
        source="configs/closedset.env",
    ),
    _field(
        "ROUTER_MEMORIZATION_LR",
        "Memorization 学习率",
        "memorization",
        "阶段超参数",
        kind="float",
    ),
    _field(
        "ROUTER_RESUME_MEMORIZATION",
        "Memorization 恢复点",
        "memorization",
        "恢复",
        kind="path",
        advanced=True,
    ),
    _field(
        "ROUTER_ALIGNMENT_EPOCHS",
        "Alignment Epochs",
        "alignment",
        "阶段超参数",
        kind="int",
        source="configs/closedset.env",
    ),
    _field(
        "ROUTER_ALIGNMENT_LR",
        "Alignment 学习率",
        "alignment",
        "阶段超参数",
        kind="float",
        source="configs/closedset.env",
    ),
    _field(
        "ROUTER_RESUME_ALIGNMENT",
        "Alignment 恢复点",
        "alignment",
        "恢复",
        kind="path",
        advanced=True,
    ),
    _field(
        "ROUTER_RETRIEVAL_EPOCHS",
        "Retrieval Epochs",
        "retrieval",
        "阶段超参数",
        kind="int",
        source="configs/closedset.env",
    ),
    _field(
        "ROUTER_RETRIEVAL_LR",
        "Retrieval 学习率",
        "retrieval",
        "阶段超参数",
        kind="float",
    ),
    _field(
        "ROUTER_RETRIEVAL_REPLAY_FRACTION",
        "Replay 比例",
        "retrieval",
        "阶段超参数",
        kind="float",
        source="configs/closedset.env",
    ),
    _field(
        "ROUTER_RESUME_RETRIEVAL",
        "Retrieval 恢复点",
        "retrieval",
        "恢复",
        kind="path",
        advanced=True,
    ),
    _field(
        "ROUTER_NUM_GPUS",
        "GPU 数量",
        "retrieval",
        "分布式与资源",
        kind="int",
    ),
    _field(
        "ROUTER_DEEPSPEED_CONFIG",
        "DeepSpeed 配置",
        "retrieval",
        "分布式与资源",
        kind="path",
    ),
    _field(
        "ROUTER_PRECISION",
        "训练精度",
        "retrieval",
        "分布式与资源",
        kind="select",
        options=("bf16", "fp16", "fp32"),
    ),
    _field(
        "ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE",
        "每卡训练 Batch",
        "retrieval",
        "Batch 与序列",
        kind="int",
    ),
    _field(
        "ROUTER_PER_DEVICE_EVAL_BATCH_SIZE",
        "每卡评估 Batch",
        "retrieval",
        "Batch 与序列",
        kind="int",
        advanced=True,
    ),
    _field(
        "ROUTER_GRADIENT_ACCUMULATION_STEPS",
        "梯度累积步数",
        "retrieval",
        "Batch 与序列",
        kind="int",
    ),
    _field(
        "ROUTER_MAX_LENGTH",
        "最大序列长度",
        "retrieval",
        "Batch 与序列",
        kind="int",
    ),
    _field(
        "ROUTER_GRADIENT_CHECKPOINTING",
        "梯度检查点",
        "retrieval",
        "显存与性能",
        kind="bool",
    ),
    _field(
        "ROUTER_GRADIENT_CHECKPOINTING_MODE",
        "梯度检查点模式",
        "retrieval",
        "显存与性能",
        kind="select",
        options=("auto", "reentrant", "non_reentrant"),
        advanced=True,
    ),
    _field(
        "ROUTER_WEIGHT_DECAY",
        "Weight Decay",
        "retrieval",
        "优化与调度",
        kind="float",
    ),
    _field(
        "ROUTER_WARMUP_RATIO",
        "Warmup Ratio",
        "retrieval",
        "优化与调度",
        kind="float",
    ),
    _field(
        "ROUTER_LOGGING_STEPS",
        "日志步数",
        "retrieval",
        "保存与日志",
        kind="int",
        advanced=True,
    ),
    _field(
        "ROUTER_SAVE_STEPS",
        "保存步数",
        "retrieval",
        "保存与日志",
        kind="int",
        source="dataset",
    ),
    _field(
        "ROUTER_EVAL_STEPS",
        "评估步数",
        "retrieval",
        "保存与日志",
        kind="int",
        source="dataset",
    ),
    _field(
        "ROUTER_SAVE_TOTAL_LIMIT",
        "最多保留 Checkpoint",
        "retrieval",
        "保存与日志",
        kind="int",
        advanced=True,
    ),
    _field(
        "ROUTER_DATALOADER_NUM_WORKERS",
        "Dataloader Workers",
        "retrieval",
        "显存与性能",
        kind="int",
        advanced=True,
    ),
    _field(
        "ROUTER_SEED",
        "训练随机种子",
        "retrieval",
        "优化与调度",
        kind="int",
        advanced=True,
    ),
    _field(
        "ROUTER_TRUST_REMOTE_CODE",
        "Trust Remote Code",
        "retrieval",
        "模型加载",
        kind="bool",
        advanced=True,
    ),
    _field(
        "ROUTER_LORA_R",
        "LoRA Rank",
        "retrieval",
        "LoRA",
        kind="int",
        advanced=True,
    ),
    _field(
        "ROUTER_LORA_ALPHA",
        "LoRA Alpha",
        "retrieval",
        "LoRA",
        kind="int",
        advanced=True,
    ),
    _field(
        "ROUTER_LORA_DROPOUT",
        "LoRA Dropout",
        "retrieval",
        "LoRA",
        kind="float",
        advanced=True,
    ),
    _field(
        "ROUTER_LORA_TARGET_MODULES",
        "LoRA Target Modules",
        "retrieval",
        "LoRA",
        advanced=True,
    ),
    _field(
        "ROUTER_LORA_MODULES_TO_SAVE",
        "LoRA Modules To Save",
        "retrieval",
        "LoRA",
        advanced=True,
    ),
    _field(
        "EVAL_PROTOCOL",
        "评估协议",
        "evaluation",
        "评估范围",
        kind="select",
        options=("closedset", "unseen", "both"),
        source="configs/closedset.env",
    ),
    _field(
        "QUERY_SET",
        "Query Set",
        "evaluation",
        "评估范围",
        kind="select",
        options=("validation", "dataset-validation", "test", "train"),
        source="configs/closedset.env",
    ),
    _field(
        "EVAL_DTYPE",
        "推理精度",
        "evaluation",
        "推理参数",
        kind="select",
        options=("bfloat16", "float16", "float32"),
    ),
    _field(
        "EVAL_BATCH_SIZE",
        "评估 Batch Size",
        "evaluation",
        "推理参数",
        kind="int",
    ),
    _field(
        "EVAL_MAX_CODE_PATHS",
        "最多生成路径",
        "evaluation",
        "检索参数",
        kind="int",
    ),
    _field(
        "EVAL_TOP_K",
        "候选 Top K",
        "evaluation",
        "检索参数",
        kind="int",
    ),
    _field(
        "EVAL_CUTOFFS",
        "Recall Cutoffs",
        "evaluation",
        "检索参数",
        kind="number_list",
    ),
    _field(
        "EVAL_DIR",
        "评估输出目录",
        "evaluation",
        "产物路径",
        kind="path",
        source="RUN_DIR",
        help="默认跟随 $RUN_DIR/evaluation；手动修改可单独覆盖。",
        advanced=True,
        derived_from="RUN_DIR",
        derived_suffix="evaluation",
    ),
)

FIELD_BY_KEY = {field.key: field for field in FIELDS}
RUN_DIR_DERIVED_PATHS = {
    field.key: field.derived_suffix
    for field in FIELDS
    if field.derived_from == "RUN_DIR" and field.derived_suffix
}
ALLOWED_KEYS = frozenset(FIELD_BY_KEY)
CONFIG_SOURCE_RUNTIME_KEYS = frozenset(
    {
        "CONDA_PREFIX",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "VIRTUAL_ENV",
    }
)
PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class ConfigValidationError(ValueError):
    """Raised when a submitted console configuration is unsafe or invalid."""

    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        super().__init__("; ".join(error["message"] for error in errors))


def _normalize_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return "1"
    if normalized in {"0", "false", "no", "off"}:
        return "0"
    raise ValueError("必须是启用或关闭")


def _validate_number(value: str, integer: bool, optional: bool = False) -> str:
    stripped = value.strip()
    if optional and not stripped:
        return ""
    if integer:
        parsed = int(stripped)
        if parsed < 0:
            raise ValueError("必须大于或等于 0")
    else:
        float(stripped)
    return stripped


_SENSITIVE_NAME_COMPONENTS = frozenset(
    {
        "APIKEY",
        "AUTH",
        "AUTHORIZATION",
        "COOKIE",
        "CREDENTIAL",
        "CREDENTIALS",
        "KEY",
        "PASSWORD",
        "PASSWD",
        "PWD",
        "SECRET",
        "SIG",
        "SIGNATURE",
        "TOKEN",
    }
)
_SENSITIVE_NAME_COMPACT_FRAGMENTS = frozenset(
    {
        "APIKEY",
        "ACCESSKEY",
        "SECRETKEY",
        "SIGNINGKEY",
        "PRIVATEKEY",
        "ACCESSTOKEN",
        "REFRESHTOKEN",
        "BEARERTOKEN",
        "AUTHTOKEN",
        "CLIENTSECRET",
        "TOKEN",
        "SECRET",
        "KEY",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "BEARER",
        "COOKIE",
        "SESSION",
        "JWT",
        "SIGNATURE",
    }
)


def _has_sensitive_name(name: str) -> bool:
    camel_separated = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        str(name),
    )
    components = {
        component
        for component in re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            camel_separated,
        ).upper().split("_")
        if component
    }
    compact = re.sub(r"[^A-Za-z0-9]+", "", camel_separated).upper()
    return bool(
        components & _SENSITIVE_NAME_COMPONENTS
        or any(
            fragment in compact
            for fragment in _SENSITIVE_NAME_COMPACT_FRAGMENTS
        )
    )


def _validate_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        # Accessing port also rejects malformed or out-of-range port values.
        parsed.port
    except ValueError as exc:
        raise ValueError("必须是合法的 HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("必须以 http:// 或 https:// 开头并包含主机名")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL 不能包含用户名或密码")
    sensitive_parameters = [
        key
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        if _has_sensitive_name(key)
    ]
    if sensitive_parameters:
        raise ValueError(
            "URL query 不能包含密钥、token、密码或其他敏感参数"
        )
    return value


def _normalize_value(spec: FieldSpec, raw: Any) -> str:
    value = "" if raw is None else str(raw).strip()
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError("不能包含换行或 NUL")
    if len(value) > 4096:
        raise ValueError("长度不能超过 4096 个字符")
    if spec.required and not value:
        raise ValueError("不能为空")
    if spec.kind == "bool":
        return _normalize_bool(raw)
    if spec.kind == "int":
        return _validate_number(value, integer=True)
    if spec.kind == "optional_int":
        return _validate_number(value, integer=True, optional=True)
    if spec.kind == "float":
        return _validate_number(value, integer=False)
    if spec.kind == "select" and value not in spec.options:
        raise ValueError(f"必须是：{', '.join(spec.options)}")
    if spec.kind == "gpu_csv":
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if not tokens or any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", token) for token in tokens):
            raise ValueError("必须是逗号分隔的 GPU 标识")
        return ",".join(tokens)
    if spec.kind == "number_list":
        tokens = value.split()
        if not tokens:
            raise ValueError("至少需要一个数值")
        for token in tokens:
            float(token)
        return " ".join(tokens)
    if spec.kind == "url" and value:
        return _validate_url(value)
    return value


def _run_dir_child(run_dir: str, suffix: str) -> str:
    """Return a stable shell-style child path without resolving it on this host."""

    stripped = run_dir.rstrip("/")
    if stripped:
        return f"{stripped}/{suffix}"
    if run_dir.startswith("/"):
        return f"/{suffix}"
    return suffix


def _run_dir_expression(suffix: str) -> str:
    return f"$RUN_DIR/{suffix}"


def _canonical_run_dir_expression(value: str) -> str:
    braced_prefix = "${RUN_DIR}/"
    if value.startswith(braced_prefix):
        return f"$RUN_DIR/{value[len(braced_prefix):]}"
    return value


def _expand_run_dir_expression(value: str, run_dir: str) -> str:
    prefix = "$RUN_DIR/"
    if value.startswith(prefix):
        return _run_dir_child(run_dir, value[len(prefix):])
    return value


class ConfigResolver:
    """Resolve trusted shell defaults and validate versioned UI overrides."""

    def __init__(
        self,
        repo_root: Path,
        *,
        inherited_env: Mapping[str, str] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.inherited_env = dict(inherited_env or os.environ)

    def _source_config(
        self,
        dataset: str,
        *,
        preset: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if dataset not in DATASETS:
            raise ConfigValidationError(
                [{"field": "DATASET", "message": f"未知数据集：{dataset}"}]
            )
        config_path = self.repo_root / DATASETS[dataset]["config"]
        if not config_path.is_file():
            raise ConfigValidationError(
                [{"field": "DATASET", "message": f"配置不存在：{config_path}"}]
            )
        # Dataset configs are trusted repository files, but the caller's shell
        # may contain BASH_ENV, a stale SKILLRET_ROOT, or undocumented training
        # switches. Only declared configuration values and the small runtime
        # substrate needed by bash may influence default resolution.
        env = {
            key: str(value)
            for key, value in self.inherited_env.items()
            if key in ALLOWED_KEYS or key in CONFIG_SOURCE_RUNTIME_KEYS
        }
        env["SKILLRET_ROOT"] = str(self.repo_root)
        env.update({key: str(value) for key, value in (preset or {}).items()})
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'set -ae; source "$1"; env -0',
                "training-console",
                str(config_path),
            ],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ConfigValidationError(
                [
                    {
                        "field": "DATASET",
                        "message": f"无法解析 {config_path.name}：{message}",
                    }
                ]
            )
        resolved: dict[str, str] = {}
        for entry in completed.stdout.split(b"\0"):
            if not entry or b"=" not in entry:
                continue
            key_bytes, value_bytes = entry.split(b"=", 1)
            key = key_bytes.decode("utf-8", errors="replace")
            if key in ALLOWED_KEYS:
                resolved[key] = value_bytes.decode("utf-8", errors="replace")
        return resolved

    def defaults(self, dataset: str) -> dict[str, str]:
        resolved = self._source_config(dataset)
        resolved["DATASET"] = dataset
        resolved["PIPELINE_COMMAND"] = "full"
        resolved.setdefault("SKIP_PREPARE", "0")
        num_gpus = max(1, int(resolved.get("ROUTER_NUM_GPUS", "1")))
        resolved.setdefault(
            "CUDA_VISIBLE_DEVICES",
            ",".join(str(index) for index in range(num_gpus)),
        )
        resolved.setdefault("EVAL_DIR", "")
        for field in FIELDS:
            resolved.setdefault(field.key, "")
        run_dir = resolved.get("RUN_DIR", "")
        if run_dir:
            resolved.update(
                {
                    key: _run_dir_child(run_dir, suffix)
                    for key, suffix in RUN_DIR_DERIVED_PATHS.items()
                }
            )
        return resolved

    def validate(
        self,
        dataset: str,
        command: str,
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        if dataset not in DATASETS:
            errors.append({"field": "DATASET", "message": "请选择有效数据集"})
        if command not in PIPELINE_COMMANDS:
            errors.append(
                {"field": "PIPELINE_COMMAND", "message": "请选择有效执行范围"}
            )
        normalized: dict[str, str] = {}
        for key, raw in overrides.items():
            spec = FIELD_BY_KEY.get(key)
            if spec is None:
                errors.append({"field": key, "message": "不允许的配置项"})
                continue
            if key in {"DATASET", "PIPELINE_COMMAND"}:
                continue
            try:
                normalized[key] = _normalize_value(spec, raw)
            except (TypeError, ValueError) as exc:
                errors.append({"field": key, "message": str(exc)})
        if errors:
            raise ConfigValidationError(errors)

        for key in RUN_DIR_DERIVED_PATHS:
            if key in normalized:
                normalized[key] = _canonical_run_dir_expression(normalized[key])

        base = self.defaults(dataset)
        effective_run_dir = normalized.get("RUN_DIR", base.get("RUN_DIR", ""))
        runtime_overrides = {
            key: (
                _expand_run_dir_expression(value, effective_run_dir)
                if key in RUN_DIR_DERIVED_PATHS
                else value
            )
            for key, value in normalized.items()
        }
        preset = dict(runtime_overrides)
        preset["DATASET"] = dataset
        preset["PIPELINE_COMMAND"] = command
        resolved = dict(base)
        resolved.update(self._source_config(dataset, preset=preset))
        resolved.update(runtime_overrides)
        resolved["DATASET"] = dataset
        resolved["PIPELINE_COMMAND"] = command
        resolved.setdefault("SKIP_PREPARE", "0")
        run_dir = resolved.get("RUN_DIR", "")
        if run_dir:
            resolved.update(
                {
                    key: _run_dir_child(run_dir, suffix)
                    for key, suffix in RUN_DIR_DERIVED_PATHS.items()
                    if key not in runtime_overrides
                }
            )

        for field in FIELDS:
            try:
                resolved[field.key] = _normalize_value(
                    field,
                    resolved.get(field.key, ""),
                )
            except (TypeError, ValueError) as exc:
                errors.append({"field": field.key, "message": str(exc)})

        try:
            num_levels = int(resolved["NUM_LEVELS"])
            branching = resolved["BRANCHING_FACTORS"].split()
            epsilons = resolved["SK_EPSILONS"].split()
            raw_utilization = resolved["CODE_MIN_RAW_LEVEL_UTILIZATION"].split()
            if len(branching) != num_levels:
                errors.append(
                    {
                        "field": "BRANCHING_FACTORS",
                        "message": "分支数数量必须等于 Code 层数",
                    }
                )
            if len(epsilons) != num_levels:
                errors.append(
                    {
                        "field": "SK_EPSILONS",
                        "message": "Epsilon 数量必须等于 Code 层数",
                    }
                )
            if len(raw_utilization) not in {1, num_levels}:
                errors.append(
                    {
                        "field": "CODE_MIN_RAW_LEVEL_UTILIZATION",
                        "message": "利用率阈值需提供一个值或每层一个值",
                    }
                )
        except (KeyError, ValueError):
            pass

        if resolved.get("DEVICE", "").startswith("cuda"):
            visible = [
                token
                for token in resolved.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if token
            ]
            requested = int(resolved.get("ROUTER_NUM_GPUS", "0") or 0)
            if requested and len(visible) != requested:
                errors.append(
                    {
                        "field": "ROUTER_NUM_GPUS",
                        "message": (
                            "GPU 数量必须与 CUDA_VISIBLE_DEVICES 中的数量一致"
                        ),
                    }
                )

        if errors:
            raise ConfigValidationError(errors)

        contextual_defaults = dict(base)
        run_dir = resolved.get("RUN_DIR", "")
        if run_dir:
            contextual_defaults.update(
                {
                    key: _run_dir_expression(suffix)
                    for key, suffix in RUN_DIR_DERIVED_PATHS.items()
                }
            )
        effective_overrides = {
            key: value
            for key, value in normalized.items()
            if value != contextual_defaults.get(key, "")
        }
        sources = {
            field.key: (
                "本版本覆盖"
                if field.key in effective_overrides
                else field.payload(dataset)["source"]
            )
            for field in FIELDS
        }
        return {
            "dataset": dataset,
            "command": command,
            "resolved": resolved,
            "overrides": effective_overrides,
            "defaults": base,
            "sources": sources,
            "warnings": self._warnings(resolved),
            "contract": self.contract(resolved, dataset, command),
        }

    def _warnings(self, resolved: Mapping[str, str]) -> list[str]:
        warnings: list[str] = []
        deepspeed = resolved.get("ROUTER_DEEPSPEED_CONFIG", "")
        if deepspeed and deepspeed != "none":
            path = Path(deepspeed)
            if not path.is_absolute():
                path = self.repo_root / path
            if not path.is_file():
                warnings.append(f"DeepSpeed 配置尚不存在：{path}")
        if resolved.get("SKIP_PREPARE") == "1":
            for key in ("PROCESSED_DIR", "EMBEDDING_DIR"):
                path = Path(resolved.get(key, ""))
                if not path.is_absolute():
                    path = self.repo_root / path
                if not path.is_dir():
                    warnings.append(f"{key} 尚不存在，不能安全跳过预处理")
        return warnings

    def contract(
        self,
        resolved: Mapping[str, str],
        dataset: str,
        command: str,
    ) -> dict[str, Any]:
        run_dir = resolved.get("RUN_DIR", "")
        visible_gpus = [
            token
            for token in resolved.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if token
        ]
        return {
            "argv": [
                "bash",
                "scripts/router_pipeline.sh",
                dataset,
                command,
            ],
            "command_text": (
                f"bash scripts/router_pipeline.sh {dataset} {command}"
            ),
            "run_dir": run_dir,
            "checkpoint_dir": f"{run_dir.rstrip('/')}/router"
            if run_dir
            else "",
            # The durable console log lives beside run.json in the independent
            # state directory and therefore has no path until a run is created.
            "log_dir": "",
            "gpus": visible_gpus,
            "num_gpus": resolved.get("ROUTER_NUM_GPUS", ""),
            "deepspeed": resolved.get("ROUTER_DEEPSPEED_CONFIG", ""),
            "precision": resolved.get("ROUTER_PRECISION", ""),
            "num_levels": resolved.get("NUM_LEVELS", ""),
            "branching_factors": resolved.get("BRANCHING_FACTORS", ""),
        }

    def schema(self, dataset: str) -> dict[str, Any]:
        defaults = self.defaults(dataset)
        display_defaults = dict(defaults)
        display_defaults.update(
            {
                key: _run_dir_expression(suffix)
                for key, suffix in RUN_DIR_DERIVED_PATHS.items()
            }
        )
        return {
            "schema_version": 1,
            "dataset": dataset,
            "datasets": [
                {"id": key, **value} for key, value in DATASETS.items()
            ],
            "commands": list(PIPELINE_COMMANDS),
            "stages": list(STAGES),
            "fields": [field.payload(dataset) for field in FIELDS],
            "defaults": display_defaults,
            "directory_contract": {
                "root": "RUN_DIR",
                "derived": [
                    {"key": key, "suffix": suffix}
                    for key, suffix in RUN_DIR_DERIVED_PATHS.items()
                ],
            },
            "default_profile_id": DATASETS[dataset]["profile"],
            "secrets": {
                "OPENAI_API_KEY": {
                    "persisted": False,
                    "label": "由独立运行器环境提供",
                }
            },
        }


def validate_profile_id(profile_id: str) -> str:
    """Validate and return a portable profile identifier."""

    normalized = str(profile_id).strip().lower()
    if not PROFILE_RE.fullmatch(normalized):
        raise ConfigValidationError(
            [
                {
                    "field": "profile_id",
                    "message": (
                        "配置 ID 需为 2–64 位小写字母、数字、短横线或下划线"
                    ),
                }
            ]
        )
    return normalized
