"""Direct candidate-name Top1 router training."""

from .top1 import (
    CONVERSATION_TEMPLATE,
    INFERENCE_DECISION_RULE,
    INFERENCE_SCORING_RULE,
    MEMORIZATION_DESCRIPTION_TYPES,
    MEMORIZATION_SOURCE_TYPE,
    ROUTING_MODE,
    TARGET_CONTRACT,
    Top1DataError,
    load_candidate_names,
    prepare_example,
    prepare_router_prompt,
    prompt_implementation_sha256,
    tokenizer_prompt_contract,
    validate_memorization_rows,
    validate_training_rows,
)

__all__ = [
    "CONVERSATION_TEMPLATE",
    "INFERENCE_DECISION_RULE",
    "INFERENCE_SCORING_RULE",
    "MEMORIZATION_DESCRIPTION_TYPES",
    "MEMORIZATION_SOURCE_TYPE",
    "ROUTING_MODE",
    "TARGET_CONTRACT",
    "Top1DataError",
    "load_candidate_names",
    "prepare_example",
    "prepare_router_prompt",
    "prompt_implementation_sha256",
    "tokenizer_prompt_contract",
    "validate_memorization_rows",
    "validate_training_rows",
]

__version__ = "0.1.0"
