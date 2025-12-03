"""Utility modules for GRPO pipeline"""

from .prompts import (
    get_code_only_prompt,
    get_thinking_prompt,
    SYSTEM_PROMPT_CODE_ONLY,
    SYSTEM_PROMPT_WITH_THINKING,
    SYSTEM_PROMPT_APPS,
)

from .code_parser import (
    parse_model_response,
    extract_thinking_and_code,
    extract_code_from_markdown,
    clean_code,
    validate_python_code,
)

from .thinking_extraction import (
    extract_qwen_thinking,
    extract_thinking_and_code_unified,
    format_response_with_thinking,
)

from .sft_data import (
    prepare_teacher_sft_batch,
    prepare_self_sft_batch,
    create_sft_labels,
    verify_labels,
)

from .dataset_loader import (
    load_mbpp_dataset,
    load_apps_dataset,
    combine_and_split_datasets,
    save_dataset,
    load_dataset_from_file,
)

__all__ = [
    # Prompts
    "get_code_only_prompt",
    "get_thinking_prompt",
    "SYSTEM_PROMPT_CODE_ONLY",
    "SYSTEM_PROMPT_WITH_THINKING",
    "SYSTEM_PROMPT_APPS",
    # Code parsing
    "parse_model_response",
    "extract_thinking_and_code",
    "extract_code_from_markdown",
    "clean_code",
    "validate_python_code",
    # Thinking extraction (Qwen-specific)
    "extract_qwen_thinking",
    "extract_thinking_and_code_unified",
    "format_response_with_thinking",
    # SFT data preparation
    "prepare_teacher_sft_batch",
    "prepare_self_sft_batch",
    "create_sft_labels",
    "verify_labels",
    # Dataset loading
    "load_mbpp_dataset",
    "load_apps_dataset",
    "combine_and_split_datasets",
    "save_dataset",
    "load_dataset_from_file",
]
