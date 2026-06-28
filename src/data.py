"""Dataset loading and formatting for GuildLM Anvil.

The *formatting* logic lives in pure-python functions (``to_chat_messages``,
``format_example``, ``to_preference_example``) that have **no** ML dependency
and are fully unit-testable. The :func:`load_sft_dataset` /
:func:`load_dpo_dataset` helpers wrap those functions with the ``datasets``
library and are only imported lazily.

Supported SFT input schemas (auto-detected per row):

* ``{"messages": [{"role": ..., "content": ...}, ...]}``        (chat / Forge)
* ``{"instruction": ..., "response": ..., "context": ...?}``    (Alpaca-like)
* ``{"prompt": ..., "completion": ...}``                        (prompt/completion)

Supported preference (DPO) schemas:

* ``{"prompt": ..., "chosen": ..., "rejected": ...}``
* ``{"messages": [...prompt turns...], "chosen": ..., "rejected": ...}``
* ``{"instruction": ..., "context": ...?, "chosen": ..., "rejected": ...}``
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "to_chat_messages",
    "format_example",
    "to_preference_example",
    "fallback_chat_format",
    "load_sft_dataset",
    "load_dpo_dataset",
    "DataError",
]

Message = dict[str, str]
# A callable matching ``tokenizer.apply_chat_template`` for the parts we use.
ChatTemplateFn = Callable[..., str]

_VALID_ROLES = {"system", "user", "assistant", "tool"}


class DataError(ValueError):
    """Raised when an input row does not match any supported schema."""


# --------------------------------------------------------------------------- #
# Pure-python formatting (no ML deps)
# --------------------------------------------------------------------------- #
def _normalize_messages(messages: list[Any]) -> list[Message]:
    """Validate and normalize a list of chat messages."""
    if not isinstance(messages, list) or not messages:
        raise DataError("'messages' must be a non-empty list")
    out: list[Message] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise DataError(f"messages[{i}] must have 'role' and 'content'")
        role = str(msg["role"]).strip()
        if role not in _VALID_ROLES:
            raise DataError(
                f"messages[{i}] has invalid role {role!r}; "
                f"expected one of {sorted(_VALID_ROLES)}"
            )
        out.append({"role": role, "content": str(msg["content"])})
    return out


def to_chat_messages(
    example: dict[str, Any], system_prompt: str | None = None
) -> list[Message]:
    """Convert a single dataset row into a list of chat messages.

    Auto-detects the input schema. When *system_prompt* is provided and the row
    does not already begin with a system turn, it is prepended.

    Raises:
        DataError: if the row matches none of the supported schemas.
    """
    if not isinstance(example, Mapping):
        raise DataError(f"example must be a mapping, got {type(example)}")
    # HuggingFace datasets.map() passes a LazyRow (a Mapping, not a dict);
    # coerce to a plain dict so .get()/indexing behave as expected.
    example = dict(example)

    if "messages" in example:
        messages = _normalize_messages(example["messages"])
    elif "instruction" in example:
        user = str(example["instruction"])
        context = example.get("context")
        if context:
            user = f"{user}\n\n{context}"
        response = example.get("response", example.get("output"))
        if response is None:
            raise DataError(
                "instruction schema requires a 'response' (or 'output') field"
            )
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(response)},
        ]
    elif "prompt" in example and "completion" in example:
        messages = [
            {"role": "user", "content": str(example["prompt"])},
            {"role": "assistant", "content": str(example["completion"])},
        ]
    else:
        raise DataError(
            "Unrecognized SFT schema. Expected one of: {messages}, "
            "{instruction,response[,context]}, {prompt,completion}. "
            f"Got keys: {sorted(example)}"
        )

    if system_prompt and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": str(system_prompt)}] + messages
    return messages


def fallback_chat_format(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """Render chat messages as ChatML when a tokenizer template is unavailable.

    This mirrors the widely-used ``<|im_start|>role\\ncontent<|im_end|>`` format
    so that training still works for base models that ship without a chat
    template. Production runs should prefer the tokenizer's own template.
    """
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def format_example(
    example: dict[str, Any],
    chat_template_fn: ChatTemplateFn | None = None,
    system_prompt: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Format one SFT row into a single training string.

    Args:
        example: A raw dataset row in any supported schema.
        chat_template_fn: Optional ``tokenizer.apply_chat_template``. When
            ``None`` (e.g. in tests or for template-less models) a ChatML
            fallback is used.
        system_prompt: Optional system turn to prepend.
        add_generation_prompt: Whether to append an empty assistant turn.

    Returns:
        The formatted text ready to be tokenized.
    """
    messages = to_chat_messages(example, system_prompt=system_prompt)
    if chat_template_fn is not None:
        return chat_template_fn(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return fallback_chat_format(messages, add_generation_prompt=add_generation_prompt)


def to_preference_example(
    example: dict[str, Any],
    chat_template_fn: ChatTemplateFn | None = None,
    system_prompt: str | None = None,
) -> dict[str, str]:
    """Convert a preference row into ``{prompt, chosen, rejected}`` strings.

    The prompt is rendered with a generation prompt appended so the model is
    cued to produce the ``chosen``/``rejected`` continuation.
    """
    if not isinstance(example, Mapping):
        raise DataError(f"example must be a mapping, got {type(example)}")
    # HuggingFace datasets.map() passes a LazyRow (a Mapping, not a dict);
    # coerce to a plain dict so .get()/indexing behave as expected.
    example = dict(example)
    if "chosen" not in example or "rejected" not in example:
        raise DataError("preference rows require 'chosen' and 'rejected' fields")

    if "messages" in example:
        prompt_messages = _normalize_messages(example["messages"])
    elif "prompt" in example:
        prompt_messages = [{"role": "user", "content": str(example["prompt"])}]
    elif "instruction" in example:
        user = str(example["instruction"])
        context = example.get("context")
        if context:
            user = f"{user}\n\n{context}"
        prompt_messages = [{"role": "user", "content": user}]
    else:
        raise DataError(
            "Unrecognized preference schema. Expected a 'prompt', 'instruction' "
            f"or 'messages' field alongside chosen/rejected. Got: {sorted(example)}"
        )

    if system_prompt and prompt_messages[0]["role"] != "system":
        prompt_messages = [
            {"role": "system", "content": str(system_prompt)}
        ] + prompt_messages

    if chat_template_fn is not None:
        prompt = chat_template_fn(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = fallback_chat_format(prompt_messages, add_generation_prompt=True)

    return {
        "prompt": prompt,
        "chosen": str(example["chosen"]),
        "rejected": str(example["rejected"]),
    }


# --------------------------------------------------------------------------- #
# Heavy helpers (lazy `datasets` import)
# --------------------------------------------------------------------------- #
def _require_datasets():  # pragma: no cover - exercised only with deps installed
    try:
        import datasets  # noqa: WPS433 (intentional lazy import)
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for dataset loading. "
            "Install the training extra: `pip install guildlm-anvil[train]`."
        ) from exc
    return datasets


def _maybe_split(dataset, val_split: float, seed: int):  # pragma: no cover
    """Split *dataset* into (train, eval) when ``val_split`` > 0."""
    if val_split and val_split > 0.0:
        split = dataset.train_test_split(test_size=val_split, seed=seed)
        return split["train"], split["test"]
    return dataset, None


def load_sft_dataset(
    data_path: str,
    chat_template_fn: ChatTemplateFn | None = None,
    *,
    eval_path: str | None = None,
    val_split: float = 0.0,
    system_prompt: str | None = None,
    text_field: str = "text",
    seed: int = 42,
    num_proc: int = 1,
) -> tuple[Any, Any]:  # pragma: no cover - requires `datasets`
    """Load and format a JSON/JSONL SFT dataset into ``(train, eval)`` datasets.

    Each output row has a single ``text_field`` column containing the formatted
    conversation. ``eval`` is ``None`` unless *eval_path* or *val_split* is set.
    """
    datasets = _require_datasets()
    logger.info("Loading SFT dataset from %s", data_path)
    raw = datasets.load_dataset("json", data_files=data_path, split="train")

    def _format(row: dict[str, Any]) -> dict[str, str]:
        return {text_field: format_example(row, chat_template_fn, system_prompt)}

    train = raw.map(_format, remove_columns=raw.column_names, num_proc=num_proc)
    train = train.filter(lambda r: len(r[text_field]) > 0)

    eval_ds = None
    if eval_path:
        raw_eval = datasets.load_dataset("json", data_files=eval_path, split="train")
        eval_ds = raw_eval.map(
            _format, remove_columns=raw_eval.column_names, num_proc=num_proc
        ).filter(lambda r: len(r[text_field]) > 0)
    else:
        train, eval_ds = _maybe_split(train, val_split, seed)

    logger.info("SFT dataset ready: %d train rows", len(train))
    return train, eval_ds


def corpus_row_to_text(row: dict[str, Any], text_field: str = "text") -> dict[str, str]:
    """Keep a raw-text corpus row as just its text field (no chat formatting).

    Continued-pretraining data is already raw text (``{"text": ..., "source":
    ...}``); training is plain next-token over it, so unlike SFT we do not apply a
    chat template — we only validate and project to the single text column.
    """
    text = row.get(text_field)
    if not isinstance(text, str):
        raise DataError(f"corpus row missing string '{text_field}' field")
    return {text_field: text}


def load_text_corpus(
    data_path: str,
    *,
    eval_path: str | None = None,
    val_split: float = 0.0,
    text_field: str = "text",
    seed: int = 42,
    num_proc: int = 1,
) -> tuple[Any, Any]:  # pragma: no cover - requires `datasets`
    """Load a raw-text corpus for continued / domain-adaptive pretraining.

    Each row keeps a single ``text_field`` column of raw text; the SFTTrainer
    then trains next-token over it (same trainer, no message formatting).
    """
    datasets = _require_datasets()
    logger.info("Loading pretrain corpus from %s", data_path)
    raw = datasets.load_dataset("json", data_files=data_path, split="train")

    def _project(row: dict[str, Any]) -> dict[str, str]:
        return corpus_row_to_text(row, text_field)

    train = raw.map(_project, remove_columns=raw.column_names, num_proc=num_proc)
    train = train.filter(lambda r: len(r[text_field]) > 0)

    eval_ds = None
    if eval_path:
        raw_eval = datasets.load_dataset("json", data_files=eval_path, split="train")
        eval_ds = raw_eval.map(
            _project, remove_columns=raw_eval.column_names, num_proc=num_proc
        ).filter(lambda r: len(r[text_field]) > 0)
    else:
        train, eval_ds = _maybe_split(train, val_split, seed)

    logger.info("Pretrain corpus ready: %d train rows", len(train))
    return train, eval_ds


def load_dpo_dataset(
    data_path: str,
    chat_template_fn: ChatTemplateFn | None = None,
    *,
    eval_path: str | None = None,
    val_split: float = 0.0,
    system_prompt: str | None = None,
    seed: int = 42,
    num_proc: int = 1,
) -> tuple[Any, Any]:  # pragma: no cover - requires `datasets`
    """Load a preference dataset into ``(train, eval)`` with prompt/chosen/rejected."""
    datasets = _require_datasets()
    logger.info("Loading DPO dataset from %s", data_path)
    raw = datasets.load_dataset("json", data_files=data_path, split="train")

    def _format(row: dict[str, Any]) -> dict[str, str]:
        return to_preference_example(row, chat_template_fn, system_prompt)

    train = raw.map(_format, remove_columns=raw.column_names, num_proc=num_proc)

    eval_ds = None
    if eval_path:
        raw_eval = datasets.load_dataset("json", data_files=eval_path, split="train")
        eval_ds = raw_eval.map(_format, remove_columns=raw_eval.column_names, num_proc=num_proc)
    else:
        train, eval_ds = _maybe_split(train, val_split, seed)

    logger.info("DPO dataset ready: %d train rows", len(train))
    return train, eval_ds
