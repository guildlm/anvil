"""Configuration schema and YAML loader for GuildLM Anvil training recipes.

This module is intentionally free of any heavy ML dependency (torch,
transformers, peft, trl, ...). It only needs ``pyyaml`` so that recipes can be
loaded, validated and unit-tested on a bare CPU runner (e.g. CI) without
installing the training stack.

A *recipe* (typically ``configs/guilds/<name>.yaml``) fully describes a training
run: the base model, the dataset, LoRA/quantization settings and the SFT (and
optionally DPO) hyper-parameters. Recipes may either inline these sections or
reference reusable building blocks stored under ``configs/base_models/`` and
``configs/lora/``::

    name: go_reviewer
    base_model: qwen2.5_7b      # -> configs/base_models/qwen2.5_7b.yaml
    lora: high_rank             # -> configs/lora/high_rank.yaml
    dataset:
      path: ./data/go_reviewer_v1.jsonl
    output_dir: ./checkpoints/go_reviewer
    sft:
      epochs: 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigError",
    "BaseModelConfig",
    "LoraConfig",
    "QuantizationConfig",
    "DatasetConfig",
    "SFTHyperParams",
    "DPOHyperParams",
    "AnvilConfig",
    "load_yaml",
    "load_recipe",
    "build_recipe",
    "DEFAULT_TARGET_MODULES",
]

# Sensible default for Llama/Qwen/Mistral-style decoder blocks.
DEFAULT_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_ALLOWED_DTYPES = {"float32", "float16", "bfloat16", "auto"}
_ALLOWED_QUANT_TYPES = {"nf4", "fp4"}
_ALLOWED_BIAS = {"none", "all", "lora_only"}


class ConfigError(ValueError):
    """Raised when a recipe is structurally or semantically invalid."""


def _filter_known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Return *data* restricted to the dataclass fields of *cls*.

    Raises :class:`ConfigError` if *data* contains keys that are not valid
    fields, which catches typos early instead of silently ignoring them.
    """
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(
            f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return {k: v for k, v in data.items() if k in valid}


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class BaseModelConfig:
    """Identity and load-time options for the base model."""

    model_id: str
    max_seq_length: int = 4096
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    torch_dtype: str = "bfloat16"
    chat_template: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not isinstance(self.model_id, str):
            raise ConfigError("base_model.model_id must be a non-empty string")
        if self.max_seq_length <= 0:
            raise ConfigError("base_model.max_seq_length must be > 0")
        if self.torch_dtype not in _ALLOWED_DTYPES:
            raise ConfigError(
                f"base_model.torch_dtype must be one of {sorted(_ALLOWED_DTYPES)}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseModelConfig:
        return cls(**_filter_known(cls, data))


@dataclass
class LoraConfig:
    """PEFT LoRA hyper-parameters (decoupled from the ``peft`` package)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_TARGET_MODULES)
    )

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ConfigError("lora.r must be > 0")
        if self.alpha <= 0:
            raise ConfigError("lora.alpha must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("lora.dropout must be in [0.0, 1.0)")
        if self.bias not in _ALLOWED_BIAS:
            raise ConfigError(f"lora.bias must be one of {sorted(_ALLOWED_BIAS)}")
        if not self.target_modules:
            raise ConfigError("lora.target_modules must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraConfig:
        return cls(**_filter_known(cls, data))


@dataclass
class QuantizationConfig:
    """bitsandbytes 4-bit (QLoRA) settings."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.bnb_4bit_quant_type not in _ALLOWED_QUANT_TYPES:
            raise ConfigError(
                f"quantization.bnb_4bit_quant_type must be one of "
                f"{sorted(_ALLOWED_QUANT_TYPES)}"
            )
        if self.bnb_4bit_compute_dtype not in _ALLOWED_DTYPES:
            raise ConfigError(
                "quantization.bnb_4bit_compute_dtype must be one of "
                f"{sorted(_ALLOWED_DTYPES)}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizationConfig:
        return cls(**_filter_known(cls, data))


@dataclass
class DatasetConfig:
    """Where the training data lives and how it should be split/formatted."""

    path: str
    eval_path: str | None = None
    val_split: float = 0.0
    max_seq_length: int | None = None
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.path or not isinstance(self.path, str):
            raise ConfigError("dataset.path must be a non-empty string")
        if not 0.0 <= self.val_split < 1.0:
            raise ConfigError("dataset.val_split must be in [0.0, 1.0)")
        if self.max_seq_length is not None and self.max_seq_length <= 0:
            raise ConfigError("dataset.max_seq_length must be > 0 when set")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        return cls(**_filter_known(cls, data))


@dataclass
class SFTHyperParams:
    """Supervised fine-tuning hyper-parameters."""

    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    save_steps: int = 50
    logging_steps: int = 10
    max_grad_norm: float = 0.3
    group_by_length: bool = True
    packing: bool = False
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        _validate_common_hparams(self, prefix="sft")
        if self.bf16 and self.fp16:
            raise ConfigError("sft: enable at most one of bf16/fp16")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SFTHyperParams:
        return cls(**_filter_known(cls, data))


@dataclass
class DPOHyperParams:
    """Direct Preference Optimization hyper-parameters."""

    beta: float = 0.1
    loss_type: str = "sigmoid"
    max_prompt_length: int = 1024
    max_length: int = 2048
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    save_steps: int = 50
    logging_steps: int = 10
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        _validate_common_hparams(self, prefix="dpo")
        if self.beta <= 0:
            raise ConfigError("dpo.beta must be > 0")
        if self.max_prompt_length <= 0 or self.max_length <= 0:
            raise ConfigError("dpo.max_prompt_length/max_length must be > 0")
        if self.max_prompt_length >= self.max_length:
            raise ConfigError("dpo.max_prompt_length must be < dpo.max_length")
        if self.bf16 and self.fp16:
            raise ConfigError("dpo: enable at most one of bf16/fp16")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DPOHyperParams:
        return cls(**_filter_known(cls, data))


def _validate_common_hparams(obj: Any, prefix: str) -> None:
    """Validate hyper-parameters shared by SFT and DPO configs."""
    if obj.batch_size <= 0:
        raise ConfigError(f"{prefix}.batch_size must be > 0")
    if obj.gradient_accumulation_steps <= 0:
        raise ConfigError(f"{prefix}.gradient_accumulation_steps must be > 0")
    if obj.learning_rate <= 0:
        raise ConfigError(f"{prefix}.learning_rate must be > 0")
    if obj.max_steps <= 0 and obj.epochs <= 0:
        raise ConfigError(
            f"{prefix}: set either epochs > 0 or max_steps > 0"
        )
    if not 0.0 <= obj.warmup_ratio < 1.0:
        raise ConfigError(f"{prefix}.warmup_ratio must be in [0.0, 1.0)")


# --------------------------------------------------------------------------- #
# Top-level recipe
# --------------------------------------------------------------------------- #
@dataclass
class AnvilConfig:
    """A complete, validated training recipe."""

    name: str
    base_model: BaseModelConfig
    dataset: DatasetConfig
    output_dir: str
    lora: LoraConfig = field(default_factory=LoraConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    sft: SFTHyperParams = field(default_factory=SFTHyperParams)
    dpo: DPOHyperParams | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ConfigError("recipe 'name' must be a non-empty string")
        if not self.output_dir or not isinstance(self.output_dir, str):
            raise ConfigError("recipe 'output_dir' must be a non-empty string")

    @property
    def effective_max_seq_length(self) -> int:
        """Sequence length to train at (dataset override or base-model default)."""
        if self.dataset.max_seq_length is not None:
            return self.dataset.max_seq_length
        return self.base_model.max_seq_length


# --------------------------------------------------------------------------- #
# YAML loading / reference resolution
# --------------------------------------------------------------------------- #
def load_yaml(path: Any) -> dict[str, Any]:
    """Load a YAML file into a dict, raising :class:`ConfigError` on problems."""
    import yaml  # local import keeps module import cheap

    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via tests
        raise ConfigError(f"Failed to parse YAML {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML in {p} must be a mapping, got {type(data)}")
    return data


def _resolve_ref(value: Any, configs_root: Path | None, subdir: str) -> dict[str, Any]:
    """Resolve an inline dict or a named/path reference to a config section.

    A string *value* is looked up (in order) as ``<root>/<subdir>/<value>``,
    ``<root>/<subdir>/<value>.yaml``, ``<root>/<value>`` and finally as a plain
    filesystem path.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ConfigError(
            f"Section '{subdir}' must be a mapping or a string reference, "
            f"got {type(value)}"
        )

    candidates: list[Path] = []
    if configs_root is not None:
        candidates.append(configs_root / subdir / value)
        candidates.append(configs_root / subdir / f"{value}.yaml")
        candidates.append(configs_root / value)
    candidates.append(Path(value))

    for cand in candidates:
        if cand.is_file():
            logger.debug("Resolved %s reference %r -> %s", subdir, value, cand)
            return load_yaml(cand)

    raise ConfigError(
        f"Could not resolve '{subdir}' reference {value!r}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def build_recipe(data: dict[str, Any], configs_root: Path | None = None) -> AnvilConfig:
    """Build and validate an :class:`AnvilConfig` from a raw mapping.

    Reference resolution (``base_model``/``lora``/``quantization`` given as a
    name or path) is performed against *configs_root* when provided.
    """
    if not isinstance(data, dict):
        raise ConfigError("Recipe must be a mapping at the top level")

    known = {
        "name",
        "base_model",
        "dataset",
        "output_dir",
        "lora",
        "quantization",
        "sft",
        "dpo",
    }
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown top-level recipe key(s): {sorted(unknown)}")

    if "base_model" not in data:
        raise ConfigError("recipe is missing required section 'base_model'")
    if "dataset" not in data:
        raise ConfigError("recipe is missing required section 'dataset'")

    base_model = BaseModelConfig.from_dict(
        _resolve_ref(data["base_model"], configs_root, "base_models")
    )
    dataset = DatasetConfig.from_dict(_resolve_ref(data["dataset"], configs_root, "datasets"))

    lora = (
        LoraConfig.from_dict(_resolve_ref(data["lora"], configs_root, "lora"))
        if "lora" in data
        else LoraConfig()
    )
    quantization = (
        QuantizationConfig.from_dict(
            _resolve_ref(data["quantization"], configs_root, "quantization")
        )
        if "quantization" in data
        else QuantizationConfig()
    )
    sft = SFTHyperParams.from_dict(data.get("sft", {}) or {})
    dpo = DPOHyperParams.from_dict(data["dpo"]) if data.get("dpo") else None

    return AnvilConfig(
        name=data.get("name", base_model.model_id.split("/")[-1]),
        base_model=base_model,
        dataset=dataset,
        output_dir=data.get("output_dir", "./checkpoints"),
        lora=lora,
        quantization=quantization,
        sft=sft,
        dpo=dpo,
    )


def load_recipe(path: Any, configs_root: Any | None = None) -> AnvilConfig:
    """Load a recipe YAML from *path* and return a validated :class:`AnvilConfig`.

    If *configs_root* is omitted it is inferred as the grandparent of *path*
    (i.e. ``configs/`` when the recipe lives in ``configs/guilds/<name>.yaml``).
    """
    p = Path(path)
    data = load_yaml(p)
    if configs_root is not None:
        root: Path | None = Path(configs_root)
    elif p.parent.name and p.parent.parent != p.parent:
        root = p.parent.parent
    else:  # pragma: no cover - defensive
        root = None
    return build_recipe(data, configs_root=root)
