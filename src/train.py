"""QLoRA supervised fine-tuning (SFT) for GuildLM Anvil.

Design
------
* **Config-driven**: every run is described by an :class:`~src.config.AnvilConfig`
  recipe (YAML), with selective CLI overrides via ``argparse``.
* **Importable without torch**: all heavy imports (torch/transformers/peft/trl)
  are performed lazily inside :func:`train`, so the pure-python kwarg builders
  below can be imported and unit-tested on a CPU-only CI runner.

The double-PEFT bug, fixed
--------------------------
The previous implementation called ``get_peft_model(model, lora_config)`` **and**
passed ``peft_config=lora_config`` to :class:`trl.SFTTrainer`, which applies LoRA
twice. We adopt **a single, documented path**: load the quantized base model and
hand a single ``peft_config`` to ``SFTTrainer``. With the pinned TRL
(``>=0.12,<0.13``) the trainer internally runs
``prepare_model_for_kbit_training`` and ``get_peft_model`` exactly once. We do
**not** pre-wrap the model ourselves. ``dataset_text_field`` and
``max_seq_length`` are set on :class:`trl.SFTConfig`, where they live in this
TRL version.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.config import AnvilConfig, LoraConfig, QuantizationConfig, load_recipe
from src.data import load_sft_dataset

logger = logging.getLogger(__name__)

__all__ = [
    "build_lora_kwargs",
    "build_bnb_kwargs",
    "build_sft_config_kwargs",
    "resolve_torch_dtype",
    "train",
    "main",
]

# Map our string dtype names to ``torch`` attribute names (resolved lazily).
_TORCH_DTYPE_NAMES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "auto": "auto",
}


# --------------------------------------------------------------------------- #
# Pure-python builders (no ML deps) -- unit tested
# --------------------------------------------------------------------------- #
def build_lora_kwargs(lora: LoraConfig) -> dict[str, Any]:
    """Translate our :class:`LoraConfig` into ``peft.LoraConfig`` kwargs."""
    return {
        "r": lora.r,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "bias": lora.bias,
        "task_type": lora.task_type,
        "target_modules": list(lora.target_modules),
    }


def build_bnb_kwargs(quant: QuantizationConfig) -> dict[str, Any]:
    """Translate our :class:`QuantizationConfig` into ``BitsAndBytesConfig`` kwargs.

    ``bnb_4bit_compute_dtype`` is returned as a *string* here; the heavy path
    resolves it to a real ``torch.dtype`` via :func:`resolve_torch_dtype`.
    """
    return {
        "load_in_4bit": quant.load_in_4bit,
        "bnb_4bit_quant_type": quant.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": quant.bnb_4bit_use_double_quant,
        "bnb_4bit_compute_dtype": quant.bnb_4bit_compute_dtype,
    }


def build_sft_config_kwargs(recipe: AnvilConfig) -> dict[str, Any]:
    """Build the kwargs for :class:`trl.SFTConfig` from a recipe (pure)."""
    sft = recipe.sft
    return {
        "output_dir": recipe.output_dir,
        "per_device_train_batch_size": sft.batch_size,
        "gradient_accumulation_steps": sft.gradient_accumulation_steps,
        "learning_rate": sft.learning_rate,
        "num_train_epochs": sft.epochs,
        "max_steps": sft.max_steps,
        "warmup_ratio": sft.warmup_ratio,
        "weight_decay": sft.weight_decay,
        "lr_scheduler_type": sft.lr_scheduler_type,
        "optim": sft.optim,
        "save_steps": sft.save_steps,
        "logging_steps": sft.logging_steps,
        "max_grad_norm": sft.max_grad_norm,
        "group_by_length": sft.group_by_length,
        "packing": sft.packing,
        "bf16": sft.bf16,
        "fp16": sft.fp16,
        "seed": sft.seed,
        "gradient_checkpointing": sft.gradient_checkpointing,
        "dataset_text_field": "text",
        "max_seq_length": recipe.effective_max_seq_length,
    }


def resolve_torch_dtype(name: str) -> Any:  # pragma: no cover - needs torch
    """Resolve a dtype *name* to a ``torch.dtype`` (or the string ``"auto"``)."""
    if name not in _TORCH_DTYPE_NAMES:
        raise ValueError(f"Unsupported dtype {name!r}; expected {list(_TORCH_DTYPE_NAMES)}")
    if name == "auto":
        return "auto"
    import torch

    return getattr(torch, name)


def _require_ml():  # pragma: no cover - needs the training stack
    """Import the heavy training stack with a friendly error if it is missing."""
    try:
        import torch  # noqa: F401
        from peft import LoraConfig as PeftLoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Supervised fine-tuning requires the training stack. Install it with "
            "`pip install guildlm-anvil[train]`."
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "PeftLoraConfig": PeftLoraConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


# --------------------------------------------------------------------------- #
# Heavy path
# --------------------------------------------------------------------------- #
def train(recipe: AnvilConfig) -> str:  # pragma: no cover - needs GPU/torch
    """Run QLoRA SFT for *recipe* and return the adapter output directory."""
    ml = _require_ml()

    logger.info("Anvil SFT | model=%s | dataset=%s", recipe.base_model.model_id, recipe.dataset.path)

    tokenizer = ml["AutoTokenizer"].from_pretrained(
        recipe.base_model.model_id,
        trust_remote_code=recipe.base_model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if recipe.base_model.chat_template:
        tokenizer.chat_template = recipe.base_model.chat_template

    chat_fn = getattr(tokenizer, "apply_chat_template", None)
    if getattr(tokenizer, "chat_template", None) is None:
        logger.warning("Tokenizer has no chat_template; using ChatML fallback.")
        chat_fn = None

    train_ds, eval_ds = load_sft_dataset(
        recipe.dataset.path,
        chat_template_fn=chat_fn,
        eval_path=recipe.dataset.eval_path,
        val_split=recipe.dataset.val_split,
        system_prompt=recipe.dataset.system_prompt,
    )

    bnb_kwargs = build_bnb_kwargs(recipe.quantization)
    bnb_kwargs["bnb_4bit_compute_dtype"] = resolve_torch_dtype(
        recipe.quantization.bnb_4bit_compute_dtype
    )
    bnb_config = ml["BitsAndBytesConfig"](**bnb_kwargs)

    logger.info("Loading base model in 4-bit (QLoRA)...")
    model = ml["AutoModelForCausalLM"].from_pretrained(
        recipe.base_model.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=recipe.base_model.trust_remote_code,
        attn_implementation=recipe.base_model.attn_implementation,
    )
    model.config.use_cache = False

    # Single PEFT application: hand the raw (quantized) model + one peft_config
    # to SFTTrainer. We deliberately do NOT call get_peft_model() ourselves.
    peft_config = ml["PeftLoraConfig"](**build_lora_kwargs(recipe.lora))
    sft_config = ml["SFTConfig"](**build_sft_config_kwargs(recipe))

    trainer = ml["SFTTrainer"](
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT...")
    trainer.train()

    out = recipe.output_dir
    Path(out).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info("Saved LoRA adapter to %s", out)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GuildLM Anvil — QLoRA SFT trainer")
    parser.add_argument("--config", "-c", required=True, help="Path to a recipe YAML")
    parser.add_argument("--configs-root", default=None, help="Override configs/ root for references")
    parser.add_argument("--model-id", default=None, help="Override base_model.model_id")
    parser.add_argument("--dataset-path", default=None, help="Override dataset.path")
    parser.add_argument("--output-dir", default=None, help="Override output_dir")
    parser.add_argument("--epochs", type=float, default=None, help="Override sft.epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override sft.learning_rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override sft.batch_size")
    parser.add_argument("--lora-r", type=int, default=None, help="Override lora.r")
    parser.add_argument("--lora-alpha", type=int, default=None, help="Override lora.alpha")
    return parser


def apply_overrides(recipe: AnvilConfig, args: argparse.Namespace) -> AnvilConfig:
    """Apply non-``None`` CLI overrides onto *recipe* in place and return it."""
    if args.model_id is not None:
        recipe.base_model.model_id = args.model_id
    if args.dataset_path is not None:
        recipe.dataset.path = args.dataset_path
    if args.output_dir is not None:
        recipe.output_dir = args.output_dir
    if args.epochs is not None:
        recipe.sft.epochs = args.epochs
    if args.learning_rate is not None:
        recipe.sft.learning_rate = args.learning_rate
    if args.batch_size is not None:
        recipe.sft.batch_size = args.batch_size
    if args.lora_r is not None:
        recipe.lora.r = args.lora_r
    if args.lora_alpha is not None:
        recipe.lora.alpha = args.lora_alpha
    return recipe


def main(argv: list | None = None) -> None:
    """CLI entry point for ``anvil-train``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _build_parser().parse_args(argv)
    recipe = load_recipe(args.config, configs_root=args.configs_root)
    recipe = apply_overrides(recipe, args)
    train(recipe)


if __name__ == "__main__":
    main()
