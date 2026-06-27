"""Direct Preference Optimization (DPO) training for GuildLM Anvil.

Trains on ``chosen``/``rejected`` preference pairs using :class:`trl.DPOTrainer`,
optionally on top of a 4-bit (QLoRA) base model with a single LoRA ``peft_config``
(the same single-PEFT discipline as :mod:`src.train`). Heavy imports are guarded
so the pure-python kwarg builder below is importable and testable without torch.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.config import AnvilConfig, DPOHyperParams, load_recipe
from src.data import load_dpo_dataset
from src.train import build_bnb_kwargs, build_lora_kwargs, resolve_torch_dtype

logger = logging.getLogger(__name__)

__all__ = ["build_dpo_config_kwargs", "train_dpo", "main"]


# --------------------------------------------------------------------------- #
# Pure-python builder (no ML deps) -- unit tested
# --------------------------------------------------------------------------- #
def build_dpo_config_kwargs(recipe: AnvilConfig) -> dict[str, Any]:
    """Build kwargs for :class:`trl.DPOConfig` from a recipe (pure).

    Raises:
        ValueError: if the recipe has no ``dpo`` section.
    """
    dpo: DPOHyperParams | None = recipe.dpo
    if dpo is None:
        raise ValueError(
            f"Recipe {recipe.name!r} has no 'dpo' section; add one to run DPO."
        )
    return {
        "output_dir": recipe.output_dir,
        "per_device_train_batch_size": dpo.batch_size,
        "gradient_accumulation_steps": dpo.gradient_accumulation_steps,
        "learning_rate": dpo.learning_rate,
        "num_train_epochs": dpo.epochs,
        "max_steps": dpo.max_steps,
        "warmup_ratio": dpo.warmup_ratio,
        "weight_decay": dpo.weight_decay,
        "lr_scheduler_type": dpo.lr_scheduler_type,
        "optim": dpo.optim,
        "save_steps": dpo.save_steps,
        "logging_steps": dpo.logging_steps,
        "bf16": dpo.bf16,
        "fp16": dpo.fp16,
        "seed": dpo.seed,
        "gradient_checkpointing": dpo.gradient_checkpointing,
        "beta": dpo.beta,
        "loss_type": dpo.loss_type,
        "max_prompt_length": dpo.max_prompt_length,
        "max_length": dpo.max_length,
    }


def _require_ml():  # pragma: no cover - needs the training stack
    try:
        import torch  # noqa: F401
        from peft import LoraConfig as PeftLoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise ImportError(
            "DPO training requires the training stack. Install it with "
            "`pip install guildlm-anvil[train]`."
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "PeftLoraConfig": PeftLoraConfig,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
    }


def train_dpo(recipe: AnvilConfig) -> str:  # pragma: no cover - needs GPU/torch
    """Run DPO training for *recipe* and return the output directory."""
    ml = _require_ml()
    dpo_kwargs = build_dpo_config_kwargs(recipe)  # validates dpo presence

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

    train_ds, eval_ds = load_dpo_dataset(
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

    logger.info("Loading policy model in 4-bit for DPO...")
    model = ml["AutoModelForCausalLM"].from_pretrained(
        recipe.base_model.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=recipe.base_model.trust_remote_code,
        attn_implementation=recipe.base_model.attn_implementation,
    )
    model.config.use_cache = False

    peft_config = ml["PeftLoraConfig"](**build_lora_kwargs(recipe.lora))
    dpo_config = ml["DPOConfig"](**dpo_kwargs)

    # With a LoRA peft_config the reference model is the frozen base (adapters
    # disabled), so we pass ref_model=None.
    trainer = ml["DPOTrainer"](
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting DPO...")
    trainer.train()

    out = recipe.output_dir
    Path(out).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info("Saved DPO adapter to %s", out)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GuildLM Anvil — DPO trainer")
    parser.add_argument("--config", "-c", required=True, help="Path to a recipe YAML")
    parser.add_argument("--configs-root", default=None, help="Override configs/ root for references")
    parser.add_argument("--model-id", default=None, help="Override base_model.model_id")
    parser.add_argument("--dataset-path", default=None, help="Override dataset.path")
    parser.add_argument("--output-dir", default=None, help="Override output_dir")
    parser.add_argument("--beta", type=float, default=None, help="Override dpo.beta")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override dpo.learning_rate")
    return parser


def apply_overrides(recipe: AnvilConfig, args: argparse.Namespace) -> AnvilConfig:
    """Apply non-``None`` CLI overrides for DPO onto *recipe* in place."""
    if args.model_id is not None:
        recipe.base_model.model_id = args.model_id
    if args.dataset_path is not None:
        recipe.dataset.path = args.dataset_path
    if args.output_dir is not None:
        recipe.output_dir = args.output_dir
    if recipe.dpo is not None:
        if args.beta is not None:
            recipe.dpo.beta = args.beta
        if args.learning_rate is not None:
            recipe.dpo.learning_rate = args.learning_rate
    return recipe


def main(argv: list | None = None) -> None:
    """CLI entry point for ``anvil-dpo``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _build_parser().parse_args(argv)
    recipe = load_recipe(args.config, configs_root=args.configs_root)
    recipe = apply_overrides(recipe, args)
    train_dpo(recipe)


if __name__ == "__main__":
    main()
