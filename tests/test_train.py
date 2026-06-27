"""Tests for the pure-python builders in train.py / dpo_train.py (no torch)."""

from __future__ import annotations

import argparse

import pytest

from src.config import build_recipe
from src.dpo_train import build_dpo_config_kwargs
from src.train import (
    apply_overrides,
    build_bnb_kwargs,
    build_lora_kwargs,
    build_sft_config_kwargs,
)


def _recipe(**extra):
    data = {
        "name": "demo",
        "base_model": {"model_id": "Qwen/Qwen2.5-7B-Instruct", "max_seq_length": 4096},
        "dataset": {"path": "./d.jsonl"},
        "output_dir": "./out",
        "lora": {"r": 32, "alpha": 64},
        "sft": {"epochs": 2, "batch_size": 8},
    }
    data.update(extra)
    return build_recipe(data)


def test_build_lora_kwargs_maps_alpha():
    recipe = _recipe()
    kw = build_lora_kwargs(recipe.lora)
    assert kw["r"] == 32
    assert kw["lora_alpha"] == 64  # alpha -> lora_alpha
    assert kw["task_type"] == "CAUSAL_LM"
    assert "q_proj" in kw["target_modules"]


def test_build_bnb_kwargs_keeps_dtype_string():
    recipe = _recipe()
    kw = build_bnb_kwargs(recipe.quantization)
    assert kw["load_in_4bit"] is True
    assert kw["bnb_4bit_quant_type"] == "nf4"
    # dtype stays a string in the pure builder (resolved later with torch)
    assert kw["bnb_4bit_compute_dtype"] == "bfloat16"


def test_build_sft_config_kwargs():
    recipe = _recipe()
    kw = build_sft_config_kwargs(recipe)
    assert kw["output_dir"] == "./out"
    assert kw["per_device_train_batch_size"] == 8
    assert kw["num_train_epochs"] == 2
    assert kw["dataset_text_field"] == "text"
    assert kw["max_seq_length"] == 4096


def test_dpo_kwargs_requires_dpo_section():
    recipe = _recipe()  # no dpo
    with pytest.raises(ValueError):
        build_dpo_config_kwargs(recipe)


def test_dpo_kwargs_present():
    recipe = _recipe(dpo={"beta": 0.3, "learning_rate": 1.0e-6})
    kw = build_dpo_config_kwargs(recipe)
    assert kw["beta"] == 0.3
    assert kw["learning_rate"] == 1.0e-6
    assert kw["max_length"] == 2048


def test_cli_overrides_applied():
    recipe = _recipe()
    args = argparse.Namespace(
        model_id="other/model",
        dataset_path=None,
        output_dir="./new",
        epochs=5.0,
        learning_rate=None,
        batch_size=None,
        lora_r=16,
        lora_alpha=None,
    )
    apply_overrides(recipe, args)
    assert recipe.base_model.model_id == "other/model"
    assert recipe.output_dir == "./new"
    assert recipe.sft.epochs == 5.0
    assert recipe.lora.r == 16
    # untouched fields stay
    assert recipe.sft.batch_size == 8
