"""Tests for recipe config loading and validation (no ML deps)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    AnvilConfig,
    ConfigError,
    DPOHyperParams,
    LoraConfig,
    SFTHyperParams,
    build_recipe,
    load_recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"


def test_lora_defaults_valid():
    lora = LoraConfig()
    assert lora.r == 16
    assert "q_proj" in lora.target_modules


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": 0},
        {"alpha": 0},
        {"dropout": 1.0},
        {"bias": "weird"},
        {"target_modules": []},
    ],
)
def test_lora_invalid(kwargs):
    with pytest.raises(ConfigError):
        LoraConfig(**kwargs)


def test_sft_requires_epochs_or_steps():
    with pytest.raises(ConfigError):
        SFTHyperParams(epochs=0, max_steps=-1)
    # max_steps alone is fine
    SFTHyperParams(epochs=0, max_steps=100)


def test_sft_bf16_fp16_exclusive():
    with pytest.raises(ConfigError):
        SFTHyperParams(bf16=True, fp16=True)


def test_dpo_validation():
    with pytest.raises(ConfigError):
        DPOHyperParams(beta=0)
    with pytest.raises(ConfigError):
        DPOHyperParams(max_prompt_length=4096, max_length=2048)
    good = DPOHyperParams(beta=0.2)
    assert good.loss_type == "sigmoid"


def test_build_recipe_inline():
    data = {
        "name": "demo",
        "base_model": {"model_id": "Qwen/Qwen2.5-7B-Instruct"},
        "dataset": {"path": "./data.jsonl"},
        "output_dir": "./out",
        "sft": {"epochs": 2},
    }
    recipe = build_recipe(data)
    assert isinstance(recipe, AnvilConfig)
    assert recipe.sft.epochs == 2
    assert recipe.dpo is None
    # max_seq_length falls back to base model default
    assert recipe.effective_max_seq_length == 4096


def test_effective_max_seq_length_override():
    data = {
        "base_model": {"model_id": "x/y", "max_seq_length": 8192},
        "dataset": {"path": "d.jsonl", "max_seq_length": 1024},
        "output_dir": "./o",
    }
    recipe = build_recipe(data)
    assert recipe.effective_max_seq_length == 1024


def test_build_recipe_unknown_top_key():
    with pytest.raises(ConfigError):
        build_recipe(
            {
                "base_model": {"model_id": "x/y"},
                "dataset": {"path": "d"},
                "output_dir": "o",
                "bogus": 1,
            }
        )


def test_build_recipe_missing_required():
    with pytest.raises(ConfigError):
        build_recipe({"dataset": {"path": "d"}, "output_dir": "o"})


def test_unknown_section_key_raises():
    with pytest.raises(ConfigError):
        build_recipe(
            {
                "base_model": {"model_id": "x/y", "typo_field": 1},
                "dataset": {"path": "d"},
                "output_dir": "o",
            }
        )


def test_load_real_go_reviewer_recipe():
    recipe = load_recipe(CONFIGS_ROOT / "guilds" / "go_reviewer.yaml")
    assert recipe.name == "go_reviewer"
    # base_model reference resolved
    assert recipe.base_model.model_id == "Qwen/Qwen2.5-7B-Instruct"
    # lora reference resolved to high_rank
    assert recipe.lora.r == 64
    assert recipe.dpo is not None
    assert recipe.dpo.beta == 0.1


def test_reference_resolution_with_explicit_root():
    data = {
        "name": "ref",
        "base_model": "mistral_7b",
        "lora": "qlora_consumer",
        "dataset": {"path": "d.jsonl"},
        "output_dir": "o",
    }
    recipe = build_recipe(data, configs_root=CONFIGS_ROOT)
    assert "Mistral" in recipe.base_model.model_id
    assert recipe.lora.r == 8


def test_unresolvable_reference():
    with pytest.raises(ConfigError):
        build_recipe(
            {
                "base_model": "does_not_exist",
                "dataset": {"path": "d"},
                "output_dir": "o",
            },
            configs_root=CONFIGS_ROOT,
        )
