"""Tests for Hub push argument validation and model-card generation.

These exercise only the pure-python helpers in ``src.hub`` — no network and no
``huggingface_hub`` import — so they pass on a bare CI runner.
"""

from __future__ import annotations

import pytest

from src.hub import (
    DEFAULT_LICENSE,
    HubError,
    build_model_card,
    validate_push_args,
)


# --------------------------------------------------------------------------- #
# validate_push_args
# --------------------------------------------------------------------------- #
def test_validate_repo_id_ok_without_dir():
    assert validate_push_args("user/go-reviewer-lora", require_dir=False) == "user/go-reviewer-lora"


@pytest.mark.parametrize(
    "repo_id",
    [
        "guildlm/go_reviewer",
        "org-name/model.v1",
        "User123/Go-Reviewer-LoRA",
    ],
)
def test_validate_repo_id_good_variants(repo_id):
    assert validate_push_args(repo_id, require_dir=False) == repo_id


@pytest.mark.parametrize(
    "repo_id",
    [
        "",
        "no-slash",
        "too/many/slashes",
        "/leading",
        "trailing/",
        "user/bad name",
        "user/bad@name",
        "user/-startsbad",
    ],
)
def test_validate_repo_id_bad(repo_id):
    with pytest.raises(HubError):
        validate_push_args(repo_id, require_dir=False)


def test_validate_repo_id_non_string():
    with pytest.raises(HubError):
        validate_push_args(None, require_dir=False)  # type: ignore[arg-type]


def test_validate_requires_existing_dir(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert validate_push_args("user/m", str(adapter)) == "user/m"


def test_validate_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(HubError):
        validate_push_args("user/m", str(missing))


def test_validate_missing_dir_arg():
    with pytest.raises(HubError):
        validate_push_args("user/m", None, require_dir=True)


def test_validate_file_is_not_dir(tmp_path):
    f = tmp_path / "afile.txt"
    f.write_text("x")
    with pytest.raises(HubError):
        validate_push_args("user/m", str(f))


# --------------------------------------------------------------------------- #
# build_model_card
# --------------------------------------------------------------------------- #
def test_model_card_contains_base_model_license_guild():
    card = build_model_card("user/go-reviewer-lora", base_model="Qwen/Qwen2.5-Coder-3B-Instruct")
    assert "Qwen/Qwen2.5-Coder-3B-Instruct" in card
    assert DEFAULT_LICENSE in card
    assert "Code Guild" in card
    assert "guildlm" in card


def test_model_card_has_frontmatter_and_repo_id():
    card = build_model_card("user/go-reviewer-lora", base_model="some/base")
    assert card.startswith("---\n")
    assert "user/go-reviewer-lora" in card
    # adapter (default) advertises the peft library
    assert "library_name: peft" in card


def test_model_card_merged_variant():
    card = build_model_card("user/go-merged", base_model="some/base", merged=True)
    assert "library_name: transformers" in card
    assert "merged model" in card


def test_model_card_without_base_model():
    card = build_model_card("user/m")
    # still valid, falls back to a placeholder, license + guild present
    assert DEFAULT_LICENSE in card
    assert "Code Guild" in card


def test_model_card_custom_license_and_guild():
    card = build_model_card("user/m", base_model="b", license="mit", guild="Code Guild (Rust)")
    assert "license: mit" in card
    assert "Code Guild (Rust)" in card
