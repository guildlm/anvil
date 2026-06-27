"""Tests for merge argument validation and step planning (no ML deps)."""

from __future__ import annotations

import pytest

from src.merge import (
    MergeError,
    find_llama_cpp_convert_script,
    plan_merge,
    resolve_dtype_name,
)


def test_resolve_dtype_valid():
    assert resolve_dtype_name("bfloat16") == "bfloat16"


def test_resolve_dtype_invalid():
    with pytest.raises(MergeError):
        resolve_dtype_name("int4")


def test_plan_merge_basic_steps():
    plan = plan_merge("base/m", "./adapter", "./out")
    assert plan.steps == ["load_base_model", "load_adapter", "merge_and_unload", "save_hf"]
    assert plan.gguf_output is None


def test_plan_merge_with_gguf():
    plan = plan_merge("base/m", "./adapter", "./out", gguf=True)
    assert "convert_gguf" in plan.steps
    assert plan.gguf_output.endswith("model.gguf")


def test_plan_merge_gguf_custom_output():
    plan = plan_merge("b", "a", "o", gguf=True, gguf_output="/tmp/x.gguf")
    assert plan.gguf_output == "/tmp/x.gguf"


def test_plan_merge_push_requires_repo():
    with pytest.raises(MergeError):
        plan_merge("b", "a", "o", push_to_hub=True)


def test_plan_merge_push_with_repo():
    plan = plan_merge("b", "a", "o", push_to_hub=True, hub_repo="guild/m")
    assert plan.steps[-1] == "push_to_hub"


def test_plan_merge_missing_args():
    with pytest.raises(MergeError):
        plan_merge("", "a", "o")


def test_find_convert_script_present(tmp_path):
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# stub")
    assert find_llama_cpp_convert_script(str(tmp_path)) == script


def test_find_convert_script_absent(tmp_path):
    assert find_llama_cpp_convert_script(str(tmp_path)) is None


def test_find_convert_script_env(tmp_path, monkeypatch):
    script = tmp_path / "convert.py"
    script.write_text("# stub")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert find_llama_cpp_convert_script(None) == script
