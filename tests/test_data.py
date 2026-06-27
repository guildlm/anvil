"""Tests for pure-python dataset formatting across all supported schemas."""

from __future__ import annotations

import pytest

from src.data import (
    DataError,
    fallback_chat_format,
    format_example,
    to_chat_messages,
    to_preference_example,
)


def fake_chat_template(messages, tokenize=False, add_generation_prompt=False):
    """Stand-in for tokenizer.apply_chat_template used to assert it's called."""
    text = "".join(f"[{m['role']}]{m['content']}" for m in messages)
    if add_generation_prompt:
        text += "[assistant]"
    return text


# --- schema detection -------------------------------------------------------
def test_messages_schema():
    ex = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    msgs = to_chat_messages(ex)
    assert msgs == ex["messages"]


def test_instruction_schema_with_context():
    ex = {"instruction": "Summarize", "context": "Long text", "response": "Short"}
    msgs = to_chat_messages(ex)
    assert msgs[0]["role"] == "user"
    assert "Long text" in msgs[0]["content"]
    assert msgs[1] == {"role": "assistant", "content": "Short"}


def test_instruction_schema_output_alias():
    ex = {"instruction": "Q", "output": "A"}
    msgs = to_chat_messages(ex)
    assert msgs[1]["content"] == "A"


def test_prompt_completion_schema():
    ex = {"prompt": "P", "completion": "C"}
    msgs = to_chat_messages(ex)
    assert msgs[0]["content"] == "P"
    assert msgs[1]["content"] == "C"


def test_unknown_schema_raises():
    with pytest.raises(DataError):
        to_chat_messages({"foo": "bar"})


def test_instruction_missing_response_raises():
    with pytest.raises(DataError):
        to_chat_messages({"instruction": "x"})


def test_invalid_role_raises():
    with pytest.raises(DataError):
        to_chat_messages({"messages": [{"role": "wizard", "content": "x"}]})


def test_system_prompt_prepended():
    ex = {"prompt": "P", "completion": "C"}
    msgs = to_chat_messages(ex, system_prompt="be nice")
    assert msgs[0] == {"role": "system", "content": "be nice"}


def test_system_prompt_not_duplicated():
    ex = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]}
    msgs = to_chat_messages(ex, system_prompt="other")
    assert msgs[0]["content"] == "sys"  # existing system kept, not duplicated


# --- formatting -------------------------------------------------------------
def test_format_example_uses_chat_template_fn():
    ex = {"prompt": "P", "completion": "C"}
    out = format_example(ex, chat_template_fn=fake_chat_template)
    assert out == "[user]P[assistant]C"


def test_format_example_fallback_chatml():
    ex = {"prompt": "P", "completion": "C"}
    out = format_example(ex)  # no template -> ChatML fallback
    assert "<|im_start|>user\nP<|im_end|>" in out
    assert "<|im_start|>assistant\nC<|im_end|>" in out


def test_fallback_generation_prompt():
    msgs = [{"role": "user", "content": "hi"}]
    out = fallback_chat_format(msgs, add_generation_prompt=True)
    assert out.endswith("<|im_start|>assistant\n")


# --- preference / DPO -------------------------------------------------------
def test_preference_prompt_schema():
    ex = {"prompt": "Why?", "chosen": "Because", "rejected": "Dunno"}
    out = to_preference_example(ex, chat_template_fn=fake_chat_template)
    assert out["prompt"] == "[user]Why?[assistant]"
    assert out["chosen"] == "Because"
    assert out["rejected"] == "Dunno"


def test_preference_instruction_schema_fallback():
    ex = {"instruction": "Do X", "context": "ctx", "chosen": "good", "rejected": "bad"}
    out = to_preference_example(ex)
    assert out["prompt"].endswith("<|im_start|>assistant\n")
    assert "ctx" in out["prompt"]
    assert out["chosen"] == "good"


def test_preference_missing_fields_raises():
    with pytest.raises(DataError):
        to_preference_example({"prompt": "p", "chosen": "c"})
