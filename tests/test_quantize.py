"""Tests for quantization argument validation and step planning (no backends)."""

from __future__ import annotations

import pytest

from src.quantize import (
    SUPPORTED_METHODS,
    QuantizeError,
    plan_quantization,
    validate_quant_args,
)


def test_supported_methods():
    assert set(SUPPORTED_METHODS) == {"gptq", "awq", "gguf"}


def test_validate_unknown_method():
    with pytest.raises(QuantizeError):
        validate_quant_args("magic", "m", "o", 4)


def test_validate_bits_per_method():
    # awq only supports 4-bit
    with pytest.raises(QuantizeError):
        validate_quant_args("awq", "m", "o", 8)
    validate_quant_args("awq", "m", "o", 4)


def test_gptq_requires_calibration():
    with pytest.raises(QuantizeError):
        validate_quant_args("gptq", "m", "o", 4)
    validate_quant_args("gptq", "m", "o", 4, calibration_dataset="c4")


def test_plan_gguf_steps():
    plan = plan_quantization("gguf", "./model", "./out", bits=4)
    assert plan.steps[0] == "load_hf_model"
    assert "convert_gguf" in plan.steps
    assert "quantize_q4" in plan.steps


def test_plan_awq_steps():
    plan = plan_quantization("awq", "./model", "./out", bits=4)
    assert "awq_quantize" in plan.steps
    assert plan.steps[-1] == "save_quantized"


def test_plan_gptq_steps():
    plan = plan_quantization("gptq", "./model", "./out", bits=4, calibration_dataset="c4")
    assert "gptq_quantize" in plan.steps
    assert plan.bits == 4


def test_invalid_group_size():
    with pytest.raises(QuantizeError):
        validate_quant_args("awq", "m", "o", 4, group_size=0)


def test_missing_model_path():
    with pytest.raises(QuantizeError):
        plan_quantization("awq", "", "o", bits=4)
