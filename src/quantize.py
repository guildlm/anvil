"""Quantization / export helpers for GuildLM Anvil.

Supports three post-training quantization backends:

* ``gptq``  — weight-only GPTQ via ``auto-gptq`` / ``optimum``.
* ``awq``   — activation-aware weight quantization via ``autoawq``.
* ``gguf``  — llama.cpp GGUF export (delegates to :mod:`src.merge`).

The *orchestration* — argument validation and step planning
(:func:`validate_quant_args`, :func:`plan_quantization`) — is pure python and
fully unit-testable. Each backend is imported lazily and raises an informative
error if its package is not installed.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    "QuantizeError",
    "QuantPlan",
    "SUPPORTED_METHODS",
    "validate_quant_args",
    "plan_quantization",
    "quantize",
    "main",
]

SUPPORTED_METHODS = ("gptq", "awq", "gguf")
# Bit-widths each backend can emit.
_ALLOWED_BITS = {"gptq": {2, 3, 4, 8}, "awq": {4}, "gguf": {2, 3, 4, 5, 6, 8, 16}}


class QuantizeError(ValueError):
    """Raised for invalid quantization arguments or missing backends."""


@dataclass
class QuantPlan:
    """A validated, ordered description of a quantization job."""

    method: str
    model_path: str
    output_dir: str
    bits: int
    steps: list[str] = field(default_factory=list)
    group_size: int = 128
    calibration_dataset: str | None = None
    llama_cpp_dir: str | None = None


def validate_quant_args(
    method: str,
    model_path: str,
    output_dir: str,
    bits: int,
    *,
    group_size: int = 128,
    calibration_dataset: str | None = None,
) -> None:
    """Validate quantization arguments, raising :class:`QuantizeError` on error."""
    if method not in SUPPORTED_METHODS:
        raise QuantizeError(
            f"Unsupported method {method!r}; expected one of {list(SUPPORTED_METHODS)}"
        )
    if not model_path:
        raise QuantizeError("model_path is required")
    if not output_dir:
        raise QuantizeError("output_dir is required")
    if bits not in _ALLOWED_BITS[method]:
        raise QuantizeError(
            f"{method} does not support {bits}-bit; allowed: "
            f"{sorted(_ALLOWED_BITS[method])}"
        )
    if group_size <= 0:
        raise QuantizeError("group_size must be > 0")
    if method == "gptq" and not calibration_dataset:
        raise QuantizeError("gptq requires a --calibration-dataset for calibration")


def plan_quantization(
    method: str,
    model_path: str,
    output_dir: str,
    *,
    bits: int = 4,
    group_size: int = 128,
    calibration_dataset: str | None = None,
    llama_cpp_dir: str | None = None,
) -> QuantPlan:
    """Validate inputs and return an ordered :class:`QuantPlan`."""
    validate_quant_args(
        method,
        model_path,
        output_dir,
        bits,
        group_size=group_size,
        calibration_dataset=calibration_dataset,
    )

    if method == "gguf":
        steps = ["load_hf_model", "convert_gguf", f"quantize_q{bits}", "write_output"]
    else:  # gptq / awq
        steps = ["load_hf_model", "load_calibration_data", f"{method}_quantize", "save_quantized"]

    return QuantPlan(
        method=method,
        model_path=model_path,
        output_dir=output_dir,
        bits=bits,
        steps=steps,
        group_size=group_size,
        calibration_dataset=calibration_dataset,
        llama_cpp_dir=llama_cpp_dir,
    )


# --------------------------------------------------------------------------- #
# Guarded backends
# --------------------------------------------------------------------------- #
def _quantize_gptq(plan: QuantPlan) -> str:  # pragma: no cover - needs auto-gptq
    try:
        from optimum.gptq import GPTQQuantizer
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise QuantizeError(
            "GPTQ quantization requires 'optimum' and 'auto-gptq'. Install with "
            "`pip install optimum auto-gptq`."
        ) from exc
    logger.info("GPTQ quantizing %s to %d-bit", plan.model_path, plan.bits)
    tokenizer = AutoTokenizer.from_pretrained(plan.model_path)
    quantizer = GPTQQuantizer(
        bits=plan.bits, group_size=plan.group_size, dataset=plan.calibration_dataset
    )
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(plan.model_path, device_map="auto")
    quantizer.quantize_model(model, tokenizer)
    model.save_pretrained(plan.output_dir)
    tokenizer.save_pretrained(plan.output_dir)
    return plan.output_dir


def _quantize_awq(plan: QuantPlan) -> str:  # pragma: no cover - needs autoawq
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise QuantizeError(
            "AWQ quantization requires 'autoawq'. Install with `pip install autoawq`."
        ) from exc
    logger.info("AWQ quantizing %s to %d-bit", plan.model_path, plan.bits)
    model = AutoAWQForCausalLM.from_pretrained(plan.model_path)
    tokenizer = AutoTokenizer.from_pretrained(plan.model_path)
    model.quantize(
        tokenizer,
        quant_config={"w_bit": plan.bits, "q_group_size": plan.group_size, "version": "GEMM"},
    )
    model.save_quantized(plan.output_dir)
    tokenizer.save_pretrained(plan.output_dir)
    return plan.output_dir


def _quantize_gguf(plan: QuantPlan) -> str:  # pragma: no cover - needs llama.cpp
    from pathlib import Path

    from src.merge import convert_to_gguf

    out = str(Path(plan.output_dir) / f"model-q{plan.bits}.gguf")
    quant_type = f"q{plan.bits}_0" if plan.bits not in (16,) else "f16"
    return convert_to_gguf(
        plan.model_path, out, llama_cpp_dir=plan.llama_cpp_dir, quant_type=quant_type
    )


_BACKENDS = {"gptq": _quantize_gptq, "awq": _quantize_awq, "gguf": _quantize_gguf}


def quantize(plan: QuantPlan) -> str:  # pragma: no cover - dispatches to backends
    """Execute a :class:`QuantPlan` using the appropriate guarded backend."""
    logger.info("Quantization plan (%s): %s", plan.method, " -> ".join(plan.steps))
    return _BACKENDS[plan.method](plan)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GuildLM Anvil — quantize & export")
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS, help="Quantization backend")
    parser.add_argument("--model-path", required=True, help="HF model directory or ID")
    parser.add_argument("--output-dir", required=True, help="Where to write the quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Target bit-width")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--calibration-dataset", default=None, help="Calibration dataset (GPTQ)")
    parser.add_argument("--llama-cpp-dir", default=None, help="Path to a llama.cpp checkout (GGUF)")
    return parser


def main(argv: list | None = None) -> None:
    """CLI entry point for ``anvil-quantize``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _build_parser().parse_args(argv)
    plan = plan_quantization(
        args.method,
        args.model_path,
        args.output_dir,
        bits=args.bits,
        group_size=args.group_size,
        calibration_dataset=args.calibration_dataset,
        llama_cpp_dir=args.llama_cpp_dir,
    )
    quantize(plan)


if __name__ == "__main__":
    main()
