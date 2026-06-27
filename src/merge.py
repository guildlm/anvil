"""Merge a LoRA adapter into its base model and export it.

Exports to HuggingFace ``safetensors`` format and provides an optional GGUF
conversion hook that shells out to llama.cpp's ``convert_hf_to_gguf.py`` when it
is available (and raises a clear error otherwise).

The argument-validation and step-planning logic (:func:`plan_merge`,
:func:`find_llama_cpp_convert_script`, :func:`resolve_dtype_name`) is pure
python and unit-testable; the actual model loading is guarded behind a lazy
import.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MergeError",
    "MergePlan",
    "resolve_dtype_name",
    "find_llama_cpp_convert_script",
    "plan_merge",
    "merge_and_export",
    "convert_to_gguf",
    "main",
]

_ALLOWED_DTYPES = {"float16", "bfloat16", "float32"}
_CONVERT_SCRIPT_NAMES = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py", "convert.py")


class MergeError(ValueError):
    """Raised for invalid merge arguments or missing external tooling."""


@dataclass
class MergePlan:
    """A validated, ordered description of a merge/export job."""

    base_model: str
    adapter_path: str
    output_dir: str
    dtype: str
    steps: list[str]
    gguf_output: str | None = None
    push_to_hub: bool = False
    hub_repo: str | None = None


def resolve_dtype_name(name: str) -> str:
    """Validate and canonicalize a dtype *name* for merging."""
    if name not in _ALLOWED_DTYPES:
        raise MergeError(
            f"Unsupported dtype {name!r}; expected one of {sorted(_ALLOWED_DTYPES)}"
        )
    return name


def find_llama_cpp_convert_script(llama_cpp_dir: str | None) -> Path | None:
    """Locate llama.cpp's HF->GGUF conversion script.

    Searches *llama_cpp_dir* (if given) then the ``LLAMA_CPP_DIR`` environment
    variable. Returns the script path, or ``None`` if not found.
    """
    search_dirs: list[Path] = []
    if llama_cpp_dir:
        search_dirs.append(Path(llama_cpp_dir))
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))

    for base in search_dirs:
        for name in _CONVERT_SCRIPT_NAMES:
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


def plan_merge(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    *,
    dtype: str = "bfloat16",
    gguf: bool = False,
    gguf_output: str | None = None,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
) -> MergePlan:
    """Validate inputs and produce an ordered :class:`MergePlan`.

    Raises:
        MergeError: on any invalid combination of arguments.
    """
    if not base_model:
        raise MergeError("base_model is required")
    if not adapter_path:
        raise MergeError("adapter_path is required")
    if not output_dir:
        raise MergeError("output_dir is required")
    dtype = resolve_dtype_name(dtype)
    if push_to_hub and not hub_repo:
        raise MergeError("push_to_hub requires --hub-repo")

    steps = ["load_base_model", "load_adapter", "merge_and_unload", "save_hf"]
    resolved_gguf: str | None = None
    if gguf:
        resolved_gguf = gguf_output or str(Path(output_dir) / "model.gguf")
        steps.append("convert_gguf")
    if push_to_hub:
        steps.append("push_to_hub")

    return MergePlan(
        base_model=base_model,
        adapter_path=adapter_path,
        output_dir=output_dir,
        dtype=dtype,
        steps=steps,
        gguf_output=resolved_gguf,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


def _resolve_torch_dtype(name: str) -> Any:  # pragma: no cover - needs torch
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _require_ml():  # pragma: no cover - needs torch/transformers/peft
    try:
        import torch  # noqa: F401
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Merging requires torch/transformers/peft. Install with "
            "`pip install guildlm-anvil[train]`."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer, PeftModel


def convert_to_gguf(
    model_dir: str,
    output_path: str,
    *,
    llama_cpp_dir: str | None = None,
    quant_type: str = "f16",
) -> str:  # pragma: no cover - requires llama.cpp on disk
    """Convert a HF model directory to GGUF via llama.cpp's converter.

    Raises:
        MergeError: if the llama.cpp conversion script cannot be found.
    """
    import subprocess
    import sys

    script = find_llama_cpp_convert_script(llama_cpp_dir)
    if script is None:
        raise MergeError(
            "GGUF conversion requested but llama.cpp's convert script was not "
            "found. Clone https://github.com/ggerganov/llama.cpp and pass "
            "--llama-cpp-dir or set LLAMA_CPP_DIR."
        )
    cmd = [sys.executable, str(script), model_dir, "--outfile", output_path, "--outtype", quant_type]
    logger.info("Running GGUF conversion: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def merge_and_export(plan: MergePlan, llama_cpp_dir: str | None = None) -> str:  # pragma: no cover
    """Execute a :class:`MergePlan`: merge, save HF, optionally GGUF + push."""
    AutoModelForCausalLM, AutoTokenizer, PeftModel = _require_ml()

    logger.info("Loading base model %s (%s)", plan.base_model, plan.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        plan.base_model,
        torch_dtype=_resolve_torch_dtype(plan.dtype),
        device_map="cpu",
        low_cpu_mem_usage=True,
        return_dict=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(plan.base_model)

    logger.info("Applying adapter %s", plan.adapter_path)
    model = PeftModel.from_pretrained(base, plan.adapter_path)
    merged = model.merge_and_unload()

    Path(plan.output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(plan.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(plan.output_dir)
    logger.info("Merged model written to %s", plan.output_dir)

    if plan.gguf_output:
        convert_to_gguf(plan.output_dir, plan.gguf_output, llama_cpp_dir=llama_cpp_dir)

    if plan.push_to_hub and plan.hub_repo:
        logger.info("Pushing merged model to %s", plan.hub_repo)
        merged.push_to_hub(plan.hub_repo)
        tokenizer.push_to_hub(plan.hub_repo)

    return plan.output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GuildLM Anvil — LoRA merge & export")
    parser.add_argument("--base-model", required=True, help="Base model ID or path")
    parser.add_argument("--adapter", required=True, dest="adapter_path", help="LoRA adapter path")
    parser.add_argument("--output-dir", required=True, help="Where to write the merged model")
    parser.add_argument("--dtype", default="bfloat16", help="Merge dtype (float16/bfloat16/float32)")
    parser.add_argument("--gguf", action="store_true", help="Also export to GGUF via llama.cpp")
    parser.add_argument("--gguf-output", default=None, help="GGUF output path")
    parser.add_argument("--llama-cpp-dir", default=None, help="Path to a llama.cpp checkout")
    parser.add_argument("--push-to-hub", action="store_true", help="Push merged model to HF Hub")
    parser.add_argument("--hub-repo", default=None, help="HF Hub repo for push")
    return parser


def main(argv: list | None = None) -> None:
    """CLI entry point for ``anvil-merge``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _build_parser().parse_args(argv)
    plan = plan_merge(
        args.base_model,
        args.adapter_path,
        args.output_dir,
        dtype=args.dtype,
        gguf=args.gguf,
        gguf_output=args.gguf_output,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )
    logger.info("Merge plan: %s", " -> ".join(plan.steps))
    merge_and_export(plan, llama_cpp_dir=args.llama_cpp_dir)


if __name__ == "__main__":
    main()
