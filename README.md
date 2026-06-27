# Anvil

[![CI](https://github.com/guildlm/anvil/actions/workflows/ci.yml/badge.svg)](https://github.com/guildlm/anvil/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

> **Anvil** is the training forge of [GuildLM](https://github.com/guildlm/guildlm.github.io).
> It turns **any base model** + **any dataset** into a specialized guild model:
> SFT, DPO preference tuning, LoRA/QLoRA, adapter merging, and quantized export —
> all driven by small, composable YAML recipes.

Anvil is the **train** stage of the GuildLM pipeline:
`forge` (data) → **`anvil` (train)** → serving.

> 💸 **Train your first Go specialist for $0.** Open
> [`notebooks/kaggle_go_reviewer.ipynb`](notebooks/kaggle_go_reviewer.ipynb) on
> Kaggle's free T4 GPU, **Run All**, then `anvil-push` the adapter to the
> HuggingFace Hub (free hosting) and serve it in Ollama. The full free recipe —
> Kaggle → HF Hub → Ollama, with an honest cost table — is in
> **[TRAINING.md](TRAINING.md)**.

---

## Why Anvil

- **Recipe-driven.** A run is a single YAML file that references reusable base-model
  and LoRA building blocks. No more 12-flag command lines.
- **QLoRA by default.** 4-bit NF4 base + LoRA adapters so you can fine-tune 7–8B
  models on a single consumer GPU.
- **SFT and DPO.** Supervised fine-tuning and preference optimization share the same
  config schema and data loaders.
- **Chat-template-aware.** Uses the tokenizer's own `apply_chat_template` when present,
  with a ChatML fallback for template-less bases.
- **Light to install, light to test.** The core (configs, data formatting,
  orchestration) needs only `pyyaml`+`typer`; the GPU stack lives in an extra.
  The whole test suite runs on CPU CI **without torch**.

---

## Install

```bash
# Core only (configs, data formatting, CLI plumbing, tests)
pip install -e ".[dev]"

# Full training stack (GPU box: torch, transformers, peft, trl, bitsandbytes, ...)
pip install -e ".[train]"
```

Console scripts installed: `anvil-train`, `anvil-dpo`, `anvil-merge`, `anvil-quantize`,
`anvil-push` (publish an adapter/merged model to the HuggingFace Hub).

---

## Quickstart: a real QLoRA SFT run

1. **Pick or write a recipe.** `configs/guilds/go_reviewer.yaml` references the
   Qwen2.5-7B base and a high-rank LoRA:

   ```yaml
   name: go_reviewer
   base_model: qwen2.5_7b        # -> configs/base_models/qwen2.5_7b.yaml
   lora: high_rank               # -> configs/lora/high_rank.yaml
   dataset:
     path: ./data/go_reviewer_v1.jsonl
     val_split: 0.05
     system_prompt: "You are a meticulous senior Go engineer performing code review."
   output_dir: ./checkpoints/go_reviewer_adapter
   sft:
     batch_size: 4
     learning_rate: 2.0e-4
     epochs: 3
   ```

2. **Train.** CLI flags override any recipe field:

   ```bash
   anvil-train --config configs/guilds/go_reviewer.yaml \
       --dataset-path ./data/go_reviewer_v1.jsonl \
       --output-dir ./checkpoints/go_reviewer_adapter
   ```

3. **Merge** the adapter back into the base and export to HuggingFace format:

   ```bash
   anvil-merge \
       --base-model Qwen/Qwen2.5-7B-Instruct \
       --adapter ./checkpoints/go_reviewer_adapter \
       --output-dir ./exports/go_reviewer_merged \
       --dtype bfloat16
   ```

4. **Quantize / export** for serving (GGUF shown):

   ```bash
   anvil-quantize --method gguf --bits 4 \
       --model-path ./exports/go_reviewer_merged \
       --output-dir ./exports/go_reviewer_gguf \
       --llama-cpp-dir /path/to/llama.cpp
   ```

5. **Publish** the adapter (or merged model) to the HuggingFace Hub for free hosting:

   ```bash
   anvil-push --repo-id your-username/go-reviewer-lora \
       --adapter ./checkpoints/go_reviewer_adapter \
       --base-model Qwen/Qwen2.5-7B-Instruct   # add --merged for a merged model
   ```

For the full **$0** path (Kaggle GPU → HF Hub → Ollama), see **[TRAINING.md](TRAINING.md)**.

---

## Dataset formats

Anvil auto-detects the schema **per row**, so mixed datasets work. For SFT:

| Schema | Example |
| --- | --- |
| Chat / Forge | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| Alpaca-like | `{"instruction": "...", "context": "...", "response": "..."}` (`output` also accepted) |
| Prompt/completion | `{"prompt": "...", "completion": "..."}` |

For DPO (preference) data, add `chosen`/`rejected` to any of:

```jsonc
{"prompt": "...",       "chosen": "...", "rejected": "..."}
{"instruction": "...",  "chosen": "...", "rejected": "..."}
{"messages": [ ... ],   "chosen": "...", "rejected": "..."}
```

The pure-python `src.data.format_example` / `to_preference_example` functions do
this conversion and are fully unit-tested without any ML dependency.

---

## Config schema

A recipe (`configs/guilds/*.yaml`) is validated into `src.config.AnvilConfig`.
`base_model`, `lora` and `quantization` may be given **inline** or as a **named
reference** resolved against `configs/<section>/<name>.yaml`.

| Section | Key fields (defaults) |
| --- | --- |
| `base_model` | `model_id` (req), `max_seq_length` (4096), `trust_remote_code` (false), `attn_implementation`, `torch_dtype` (bfloat16), `chat_template` |
| `lora` | `r` (16), `alpha` (32), `dropout` (0.05), `bias` (none), `task_type` (CAUSAL_LM), `target_modules` |
| `quantization` | `load_in_4bit` (true), `bnb_4bit_quant_type` (nf4), `bnb_4bit_use_double_quant` (true), `bnb_4bit_compute_dtype` (bfloat16) |
| `dataset` | `path` (req), `eval_path`, `val_split` (0.0), `max_seq_length`, `system_prompt` |
| `sft` | `batch_size`, `gradient_accumulation_steps`, `learning_rate`, `epochs`, `max_steps`, `warmup_ratio`, `group_by_length`, `bf16`/`fp16`, ... |
| `dpo` | `beta` (0.1), `loss_type` (sigmoid), `max_prompt_length`, `max_length`, plus shared hyper-params |

Bundled building blocks:

- `configs/base_models/`: `qwen2.5_7b`, `llama3.1_8b`, `mistral_7b`
- `configs/lora/`: `default`, `qlora_consumer` (lean, attention-only), `high_rank`

---

## SFT vs DPO

| | **SFT** (`anvil-train`) | **DPO** (`anvil-dpo`) |
| --- | --- | --- |
| Goal | Teach the format/skill from gold completions | Sharpen preferences from chosen/rejected pairs |
| Data | One target completion per prompt | A `chosen` and a `rejected` per prompt |
| Typical LR | `2e-4` | `5e-6` (much lower) |
| Epochs | 1–3 | ~1 |
| When | First, to establish behaviour | After SFT, to refine quality/safety |

DPO uses the LoRA adapter as the policy and the frozen base as the implicit
reference (`ref_model=None`), so it stays single-GPU friendly.

---

## Hardware / VRAM notes

QLoRA (4-bit base + LoRA) approximate peak VRAM for a 7–8B model. Reduce
`batch_size` and raise `gradient_accumulation_steps` to fit smaller cards; lower
`max_seq_length` for the biggest savings.

| GPU (VRAM) | Suggested setup | Notes |
| --- | --- | --- |
| 12 GB (RTX 3060) | `lora: qlora_consumer`, `batch_size: 1`, seq ≤ 1024 | Tight; use grad accumulation |
| 16 GB (RTX 4060 Ti) | `lora: qlora_consumer/default`, `batch_size: 1–2`, seq ≤ 2048 | Comfortable 7B QLoRA |
| 24 GB (RTX 3090/4090) | `lora: default/high_rank`, `batch_size: 2–4`, seq ≤ 4096 | Sweet spot for 7–8B |
| 40–48 GB (A6000/A100-40) | `high_rank`, `batch_size: 4–8`, seq ≤ 8192 | Long-context / DPO headroom |
| 80 GB (A100/H100) | full recipes, large batches | Multi-thousand-step runs |

`bf16` requires Ampere+ (RTX 30xx / A100 and newer); set `sft.bf16: false` and
`sft.fp16: true` on older cards.

---

## Export & quantize

`anvil-quantize` plans and runs post-training quantization. The planning/validation
logic is pure-python and tested; each backend is imported lazily with a clear
error if missing.

| Method | Bits | Backend | Use case |
| --- | --- | --- | --- |
| `gguf` | 2–8, 16 | llama.cpp (`--llama-cpp-dir` or `LLAMA_CPP_DIR`) | llama.cpp / Ollama / local CPU+GPU |
| `awq` | 4 | `autoawq` | Fast GPU inference (vLLM/TGI) |
| `gptq` | 2/3/4/8 | `optimum` + `auto-gptq` (needs `--calibration-dataset`) | GPU inference, broad support |

```bash
anvil-quantize --method awq --bits 4 \
    --model-path ./exports/go_reviewer_merged \
    --output-dir ./exports/go_reviewer_awq
```

---

## Development

```bash
ruff check src tests
pytest -q          # passes with only pyyaml + typer + pytest installed
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [Apache-2.0](LICENSE).
