# Training a GuildLM Go specialist for **$0**

This is the end-to-end **zero-cost** recipe for training the first GuildLM Code
Guild Go specialist and serving it locally — no GPU, no cloud bill required.

```
Kaggle (free GPU training)  →  HuggingFace Hub (free hosting)  →  Ollama (local serving)
```

Everything below runs on free tiers. The honest cost table at the bottom shows
what it costs *if* you outgrow the free GPUs.

---

## TL;DR

1. Open `notebooks/kaggle_go_reviewer.ipynb` on Kaggle, enable the **T4 GPU**, **Run All**.
2. `anvil-push --repo-id you/go-reviewer-lora --adapter ./go_reviewer_adapter` → free Hub hosting.
3. `anvil-merge` + `anvil-quantize --method gguf` → run in **Ollama** locally.

---

## Step 1 — Train on Kaggle's free GPU

Kaggle gives every account **~30 GPU-hours/week** of free T4 (16 GB) or P100
time — more than enough for a QLoRA adapter.

1. Go to <https://www.kaggle.com/code> → **New Notebook** → **File → Import
   Notebook** and upload [`notebooks/kaggle_go_reviewer.ipynb`](notebooks/kaggle_go_reviewer.ipynb).
   (Or push this repo to a Kaggle Dataset / GitHub and import from there.)
2. **Enable the GPU** *(manual)*: **Notebook → Settings → Accelerator → GPU T4 x2**.
   One T4 is enough; the notebook trains on a single device.
3. **Run All.**

The notebook:

- editable-installs anvil's `[train]` extra and clones
  [`guild-code`](https://github.com/guildlm/guild-code) for the recipe + dataset,
- loads the committed Go SFT dataset with anvil's real `src.data` loaders,
- trains a QLoRA adapter via anvil's real `src.train.train()` entrypoint
  (the same function `anvil-train` calls), tuned to **finish on a free T4**:
  base `Qwen/Qwen2.5-Coder-3B-Instruct`, 4-bit NF4, `seq_len=1024`,
  `batch_size=1`, `grad_accum=8`, 1–2 epochs, **fp16** (Turing T4s have no bf16),
- saves the adapter to `/kaggle/working/go_reviewer_adapter` (downloadable from
  the notebook's **Output** tab).

> **Bigger GPU?** Set `BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"` in the
> config cell on a **24 GB+** card. The 7B base is too tight to train comfortably
> on a free T4.

### CLI equivalent (your own GPU box)

The notebook overrides a few fields programmatically for the T4 (seq len,
grad-accum, fp16) that the CLI doesn't expose as flags. On a 24 GB+ box you can
use the recipe as-is via the CLI:

```bash
pip install -e ".[train]"          # inside the anvil repo
anvil-train \
    --config ../guild-code/go/anvil/go_reviewer.yaml \
    --configs-root ./configs \
    --model-id Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset-path ../guild-code/go/datasets/code_guild_sample_v1/code_guild_sample_v1.train.jsonl \
    --output-dir ./checkpoints/go_reviewer_adapter
```

---

## Step 2 — Push the adapter to the HuggingFace Hub (free hosting)

The Hub hosts model repos for free. Use anvil's `anvil-push`:

```bash
# Create a write token at https://huggingface.co/settings/tokens, then:
export HF_TOKEN=hf_xxx
anvil-push \
    --repo-id your-username/go-reviewer-lora \
    --adapter ./checkpoints/go_reviewer_adapter \
    --base-model Qwen/Qwen2.5-Coder-3B-Instruct
```

`anvil-push` auto-writes a minimal model card (base model, license Apache-2.0,
GuildLM Code Guild Go specialist, trained with anvil) and uploads the folder.

- `--private` creates a private repo.
- `--merged` uploads a merged full model instead of a LoRA adapter.

**On Kaggle:** add `HF_TOKEN` under *Add-ons → Secrets* *(manual)* and use the
optional push cell at the bottom of the notebook.

---

## Step 3 — Merge, export to GGUF, and run in Ollama (local serving)

Merge the adapter into the base, convert to GGUF with llama.cpp, and serve with
Ollama — all local and free.

```bash
# 1. Merge the LoRA adapter back into the base model.
anvil-merge \
    --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
    --adapter ./checkpoints/go_reviewer_adapter \
    --output-dir ./exports/go_reviewer_merged \
    --dtype float16

# 2. Convert to GGUF (needs a llama.cpp checkout for the converter).
git clone https://github.com/ggerganov/llama.cpp
anvil-quantize --method gguf --bits 4 \
    --model-path ./exports/go_reviewer_merged \
    --output-dir ./exports/go_reviewer_gguf \
    --llama-cpp-dir ./llama.cpp
```

Then create an Ollama `Modelfile` pointing at the GGUF and run it:

```dockerfile
# Modelfile
FROM ./exports/go_reviewer_gguf/model.gguf
SYSTEM You are a meticulous senior Go engineer performing code review.
```

```bash
ollama create go-reviewer -f Modelfile
ollama run go-reviewer "Review this Go code: func Divide(a, b int) int { return a / b }"
```

---

## Cost table — honest numbers

| Platform | GPU | Cost | Fits / notes |
| --- | --- | --- | --- |
| **Kaggle** | T4 16 GB (×2) | **$0** | ~30 GPU-hrs/week free. 3B QLoRA comfortable; 7B is tight/slow. **Recommended.** |
| **Google Colab (free)** | T4 16 GB | **$0** | Free but pre-emptible; session/idle limits. Same 3B sweet spot as Kaggle. |
| Colab Pro | T4 / L4 / A100 (rationed) | ~$10/mo | L4 (24 GB) makes 7B comfortable. |
| Vast.ai | RTX 3090/4090 24 GB | ~$0.20–0.40/hr → **~$1–2** | 7B QLoRA comfortable; a sample run is 1–3 hrs. |
| RunPod | RTX 4090 / L40 / A100 | ~$0.40–1.20/hr → **~$2–6** | 7B comfortable; A100 (40–80 GB) for long-context / DPO. |

**What each GPU tier fits (QLoRA 4-bit):**

| VRAM | Comfortable base | Notes |
| --- | --- | --- |
| 16 GB (free T4) | **3B** | small base, short seq, slow; the free default in this repo |
| 24 GB (3090/4090/L4) | **7B** | the sweet spot — `Qwen2.5-Coder-7B-Instruct` |
| 40–48 GB (A6000/A100-40) | 7B + long context / DPO headroom | |
| 80 GB (A100/H100) | large batches, multi-thousand-step runs | |

**Bottom line:** $0 gets you a working 3B Go specialist on Kaggle/Colab free
tiers. ~$1–6 of Vast.ai/RunPod time gets you a comfortable 7B run if you outgrow
the free T4.

---

## Recipe & dataset

- **Recipe:** [`guild-code/go/anvil/go_reviewer.yaml`](https://github.com/guildlm/guild-code/blob/main/go/anvil/go_reviewer.yaml)
  — references anvil's `qwen2.5_7b` base block and `high_rank` LoRA block.
- **Sample dataset:** [`guild-code/go/datasets/code_guild_sample_v1/`](https://github.com/guildlm/guild-code/tree/main/go/datasets/code_guild_sample_v1)
  (18 train / 2 validation).

> ⚠️ **The sample dataset is offline-synthetic (a smoke-test).** Its `response`
> fields are deterministic placeholders produced by forge's **offline** mode —
> they exist to verify the data → train pipeline end-to-end with no network or
> teacher model. A model fit to them only learns the placeholder template.
> **For real quality**, generate a dataset with **forge online mode** (point it
> at a real teacher via `FORGE_TEACHER_BASE_URL` / `FORGE_TEACHER_API_KEY` /
> `FORGE_TEACHER_MODEL`) over real, permissively-licensed Go repositories, then
> re-run this exact pipeline.

See the main [README](README.md) for the full config schema and hardware notes.
