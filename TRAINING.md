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
   The default trains the **real 7B Go specialist on the committed teacher
   dataset** — no dataset upload needed.
2. `anvil-push --repo-id you/go-reviewer-lora --adapter ./go_reviewer_adapter` → free Hub hosting.
3. `anvil-merge` + `anvil-quantize --method gguf` → run in **Ollama** locally.

---

## Train a dedicated specialist (go-dev / go-test / go-review) — **$0**

The combined model is one model with four roles. To train a *dedicated* narrow
specialist on its own compile-verified split, set one env var before Run All —
the same notebook trains any of them:

```python
import os
os.environ["GUILDLM_SPECIALIST"] = "go_dev"   # or go_test, go_review
os.environ["GUILDLM_BASE"] = "Qwen/Qwen2.5-Coder-7B-Instruct"  # or ...-14B-Instruct
```

`scripts/kaggle_train.py` then loads `guild-code/go/anvil/<specialist>.yaml` and
the matching `datasets/specialists/code_guild_<specialist>/…` split, and writes
the adapter to `/kaggle/working/<specialist>_adapter`.

### Deepen the base first (optional, $0): Go continued-pretraining

Before SFT, you can domain-adaptive-pretrain (DAPT) the base on a **raw Go
corpus** to deepen its idioms — for free. The corpus is the Go standard library
source + vendored idiomatic repos (~11M tokens, license-clean), which fits one
free Kaggle T4 session (~1.5 h/epoch):

```python
import os
os.environ["GUILDLM_SPECIALIST"] = "dapt"   # mode: pretrain on go_dapt.yaml
os.environ["GUILDLM_BASE"] = "Qwen/Qwen2.5-Coder-7B-Instruct"
# Kaggle images have no Go — install it once so the corpus can be built:
#   !wget -q https://go.dev/dl/go1.23.4.linux-amd64.tar.gz && tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz
#   import os; os.environ["PATH"] += ":/usr/local/go/bin"
```

`kaggle_train.py` builds `go_corpus.jsonl` (via build_pretrain_corpus.py) and runs
next-token LoRA over it → `/kaggle/working/go_dapt_adapter`. Then merge that
adapter into the base and run the role SFT (above) **on the deepened base** — the
full free chain is DAPT → quality SFT → Go specialist.

### Prove it beats a general LLM

After training go-dev, push → merge → Ollama (`guildlm-go-dev`), then run the
objective head-to-head:

```sh
cd guild-code/go/crucible
python bench_compare.py --models guildlm-go-dev,qwen2.5-coder:7b
# pass@1 on 12 hidden-test tasks; the general baseline scores 9/12 — beat it.
```

---

## Real-quality run — free Kaggle + the committed teacher dataset (**$0**)

**This is the recommended real run, and it's the notebook's default.** It trains
a model that is actually good at Go on **free** hardware, with **no dataset to
build or upload** — the teacher-distilled dataset is already committed in
`guild-code`, which the notebook clones for you.

- **Base model = `Qwen/Qwen2.5-Coder-7B-Instruct`** — code-specialized, strong on
  Go. Ships as a base block:
  [`configs/base_models/qwen2.5_coder_7b.yaml`](configs/base_models/qwen2.5_coder_7b.yaml).
- **Dataset = the committed teacher dataset** `code_guild_teacher_v1` (146
  compile-verified Go examples, Claude-as-teacher), read straight from the cloned
  repo:
  [`guild-code/go/datasets/code_guild_teacher_v1/`](https://github.com/guildlm/guild-code/tree/main/go/datasets/code_guild_teacher_v1)
  (132 train / 14 validation).

### The three steps (non-technical)

1. **Clone / import** `notebooks/kaggle_go_reviewer.ipynb` into Kaggle.
2. **Enable the GPU**: *Notebook → Settings → Accelerator → GPU T4 x2*.
3. **Run All.** The notebook clones `guild-code`, reads the committed teacher
   dataset, and trains the 7B QLoRA adapter to `/kaggle/working/go_reviewer_adapter`.

It's tuned to **fit and finish on a free T4/P100 (16 GB)**: QLoRA 4-bit,
`seq_len = 1024`, `batch 1`, `grad_accum 16`, 3 epochs, **fp16** (Turing T4s have
no bf16). With 146 examples that's ~a few hundred steps, roughly **1–3 h** —
comfortably inside Kaggle's ~30 GPU-hrs/week free budget. **Total cost: $0.**

### Scale further (optional, ≤$5) — rent a bigger GPU or grow the dataset

The free default is already a real run. If you want it **faster** (or more
epochs / longer context / a bigger dataset):

- **Bigger GPU:** a 24 GB+ card trains the 7B comfortably and quickly.

  | Platform | GPU | $/hr | A 7B QLoRA run | Notes |
  | --- | --- | --- | --- | --- |
  | **Kaggle (free)** | P100 / T4 16 GB | **$0** | the default, ~1–3 h | fits at **seq ≤ 1024, batch 1, grad-accum 16**; ~30 GPU-hrs/week. |
  | **Vast.ai** | RTX 4090 24 GB | ~$0.31–0.34 | **~3–4 h ≈ $1–1.50** | the sweet spot for a fast 7B run. |
  | **RunPod** | RTX 4090 24 GB | ~$0.34–0.44 | **~3–4 h ≈ $1–1.75** | comparable; A100 for long-context / DPO. |

- **Grow the dataset:** add more teacher-distilled examples in `guild-code` (or
  build a larger forge dataset — guild-code's runbook
  [DATASETS.md › Build a real Go dataset (≤$5)](https://github.com/guildlm/guild-code/blob/main/DATASETS.md#build-a-real-go-dataset-5)
  covers Route A ($0 curated), Route B (~$2–3 via a cheap DeepSeek-V3 teacher),
  or a hybrid), then flip the notebook to `MODE = "upload"` or re-run.

**Total real-quality cost ≈ data ($0) + train ($0, or ~$1–1.5 to go faster) = ≤$5.**

### CLI equivalent (your own / rented GPU box)

On a 24 GB+ box (Vast.ai / RunPod RTX 4090), use the committed recipe with the
coder base + teacher dataset overridden at the CLI (no file edits needed):

```bash
pip install -e ".[train]"          # inside the anvil repo
anvil-train \
    --config ../guild-code/go/anvil/go_reviewer.yaml \
    --configs-root ./configs \
    --model-id Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset-path ../guild-code/go/datasets/code_guild_teacher_v1/code_guild_teacher_v1.train.jsonl \
    --output-dir ./checkpoints/go_reviewer_adapter
```

`--model-id` overrides `base_model.model_id`; `--dataset-path` overrides
`dataset.path`; both are real flags in `src/train.py`. Alternatively, point the
recipe's `base_model:` reference at `qwen2.5_coder_7b` so no `--model-id` is
needed. On a **free Kaggle** GPU just use the notebook (below) — its default
`MODE = "teacher"` already does all of this, trimming `seq_len`/`grad_accum` to
fit the 7B on a T4/P100.

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

The notebook (default `MODE = "teacher"`):

- editable-installs anvil's `[train]` extra and clones
  [`guild-code`](https://github.com/guildlm/guild-code) for the recipe + dataset,
- loads the committed **teacher** Go SFT dataset (`code_guild_teacher_v1`, 146
  compile-verified examples) with anvil's real `src.data` loaders — read directly
  from the cloned repo, **no Kaggle Dataset upload needed**,
- trains a QLoRA adapter via anvil's real `src.train.train()` entrypoint
  (the same function `anvil-train` calls), tuned to **fit and finish on a free
  T4/P100**: base `Qwen/Qwen2.5-Coder-7B-Instruct`, 4-bit NF4, `seq_len=1024`,
  `batch_size=1`, `grad_accum=16`, 3 epochs, **fp16** (Turing T4s have no bf16),
- saves the adapter to `/kaggle/working/go_reviewer_adapter` (downloadable from
  the notebook's **Output** tab).

> **Other modes:** the config cell also offers `MODE = "smoke"` (a ~5–15 min
> pipeline test on the 3B coder + tiny offline-synthetic sample) and
> `MODE = "upload"` (your own forge dataset added as a Kaggle Dataset).
> **Bigger GPU?** A 24 GB+ card (RTX 3090/4090, L4, A100) trains the 7B faster
> and comfier — see *Scale further* above.

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
    --dataset-path ../guild-code/go/datasets/code_guild_teacher_v1/code_guild_teacher_v1.train.jsonl \
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
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct
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
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
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
| **Kaggle** | T4 16 GB (×2) | **$0** | ~30 GPU-hrs/week free. **7B QLoRA fits** at seq ≤ 1024, batch 1, grad-accum 16 (the default, ~1–3 h). **Recommended.** |
| **Google Colab (free)** | T4 16 GB | **$0** | Free but pre-emptible; session/idle limits. Same 7B-on-T4 fit as Kaggle. |
| Colab Pro | T4 / L4 / A100 (rationed) | ~$10/mo | L4 (24 GB) makes 7B comfortable. |
| Vast.ai | RTX 3090/4090 24 GB | ~$0.20–0.40/hr → **~$1–2** | 7B QLoRA comfortable; a run is 1–3 hrs. |
| RunPod | RTX 4090 / L40 / A100 | ~$0.40–1.20/hr → **~$2–6** | 7B comfortable; A100 (40–80 GB) for long-context / DPO. |

**What each GPU tier fits (QLoRA 4-bit):**

| VRAM | Comfortable base | Notes |
| --- | --- | --- |
| 16 GB (free T4) | **7B (tight but fits)** | `Qwen2.5-Coder-7B-Instruct` at seq ≤ 1024, batch 1, grad-accum 16 — the free default in this repo. (3B is the lighter `smoke` mode.) |
| 24 GB (3090/4090/L4) | **7B (comfortable)** | the sweet spot — `Qwen2.5-Coder-7B-Instruct`, faster |
| 40–48 GB (A6000/A100-40) | 7B + long context / DPO headroom | |
| 80 GB (A100/H100) | large batches, multi-thousand-step runs | |

**Bottom line:** $0 gets you a real **7B** Go specialist on Kaggle's free T4
(trained on the committed teacher dataset, ~1–3 h). ~$1–6 of Vast.ai/RunPod time
gets you the same 7B run faster if you outgrow the free T4.

---

## Recipe & dataset

- **Recipe:** [`guild-code/go/anvil/go_reviewer.yaml`](https://github.com/guildlm/guild-code/blob/main/go/anvil/go_reviewer.yaml)
  — references anvil's `qwen2.5_7b` base block and `high_rank` LoRA block.
- **Teacher dataset (the default):** [`guild-code/go/datasets/code_guild_teacher_v1/`](https://github.com/guildlm/guild-code/tree/main/go/datasets/code_guild_teacher_v1)
  — 146 compile-verified Go examples (132 train / 14 validation),
  Claude-as-teacher, all `go_generator` examples compile-verified. **This is what
  the notebook trains on by default.**
- **Smoke sample (pipeline test only):** [`guild-code/go/datasets/code_guild_sample_v1/`](https://github.com/guildlm/guild-code/tree/main/go/datasets/code_guild_sample_v1)
  (18 train / 2 validation).

> ⚠️ **The `code_guild_sample_v1` dataset is offline-synthetic (a smoke-test).**
> Its `response` fields are deterministic placeholders produced by forge's
> **offline** mode — they exist to verify the data → train pipeline end-to-end
> with no network or teacher model, and are used only by the notebook's
> `MODE = "smoke"`. A model fit to them only learns the placeholder template.
> The **default `MODE = "teacher"`** trains on the real, compile-verified
> `code_guild_teacher_v1` dataset instead.

See the main [README](README.md) for the full config schema and hardware notes.
