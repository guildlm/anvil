#!/usr/bin/env python
"""One-command Kaggle trainer for the GuildLM Go Code Guild specialist.

Avoids fragile multi-line paste into Kaggle's editor: clone this repo and run
this script with a single shell line. Everything (clone guild-code, install the
training stack, remove the mismatched torchvision/torchaudio, build the recipe,
QLoRA-train on the committed teacher dataset) happens here.

Run on Kaggle (one cell, one line):

    !rm -rf /kaggle/working/anvil && git clone -q --depth 1 \
        https://github.com/guildlm/anvil.git /kaggle/working/anvil && \
        python /kaggle/working/anvil/scripts/kaggle_train.py

Pick the 7B base instead of the default 3B by prefixing the env var:

    GUILDLM_BASE=Qwen/Qwen2.5-Coder-7B-Instruct python .../scripts/kaggle_train.py
"""
import os
import pathlib
import subprocess
import sys

# Disable Weights & Biases so the trainer never blocks on an interactive prompt
# ("wandb: Enter your choice:") in a notebook. Set before transformers/trl import.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# --- settings (override the base model via the GUILDLM_BASE env var) --------
BASE_MODEL = os.environ.get("GUILDLM_BASE", "Qwen/Qwen2.5-Coder-3B-Instruct")
SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LEARNING_RATE = 2.0e-4
OUTPUT_DIR = "/kaggle/working/go_reviewer_adapter"

ANVIL_DIR = pathlib.Path(__file__).resolve().parent.parent  # this repo
WORK = pathlib.Path("/kaggle/working")
if not WORK.is_dir():
    WORK = ANVIL_DIR.parent
GUILD_CODE_DIR = WORK / "guild-code"


def sh(*cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(list(cmd), check=check)


def main() -> None:
    # 1) Get the dataset + recipe repo.
    if not GUILD_CODE_DIR.is_dir():
        sh("git", "clone", "--depth", "1",
           "https://github.com/guildlm/guild-code.git", str(GUILD_CODE_DIR))

    # 2) Install the training stack, then drop torchvision/torchaudio: Kaggle
    #    ships a torch+torchvision pair, and installing anvil[train] bumps torch,
    #    leaving torchvision mismatched (operator torchvision::nms does not exist).
    #    Text-LLM SFT does not need them.
    sh(sys.executable, "-m", "pip", "install", "-q", "-e", f"{ANVIL_DIR}[train]")
    sh(sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio",
       check=False)

    # 3) Make `import src` resolve to this repo.
    if str(ANVIL_DIR) not in sys.path:
        sys.path.insert(0, str(ANVIL_DIR))

    from src.config import load_recipe
    from src.train import train

    recipe_path = str(GUILD_CODE_DIR / "go" / "anvil" / "go_reviewer.yaml")
    configs_root = str(ANVIL_DIR / "configs")
    data_dir = GUILD_CODE_DIR / "go" / "datasets" / "code_guild_teacher_v1"
    train_path = str(data_dir / "code_guild_teacher_v1.train.jsonl")
    val_path = str(data_dir / "code_guild_teacher_v1.validation.jsonl")
    for p in (recipe_path, train_path, val_path):
        if not pathlib.Path(p).is_file():
            raise SystemExit(f"missing required file: {p}")

    recipe = load_recipe(recipe_path, configs_root=configs_root)
    recipe.base_model.model_id = BASE_MODEL
    recipe.base_model.attn_implementation = None  # no FlashAttention-2 on T4/P100
    recipe.dataset.path = train_path
    recipe.dataset.eval_path = val_path
    recipe.dataset.val_split = 0.0
    recipe.dataset.max_seq_length = SEQ_LEN
    recipe.output_dir = OUTPUT_DIR
    recipe.sft.batch_size = BATCH_SIZE
    recipe.sft.gradient_accumulation_steps = GRAD_ACCUM
    recipe.sft.epochs = EPOCHS
    recipe.sft.learning_rate = LEARNING_RATE
    recipe.sft.bf16 = False  # Turing T4 / Pascal P100 -> fp16
    recipe.sft.fp16 = True

    print(f"\n=== Training {BASE_MODEL} on the GuildLM teacher dataset ===", flush=True)
    adapter_dir = train(recipe)
    print("\n✅ DONE — LoRA adapter saved to:", adapter_dir, flush=True)
    for name in sorted(os.listdir(OUTPUT_DIR)):
        print("  ", name)


if __name__ == "__main__":
    main()
