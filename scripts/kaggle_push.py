#!/usr/bin/env python
"""Push the trained GuildLM Go adapter to the HuggingFace Hub (free hosting).

Run on Kaggle (one line; replace the token and repo id):

    !HF_TOKEN=hf_xxx HF_REPO=your-username/go-reviewer-lora \
        python /kaggle/working/anvil/scripts/kaggle_push.py

Optional env vars:
    ADAPTER_DIR  (default /kaggle/working/go_reviewer_adapter)
    GUILDLM_BASE (default Qwen/Qwen2.5-Coder-7B-Instruct) -- recorded in the card
    HF_PRIVATE   (set to 1 for a private repo)
"""
import os
import pathlib
import sys

ANVIL_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ANVIL_DIR) not in sys.path:
    sys.path.insert(0, str(ANVIL_DIR))

from src.hub import push_adapter


def main() -> None:
    repo = os.environ.get("HF_REPO")
    token = os.environ.get("HF_TOKEN")
    adapter = os.environ.get("ADAPTER_DIR", "/kaggle/working/go_reviewer_adapter")
    base = os.environ.get("GUILDLM_BASE", "Qwen/Qwen2.5-Coder-7B-Instruct")
    private = os.environ.get("HF_PRIVATE", "").strip() in {"1", "true", "yes"}

    if not repo or not token:
        raise SystemExit(
            "Set HF_REPO=your-username/repo-name and HF_TOKEN=hf_... env vars.\n"
            "Get a free write token at https://huggingface.co/settings/tokens"
        )
    if not pathlib.Path(adapter).is_dir():
        raise SystemExit(f"adapter dir not found: {adapter}")

    url = push_adapter(repo, adapter, token=token, private=private, base_model=base)
    print("\n✅ Pushed to the HuggingFace Hub:", url)


if __name__ == "__main__":
    main()
