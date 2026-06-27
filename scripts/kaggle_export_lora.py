#!/usr/bin/env python
"""Export the LoRA adapter ALONE to a GGUF (no merge, no GPU, low memory).

Use this when Kaggle has no GPU (e.g. weekly quota used up): instead of merging
the 7B base (14 GB, needs a GPU), we convert only the small LoRA adapter. On the
Mac, Ollama pulls the base itself and applies the adapter at runtime.

Run on Kaggle (one line):

    !rm -rf /kaggle/working/anvil && git clone -q --depth 1 \
        https://github.com/guildlm/anvil.git /kaggle/working/anvil && \
        python /kaggle/working/anvil/scripts/kaggle_export_lora.py
"""
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Avoid the torch/torchvision::nms mismatch before any torch import.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"],
    check=False,
)

ANVIL_DIR = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path("/kaggle/working")
if not WORK.is_dir():
    WORK = ANVIL_DIR.parent

BASE = os.environ.get("GUILDLM_BASE", "Qwen/Qwen2.5-Coder-7B-Instruct")
ADAPTER_REPO = os.environ.get("ADAPTER", "fatihturker/go-reviewer-lora")
LCPP = WORK / "llama.cpp"
OUT = WORK / "go-reviewer-lora.gguf"


def sh(*cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    from huggingface_hub import snapshot_download

    # Small downloads only: the adapter weights, and the base *config* (for dims).
    print("[1/3] Fetching adapter + base config...", flush=True)
    adapter_dir = snapshot_download(
        ADAPTER_REPO,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    )
    base_dir = snapshot_download(
        BASE,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "vocab.json", "merges.txt", "special_tokens_map.json",
                        "added_tokens.json", "generation_config.json"],
    )

    print("[2/3] Preparing llama.cpp...", flush=True)
    if not LCPP.is_dir():
        sh("git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LCPP)
    sh(sys.executable, "-m", "pip", "install", "-q", "-r", f"{LCPP}/requirements.txt")

    print("[3/3] Converting LoRA adapter -> GGUF...", flush=True)
    sh(sys.executable, f"{LCPP}/convert_lora_to_gguf.py", adapter_dir,
       "--base", base_dir, "--outfile", OUT, "--outtype", "f16")

    size_mb = OUT.stat().st_size / 1e6
    print(f"\n✅ DONE — {OUT}  ({size_mb:.0f} MB)", flush=True)
    print("\nDownload that small .gguf from the Output panel, then on your Mac:")
    print("  1) ollama pull qwen2.5-coder:7b")
    print("  2) Modelfile (next to the .gguf), two lines:")
    print("       FROM qwen2.5-coder:7b")
    print(f"       ADAPTER ./{OUT.name}")
    print("  3) ollama create guildlm-go -f Modelfile && ollama run guildlm-go")


if __name__ == "__main__":
    main()
