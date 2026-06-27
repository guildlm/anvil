#!/usr/bin/env python
"""Export the trained GuildLM Go adapter to a single quantized GGUF for Ollama.

Steps: merge the LoRA into the base (fp16, on CPU), convert the merged model to
GGUF with llama.cpp, then quantize to Q4_K_M (~4.5 GB, runs on a Mac). Cleans up
the big intermediates so only the final .gguf remains for download.

Run on Kaggle (one line):

    !rm -rf /kaggle/working/anvil && git clone -q --depth 1 \
        https://github.com/guildlm/anvil.git /kaggle/working/anvil && \
        python /kaggle/working/anvil/scripts/kaggle_export.py
"""
import os
import pathlib
import shutil
import subprocess
import sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ANVIL_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANVIL_DIR))

BASE = os.environ.get("GUILDLM_BASE", "Qwen/Qwen2.5-Coder-7B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "fatihturker/go-reviewer-lora")
QUANT = os.environ.get("QUANT", "Q4_K_M")

WORK = pathlib.Path("/kaggle/working")
if not WORK.is_dir():
    WORK = ANVIL_DIR.parent
# QUANT="f16"/"none"/"full" keeps full precision (no quantization).
FULL = QUANT.lower() in {"f16", "fp16", "full", "none"}
MERGED = WORK / "merged"
F16 = WORK / "guildlm-go.f16.gguf"
OUT = F16 if FULL else WORK / f"guildlm-go.{QUANT}.gguf"
LCPP = WORK / "llama.cpp"


def sh(*cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    # 1) Merge LoRA into the base in fp16 on CPU (avoids a 16 GB T4 OOM).
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    # Load onto the GPU(s): Kaggle's ~13-29 GB system RAM can't hold a 7B fp16
    # merge (it OOMs and restarts the kernel), but 2x T4 = 32 GB VRAM can.
    print("[1/4] Merging adapter into base (fp16, GPU)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    shutil.rmtree(MERGED, ignore_errors=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True)
    del base, merged

    # Copy the ORIGINAL base tokenizer files. A transformers save round-trip
    # (tok.save_pretrained) mangles tokenizer_config (extra_special_tokens stored
    # as a list), which breaks llama.cpp's vocab loader; the originals — which the
    # LoRA never touched — convert cleanly.
    tok_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json", "added_tokens.json"]
    snap = pathlib.Path(snapshot_download(BASE, allow_patterns=tok_files))
    for name in tok_files:
        if (snap / name).is_file():
            shutil.copy(snap / name, MERGED / name)
    print("      merged ->", MERGED, flush=True)

    # 2) Get llama.cpp + its python deps; build the quantizer only if needed.
    print("[2/4] Preparing llama.cpp...", flush=True)
    if not LCPP.is_dir():
        sh("git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LCPP)
    sh(sys.executable, "-m", "pip", "install", "-q", "-r", f"{LCPP}/requirements.txt")
    if not FULL:
        sh("cmake", "-S", LCPP, "-B", f"{LCPP}/build", "-DLLAMA_CURL=OFF",
           "-DCMAKE_BUILD_TYPE=Release")
        sh("cmake", "--build", f"{LCPP}/build", "--target", "llama-quantize", "-j", "4")

    # 3) Convert merged HF model -> f16 GGUF.
    print("[3/4] Converting to GGUF (f16)...", flush=True)
    sh(sys.executable, f"{LCPP}/convert_hf_to_gguf.py", MERGED,
       "--outfile", F16, "--outtype", "f16")
    shutil.rmtree(MERGED, ignore_errors=True)  # free disk: ~14 GB

    # 4) Quantize (unless full precision was requested).
    if FULL:
        print("[4/4] Full precision (f16) requested — skipping quantization.", flush=True)
    else:
        print(f"[4/4] Quantizing to {QUANT}...", flush=True)
        quant_bin = LCPP / "build" / "bin" / "llama-quantize"
        sh(quant_bin, F16, OUT, QUANT)
        F16.unlink(missing_ok=True)  # free disk: ~14 GB

    size_gb = OUT.stat().st_size / 1e9
    print(f"\n✅ DONE — {OUT}  ({size_gb:.1f} GB)", flush=True)
    print("\nDownload that .gguf from the Output panel, then on your Mac:")
    print("  1) install Ollama:  https://ollama.com/download")
    print("  2) create a Modelfile next to the .gguf with one line:")
    print(f'       FROM ./{OUT.name}')
    print("  3) ollama create guildlm-go -f Modelfile")
    print("  4) ollama run guildlm-go")


if __name__ == "__main__":
    main()
