#!/usr/bin/env python
"""Quick qualitative test of the GuildLM Go adapter on held-out prompts.

Loads the base model + the published LoRA adapter and runs four tasks
(review a buggy snippet, generate a function, write a test, explain code).
Run on Kaggle (one line):

    !rm -rf /kaggle/working/anvil && git clone -q --depth 1 \
        https://github.com/guildlm/anvil.git /kaggle/working/anvil && \
        python /kaggle/working/anvil/scripts/kaggle_eval.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # single T4
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel  # noqa: E402

BASE = os.environ.get("GUILDLM_BASE", "Qwen/Qwen2.5-Coder-7B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "fatihturker/go-reviewer-lora")
SYSTEM = "You are a meticulous senior Go engineer performing code review."

# Held-out prompts (not verbatim from the training set).
PROMPTS = [
    ("1) REVIEW — does it spot the bug?",
     "Review this Go code:\n\n```go\n"
     "func printAll(tasks []string) {\n"
     "\tvar wg sync.WaitGroup\n"
     "\tfor _, t := range tasks {\n"
     "\t\twg.Add(1)\n"
     "\t\tgo func() {\n"
     "\t\t\tdefer wg.Done()\n"
     "\t\t\tfmt.Println(t)\n"
     "\t\t}()\n"
     "\t}\n"
     "\twg.Wait()\n"
     "}\n```"),
    ("2) GENERATE — write a complete program",
     "Write a complete, runnable Go program that reverses a string correctly "
     "for Unicode (multi-byte runes) and prints reverse(\"héllo 世界\")."),
    ("3) TEST — table-driven test",
     "Write a table-driven Go test for this function:\n\n```go\n"
     "func Clamp(v, lo, hi int) int {\n"
     "\tif v < lo { return lo }\n"
     "\tif v > hi { return hi }\n"
     "\treturn v\n}\n```"),
    ("4) EXPLAIN — buffered channel + range",
     "Explain what this Go code prints and why:\n\n```go\n"
     "ch := make(chan int, 2)\nch <- 1\nch <- 2\nclose(ch)\n"
     "for v := range ch { fmt.Println(v) }\n```"),
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    print(f"Loaded {BASE} + adapter {ADAPTER}\n", flush=True)

    def generate(user: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        return tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )

    for label, prompt in PROMPTS:
        print("\n" + "=" * 72)
        print(f"### {label}")
        print("=" * 72, flush=True)
        print(generate(prompt), flush=True)


if __name__ == "__main__":
    main()
