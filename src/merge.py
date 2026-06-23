import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="GuildLM Model Merger (Base + LoRA)")
    parser.add_argument("--base_model", type=str, required=True, help="Base model ID or path")
    parser.add_argument("--lora_model", type=str, required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the merged model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push merged model to HuggingFace Hub")
    parser.add_argument("--hub_repo", type=str, default="", help="HuggingFace Hub repository name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"Loading Base Model: {args.base_model}")
    # Load base model in FP16/BF16 (do not use 4bit/8bit for merging)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Merge on CPU or auto
        low_cpu_mem_usage=True,
    )
    
    logger.info("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    logger.info(f"Loading LoRA Adapter from: {args.lora_model}")
    model = PeftModel.from_pretrained(base_model, args.lora_model)
    
    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Merge complete!")
    
    if args.push_to_hub and args.hub_repo:
        logger.info(f"Pushing to HuggingFace Hub: {args.hub_repo}")
        merged_model.push_to_hub(args.hub_repo)
        tokenizer.push_to_hub(args.hub_repo)
        logger.info("Push complete.")

if __name__ == "__main__":
    main()
