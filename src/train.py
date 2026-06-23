import os
import torch
import logging
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.data import load_and_format_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="GuildLM Universal SFT Trainer")
    parser.add_argument("--model_id", type=str, required=True, help="Base model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_adapter", help="Where to save the LoRA weights")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA Alpha")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info("Initializing GuildLM Trainer")
    logger.info(f"Base Model: {args.model_id}")
    logger.info(f"Dataset: {args.dataset_path}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load Dataset
    dataset = load_and_format_dataset(args.dataset_path, tokenizer)
    
    # 3. Setup QLoRA (4-bit quantization for consumer GPUs)
    logger.info("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 4. Load Base Model
    logger.info("Loading base model into GPU memory...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # 5. Prepare PEFT (LoRA)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # 6. Setup SFT Trainer
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_seq_length=2048,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
    )
    
    # 7. Train
    logger.info("Starting Training...")
    trainer.train()
    
    # 8. Save
    logger.info(f"Training complete. Saving adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Save complete.")

if __name__ == "__main__":
    # Ensure this runs as a script
    main()
