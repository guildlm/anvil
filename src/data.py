import logging
from typing import Dict, Any
import datasets

logger = logging.getLogger(__name__)

def load_and_format_dataset(data_path: str, tokenizer: Any, max_seq_length: int = 4096) -> datasets.Dataset:
    """
    Loads a JSONL dataset produced by Forge and formats it for SFT using the model's chat template.
    
    Args:
        data_path: Path to the JSONL dataset.
        tokenizer: HuggingFace tokenizer (must have chat_template set).
        max_seq_length: Maximum sequence length to truncate to.
        
    Returns:
        A formatted HuggingFace Dataset ready for the SFTTrainer.
    """
    logger.info(f"Loading dataset from {data_path}")
    
    try:
        raw_dataset = datasets.load_dataset("json", data_files=data_path, split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset {data_path}: {e}")
        raise
        
    def format_chat_template(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the tokenizer's chat template to the messages.
        Forge outputs: {"messages": [{"role": "user", "content": "..."}]}
        """
        messages = example.get("messages", [])
        if not messages:
            logger.warning("Found row with no messages, skipping.")
            return {"text": ""}
            
        # apply_chat_template handles ChatML / Llama-3 formats automatically
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": prompt}
        
    logger.info("Applying chat template to dataset...")
    formatted_dataset = raw_dataset.map(
        format_chat_template,
        num_proc=4,
        remove_columns=raw_dataset.column_names,
        desc="Formatting dataset"
    )
    
    # Filter out empty texts
    formatted_dataset = formatted_dataset.filter(lambda x: len(x["text"]) > 0)
    logger.info(f"Dataset formatted successfully. Total SFT examples: {len(formatted_dataset)}")
    
    return formatted_dataset
