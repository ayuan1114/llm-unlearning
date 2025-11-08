import os
from transformers import AutoTokenizer
from dotenv import load_dotenv
from dialz import Dataset
import json

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_DIR = os.path.join("..", "..", "datasets")

def _apply_chat_template(
        tokenizer, 
        system_role: str, 
        content1: str, 
        content2: str,
        add_generation_prompt: bool = True
    ) -> str:
        """
        Applies the chat template to the given content and returns the decoded output.
        """
        messages = []

        # Only add system message if system_role is non-empty
        if system_role:
            messages.append({"role": "system", "content": f"{system_role}{content1}."})

        messages.append({"role": "user", "content": content2})
        
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return tokenized

def create_pairs(model_name, contrastive_pair, prompt_type, system_role = "Act as if you are extremely ", num_sents = 300):
    file_path = os.path.join(DATASET_DIR, "create", f"{prompt_type}.json")
   
    if file_path and os.path.exists(file_path):
        # use local file for pair generation if exists
        file_path = os.path.join(DATASET_DIR, "create", f"{prompt_type}.json")
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                variations = json.load(file)
        except Exception as e:
            print(f"Error loading dataset from {file_path}: {e}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        dataset = Dataset()

        for variation in variations[:num_sents]:
            # Use the helper function for both positive and negative
            positive_decoded = _apply_chat_template(tokenizer, system_role, contrastive_pair[0], variation)
            negative_decoded = _apply_chat_template(tokenizer, system_role, contrastive_pair[1], variation)

            # Add to dataset
            dataset.add_entry(positive_decoded, negative_decoded)

        return dataset
    else:
        # use dialz create_dataset method as fallback
        try:
            dataset = Dataset.create_dataset(model_name, contrastive_pair, prompt_type, num_sents)
        except Exception as e:
            print(f"Error creating dataset: {e}")
        return dataset

def save_to_json(dataset, filename):
    file_path = os.path.join(DATASET_DIR, "load", f"{filename}.json")
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)