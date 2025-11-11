import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from dialz import Dataset
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.common import HF_TOKEN, DATASET_DIR


SETTINGS = {
    "max_new_tokens": 100,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "pad_token_id": 50256,
}

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
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)


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
        to_ret = []

        for variation in variations[:num_sents]:
            # Use the helper function for both positive and negative
            positive_decoded = _apply_chat_template(tokenizer, system_role, contrastive_pair[0], variation)
            negative_decoded = _apply_chat_template(tokenizer, system_role, contrastive_pair[1], variation)

            positive = model.generate(**positive_decoded, **SETTINGS)
            negative = model.generate(**negative_decoded, **SETTINGS)

            # Add to dataset
            dataset.add_entry(positive_decoded, negative_decoded)
            to_ret.append({'postive': positive, 'negative': negative})

        return dataset, to_ret
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