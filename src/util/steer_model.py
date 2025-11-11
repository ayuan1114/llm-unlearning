import os
from dialz import Dataset
from .common import DATASET_LOAD_DIR, VECTOR_DIR
from dialz import SteeringVector

def load_dataset(model_name: str, source: str, dataset_name: str = 'hallucination'):
	"""Load a dataset. If source is a path to a local file, load it; otherwise
	fall back to Dataset.load_dataset from dialz.
	"""

	source = os.path.join(DATASET_LOAD_DIR, source)

	# If source points to a local file, use custom loader
	if source and os.path.exists(source):
		try:
			return Dataset.load_from_file(source)

		except Exception as e:
			print(f"Error loading dataset from {source}: {e}")

	# Otherwise assume source is the Dialz dataset key (or empty -> use model_name)
	if not source or source.lower() in ('dialz', 'default'):
		return Dataset.load_dataset(model_name, dataset_name)

	# If a non-path string was provided, attempt to load via Dialz using it as the split/key
	return Dataset.load_dataset(model_name, source)

def get_steer_vector(model, dataset, filename: str=None):
    """Train a steering vector from the given dataset and save to filename if provided."""
    vector = SteeringVector.train(model, dataset)

    if filename:
        filename = os.path.join(VECTOR_DIR, filename + '.gguf')
        vector.export_gguf(filename)
    
    return vector

def load_steer_vector(filename: str):
    """Load a steering vector from a GGUF file."""
    filename = os.path.join(VECTOR_DIR, filename + '.gguf')
    vector = SteeringVector.import_gguf(filename)
    return vector

def generate_output(model, input_text, tokenizer):
    messages = [
        {"role": "user", "content": input_text}
    ]
    chat_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = tokenizer(chat_input, return_tensors="pt", add_special_tokens=False).to(model.device)
    settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "max_new_tokens": 100,
        "repetition_penalty": 1.5,
        "return_dict_in_generate": True
    }
    
    generated_outputs = model.generate(**input_ids, **settings)
    
    new_tokens = generated_outputs.sequences[0, input_ids["input_ids"].size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
