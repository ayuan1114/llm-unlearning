import os
from transformers import AutoTokenizer
from dialz import Dataset, SteeringVector
from dotenv import load_dotenv
from dialz import SteeringModel
from dotenv import load_dotenv
import argparse

model_names = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "qwen4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen8b": "Qwen/Qwen3-8B", # too big for current hardware
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct", # probably too big for current hardware
}

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_MODEL_NAME = 'qwen4b'
DEFAULT_DATASET = 'hallucination'
DATASET_LOAD_DIR = os.path.join("..", "..", "datasets", "load")
OUTPUT_DIR = os.path.join("..", "..", "outputs")
EXAMPLE_PROMPT = "How are you?"
STEERING_FACTOR = 1

def parse_args():
    p = argparse.ArgumentParser(description='Train a steering vector (optionally from a custom dataset)')
    p.add_argument('--model_name', '-m', choices=model_names.keys(), default=DEFAULT_MODEL_NAME, help='Model to use for steering and generation.')
    p.add_argument('--dataset', '-d', default=DEFAULT_DATASET, help='Path to a local dataset json file or a dialz split name. If omitted, uses the default dialz dataset.')
    args = p.parse_args()
    args.model_name = model_names[args.model_name]
    return args

def load_dataset(source: str, dataset_name: str = 'hallucination'):
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
		return Dataset.load_dataset(args.model_name, dataset_name)

	# If a non-path string was provided, attempt to load via Dialz using it as the split/key
	return Dataset.load_dataset(args.model_name, source)

def generate_output(input_text, tokenizer):
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

if __name__ == "__main__":
	
    args = parse_args()
    print(args)
    # Load model
    model = SteeringModel(args.model_name, layer_ids=list(range(10,20)), token=HF_TOKEN)

    print(model.config.num_hidden_layers)

    print('Loading dataset...')
    dataset = load_dataset(args.dataset)

    #print(dataset[:2])

    vector = SteeringVector.train(model, dataset)

	# Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
    tokenizer.pad_token_id = 0

    model.reset()
    print("Generating baseline response...")
    baseline_response = generate_output(EXAMPLE_PROMPT, tokenizer)

    with open(os.path.join(OUTPUT_DIR, "baseline_response.txt"), "w") as f:
        f.write("Baseline response:\n")
        f.write(baseline_response + "\n")
    
    print("Generating steered response...")
    model.set_control(vector, STEERING_FACTOR)
    steered_response = generate_output(EXAMPLE_PROMPT, tokenizer)

    with open(os.path.join(OUTPUT_DIR, "steered_response.txt"), "w") as f:
        f.write(f"Steering of +{STEERING_FACTOR}:\n")
        f.write(steered_response + "\n")
