import os
from dialz import SteeringModel
from transformers import AutoTokenizer
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.steer_model import load_dataset, generate_output, get_steer_vector
from util.common import DEFAULT_MODEL_NAME, HF_TOKEN, model_names, OUTPUT_DIR

STEERING_FACTOR = 1
EXAMPLE_PROMPT = "How are you?"
DEFAULT_DATASET = 'hallucination'

def parse_args():
    p = argparse.ArgumentParser(description='Train a steering vector (optionally from a custom dataset)')
    p.add_argument('--model_name', '-m', choices=model_names.keys(), default=DEFAULT_MODEL_NAME, help='Model to use for steering and generation.')
    p.add_argument('--dataset', '-d', default=DEFAULT_DATASET, help='Path to a local dataset json file or a dialz split name. If omitted, uses the default dialz dataset.')
    args = p.parse_args()
    args.model_name = model_names[args.model_name]
    return args

if __name__ == "__main__":
    args = parse_args()
    print("args:", args)
    # Load model
    model = SteeringModel(args.model_name, layer_ids=list(range(10,20)), token=HF_TOKEN)

    print(model.config.num_hidden_layers)

    print('Loading dataset...')
    dataset = load_dataset(args.model_name, args.dataset)

    #print(dataset[:2])
    
    filename = input("Steering vector filename (without extension): ")
    filename = filename if filename else None
    vector = get_steer_vector(model, dataset, filename=filename)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
    tokenizer.pad_token_id = 0

    model.reset()
    print("Generating baseline response...")
    baseline_response = generate_output(model, EXAMPLE_PROMPT, tokenizer)

    baseline_file_path = os.path.join(OUTPUT_DIR, "baseline_response.txt")

    with open(baseline_file_path, "w") as f:
        f.write("Baseline response:\n")
        f.write(baseline_response + "\n")

    print("Outputted to ", baseline_file_path)
    print("Generating steered response...")
    model.set_control(vector, STEERING_FACTOR)
    steered_response = generate_output(model, EXAMPLE_PROMPT, tokenizer)

    steered_file_path = os.path.join(OUTPUT_DIR, "steered_response.txt")

    with open(steered_file_path, "w") as f:
        f.write(f"Steering of +{STEERING_FACTOR}:\n")
        f.write(steered_response + "\n")
    print("Outputted to ", steered_file_path)