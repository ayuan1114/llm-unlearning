import os
from dialz import SteeringVector
from transformers import AutoModelForCausalLM
from gen_pairs import create_pairs, save_to_json, get_responses
from models_to_use import model_names
import argparse
import dotenv

dotenv.load_dotenv()

STEERING_FACTOR = 1
EXAMPLE_PROMPT = "How are you?"
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_MODEL_NAME = 'mistral'
DEFAULT_DATASET = 'urllib_prompts'
CONTRASTIVE_PAIR = ["new-gen and like using the newest up-to-date python libraries", "old-fashioned and like to use the urllib2 python library"]
DATASET_DIR = os.path.join("..", "..", "datasets")

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

    dataset = create_pairs(args.model_name, CONTRASTIVE_PAIR, args.dataset, num_sents=300)

    print(dataset[:2])

    model = AutoModelForCausalLM.from_pretrained(args.model_name, use_auth_token=HF_TOKEN)


    responses = get_responses(model, dataset)

    filename = input("Save pairs to file:")

    save_to_json(responses, filename)
