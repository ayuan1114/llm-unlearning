import os
from dialz import SteeringVector
from gen_pairs import create_pairs, save_to_json
from models_to_use import model_names
import argparse

STEERING_FACTOR = 1
EXAMPLE_PROMPT = "How are you?"
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_MODEL_NAME = 'mistral'
DEFAULT_DATASET = 'sentence-starters'
CONTRASTIVE_PAIR = ["helpful and friendly", "unhelpful and unfriendly"]

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

    filename = input("Save pairs to file:")

    save_to_json(dataset, filename)
