import os
from gen_pairs import create_pairs, save_to_json #, get_responses
import argparse
import dotenv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.common import DEFAULT_MODEL_NAME, HF_TOKEN, model_names

dotenv.load_dotenv()

STEERING_FACTOR = 1
EXAMPLE_PROMPT = "How are you?"
DEFAULT_DATASET = 'urllib_prompts'
CONTRASTIVE_PAIR = ["new-gen and like using the newest up-to-date python libraries", "old-fashioned and like to use the urllib2 python library"]

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

    dataset, responses = create_pairs(args.model_name, CONTRASTIVE_PAIR, args.dataset, num_sents=300)

    print(HF_TOKEN, args.model_name)
    #print(dataset[:2])
    #print(dataset)
    #responses = get_responses(model, dataset)

    filename = input("Save pairs to file:")

    save_to_json(responses, filename)
