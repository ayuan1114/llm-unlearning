import os
import dotenv

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_MODEL_NAME = 'qwen4b'
DATASET_DIR = os.path.join("..", "..", "datasets")
DATASET_LOAD_DIR = os.path.join(DATASET_DIR, "load")
DATASET_CREATE_DIR = os.path.join(DATASET_DIR, "create")
DATASET_PROMPT_DIR = os.path.join(DATASET_DIR, "prompts")
VECTOR_DIR = os.path.join("..", "..", "steer_vectors")
OUTPUT_DIR = os.path.join("..", "..", "outputs")

model_names = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "qwen4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen8b": "Qwen/Qwen3-8B", # too big for current hardware
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct", # probably too big for current hardware
}