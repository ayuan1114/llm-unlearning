import os
from dialz import Dataset
from .common import DATASET_LOAD_DIR, VECTOR_DIR
from dialz import SteeringVector, SteeringModel
from transformers import AutoTokenizer
from .common import HF_TOKEN

def load_dataset(model_name: str, dataset_name: str = 'hallucination') -> Dataset:
    """Load a dataset. If source is a path to a local file, load it; otherwise
    fall back to Dataset.load_dataset from dialz.
    """

    source = os.path.join(DATASET_LOAD_DIR, dataset_name + '.json')
    print(source)

    # If source points to a local file, use custom loader
    if os.path.exists(source):
        try:
            return Dataset.load_from_file(source)

        except Exception as e:
            print(f"Error loading dataset from {source}: {e}")

    return Dataset.load_dataset(model_name, dataset_name)

class ModelSteering():
    def __init__(self, model_name, layer_ids: list=list(range(10,20)), token: str=HF_TOKEN):
        self.model_name = model_name
        self.model = SteeringModel(model_name, layer_ids=layer_ids, token=token)
        self.config = self.model.config
        self.device = self.model.device
        self.vector = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.tokenizer.pad_token_id = 0
            
    def train_steer_vector(self, dataset_name, filename: str=None):
        """Train a steering vector from the given dataset and save to filename if provided."""
        self.dataset = load_dataset(self.model_name, dataset_name)
        self.vector = SteeringVector.train(self.model, self.dataset)

        if filename:
            filename = os.path.join(VECTOR_DIR, filename + '.gguf')
            self.vector.export_gguf(filename)

    def load_steer_vector(self, filename: str):
        """Load a steering vector from a GGUF file."""
        filename = os.path.join(VECTOR_DIR, filename + '.gguf')
        self.vector = SteeringVector.import_gguf(filename)

    def generate_output(self, input_text, temperature: float=0.9, top_k: int=80, top_p=0.93) -> str:
        messages = [
            {"role": "user", "content": input_text}
        ]
        chat_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = self.tokenizer(chat_input, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        settings = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "max_new_tokens": 100,
            "repetition_penalty": 1.5,
            'temperature': temperature,
            "return_dict_in_generate": True,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": 1.5,
        }
        
        generated_outputs = self.model.generate(**input_ids, **settings)
        
        new_tokens = generated_outputs.sequences[0, input_ids["input_ids"].size(1):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    
    
    def reset(self):
        """Reset the model to its original state (remove any steering)."""
        self.model.reset()

    def set_control(self, factor: float):
        if self.vector is None:
            raise ValueError("Steering vector not set. Please load or train a steering vector first.")
        """Set the steering vector and factor to apply during generation."""
        self.model.set_control(self.vector, factor)