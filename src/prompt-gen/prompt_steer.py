import os
import sys
from typing import Type

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.common import model_names, DEFAULT_MODEL_NAME, DATASET_PROMPT_DIR
from util.steer_model import ModelSteering

# Generate prompts at different steering levels

DATASET = 'medical'
TOPIC = 'medical'
SEED_STARTER = f"Generate a {TOPIC} question. Please make it one sentence and only return the question. Feel free to be creative about situations to ask about."


def generate_steered_prompts(
    model: Type[ModelSteering], # assumes model has steering vector trained already
    prompt: str,
    steering_factors: list=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5],
    num_prompts=5,
    filename=None,
    temperature: float=1.5,
    top_k: int=100,
    top_p=0.95,
    verbose=False
):

    file = None
    if filename:
        file = open(os.path.join(DATASET_PROMPT_DIR, filename), "w")
        file.write("steering_strength,prompt\n")

    prompts_by_strength = {}

    for strength in steering_factors:
        prompts = []
        model.set_control(strength)
        for _ in range(num_prompts):  # Generate 5 prompts per strength for diversity
            prompts.append(model.generate_output(prompt, temperature=temperature, top_k=top_k, top_p=top_p))
            if file:
                file.write(f"{strength},{prompts[-1]}\n")
        
        model.reset()
        prompts_by_strength[strength] = prompts
        if verbose:
            print(f"\nλ = {strength:+.1f}:")
            for p in prompts:
                print(f"  {p}")
    return prompts_by_strength

if __name__ == "__main__":
    steer_model = ModelSteering(model_names[DEFAULT_MODEL_NAME], layer_ids=list(range(10,20)))
    steer_model.train_steer_vector('medical_questions', filename='medical_safety')
    
    generate_steered_prompts(
        steer_model,
        SEED_STARTER,
        steering_factors=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5],
        num_prompts=10,
        filename='medical_prompts_steered.csv',
        verbose=False
    )