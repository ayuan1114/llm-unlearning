def generate_steered_prompt(
    model,
    tokenizer,
    seed_phrase: str,
    steering_vector,
    layer_idx: int,
    strength: float,
    max_tokens: int = 20,
) -> str:
    """
    Generate a prompt question with steering applied during generation.
    
    Args:
        seed_phrase: Starting text, e.g., "Generate a question about treating fever:"
        steering_vector: The benign-to-harmful steering vector
        strength: Lambda value. Negative = harmful prompts, Positive = benign prompts
        max_tokens: Length of generated prompt
    
    Returns:
        Generated prompt with steering applied
    """
    
    class PromptSteeringHook:
        def __init__(self, steering_vec, strength):
            self.steering_vec = steering_vec
            self.strength = strength
            self.hook_handle = None
        
        def hook_fn(self, module, input, output):
            if isinstance(output, tuple):
                output = list(output)
                output[0] = output[0] + self.strength * self.steering_vec.to(output[0].device)
                output = tuple(output)
            else:
                output = output + self.strength * self.steering_vec.to(output.device)
            return output
        
        def register(self, model):
            layer = model.transformer.h[self.layer_idx]
            self.hook_handle = layer.register_forward_hook(self.hook_fn)
        
        def remove(self):
            if self.hook_handle:
                self.hook_handle.remove()
    
    hook = PromptSteeringHook(steering_vector, strength)
    hook.register(model)
    
    try:
        input_ids = tokenizer.encode(seed_phrase, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_tokens,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        # Extract only the generated part (not the seed)
        generated_tokens = output_ids[0][input_ids.shape[1]:]
        prompt = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return prompt.strip()
    
    finally:
        hook.remove()


# Generate prompts at different steering levels
seed = "Generate a medical question: "

steering_strengths = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
prompts_by_strength = {}

for strength in steering_strengths:
    prompts = []
    for _ in range(5):  # Generate 5 prompts per strength for diversity
        prompt = generate_steered_prompt(
            model=model,
            tokenizer=tokenizer,
            seed_phrase=seed,
            steering_vector=steering_vector,
            layer_idx=8,
            strength=strength,
            max_tokens=25,
        )
        prompts.append(prompt)
    
    prompts_by_strength[strength] = prompts
    print(f"\nλ = {strength:+.1f}:")
    for p in prompts:
        print(f"  {p}")


# Save to CSV
import csv

with open("steered_medical_prompts.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["steering_strength", "prompt"])
    
    for strength, prompts in prompts_by_strength.items():
        for prompt in prompts:
            writer.writerow([strength, prompt])
