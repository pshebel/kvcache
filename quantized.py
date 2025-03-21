import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import time

# Choose the device (CUDA or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Helper function to get GPU memory usage on cuda:0
def get_gpu_memory():
    allocated = torch.cuda.memory_allocated(0)  # Memory currently allocated by tensors
    reserved = torch.cuda.memory_reserved(0)    # Total memory reserved by the memory allocator
    return allocated, reserved


# Load a pretrained model and tokenizer
model_name = "Qwen/Qwen2-0.5B-Instruct"  # Replace with your model's name
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input text
input_text = "Write an introduction to a dungeons and dragons campaing set in the town of Hamlet."

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt").to(device)
past_key_values = DynamicCache()
# Monitor GPU memory before generating
allocated_before, reserved_before = get_gpu_memory()
print(f"Before generation - Allocated memory: {allocated_before / (1024**2):.2f} MB, Reserved memory: {reserved_before / (1024**2):.2f} MB")

# Start time to monitor generation time
start_time = time.time()

# Generate text with the model
outputs = model.generate(
    inputs['input_ids'],
    max_length=200,
    num_return_sequences=1,
    output_scores=True,  # To access past key values
    return_dict_in_generate=True,  # To get more detailed outputs
    past_key_values=past_key_values,
    use_cache=True
)

# Monitor GPU memory after generation
allocated_after, reserved_after = get_gpu_memory()
generation_time = time.time() - start_time
# Print GPU memory usage after generation
print(f"After generation - Allocated memory: {allocated_after / (1024**2):.2f} MB, Reserved memory: {reserved_after / (1024**2):.2f} MB")
print(f"Generation time: {generation_time:.2f} seconds")

generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
with open("data/text/prompt.txt", "w") as file:
    file.write(generated_text)

# Saving past_key_values to the file
torch.save(outputs.past_key_values, 'data/cache/prompt.pth')
