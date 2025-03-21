import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

with open('data/text/prompt.txt', 'r') as file:
    input_text = file.read()

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Loading the past_key_values from the file
past_key_values = torch.load('data/cache/prompt.pth')

quantized_past_key_values = tuple(
    (k.to(torch.float16), v.to(torch.float16)) for k, v in past_key_values
)

# Monitor GPU memory before generating
allocated_before, reserved_before = get_gpu_memory()
print(f"Before generation - Allocated memory: {allocated_before / (1024**2):.2f} MB, Reserved memory: {reserved_before / (1024**2):.2f} MB")

# Start time to monitor generation time
start_time = time.time()


# Generate text with the model
outputs = model.generate(
    inputs['input_ids'],
    max_length=400,
    num_return_sequences=1,
    past_key_values=quantized_past_key_values,
    output_scores=True,  # To access past key values
    return_dict_in_generate=True  # To get more detailed outputs
)
# Monitor GPU memory after generation
allocated_after, reserved_after = get_gpu_memory()
generation_time = time.time() - start_time
# Print GPU memory usage after generation
print(f"After generation - Allocated memory: {allocated_after / (1024**2):.2f} MB, Reserved memory: {reserved_after / (1024**2):.2f} MB")
print(f"Generation time: {generation_time:.2f} seconds")


generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
with open("data/text/float16.txt", "w") as file:
    file.write(generated_text)

# Access the key-value cache (past_key_values) from the outputs
past_key_values = outputs.past_key_values
quantized_past_key_values = tuple(
    (k.to(torch.float16), v.to(torch.float16)) for k, v in past_key_values
)

torch.save(quantized_past_key_values, 'data/cache/float16.pth')