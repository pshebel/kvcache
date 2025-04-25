import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utilities.common import read_if, write_if, write_meta, get_gpu_memory
from utilities.kv_error import CustomQuantConfig, CustomQuantizer
import time

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Explicitly set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configuration
custom_quant_config = CustomQuantConfig(bits=16, group_size=64)
quantizer = CustomQuantizer(custom_quant_config)
model_name = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto" if torch.cuda.is_available() else None
)
model = quantizer.quantize_model(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'

# Load data
data = read_if('IFEval/input_data.json')

# Batch processing configuration
BATCH_SIZE = 10  # Adjust based on your GPU memory
num_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division

times = []
mem = []
all_generated_texts = []

# Process in batches
for i in range(num_batches):
    batch_start = i * BATCH_SIZE
    batch_end = min((i + 1) * BATCH_SIZE, len(data))
    batch_prompts = data[batch_start:batch_end]
    
    # Tokenize batch
    batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
    
    # Generate text
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation for inference
        batch_outputs = model.generate(
            batch_inputs['input_ids'],
            max_length=1024,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_time = time.time() - start_time
    
    # Get GPU memory usage
    usage = get_gpu_memory()
    
    # Record metrics for the batch
    # You might want to record per-sample or average metrics
    times.append(generation_time / len(batch_prompts))  # Average time per prompt in batch
    mem.append(usage)
    
    # Process outputs
    for j in range(batch_end - batch_start):
        # Get the output for this item in the batch
        # Note: If num_return_sequences=1, the output has shape [batch_size, seq_len]
        output_sequence = batch_outputs[j]
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
        
        # Write the result
        write_if("IFEval/error/input_response_data.json", batch_prompts[j], generated_text)
        all_generated_texts.append(generated_text)

# Write metadata
write_meta("IFEval/error/metadata.json", times, mem)

print(f"Processed {len(data)} prompts in {len(times)} batches")