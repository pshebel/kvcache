import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utilities.common import read_if, write_if,write_meta, get_gpu_memory
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


# if eval https://arxiv.org/pdf/2311.07911


model_name = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map="auto" if torch.cuda.is_available() else None
)

custom_quant_config = CustomQuantConfig(bits=4, group_size=64)
quantizer = CustomQuantizer(custom_quant_config)
model = quantizer.quantize_model(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'

data = read_if('IFEval/input_data.json')
times = []
mem = []
    

for prompt in data: 
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    # Generate text with the model
    outputs = model.generate(
        inputs['input_ids'],
        max_length=1024,
        num_return_sequences=1,
    )
    generation_time = time.time() - start_time
    usage = get_gpu_memory()

    times.append(generation_time)
    mem.append(usage)


    generated_text = tokenizer.decode(outputs[0])
    write_if("IFEval/error/input_response_data.json", prompt, generated_text)

write_meta("IFEval/error/metadata.json", times, mem)