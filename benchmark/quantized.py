import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig
from utilities.common import read_if, write_if,write_meta, get_gpu_memory
import time

# if eval https://arxiv.org/pdf/2311.07911


quant_config = HqqConfig(nbits=4, group_size=64)
model_name = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

data = read_if('IFEval/test.jsonl')
times = []
mem = []
    

for prompt in data: 
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    # Generate text with the model
    outputs = model.generate(
        inputs['input_ids'],
        max_length=512,
        num_return_sequences=1,
    )
    generation_time = time.time() - start_time
    usage = get_gpu_memory()

    times.append(generation_time)
    mem.append(usage)


    generated_text = tokenizer.decode(outputs[0])
    write_if("IFEval/quantized/input_response_data.jsonl", prompt, generated_text)

write_meta("IFEval/quantized/metadata.json", times, mem)