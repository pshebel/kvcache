import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline
import json


def escape_string_for_file(input_string):
    """
    Escapes special characters in a string to make it safe for file writing.
    
    Args:
        input_string (str): The original string to be escaped
    
    Returns:
        str: The escaped string with special characters replaced
    """
    # Define a mapping of special characters to their escaped representations
    escape_map = {
        '\\': '\\\\',  # Backslash
        '\n': '\\n',   # Newline
        '\r': '\\r',   # Carriage return
        '\t': '\\t',   # Tab
        '\b': '\\b',   # Backspace
        '\f': '\\f',   # Form feed
        '"': '\\"',    # Double quote
        "'": "\\'",    # Single quote
    }
    
    # Use translate method with a custom translation table
    escaped_string = ''.join(escape_map.get(char, char) for char in input_string)
    
    return escaped_string


# if eval https://arxiv.org/pdf/2311.07911

# Choose the device (CUDA or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name = "Qwen/Qwen2-0.5B-Instruct"  # Replace with your model's name
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
past_key_values = DynamicCache()

data = []
with open('IFEval/test.jsonl', 'r') as file:
    j = json.load(file)
    for ele in j:
        data.append(ele['prompt'])
    

p = pipeline("text-generation", model=model, tokenizer=tokenizer, past_key_values=past_key_values)
p(data)

# for prompt in data: 

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     # Generate text with the model
#     outputs = model.generate(
#         inputs['input_ids'],
#         max_length=1024,
#         num_return_sequences=1,
#         output_scores=True,  # To access past key values
#         return_dict_in_generate=True,  # To get more detailed outputs
#         past_key_values=past_key_values,
#         use_cache=True
#     )

#     generated_text = tokenizer.decode(outputs.sequences[0])

#     # print(generated_text)
#     with open("IFEval/input_response_data.jsonl", "a") as file:
#         t = escape_string_for_file(generated_text)
#         p = escape_string_for_file(prompt)
#         file.write('{"prompt": "'+p+'", "response": "'+t+'"}')