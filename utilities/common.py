import json
import torch
import time

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

def read_if(path):
    data = []
    with open(path, 'r') as file:
        j = json.load(file)
        for ele in j:
            data.append(ele['prompt'])
    return data

def write_if(path, prompt, text):
    with open(path, "a") as file:
        line = json.dumps([{"prompt": prompt, "response": text}], separators=(',', ':'))
        file.write(line)
        

def get_gpu_memory():
    allocated = torch.cuda.memory_allocated(0)  # Memory currently allocated by tensors
    # reserved = torch.cuda.memory_reserved(0)    # Total memory reserved by the memory allocator
    # return allocated, reserved
    return allocated


def write_meta(path, times, mem):
    timestamp = str(int(time.time()))
    with open(path+"_"+timestamp, "w") as file:
        file.write(
            '{"runtime": ['+", ".join(str(x) for x in times)+']},'+
            '{"memory_usage": ['+", ".join(str(x) for x in mem)+']},'+
            '{"avg_runtime": '+ str(sum(times) / len(times)) +']},'+
            '{"avg_memory_usage": '+ str(sum(mem) / len(mem)) +']}'
        )