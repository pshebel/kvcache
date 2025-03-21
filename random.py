import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import time

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_6

class RandomOffloadedCache(DynamicCache):
    def __init__(self) -> None:
        if not (
            torch.cuda.is_available()
            or (is_torch_greater_or_equal_than_2_6("2.7", accept_dev=True) and torch.xpu.is_available())
        ):
            raise RuntimeError(
                "OffloadedCache can only be used with a GPU"
                + (" or XPU" if is_torch_greater_or_equal_than_2_6("2.7", accept_dev=True) else "")
            )

        super().__init__()
        self.original_device = []
        self.prefetch_stream = None
        self.prefetch_stream = (
            torch.Stream() if is_torch_greater_or_equal_than_2_6("2.7", accept_dev=True) else torch.cuda.Stream()
        )
        self.beam_idx = None  # used to delay beam search operations

    def prefetch_layer(self, layer_idx: int):
        "Starts prefetching the next layer cache"
        if layer_idx < len(self):
            with (
                self.prefetch_stream
                if is_torch_greater_or_equal_than_2_6("2.7", accept_dev=True)
                else torch.cuda.stream(self.prefetch_stream)
            ):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        if len(self) > 2:
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            prev_layer_idx = (layer_idx - 1) % len(self)
            self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
            self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            # Evict the previous layer if necessary
            if is_torch_greater_or_equal_than_2_6("2.7", accept_dev=True):
                torch.accelerator.current_stream().synchronize()
            else:
                torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            # Load current layer cache to its original device if not already there
            original_device = self.original_device[layer_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            # Now deal with beam search ops which were delayed
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(original_device)
                key_tensor = key_tensor.index_select(0, self.beam_idx)
                value_tensor = value_tensor.index_select(0, self.beam_idx)
            # Prefetch the next layer
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Saves the beam indices and reorders the cache when the tensor is back to its device."""
        # We delay this operation until the tensors are back to their original
        # device because performing torch.index_select on the CPU is very slow
        del self.beam_idx
        self.beam_idx = beam_idx.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            self.evict_previous_layer(layer_idx)
        else:
            key_tensor, value_tensor = self[layer_idx]
            self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # According to https://docs.python.org/3/library/exceptions.html#NotImplementedError
    # if a method is not supposed to be supported in a subclass we should set it to None
    from_legacy_cache = None

    to_legacy_cache = None

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
