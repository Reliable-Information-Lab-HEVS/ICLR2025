import os

# os.environ['TORCH_LOGS' ] = "+dynamo"
os.environ['TORCH_LOGS' ] = "recompiles"
# os.environ['TORCHDYNAMO_VERBOSE'] = "1"
# os.environ['TRITON_PTXAS_PATH'] = "/cluster/raid/home/vacy/miniforge3/envs/llm/bin/ptxas"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = "/cluster/raid/home/vacy/miniforge3/envs/llm/bin/ptxas"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import time
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

# model_name = 'google/gemma-2b'
model_name = 'meta-llama/Meta-Llama-3-8B'

# Copied from the gpt-fast repo
def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def decode_one_tokens(model, cur_token, cache_position, past_key_values=None):
    logits = model(cur_token, cache_position=cache_position, return_dict=False, use_cache = True,
                   past_key_values=past_key_values)[0]
    new_token = sample(logits,temperature=0.6, top_k=5)[0]
    return new_token
            

device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = model.to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = ["My favourite condiment is"]*10
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

batch_size, sequence_length = input_ids.shape
max_cache_length = 2048
max_new_tokens = 100

past_key_values = StaticCache(
    config=model.config,
    max_batch_size=batch_size,
    max_cache_len=max_cache_length,
    device=model.device,
    dtype=model.dtype,
)

generated_ids = torch.zeros((batch_size, max_new_tokens+sequence_length), dtype = torch.int, device=device)
generated_ids[:,:sequence_length] = input_ids


# Prefill
inputs = torch.tensor([[0]]*batch_size, device=device, dtype=int)
torch._dynamo.mark_static_address(inputs)

logits = model(input_ids, cache_position=torch.arange(sequence_length, device=device), past_key_values=past_key_values)[0]
inputs[:, :] = sample(logits, temperature=0.6, top_k=5)[0][:, :]
generated_ids[:,sequence_length] = inputs[:, 0]

cache_position = torch.tensor([sequence_length], device=device)
torch._dynamo.mark_static_address(cache_position)


torch.compiler.reset()

print('USUAL:')

with torch.no_grad():
    for i in range(10):
        # torch.cuda.synchronize()
        # t0 = time.time()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        inputs[:, :] = decode_one_tokens(model, inputs, cache_position, past_key_values=past_key_values)[:, :]
            # generated_ids.index_copy_(1, cache_position, input_id)
        end.record()
        torch.cuda.synchronize()
        dt0 = start.elapsed_time(end) / 1000
        # dt0 = time.time() - t0
        print(f'Time: {dt0:.2e} s')
        cache_position += 1


torch.compiler.reset()
decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead",fullgraph=True)

print('COMPILE:')

with torch.no_grad():
    for i in range(10):
        # torch.cuda.synchronize()
        # t0 = time.time()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        inputs[:, :] = decode_one_tokens(model, inputs, cache_position, past_key_values=past_key_values)[:, :]
            # generated_ids.index_copy_(1, cache_position, input_id)
        end.record()
        torch.cuda.synchronize()
        dt0 = start.elapsed_time(end) / 1000
        # dt0 = time.time() - t0
        print(f'Time: {dt0:.2e} s')
        cache_position += 1



torch.compiler.reset()
decode_one_tokens = torch.compile(decode_one_tokens, mode="max-autotune",fullgraph=True)

print('COMPILE AUTOTUNE:')

with torch.no_grad():
    for i in range(10):
        # torch.cuda.synchronize()
        # t0 = time.time()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        inputs[:, :] = decode_one_tokens(model, inputs, cache_position, past_key_values=past_key_values)[:, :]
            # generated_ids.index_copy_(1, cache_position, input_id)
        end.record()
        torch.cuda.synchronize()
        dt0 = start.elapsed_time(end) / 1000
        # dt0 = time.time() - t0
        print(f'Time: {dt0:.2e} s')
        cache_position += 1