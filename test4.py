from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
import torch
import warnings
import time

import engine

model_name = 'bloom-7.1B'
max_tokens = 50
batch_size = 20
num_sequences = 10 * batch_size


def expand_past_keys(past_key_values, batch_size):

    new = []

    with torch.no_grad():
        for i in range(len(past_key_values)):
            new_ = []
            for j in range(len(past_key_values[i])):
                new_.append(past_key_values[i][j].repeat(batch_size, 1, 1))
            new.append(tuple(new_))

    return tuple(new)


prompt = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.

Monkeys are highly social animals, living in complex social structures. They form troops or bands that can range in size from a few individuals to several hundred members, depending on the species. Within these groups, monkeys establish hierarchies through social interactions, with dominant individuals enjoying certain privileges and responsibilities. Social bonds are crucial for their survival, as they provide protection from predators and facilitate cooperative behaviors, such as foraging and caring for young ones.

Another remarkable aspect of monkeys is their exceptional cognitive abilities. They exhibit problem-solving skills, tool usage, and the ability to learn from each other. For instance, certain species of monkeys have been observed using rocks to crack open nuts or sticks to fish for termites. They demonstrate an understanding of cause-and-effect relationships and exhibit a sense of self-awareness. Researchers have conducted numerous studies to explore the cognitive abilities of monkeys, revealing their impressive intellectual capacities.
"""

model = engine.HFModel(model_name)

for i in range(torch.cuda.device_count()):
    print(f'Memory of the model gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GiB')
    torch.cuda.reset_peak_memory_stats(device=i)

t0 = time.time()
with torch.no_grad():
    input_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda(0)
    past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values
    past_key_values = expand_past_keys(past_key_values, batch_size)

for i in range(torch.cuda.device_count()):
    print(f'Past keys computation gpu {i}: {(torch.cuda.max_memory_allocated(0) / 1024**3):.5f} GB')
    torch.cuda.reset_peak_memory_stats(device=i)

out1 = model(prompt, num_return_sequences=num_sequences, max_new_tokens=max_tokens, seed=1,
             batch_size=batch_size, past_key_values=past_key_values)
dt = time.time() - t0

for i in range(torch.cuda.device_count()):
    print(f'Generation with past keys gpu {i}: {(torch.cuda.max_memory_allocated(0) / 1024**3):.5f} GB')
    torch.cuda.reset_peak_memory_stats(device=i)

del model, past_key_values

model = engine.HFModel(model_name)

for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(device=i)

t1 = time.time()
out2 = model(prompt, num_return_sequences=num_sequences, max_new_tokens=max_tokens, seed=1,
             batch_size=batch_size)
dt1 = time.time() - t1

for i in range(torch.cuda.device_count()):
    print(f'Generation withOUT past keys gpu {i}: {(torch.cuda.max_memory_allocated(0) / 1024**3):.5f} GB')
    torch.cuda.reset_peak_memory_stats(device=i)

print(f'Outputs are the same: {out1 == out2}')
print(f'Time with past keys: {dt:.2f} s')
print(f'Time without past keys: {dt1:.2f} s')