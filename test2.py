import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time
import resource

# prompt = "# Write a python function to multiply 2 numbers"

# model = engine.HFModel('bloom-3B', dtype=torch.float32)

# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
# else:
#     print(f'Before generation cpu: {(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3):.2f} GB')


# t0 = time.time()
# out = model(prompt, max_new_tokens=512, num_return_sequences=200, batch_size=200,
#             stopping_patterns=stopping.CODE_STOP_PATTERNS)
# dt = time.time() - t0

# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f'After generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.2f} GB')
# else:
#     print(f'After generation cpu: {(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3):.2f} GB')
# print(f'Time for generation: {dt:.2f} s')



#
#
#

import engine
import torch

prompt = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.

Monkeys are highly social animals, living in complex social structures. They form troops or bands that can range in size from a few individuals to several hundred members, depending on the species. Within these groups, monkeys establish hierarchies through social interactions, with dominant individuals enjoying certain privileges and responsibilities. Social bonds are crucial for their survival, as they provide protection from predators and facilitate cooperative behaviors, such as foraging and caring for young ones.

Another remarkable aspect of monkeys is their exceptional cognitive abilities. They exhibit problem-solving skills, tool usage, and the ability to learn from each other. For instance, certain species of monkeys have been observed using rocks to crack open nuts or sticks to fish for termites. They demonstrate an understanding of cause-and-effect relationships and exhibit a sense of self-awareness. Researchers have conducted numerous studies to explore the cognitive abilities of monkeys, revealing their impressive intellectual capacities.
"""

# torch.cuda.reset_peak_memory_stats(device=0)

model = engine.HFModel('bloom-7.1B')

model_memory = torch.cuda.max_memory_allocated(0) / 1024**3

input_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda(0)
input_ids, _ = model.model._expand_inputs_for_generation(expand_size=5, input_ids=input_ids)

torch.cuda.reset_peak_memory_stats(device=0)

t0 = time.time()
past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values
dt = time.time() - t0

memory_with_grad = torch.cuda.max_memory_allocated(0) / 1024**3 - model_memory

del past_key_values
torch.cuda.reset_peak_memory_stats(device=0)

t1 = time.time()
with torch.no_grad():
    past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values
dt1 = time.time() - t1

memory_without_grad = torch.cuda.max_memory_allocated(0) / 1024**3 - model_memory

print(f'Memory with grad: {memory_with_grad:.5f} GiB')
print(f'Memory without grad: {memory_without_grad:.5f} GiB')

print(f'Time with grad: {dt:.2f} s')
print(f'Time without grad: {dt1:.2f} s')