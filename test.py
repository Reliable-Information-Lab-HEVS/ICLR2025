import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time
import resource
import gc
import math

from typing import Optional, Tuple, Union
# from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
import torch
import warnings
import time
import numpy as np
import os
from tqdm import tqdm

import engine
from helpers import utils, datasets

large_text = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.

Monkeys are highly social animals, living in complex social structures. They form troops or bands that can range in size from a few individuals to several hundred members, depending on the species. Within these groups, monkeys establish hierarchies through social interactions, with dominant individuals enjoying certain privileges and responsibilities. Social bonds are crucial for their survival, as they provide protection from predators and facilitate cooperative behaviors, such as foraging and caring for young ones.

Another remarkable aspect of monkeys is their exceptional cognitive abilities. They exhibit problem-solving skills, tool usage, and the ability to learn from each other. For instance, certain species of monkeys have been observed using rocks to crack open nuts or sticks to fish for termites. They demonstrate an understanding of cause-and-effect relationships and exhibit a sense of self-awareness. Researchers have conducted numerous studies to explore the cognitive abilities of monkeys, revealing their impressive intellectual capacities.

Monkeys are primarily herbivorous but have a diverse diet that includes fruits, leaves, seeds, and insects. Some species, like the howler monkey, are specialized folivores, consuming mainly leaves to meet their nutritional needs. Others, such as the capuchin monkey, are known for their omnivorous diet, which includes fruits, nuts, insects, and even small vertebrates. Their varied diet contributes to the dispersal of seeds, making monkeys important agents in forest regeneration and maintaining biodiversity.

Monkeys play a crucial role in their ecosystems. As both predators and prey, they contribute to the balance of their habitats. They aid in seed dispersal, pollination, and nutrient cycling, thereby influencing the structure and dynamics of plant communities. Additionally, monkeys are indicators of ecosystem health, as their presence or absence can reflect the overall well-being of an ecosystem.

Despite their significance, monkeys face numerous challenges and threats. Habitat loss due to deforestation, fragmentation, and human encroachment is one of the primary concerns. Additionally, illegal wildlife trade and hunting pose significant risks to monkey populations. Conservation efforts, including protected areas and education campaigns, are vital to ensure the survival of these remarkable creatures.

In conclusion, monkeys are extraordinary creatures that captivate us with their diversity, social structures, cognitive abilities, and ecological importance. Their lives are intricately woven into the tapestry of their respective habitats, and understanding and protecting them is crucial for maintaining the balance of our planet's ecosystems. By appreciating and conserving these fascinating animals, we can continue to learn from them and be inspired by their remarkable qualities.
"""


# model_name = 'star-coder'
# model_name = 'llama2-7B'
# model_name = 'codegen-16B'
# model_name = 'bloom-1.7B'
# model_name = 'star_coder'


# model_name = 'llama2-7B'
# # model_name = 'opt-66B'
# input_size = 400
# max_new_tokens = 100

# t0 = time.time()
# model = engine.HFModel(model_name)
# dt0 = time.time() - t0
# print(f'Time to load the model {dt0:.2f} s')

# large_tokens = model.tokenizer.encode(large_text, return_tensors='pt')
# prompt = model.tokenizer.batch_decode(large_tokens[:, :input_size], skip_special_tokens=True)[0]

# memories = []
# for i in range(torch.cuda.device_count()):
#     torch.cuda.reset_peak_memory_stats(i)
#     memories.append(torch.cuda.max_memory_allocated(i) / 1024**3)

# t1 = time.time()
# foo = model(prompt, max_new_tokens=max_new_tokens, batch_size=1, num_return_sequences=1)
# dt1 = time.time() - t1
# print(f'Time for forward: {dt1:.2f} s')

# consumptions = []
# for i in range(torch.cuda.device_count()):
#     consumptions.append(torch.cuda.max_memory_allocated(i) / 1024**3 - memories[i])

# print(f'Max memory allocated of the gpus: {max(consumptions)} GiB')

# print(f'All memories used: {*consumptions,}')

# import bitsandbytes

import gc

model = engine.HFModel('star-chat-beta')
model.extra_eos_tokens = []

prompt = ' '.join(large_text.split(' ')[0:100])
use_cache = True


t0 = time.time()
torch.cuda.reset_peak_memory_stats(0)
actual_peak = torch.cuda.max_memory_allocated(0) / 1024**3
foo = model(prompt, batch_size=1, max_new_tokens=5, min_new_tokens=0, seed=12, post_process_output=False,
            use_cache=use_cache)
mem = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak
dt0 = time.time() - t0

print(f'Mem first time: {mem} GiB')
print(f'Time first time: {dt0:.2f} s')


t1 = time.time()
torch.cuda.reset_peak_memory_stats(0)
actual_peak2 = torch.cuda.max_memory_allocated(0) / 1024**3
print(actual_peak2)
foo2 = model(prompt, batch_size=1, max_new_tokens=5, min_new_tokens=5, seed=12, post_process_output=False,
             use_cache=use_cache)
print(torch.cuda.max_memory_allocated(0) / 1024**3)
mem2 = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak2
dt1 = time.time() - t1

print(f'Mem second time: {mem2} GiB')
print(f'Time second time: {dt1:.2f} s')


t2 = time.time()
torch.cuda.reset_peak_memory_stats(0)
actual_peak4 = torch.cuda.max_memory_allocated(0) / 1024**3
foo4 = model(prompt, batch_size=1, max_new_tokens=200, min_new_tokens=200, seed=12, post_process_output=False,
             use_cache=use_cache)
mem4 = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak4
dt2 = time.time() - t2

print(f'Mem with large max new tokens: {mem4} GiB')
print(f'Time with large max new tokens: {dt2:.2f} s')


new_prompt = prompt + """
What is the main point of this paragraph?<|end|>
<|system|>
<|end|>
<|user|>
Can you help me write a story? It should be about a teenage girl who is obsessed with anime, but has a very normal life.<|end|>
<|assistant|>
Sure! Here's a short story about a teenage girl named Ava who is obsessed with anime and has a pretty normal life:

Ava had always been a quiet and shy girl, but her love for anime had given her a sense of belonging. She spent hours watching her favorite shows and movies, imagining that she was one of the characters in the story."""


t3 = time.time()
torch.cuda.reset_peak_memory_stats(0)
actual_peak5 = torch.cuda.max_memory_allocated(0) / 1024**3
foo5 = model(new_prompt, batch_size=1, max_new_tokens=2, min_new_tokens=1, seed=12, post_process_output=False,
             use_cache=use_cache)
mem5 = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak5
dt3 = time.time() - t3

print(f'Mem with new prompt: {mem5} GiB')
print(f'Time with new prompt: {dt3:.2f} s')