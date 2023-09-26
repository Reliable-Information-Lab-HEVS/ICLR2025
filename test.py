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

model = engine.HFModel('star-chat-beta')

prompt = ' '.join(large_text.split(' ')[0:100])

actual_peak = torch.cuda.memory_allocated(0) / 1024**3
torch.cuda.reset_peak_memory_stats(0)
foo = model(prompt, batch_size=1, max_new_tokens=2, min_new_tokens=0)
mem = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak

print(f'Mem : {mem} GiB')

actual_peak2 = torch.cuda.memory_allocated(0) / 1024**3
torch.cuda.reset_peak_memory_stats(0)
foo = model(prompt, batch_size=1, max_new_tokens=100, min_new_tokens=0)
mem2 = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak2

print(f'Mem with large max new tokens : {mem2} GiB')