import torch
import numpy as np
import argparse
import time
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain

import engine
from engine import stopping
from engine import loader, generation
from helpers import utils

# model = engine.HFModel('bloom-7.1B', device_map='balanced')
# model_memory = torch.cuda.max_memory_allocated(0) / 1024**3
# print(model.device_map)
# print(f'From peak: {model_memory} GiB')
# mem = model.model.get_memory_footprint() / 1024**3
# print(f'From func estimation: {mem} GiB')
# mem_per_device = generation.get_memory_footprint(model.model)
# print(f'Memory per device: {mem_per_device}')
# print(f'The sum is equal: {(sum(mem_per_device.values()) / 1024**3) == mem}')

# a = torch.rand(10000, 10000, 5, device=0)

# def get_mem(a):
#     return a.nelement() * a.element_size() / 1024**3

# mem_a = get_mem(a)
# print(f'Memory of tensor allocated outside func: {mem_a:.2f} GiB')

# def test_alloc():
#     ref = torch.cuda.memory_allocated() / 1024**3
#     b = torch.rand(10000, 10000, 3, device=0)
#     mem_b = get_mem(b)
#     print(f'Memory of tensor allocated in func: {mem_b:.2f} GiB')
#     max_mem = torch.cuda.max_memory_allocated() / 1024**3
#     print(f'Max memory: {max_mem-ref:.2f} GiB')
#     print(f'Sum of both: {(mem_a + mem_b):.2f} GiB')

# test_alloc()
# print('After func:')
# max_mem = torch.cuda.max_memory_allocated() / 1024**3
# print(f'Max memory: {max_mem:.2f} GiB')

model = engine.HFModel('bloom-1.7B')
print(f'Max memory: {(torch.cuda.max_memory_allocated() / 1024**3):.2f} GiB')
print(f'Memory footprint: {model.memory_footprint:.2f} GiB')
print(f'Memory map: {model.memory_map}')






# large_text = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

# One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.

# Monkeys are highly social animals, living in complex social structures. They form troops or bands that can range in size from a few individuals to several hundred members, depending on the species. Within these groups, monkeys establish hierarchies through social interactions, with dominant individuals enjoying certain privileges and responsibilities. Social bonds are crucial for their survival, as they provide protection from predators and facilitate cooperative behaviors, such as foraging and caring for young ones.

# Another remarkable aspect of monkeys is their exceptional cognitive abilities. They exhibit problem-solving skills, tool usage, and the ability to learn from each other. For instance, certain species of monkeys have been observed using rocks to crack open nuts or sticks to fish for termites. They demonstrate an understanding of cause-and-effect relationships and exhibit a sense of self-awareness. Researchers have conducted numerous studies to explore the cognitive abilities of monkeys, revealing their impressive intellectual capacities.

# Monkeys are primarily herbivorous but have a diverse diet that includes fruits, leaves, seeds, and insects. Some species, like the howler monkey, are specialized folivores, consuming mainly leaves to meet their nutritional needs. Others, such as the capuchin monkey, are known for their omnivorous diet, which includes fruits, nuts, insects, and even small vertebrates. Their varied diet contributes to the dispersal of seeds, making monkeys important agents in forest regeneration and maintaining biodiversity.

# Monkeys play a crucial role in their ecosystems. As both predators and prey, they contribute to the balance of their habitats. They aid in seed dispersal, pollination, and nutrient cycling, thereby influencing the structure and dynamics of plant communities. Additionally, monkeys are indicators of ecosystem health, as their presence or absence can reflect the overall well-being of an ecosystem.

# Despite their significance, monkeys face numerous challenges and threats. Habitat loss due to deforestation, fragmentation, and human encroachment is one of the primary concerns. Additionally, illegal wildlife trade and hunting pose significant risks to monkey populations. Conservation efforts, including protected areas and education campaigns, are vital to ensure the survival of these remarkable creatures.

# In conclusion, monkeys are extraordinary creatures that captivate us with their diversity, social structures, cognitive abilities, and ecological importance. Their lives are intricately woven into the tapestry of their respective habitats, and understanding and protecting them is crucial for maintaining the balance of our planet's ecosystems. By appreciating and conserving these fascinating animals, we can continue to learn from them and be inspired by their remarkable qualities.
# """

# model_name = 'bloom-7.1B'
# batch_size = 10
# max_tokens = 512
# input_size = 500
# num_sequences = 40

# model = engine.HFModel(model_name)
# print(model.device_map)

# large_tokens = model.tokenizer.encode(large_text, return_tensors='pt')
# prompt = model.tokenizer.batch_decode(large_tokens[:, :input_size], skip_special_tokens=True)[0]

# t0 = time.time()
# foo = model(prompt, max_new_tokens=max_tokens, num_return_sequences=num_sequences, batch_size=batch_size)
# dt = time.time() - t0

# del model
# gc.collect()

# model = engine.HFModel(model_name, device_map='balanced')
# print(model.device_map)
# t1 = time.time()
# foo = model(prompt, max_new_tokens=max_tokens, num_return_sequences=num_sequences, batch_size=15)
# dt1 = time.time() - t1

# print(f'Single gpu {dt:.2f} s')
# print(f'3 gpus with larger batch {dt1:.2f} s')