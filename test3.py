from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
import torch
import warnings
import time
import numpy as np
import os
from tqdm import tqdm

import engine
from helpers import utils

large_text = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.

Monkeys are highly social animals, living in complex social structures. They form troops or bands that can range in size from a few individuals to several hundred members, depending on the species. Within these groups, monkeys establish hierarchies through social interactions, with dominant individuals enjoying certain privileges and responsibilities. Social bonds are crucial for their survival, as they provide protection from predators and facilitate cooperative behaviors, such as foraging and caring for young ones.

Another remarkable aspect of monkeys is their exceptional cognitive abilities. They exhibit problem-solving skills, tool usage, and the ability to learn from each other. For instance, certain species of monkeys have been observed using rocks to crack open nuts or sticks to fish for termites. They demonstrate an understanding of cause-and-effect relationships and exhibit a sense of self-awareness. Researchers have conducted numerous studies to explore the cognitive abilities of monkeys, revealing their impressive intellectual capacities.

Monkeys are primarily herbivorous but have a diverse diet that includes fruits, leaves, seeds, and insects. Some species, like the howler monkey, are specialized folivores, consuming mainly leaves to meet their nutritional needs. Others, such as the capuchin monkey, are known for their omnivorous diet, which includes fruits, nuts, insects, and even small vertebrates. Their varied diet contributes to the dispersal of seeds, making monkeys important agents in forest regeneration and maintaining biodiversity.

Monkeys play a crucial role in their ecosystems. As both predators and prey, they contribute to the balance of their habitats. They aid in seed dispersal, pollination, and nutrient cycling, thereby influencing the structure and dynamics of plant communities. Additionally, monkeys are indicators of ecosystem health, as their presence or absence can reflect the overall well-being of an ecosystem.

Despite their significance, monkeys face numerous challenges and threats. Habitat loss due to deforestation, fragmentation, and human encroachment is one of the primary concerns. Additionally, illegal wildlife trade and hunting pose significant risks to monkey populations. Conservation efforts, including protected areas and education campaigns, are vital to ensure the survival of these remarkable creatures.

In conclusion, monkeys are extraordinary creatures that captivate us with their diversity, social structures, cognitive abilities, and ecological importance. Their lives are intricately woven into the tapestry of their respective habitats, and understanding and protecting them is crucial for maintaining the balance of our planet's ecosystems. By appreciating and conserving these fascinating animals, we can continue to learn from them and be inspired by their remarkable qualities.
"""

model_name = 'bloom-7.1B'
input_size = 300
max_new_tokens = 512
num_sequences = 100
batch_size = 1

model = engine.HFModel(model_name)

large_tokens = model.tokenizer.encode(large_text, return_tensors='pt')
prompt = model.tokenizer.batch_decode(large_tokens[:, :input_size], skip_special_tokens=True)[0]

t0 = time.time()

with torch.no_grad():
    input_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda(0)
    past_key_values = model.model.transformer(input_ids[:, :-1], return_dict=True).past_key_values

foo = model(prompt, num_return_sequences=num_sequences, max_new_tokens=max_new_tokens, seed=1,
            batch_size=batch_size, past_key_values=past_key_values)

dt = time.time() - t0

print(f'Time with past: {dt:.2f} s')

t1 = time.time()
foo2 = model(prompt, num_return_sequences=num_sequences, max_new_tokens=max_new_tokens, seed=1,
             batch_size=batch_size)
dt1 = time.time() - t1

print(f'Time without past: {dt1:.2f} s')
print(f'Results are the same: {foo == foo2}')

