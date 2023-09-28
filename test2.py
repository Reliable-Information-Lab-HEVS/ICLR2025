import torch


import engine
from engine import generation
from engine import stopping
import time
import resource
import gc
import math
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer

import engine
from engine import loader
from helpers import utils, datasets

large_text = """Monkeys: Nature's Pranksters, Social Geniuses, and Ecological Wonders

Introduction

Monkeys, the enchanting inhabitants of our planet's diverse ecosystems, have long been subjects of fascination and study for scientists, wildlife enthusiasts, and the curious at heart. Their intriguing behaviors, remarkable adaptations, and complex social structures invite us into a world that is both astonishing and deeply entwined with our understanding of life on Earth. In this essay, we will embark on a journey through the realms of monkeys, exploring their evolutionary history, the rich tapestry of their classifications, the ecological significance they hold, the intricate dynamics of their societies, their modes of communication, and the critical importance of their conservation.

Evolutionary History

The tale of monkeys begins in the annals of evolutionary history, stretching back millions of years to a time when Earth was teeming with diverse life forms. Monkeys belong to the grand order of Primates, a lineage that emerged around 60 million years ago. This group shared common traits that set them apart from other mammals: agile grasping hands and feet, forward-facing eyes for depth perception, and increasingly enlarged brains, which laid the foundation for their remarkable cognitive abilities.

Around 35 million years ago, a pivotal moment occurred in the primate story - the division of the evolutionary tree. This division led to the emergence of two significant lineages: New World monkeys (Platyrrhini) and Old World monkeys (Catarrhini). The geographical separation between these groups set the stage for a diversity of adaptations and characteristics unique to each.

Classification and Diversity

Monkeys are a diverse family, representing over 260 species and encompassing a remarkable range of shapes, sizes, and behaviors. These fascinating creatures are categorized into two major families: Cebidae, encompassing New World monkeys, and Cercopithecidae, home to the Old World monkeys. The distinction between these two families extends beyond mere taxonomy; it shapes their ecology, behavior, and evolutionary pathways.

New World Monkeys (Family: Cebidae)

Within the New World monkeys, we encounter an astonishing array of life. These monkeys have mastered the art of adaptation to their specific environments in the Americas. Some swing effortlessly through the treetops, while others are known for their resonant vocalizations that echo through the forests. The capuchin monkeys exhibit remarkable intelligence, utilizing tools to extract food from hard-to-reach places. In contrast, tamarins and marmosets have evolved unique cooperative breeding systems that govern their social structures.

Old World Monkeys (Family: Cercopithecidae)

The Old World monkeys, found in Africa, Asia, and parts of Gibraltar, have charted their own evolutionary course. Here, we encounter the imposing baboons, known for their dog-like faces and robust physiques. Macaques, on the other hand, are exemplars of adaptability, thriving in diverse habitats, from snow-clad mountains to lush tropical forests. Mandrills, with their striking facial colors and imposing canines, claim the title of the world's largest monkeys, inhabiting the lush rainforests of Central Africa.

Ecological Significance

Beyond their aesthetic appeal and charismatic behaviors, monkeys play a pivotal role in the delicate balance of the ecosystems they call home. They are often referred to as "keystone species" because their presence or absence can trigger cascading effects throughout their habitats.

One of the most significant contributions of monkeys to their ecosystems is seed dispersal. As avid fruit consumers, they play a vital role in the regeneration of plant populations by spreading seeds as they move through forests. This not only helps maintain forest diversity but also contributes to the health and stability of these ecosystems.

Moreover, monkeys' selective feeding habits, such as leaf consumption by howler monkeys and colobus monkeys, shape the structure of forests by influencing the composition and density of vegetation. Their predation on insects, small mammals, and bird eggs helps regulate prey populations, ensuring the intricate balance of food webs within their habitats.

Monkeys, as indicator species, offer valuable insights into the health of their ecosystems. Their population trends can serve as early warnings of environmental disturbances, including habitat loss, deforestation, and climate change. As such, preserving monkey populations is not merely an act of conservation but also a means of safeguarding the well-being of entire ecosystems.

Social Structures and Behavior

Monkeys are renowned for their complex social structures and behaviors, which vary widely among species and are shaped by a multitude of ecological and environmental factors. These social dynamics are essential for their survival and reproduction, shedding light on the intricacies of primate evolution and behavior.

In the world of monkeys, social life revolves around the concept of the "troop." These groups, varying greatly in size and composition, are the core units of monkey society. Within these troops, a dominance hierarchy often emerges, dictating access to valuable resources such as food and mates. Dominant individuals assert their authority through various displays of dominance, including vocalizations and physical posturing.

The intricacies of monkey societies extend to mating strategies, parenting, and cooperation. Some species, like capuchin monkeys, display impressive problem-solving abilities and tool use, revealing the depths of their intelligence. Others, such as tamarins and marmosets, engage in cooperative breeding systems, with non-breeding individuals helping to care for the offspring of dominant breeding pairs.

Communication and Language

Monkeys communicate through a diverse array of vocalizations, gestures, and behaviors, revealing a rich tapestry of interaction within their social groups. Vocalizations range from the booming howls of howler monkeys to the more subtle chirps and calls of capuchins. Each species has developed its unique repertoire of sounds to convey information about food, danger, and social dynamics.

Beyond vocalizations, monkeys employ body language and facial expressions to convey emotions, assert dominance, or establish social bonds. The intricate dance of gestures and postures within monkey societies is a testament to their highly evolved communication systems.

Conservation Challenges

Despite their resilience and adaptability, monkeys face a host of challenges in the modern world. Habitat loss due to deforestation, logging, and urban expansion is a severe threat to their survival. The fragmentation of their habitats isolates populations, limiting genetic diversity and making them more vulnerable to diseases and other environmental pressures.

Illegal wildlife trade also poses a grave danger to monkeys. Poaching for the pet trade and traditional medicine markets, as well as the loss of individuals to captivity, disrupts natural populations and can have devastating consequences.

Moreover, monkeys often come into conflict with humans due to habitat encroachment. This can lead to retaliation killings, further endangering their populations. Climate change, with its unpredictable effects on ecosystems, adds another layer of uncertainty to their future.

Conclusion

Monkeys, these captivating creatures of the wild, offer us a glimpse into the wonders of the natural world. Their evolutionary journey, intricate social structures, communication methods, and ecological significance remind us of the interconnectedness of all life on Earth. As we strive to conserve these remarkable beings and their habitats, we not only safeguard the diversity of our planet but also enrich our understanding of the delicate tapestry of life that surrounds us. Monkeys are not merely subjects of fascination; they are ambassadors of the wild, beckoning us to protect and preserve the ecosystems they call home. In doing so, we honor our shared heritage and ensure a brighter future for all living creatures.
"""

# # model_name = 'llama2-7B'
# model_name = 'code-llama-7B-python'

# t0 = time.time()
# model = engine.HFModel(model_name)
# dt = time.time() - t0
# print(f'Time for loading: {dt:.2f} s')

# dataset = datasets.HumanEval()
# prompt = dataset[0]['prompt']

# t0 = time.time()
# out1 = model(prompt, max_new_tokens=512, do_sample=False, batch_size=1, num_return_sequences=1,
#              stopping_patterns=True)

# print(out1)
# print('\n\n')

# out2 = model(prompt.strip(), max_new_tokens=512, do_sample=False, batch_size=1, num_return_sequences=1,
#              stopping_patterns=True)
# dt = time.time() - t0

# print(out2)

# print('\n\n')

# print(f'Time for 2 inferences: {dt:.2f} s')





# model_name = 'bloom-560M'

# model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m', low_cpu_mem_usage=True)
# model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m')

# model = load_model(model_name)

# model = engine.HFModel(model_name)
# print(model.dtype_category())
# print(model.get_gpu_memory_footprint())

# from transformers import AutoModelForCausalLM


# foo = loader.estimate_model_gpu_footprint('bloom-560M')


# LARGE_MODELS = (
#     'gpt-neoX-20B',
#     'opt-30B',
#     'opt-66B',
#     'llama2-70B',
#     'llama2-70B-chat',
#     'bloom-176B',
# )

# model_footprints = []
# for model in LARGE_MODELS:
#     # Override quantization for bloom because it's too big
#     if model == 'bloom-176B':
#         gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization_8bits=True,
#                                                             quantization_4bits=False)
#     else:
#         gpu_needed, _ = loader.estimate_model_gpu_footprint(model)
#     model_footprints.append(gpu_needed)

# print(model_footprints)

def memory_usage(past_key_values):
    """Recursively compute the memory footprint of past key values (in bytes).
    """

    if isinstance(past_key_values, torch.Tensor):
        return past_key_values.nelement() * past_key_values.element_size()
    elif isinstance(past_key_values[0], torch.Tensor):
        return sum([x.nelement() * x.element_size() for x in past_key_values])
    else:
        return sum([memory_usage(x) for x in past_key_values])


model = AutoModelForCausalLM.from_pretrained('HuggingFaceH4/starchat-alpha', device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/starchat-alpha')

ids = tokenizer.encode(large_text)
prompt = tokenizer.decode(ids[0:1500], skip_special_tokens=True)


with torch.no_grad():
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    
    actual_peaks = {}
    for gpu_rank in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(gpu_rank)
        actual_peaks[gpu_rank] = torch.cuda.max_memory_allocated(gpu_rank) / 1024**3

    # Single forward pass, caching past key values
    output = model(prompt_ids, use_cache=True)

    memory_used = {}
    for gpu_rank in range(torch.cuda.device_count()):
        memory_used[gpu_rank] = (torch.cuda.max_memory_allocated(gpu_rank) / 1024**3) - actual_peaks[gpu_rank]
    
# Actual largest memory usage peak accross gpus
max_peak = max(memory_used.values())
# Compute size of past K-V
past_key_values_memory = memory_usage(output.past_key_values) / 1024**3

print(f'max peak: {max_peak}')
print(f'KV: {past_key_values_memory}')