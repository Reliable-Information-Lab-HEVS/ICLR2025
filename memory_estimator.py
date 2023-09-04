import os
import gc
import argparse
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np

import engine
from engine import loader
from helpers import utils


# Random long text about monkeys (thanks ChatGPT!!)
large_text = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.

Monkeys are highly social animals, living in complex social structures. They form troops or bands that can range in size from a few individuals to several hundred members, depending on the species. Within these groups, monkeys establish hierarchies through social interactions, with dominant individuals enjoying certain privileges and responsibilities. Social bonds are crucial for their survival, as they provide protection from predators and facilitate cooperative behaviors, such as foraging and caring for young ones.

Another remarkable aspect of monkeys is their exceptional cognitive abilities. They exhibit problem-solving skills, tool usage, and the ability to learn from each other. For instance, certain species of monkeys have been observed using rocks to crack open nuts or sticks to fish for termites. They demonstrate an understanding of cause-and-effect relationships and exhibit a sense of self-awareness. Researchers have conducted numerous studies to explore the cognitive abilities of monkeys, revealing their impressive intellectual capacities.

Monkeys are primarily herbivorous but have a diverse diet that includes fruits, leaves, seeds, and insects. Some species, like the howler monkey, are specialized folivores, consuming mainly leaves to meet their nutritional needs. Others, such as the capuchin monkey, are known for their omnivorous diet, which includes fruits, nuts, insects, and even small vertebrates. Their varied diet contributes to the dispersal of seeds, making monkeys important agents in forest regeneration and maintaining biodiversity.

Monkeys play a crucial role in their ecosystems. As both predators and prey, they contribute to the balance of their habitats. They aid in seed dispersal, pollination, and nutrient cycling, thereby influencing the structure and dynamics of plant communities. Additionally, monkeys are indicators of ecosystem health, as their presence or absence can reflect the overall well-being of an ecosystem.

Despite their significance, monkeys face numerous challenges and threats. Habitat loss due to deforestation, fragmentation, and human encroachment is one of the primary concerns. Additionally, illegal wildlife trade and hunting pose significant risks to monkey populations. Conservation efforts, including protected areas and education campaigns, are vital to ensure the survival of these remarkable creatures.

In conclusion, monkeys are extraordinary creatures that captivate us with their diversity, social structures, cognitive abilities, and ecological importance. Their lives are intricately woven into the tapestry of their respective habitats, and understanding and protecting them is crucial for maintaining the balance of our planet's ecosystems. By appreciating and conserving these fascinating animals, we can continue to learn from them and be inspired by their remarkable qualities.
"""


SMALL_MODELS = (
    'bloom-560M',
    'bloom-1.7B',
    'bloom-3B',
    'bloom-7.1B',
    'dialo-gpt-small',
    'dialo-gpt-medium',
    'dialo-gpt-large',
    'stable-lm-3B',
    'stable-lm-7B',
    'star-coder-base',
    'star-coder',
    'star-coder-plus',
    'star-chat-alpha',
    'star-chat-beta',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'gpt-j-6B',
    'gpt-neo-125M',
    'gpt-neo-1.3B',
    'gpt-neo-2.7B',
    'opt-125M',
    'opt-350M',
    'opt-1.3B',
    'opt-2.7B',
    'opt-6.7B',
    'opt-13B',
    'codegen-350M',
    'codegen-2B',
    'codegen-6B',
    'codegen-16B',
    'codegen2-1B',
    'codegen2-3.7B',
    'codegen2-7B',
    'codegen2-16B',
    'codegen25-7B',
    'codegen25-7B-instruct',
    'vicuna-7B',
    'vicuna-13B',
    'llama2-7B',
    'llama2-7B-chat',
    'llama2-13B',
    'llama2-13B-chat',
)


LARGE_MODELS = (
    'gpt-neoX-20B',
    'opt-30B',
    'opt-66B',
    'llama2-70B',
    'llama2-70B-chat',
    'bloom-176B',
)


input_sizes = [50*i for i in range(1, 11)]
max_tokens = [50*i for i in range(1, 11)]
max_tokens += [512]


@utils.duplicate_function_for_gpu_dispatch
def memory_estimation(model_name: str, quantization_8bits: bool, quantization_4bits: bool, N_repeat: int = 10):

    # Override quantization for bloom because it's too big
    if model_name == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
        quantization_8bits = True

    model = engine.HFModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)
    gpus = model.get_gpu_devices()
    large_tokens = model.tokenizer.encode(large_text, return_tensors='pt')

    # Initialize dicts
    model_memory_consumption = {}
    model_time_consumption = {}

    # Initialize filenames and return if files already exist
    dtype_name = model.dtype_category()
    filename_memory = os.path.join(utils.ROOT_FOLDER, 'memory_estimator', model_name, f'{dtype_name}.json')
    filename_time = os.path.join(utils.ROOT_FOLDER, 'time_estimator', model_name, f'{dtype_name}.json')
    if os.path.exists(filename_memory) and os.path.exists(filename_time):
        return

    for i, input_size in enumerate(input_sizes):

        prompt = model.tokenizer.batch_decode(large_tokens[:, :input_size], skip_special_tokens=True)[0]
        input_size_memory_consumption = {}
        input_size_time_consumption = {}

        for j, max_token in enumerate(max_tokens):

            results = []
            times = []
            for k in range(N_repeat):
                
                actual_peaks = {}
                for gpu_rank in gpus:
                    torch.cuda.reset_peak_memory_stats(gpu_rank)
                    actual_peaks[gpu_rank] = torch.cuda.max_memory_allocated(gpu_rank) / 1024**3

                t0 = time.time()
                foo = model(prompt, num_return_sequences=1, max_new_tokens=max_token, batch_size=1)
                dt = time.time() - t0
                
                memory_used = {}
                for gpu_rank in gpus:
                    memory_used[gpu_rank] = (torch.cuda.max_memory_allocated(gpu_rank) / 1024**3) - actual_peaks[gpu_rank]
                
                # Actual largest memory usage peak accross gpus
                max_peak = max(memory_used.values())
                results.append(max_peak)
                # Time consumption
                times.append(dt)

            input_size_memory_consumption[max_token] = np.mean(results)
            input_size_time_consumption[max_token] = np.mean(times)

        model_memory_consumption[input_size] = input_size_memory_consumption
        model_time_consumption[input_size] = input_size_time_consumption

    # Save results
    utils.save_json(model_memory_consumption, filename_memory)
    utils.save_json(model_time_consumption, filename_time)

    print(f'Done with {model_name}!')

    del model
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Memory estimator')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    parser.add_argument('--big_models', action='store_true',
                        help='If given, also run on large models that do not fit on a single gpu.')
    parser.add_argument('--big_models_only', action='store_true',
                        help='If given, only run on large models that do not fit on a single gpu.')
    
    args = parser.parse_args()
    int8 = args.int8
    int4 = args.int4
    big_models = args.big_models
    big_models_only = args.big_models_only

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')

    num_gpus = torch.cuda.device_count()

    if not big_models_only:

        if num_gpus > 1:
            int8_iter = (int8,)*len(SMALL_MODELS)
            int4_iter = (int4,)*len(SMALL_MODELS)
            with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context('spawn'),
                                        initializer=utils.set_cuda_visible_device_of_subprocess) as pool:
                _ = list(pool.map(memory_estimation, SMALL_MODELS, int8_iter, int4_iter, chunksize=1))
        else:
            for model in SMALL_MODELS:
                memory_estimation(model, int8, int4)

    if big_models or big_models_only:

        model_footprints = []
        for model in LARGE_MODELS:
            # Override quantization for bloom because it's too big
            if model == 'bloom-176B' and not (int8 or int4):
                int8 = True
            gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization_8bits=int8,
                                                                quantization_4bits=int4)
            model_footprints.append(gpu_needed)

        utils.dispatch_jobs(LARGE_MODELS, model_footprints, num_gpus, memory_estimation, func_args=(int8, int4))