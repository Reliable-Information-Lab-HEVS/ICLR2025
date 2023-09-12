import torch


import engine
from engine import generation
from engine import stopping
import time
import resource
import gc
import math

import engine
from helpers import utils, datasets

# model_name = 'llama2-7B'

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


model_name = 'bloom-560M'

# model = engine.HFModel(model_name)
# print(model.get_gpu_memory_footprint())

from transformers import AutoModelForCausalLM
from engine import loader


def estimate_model_gpu_footprint(model_name, quantization_8bits: bool = False, quantization_4bits: bool = False,
                                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8,
                                 max_fraction_gpus: float = 0.8) -> tuple[int, dict]:
    """Estimate the minimum number of gpus needed to perform inference with a model, given the maximum gpu memory
    proportion `max_fraction_gpu_0` and `max_fraction_gpus` that we allow for the model. This relies on
    simple heuristics. This also computes the corresponding `memory_map` to use when creating a `device_map`.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 8 bits mode, by default False.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8.

    Returns
    -------
    tuple[int, dict]
        Tuple containing the minimum number of gpus needed, the `memory_map`, i.e. a dictionary mapping each gpu
        needed to the maximum size reserved by the model for this gpu.
    """

    if max_fraction_gpu_0 < 0 or max_fraction_gpus < 0:
        raise ValueError('The maximum fraction of gpu memory to use cannot be negative.')
    
    if max_fraction_gpu_0 > 0.85 or max_fraction_gpus > 0.85:
        raise ValueError(('The maximum fraction of gpu memory to use cannot be larger than 0.85 because some '
                         'memory need to stay free for the forward pass and other computations.'))
    
    # Silently use 4bits when both are True
    if quantization_4bits and quantization_8bits:
        quantization_8bits = False

    # If not provided take the default one
    if dtype is None:
        dtype = loader.get_model_dtype(model_name)

    if quantization_4bits:
        size_multiplier = 1/2
    elif quantization_8bits:
        size_multiplier = 1
    elif (dtype == torch.float16) or (dtype == torch.bfloat16):
        size_multiplier = 2
    else:
        size_multiplier = 4

    # Estimate of the memory size of the model
    rough_model_size_estimate = loader.ALL_MODELS_PARAMS[model_name] * size_multiplier
    
    # We assume that we always have identical gpus when using multiple gpus
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Say we only have access to a portion of that memory for our model
    gpu_0_available_memory = max_fraction_gpu_0 * gpu_memory
    gpus_available_memory = max_fraction_gpus * gpu_memory

    # Heuristic: if the remainder is smaller than 2% of gpu_memory, do not add a gpu 
    if rough_model_size_estimate <= gpu_0_available_memory + 0.02 * gpu_memory:
        return 1, None
    
    else:
        max_memory_map = {0: f'{math.ceil(gpu_0_available_memory)}GiB'}
        to_fit_on_other_gpus = rough_model_size_estimate - gpu_0_available_memory
        additional_gpus_needed = int(to_fit_on_other_gpus // gpus_available_memory)

        # Heuristic: if the remainder is smaller than 2% of each gpu_memory, do not add a gpu and distill
        # the small excess between existing gpus
        if to_fit_on_other_gpus % gpus_available_memory >= (0.02 * gpu_memory) * additional_gpus_needed:
            additional_gpus_needed += 1
            available_gpu_size = math.ceil(gpus_available_memory)
        else:
            # Add the 2% to the gpus size requirements
            available_gpu_size = math.ceil((max_fraction_gpus + 0.02) * gpu_memory)

        gpu_needed = 1 + additional_gpus_needed
        for i in range(1, gpu_needed):
            max_memory_map[i] = f'{available_gpu_size}GiB'

        return gpu_needed, max_memory_map



# foo = estimate_model_gpu_footprint('bloom-560M')