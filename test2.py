import torch


import engine
from engine import generation
from engine import stopping
import time
import resource
import gc
import math
import warnings

from transformers import AutoModelForCausalLM

import engine
from engine import loader
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

def load_model(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
               dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
               device_map: dict | None = None, gpu_rank: int = 0):
    """Load one of the supported pretrained model.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_4bits`.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_8bits`.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    device_map : dict | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model dispatch to GPU(s) is made according to `max_fraction_gpu_0` and `max_fraction_gpus`
        in such a way to use the smallest number of gpus that respect these two values.
    gpu_rank : int, optional
        The gpu rank on which to put the model if it can fit on a single gpu. This is ignored if `device_map`
        is provided. By default 0.

    Returns
    -------
        The model.
    """

    if model_name not in loader.ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*loader.ALLOWED_MODELS,}.')
    
    # Set the dtype if not provided
    if dtype is None:
        dtype = loader.ALL_MODELS_DTYPES[model_name]

    if dtype not in loader.ALLOWED_DTYPES:
        raise ValueError(f'The dtype must be one of {*ALLOWED_DTYPES,}.')
    
    if quantization_8bits and quantization_4bits:
        raise ValueError(('You cannot load a model with both `quantization_8bits` and `quantization_4bits`. '
                         'Please choose one'))
    
    if isinstance(device_map, str):
        raise ValueError(("You cannot provide the `device_map` as a string. Please use the `max_fraction_gpu_0` "
                         "and `max_fraction_gpus` arguments to mimic `device_map='balanced'` or "
                         "`device_map='balanced_low_0'` in a more coherent way."))
    
    # torch.float16 is not supported on cpu
    if not torch.cuda.is_available() and dtype != torch.float32:
        dtype = torch.float32
    
    # Override quantization if we don't have access to GPUs
    if not torch.cuda.is_available() and (quantization_8bits or quantization_4bits):
        quantization_4bits = False
        quantization_8bits = False
        warnings.warn('There are no GPUs available. The model will NOT be quantized.', RuntimeWarning)

    # Flag to know if the model is quantized
    quantization = quantization_8bits or quantization_4bits

    # Override dtype if we quantize the model as only float16 is acceptable for quantization
    dtype = torch.float16 if quantization else dtype

    # Add possible additional kwargs
    if model_name in loader.ALL_MODELS_ADDITIONAL_MODEL_KWARGS.keys():
        additional_kwargs = loader.ALL_MODELS_ADDITIONAL_MODEL_KWARGS[model_name]
    else:
        additional_kwargs = {}


    # Flag that will be set to True if we don't even need a device_map and can just put the model on one gpu
    only_move_to_one_gpu = False
    
    # Automatically find the best device_map depending on the model size and gpu size.
    # Try to minimize the number of gpus to use because using more will slow inference (but allow larger
    # batch size -> hard trade-off to find). Indeed, the parallelism of device_map is naive and gpus are only
    # used sequentially
    if (device_map is None) and torch.cuda.is_available():
    
        min_gpu_needed, max_memory_map = loader.estimate_model_gpu_footprint(model_name, quantization_8bits=quantization_8bits,
                                                                      quantization_4bits=quantization_4bits, dtype=dtype,
                                                                      max_fraction_gpu_0=max_fraction_gpu_0,
                                                                      max_fraction_gpus=max_fraction_gpus)
        gpu_number = torch.cuda.device_count()

        if min_gpu_needed > gpu_number:
            raise RuntimeError(("The model seems too big for the gpu resources you have. To offload to the cpu as well, "
                               "explicitly pass a `device_map`, e.g device_map='balanced'."))
        
        # In this case we don't need a device_map, we just move the model to the 1st gpu. Most models are 
        # relatively small and should fall on this category.
        if min_gpu_needed == 1:
            only_move_to_one_gpu = True
        # In this case, we need more than 1 gpu so we create a device_map between different gpus. However, 
        # we minimize the number of gpus used with the max_memory arg instead of naively using device_map='balanced'
        # between all gpus, because the parallelism is not optimized and thus using a lot of gpus is not efficient
        # if not needed
        else:
            additional_kwargs['max_memory'] = max_memory_map
            # Providing 'balanced' dispatch correctly with respect to the max_memory_map we provide
            device_map = 'balanced'

    
    # # Load model
    # model = AutoModelForCausalLM.from_pretrained(loader.ALL_MODELS_MAPPING[model_name], device_map=device_map,
    #                                             torch_dtype=dtype, load_in_8bit=quantization_8bits,
    #                                             load_in_4bit=quantization_4bits, low_cpu_mem_usage=True,
    #                                             **additional_kwargs)
    
    # # If the flag is active we directly put our model on one gpu without using any device_map (this is 
    # # more efficient). But if the model is quantized, this is already done automatically because quantization
    # # happen only on gpu
    # if only_move_to_one_gpu and not quantization:
    #     # This operation is in-place for nn.Module
    #     model.cuda(gpu_rank)

    # # For some reason bettertransformer is supported for codegen2 models but makes them crash during the forward
    # if not ('codegen2-' in model_name):
    #     # Convert to better transformer to use Pytorch optimizations if supported by the model
    #     try:
    #         model = model.to_bettertransformer()
    #     except:
    #         pass
        
    # model.eval()

    # return model





model_name = 'bloom-560M'

model = load_model(model_name)

# model = engine.HFModel(model_name)
# print(model.dtype_category())
# print(model.get_gpu_memory_footprint())

# from transformers import AutoModelForCausalLM


# foo = loader.estimate_model_gpu_footprint('bloom-560M')