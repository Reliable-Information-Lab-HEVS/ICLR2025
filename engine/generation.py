import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import StoppingCriteriaList
import numpy as np
import math
import gc
import psutil
import os

from engine import loader
from engine import stopping
from helpers import utils


class HFModel(object):
    """Class encapsulating a HuggingFace model and its tokenizer to generate text. 
    """

    def __init__(self, model_name: str, quantization: bool = False, device_map: str | None = None,
                 gpu_rank: int = 0, dtype: torch.dtype | None = None):
        
        # Save the current allocated memory on each gpu to estimate model size after loading
        if torch.cuda.is_available():
            reference_memory = {}
            for i in range(torch.cuda.device_count()):
                reference_memory[i] = torch.cuda.memory_allocated(i)

        self.model, self.tokenizer = loader.load_model_and_tokenizer(model_name, quantization=quantization,
                                                                     device_map=device_map, gpu_rank=gpu_rank,
                                                                     dtype=dtype)
        
        # Compute the memory footprint of the model on each gpu
        self.gpu_memory_map = {}
        if hasattr(self.model, 'hf_device_map'):
            devices = set(self.model.hf_device_map.values())
            devices.discard('cpu')
            for device in devices:
                self.gpu_memory_map[device] = (torch.cuda.memory_allocated(device) - reference_memory[device]) / 1024**3
        # In this case the model is on a single device thus we directly estimate its size with the number of parameters
        else:
            self.gpu_memory_map[gpu_rank] = self.model.get_memory_footprint() / 1024**3

        # Maximum memory taken by the model on a single gpu, or on the cpu
        self.max_memory_footprint = max(self.gpu_memory_map.values())

        self.model_name = model_name
        self.quantization = quantization
        # The model is on multiple devices
        if hasattr(self.model, 'hf_device_map'):
            self.device_map = self.model.hf_device_map
            self.input_device = min(devices) if len(devices) > 0 else 'cpu'
        # The model is on a single device
        else:
            device = next(self.model.parameters()).get_device()
            self.device_map = 'cpu' if device == -1 else f'cuda:{device}'
            self.input_device = 'cpu' if device == -1 else device
        self.dtype = self.model.dtype

        # In this case the gpu memory map is erroneous
        if self.device_map == 'cpu':
            self.gpu_memory_map = {}

    
    def __repr__(self) -> str:
        return f'HFModel({self.model_name}, {self.quantization}, {self.device_map})'
    
    def __str__(self) -> str:
        if self.quantization:
            return f'{self.model_name} model, quantized 8 bits version'
        else:
            return f'{self.model_name} model, original (not quantized) version'
        

    def __call__(self, prompt: str, max_new_tokens: int = 60, min_new_tokens: int = 5, do_sample: bool = True,
                 top_k: int = 40, top_p: float = 0.90, temperature: float = 0.9, num_return_sequences: int = 1,
                 batch_size: int | None = None, seed: int | None = None, truncate_prompt_from_output: bool = False,
                 stopping_patterns: list[str] | bool | None = None, **kwargs) -> str | list[str]:
        """Generate text according to `prompt` using the parameters specified.

        Parameters
        ----------
        prompt : str
            The prompt to the model.
        max_new_tokens : int, optional
            How many new tokens to generate, by default 60.
        min_new_tokens : int, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
            force the model to generate an output, instead of immediately generating EOS, by default 5.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int, optional
            How many tokens with max probability to consider for randomness, by default 50.
        top_p : float, optional
            The probability density covering the new tokens to consider for randomness, by default 0.92.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.9.
        num_return_sequences : int, optional
            How many sequences to generate according to the `prompt`, by default 1.
        batch_size : int | None, optional
            Max batch size for the model forward pass, in case `num_return_sequences` is large, by default None.
            If None, will try to determine the largest possible batch size that does not result in memory error.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible.
        truncate_prompt_from_output : bool, optional
            Whether to remove the prompt from the model answer or not, by default False.
        stopping_patterns: list[str] | bool | None
            List of words/patterns to stop the generation. Pass `True` to use the default `CODE_STOP_PATTERNS` patterns.
            If `None`, no early stopping is performed, by default None.
        input_device : int | str, optional
            The device on which to put the inputs, by default 0.

        Returns
        -------
        str | list[str]
            Str containing the generated sequence, or list[str] if `num_return_sequences` > 1.
        """
        
        return self.generate_text(prompt, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, do_sample=do_sample,
                                  top_k=top_k, top_p=top_p, temperature=temperature, num_return_sequences=num_return_sequences,
                                  batch_size=batch_size, seed=seed, truncate_prompt_from_output=truncate_prompt_from_output,
                                  stopping_patterns=stopping_patterns, input_device=self.input_device, **kwargs)
    

    def generate_text(self, prompt: str, max_new_tokens: int = 60, min_new_tokens: int = 5, do_sample: bool = True,
                      top_k: int = 40, top_p: float = 0.90, temperature: float = 0.9, num_return_sequences: int = 1,
                      batch_size: int | None = None, seed: int | None = None, truncate_prompt_from_output: bool = False,
                      stopping_patterns: list[str] | bool | None = None, input_device: int | str = 0,
                      **kwargs) -> str | list[str]:
        """Generate text according to `prompt` using the parameters specified.

        Parameters
        ----------
        prompt : str
            The prompt to the model.
        max_new_tokens : int, optional
            How many new tokens to generate, by default 60.
        min_new_tokens : int, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
            force the model to generate an output, instead of immediately generating EOS, by default 5.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int, optional
            How many tokens with max probability to consider for randomness, by default 50.
        top_p : float, optional
            The probability density covering the new tokens to consider for randomness, by default 0.92.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.9.
        num_return_sequences : int, optional
            How many sequences to generate according to the `prompt`, by default 1.
        batch_size : int | None, optional
            Max batch size for the model forward pass, in case `num_return_sequences` is large, by default None.
            If None, will try to determine the largest possible batch size that does not result in memory error.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible.
        truncate_prompt_from_output : bool, optional
            Whether to remove the prompt from the model answer or not, by default False.
        stopping_patterns: list[str] | bool | None
            List of words/patterns to stop the generation. Pass `True` to use the default `CODE_STOP_PATTERNS` patterns.
            If `None`, no early stopping is performed, by default None.
        input_device : int | str, optional
            The device on which to put the inputs, by default 0.

        Returns
        -------
        str | list[str]
            Str containing the generated sequence, or list[str] if `num_return_sequences` > 1.
        """
    
        if seed is not None:
            utils.set_all_seeds(seed)

        input = self.tokenizer.encode(prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=input_device)

        # Possible early stopping
        if type(stopping_patterns) is list:
            stopping_criteria = stopping.TextPatternStopping(input_length, self.tokenizer, stopping_patterns)
            stopping_criteria = StoppingCriteriaList([stopping_criteria])
            post_process = True
            post_process_list = stopping_patterns
        elif type(stopping_patterns) is bool and stopping_patterns:
            stopping_criteria = stopping.TextPatternStopping(input_length, self.tokenizer)
            stopping_criteria = StoppingCriteriaList([stopping_criteria])
            post_process = True
            post_process_list = stopping.CODE_STOP_PATTERNS
        else:
            stopping_criteria = None
            post_process = False

        if batch_size is None:
            batch_size = self.infer_best_batch_size(input_length, max_new_tokens, num_return_sequences)

        print(f'Inside the function, the batch size is: {batch_size}')

        # If we require more sequences than the allowed batch size, we need to split the generation into
        # multiple passes
        if num_return_sequences > batch_size:
            batch_sizes = [batch_size]*(num_return_sequences // batch_size)
            remainder = num_return_sequences % batch_size
            if remainder != 0:
                batch_sizes += [remainder]
            assert sum(batch_sizes) == num_return_sequences
        else:
            batch_sizes = [num_return_sequences]

        past = False
        # Past key values generation
        if 'past_key_values' in kwargs.keys():
            unique_batch_sizes = np.unique(batch_sizes) # already sorted
            past_key_values = kwargs.pop('past_key_values')
            # Maximum 2 elements in the following dict: one for common batch size and one for remainder
            past_keys = {size: expand_past_keys(past_key_values, size) for size in unique_batch_sizes}
            past = True

        # Suppress pad_token_id warning
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        generated_text = []

        for size in batch_sizes:

            if past:
                outputs = self.model.generate(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                              do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                              num_return_sequences=size, stopping_criteria=stopping_criteria,
                                              pad_token_id=pad_token_id, past_key_values=past_keys[size], **kwargs)
                
            else:
                outputs = self.model.generate(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                              do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                              num_return_sequences=size, stopping_criteria=stopping_criteria,
                                              pad_token_id=pad_token_id, **kwargs)
                
            # In this case truncate the prompt from the output
            if truncate_prompt_from_output:
                outputs = outputs[:, input_length:]

            generated_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True) 
            if post_process:
                generated_batch = stopping.post_process_sequences(generated_batch, prompt, post_process_list)
            
            generated_text += generated_batch

        # In this case return a str instead of list[str]
        if num_return_sequences == 1:
            generated_text = generated_text[0]

        return generated_text
    
    

    def infer_best_batch_size(self, input_size: int, max_new_tokens: int, num_return_sequences: int):
    
        if not torch.cuda.is_available():
            memory = psutil.virtual_memory().total / 1024**3
        else:
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        available_memory = memory*0.9 - self.max_memory_footprint

        try:
            batch_footprint = utils.load_json(os.path.join(utils.ROOT_FOLDER, 'memory_estimator', f'{self.model_name}.json'))
            # Convert keys to int
            batch_footprint = {int(k): {int(k1):v1 for k1,v1 in batch_footprint[k].items()} for k in batch_footprint.keys()}
        except FileNotFoundError:
            pass

        # Find the reference input size immediately larger than the current input size
        input_sizes = np.sort(list(batch_footprint.keys()))
        indices = np.nonzero(input_sizes >= input_size)
        if len(indices) == 0:
            pass
        else:
            ref_input_size = input_sizes[indices[0][0]]

        # Find the reference max new tokens immediately larger than the current max new tokens
        max_tokens = np.sort(list(batch_footprint[ref_input_size].keys()))
        indices = np.nonzero(max_tokens >= max_new_tokens)
        if len(indices) == 0:
            pass
        else:
            ref_max_tokens = max_tokens[indices[0][0]]

        ref_batch_footprint = batch_footprint[ref_input_size][ref_max_tokens]

        if ref_batch_footprint < 0:
            return num_return_sequences

        return int(available_memory // ref_batch_footprint)


    def generate_batch_recursive_oom_safe(self, input: torch.Tensor, max_new_tokens: int, min_new_tokens: int, do_sample: bool,
                                top_k: int, top_p: float, temperature: float, batch_size: int,
                                stopping_criteria: StoppingCriteriaList | None, pad_token_id: int,
                                past_key_values: tuple | None, **kwargs):
        """Generate text by recursively recovering from possible memory errors (OOMs) by lowering the batch size.
        Note that it is not possible to retry immediately in the except block because the exception retains the
        tensors already allocated in the try block which causes an immediate new OOM
        (see https://github.com/pytorch/pytorch/issues/18853)
        """
        retry = False

        # Try generating result
        try:
            if past_key_values is not None:
                out = self.model.generate(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                          do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                          num_return_sequences=batch_size, stopping_criteria=stopping_criteria,
                                          pad_token_id=pad_token_id, past_key_values=past_key_values, **kwargs)
            else:
                out = self.model.generate(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                          do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                          num_return_sequences=batch_size, stopping_criteria=stopping_criteria,
                                          pad_token_id=pad_token_id, **kwargs)
        
        except RuntimeError as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                retry = True
            else:
                raise e

        if retry:
            if batch_size == 1:
                raise RuntimeError('Even a batch size of 1 causes an OOM. Cannot generate with current config.')
            batch_size = max(1, math.floor(batch_size*0.8))
            gc.collect()
            torch.cuda.empty_cache()
            return self.generate_batch_recursive_oom_safe(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                                          do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                                          batch_size=batch_size, stopping_criteria=stopping_criteria,
                                                          pad_token_id=pad_token_id, past_key_values=past_key_values,
                                                          **kwargs)
        else:
            return out, batch_size
   

def expand_past_keys(past_key_values, batch_size):

    if batch_size <=1:
        return past_key_values
    
    new = []
    with torch.no_grad():
        for i in range(len(past_key_values)):
            new_ = []
            for j in range(len(past_key_values[i])):
                new_.append(past_key_values[i][j].repeat(batch_size, 1, 1))
            new.append(tuple(new_))

    return tuple(new)




def load_and_generate_text(model_name: str, prompt: str, quantization: bool = False, device_map: str | None = None,
                           dtype: torch.dtype | None = None, max_new_tokens: int = 60, min_new_tokens: int = 5,
                           do_sample: bool = True, top_k: int = 100, top_p: float = 0.92, temperature: float = 0.9,
                           batch_size: int | None = None, num_return_sequences: int = 1, seed: int | None = None,
                           truncate_prompt_from_output: bool = False,
                           stopping_patterns: list[str] | bool | None = None, input_device: int | str = 0,
                           **kwargs) -> str | list[str]:
    """Load a model and its tokenizer and generate text according to `prompt`.

    Parameters
    ----------
    model_name : str
        The model used to generate the text.
    prompt : str
        The prompt to the model.
    quantization : bool, optional
        Whether to load the model in 8 bits mode to save memory, by default False.
    device_map : str | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model will be put on a single GPU if relatively small, else split using 'auto'.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.
    min_new_tokens : int, optional
        The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
        force the model to generate an output, instead of immediately generating EOS, by default 5.
    do_sample : bool, optional
        Whether to introduce randomness in the generation, by default True.
    top_k : int, optional
        How many tokens with max probability to consider for randomness, by default 100.
    top_p : float, optional
        The probability density covering the new tokens to consider for randomness, by default 0.92.
    temperature : float, optional
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness), by default 0.9.
    num_return_sequences : int, optional
        How many sequences to generate according to the `prompt`, by default 1.
    batch_size : int | None, optional
        Max batch size for the model forward pass, in case `num_return_sequences` is large, by default None.
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.
    truncate_prompt_from_output : bool, optional
        Whether to remove the prompt from the model answer or not, by default False.
    stopping_patterns: list[str] | bool | None
        List of words/patterns to stop the generation. Pass `True` to use the default `CODE_STOP_PATTERNS` patterns.
        If `None`, no early stopping is performed, by default None.
    input_device : int | str, optional
        The device on which to put the inputs, by default 0.


    Returns
    -------
    str | list[str]
        Str containing the generated sequence, or list[str] if `num_return_sequences` > 1.
    """
    
    model = HFModel(model_name, quantization=quantization, device_map=device_map, dtype=dtype)

    return model(prompt, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                num_return_sequences=num_return_sequences, batch_size=batch_size, seed=seed,
                truncate_prompt_from_output=truncate_prompt_from_output,
                stopping_patterns=stopping_patterns, input_device=input_device, **kwargs)



