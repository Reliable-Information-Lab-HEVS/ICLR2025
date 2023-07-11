import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import StoppingCriteriaList

from engine import loader
from engine import stopping
from helpers import utils


def generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str, max_new_tokens: int = 60,
                  min_new_tokens: int = 5, do_sample: bool = True, top_k: int = 40, top_p: float = 0.90,
                  temperature: float = 0.9, num_return_sequences: int = 1, batch_size: int | None = None,
                  seed: int | None = None, truncate_prompt_from_output: bool = False,
                  stopping_patterns: list[str] | bool | None = None, **kwargs) -> str | list[str]:
    """Generate text according to `prompt` using the `model` and `tokenizer` specified.

    Parameters
    ----------
    model : PreTrainedModel
        The model used to generate the text.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use to process the input and output text.
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
    seed : int | None, optional
        An optional seed to force the generation to be reproducible.
    truncate_prompt_from_output : bool, optional
        Whether to remove the prompt from the model answer or not, by default False.
    stopping_patterns: list[str] | bool | None
        List of words/patterns to stop the generation. Pass `True` to use the default `CODE_STOP_PATTERNS` patterns.
        If `None`, no early stopping is performed, by default None.

    Returns
    -------
    str | list[str]
        Str containing the generated sequence, or list[str] if `num_return_sequences` > 1.
    """
    
    if seed is not None:
        utils.set_all_seeds(seed)

    input = tokenizer.encode(prompt, return_tensors='pt')
    input_length = input.shape[-1]
    if torch.cuda.is_available():
        input = input.to('cuda')

    # Possible early stopping
    if type(stopping_patterns) is list:
        stopping_criteria = stopping.TextPatternStopping(input_length, tokenizer, stopping_patterns)
        stopping_criteria = StoppingCriteriaList([stopping_criteria])
        post_process = True
        post_process_list = stopping_patterns
    elif type(stopping_patterns) is bool and stopping_patterns:
        stopping_criteria = stopping.TextPatternStopping(input_length, tokenizer)
        stopping_criteria = StoppingCriteriaList([stopping_criteria])
        post_process = True
        post_process_list = stopping.CODE_STOP_PATTERNS
    else:
        stopping_criteria = None
        post_process = False


    # If we require more sequences than the allowed batch size, we need to split the generation into
    # multiple passes
    if (batch_size is not None) and (num_return_sequences > batch_size):
        batch_sizes = [batch_size]*(num_return_sequences // batch_size)
        remainder = num_return_sequences % batch_size
        if remainder != 0:
              batch_sizes += [remainder]
        assert sum(batch_sizes) == num_return_sequences
    else:
        batch_sizes = [num_return_sequences]

    generated_text = []

    for size in batch_sizes:

        with torch.no_grad():
            outputs = model.generate(input, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                     do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                                     num_return_sequences=size, stopping_criteria=stopping_criteria, **kwargs)
            
        # In this case truncate the prompt from the output
        if truncate_prompt_from_output:
            outputs = outputs[:, input_length:]

        generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True) 
        if post_process:
            generated_batch = stopping.post_process_sequences(generated_batch, prompt, post_process_list)
        
        generated_text += generated_batch

    # In this case return a str instead of list[str]
    if num_return_sequences == 1:
        generated_text = generated_text[0]

    return generated_text



def load_and_generate_text(model_name: str, prompt: str, quantization: bool = False, max_new_tokens: int = 60,
                           min_new_tokens: int = 5, do_sample: bool = True, top_k: int = 100, top_p: float = 0.92,
                           temperature: float = 0.9, batch_size: int | None = None, num_return_sequences: int = 1,
                           seed: int | None = None, truncate_prompt_from_output: bool = False,
                           stopping_patterns: list[str] | bool | None = None, **kwargs) -> str | list[str]:
    """Load a model and its tokenizer and generate text according to `prompt`.

    Parameters
    ----------
    model_name : str
        The model used to generate the text.
    prompt : str
        The prompt to the model.
    quantization : bool, optional
        Whether to load the model in 8 bits mode to save memory, by default False.
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


    Returns
    -------
    str | list[str]
        Str containing the generated sequence, or list[str] if `num_return_sequences` > 1.
    """

    model, tokenizer = loader.load_model_and_tokenizer(model_name, quantization)

    return generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                         do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
                         num_return_sequences=num_return_sequences, batch_size=batch_size, seed=seed,
                         truncate_prompt_from_output=truncate_prompt_from_output,
                         stopping_patterns=stopping_patterns, **kwargs)




class HFModel(object):
    """Class encapsulating a HuggingFace model and its tokenizer to generate text. 
    """

    def __init__(self, model_name: str, quantization: bool = False, device_map: str = 'auto'):

        self.model, self.tokenizer = loader.load_model_and_tokenizer(model_name, quantization=quantization,
                                                                     device_map=device_map)
        self.model_name = model_name
        self.quantization = quantization
        self.device_map = device_map

    
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
        """Generate text according to `prompt`.

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
    seed : int | None, optional
        An optional seed to force the generation to be reproducible.
    truncate_prompt_from_output : bool, optional
        Whether to remove the prompt from the model answer or not, by default False.
    stopping_patterns: list[str] | bool | None
        List of words/patterns to stop the generation. Pass `True` to use the default `CODE_STOP_PATTERNS` patterns.
        If `None`, no early stopping is performed, by default None.

    Returns
    -------
    str | list[str]
        Str containing the generated sequence, or list[str] if `num_return_sequences` > 1.
    """
        
        return generate_text(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens,
                             min_new_tokens=min_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                             temperature=temperature, num_return_sequences=num_return_sequences,
                             batch_size=batch_size, seed=seed, truncate_prompt_from_output=truncate_prompt_from_output,
                             stopping_patterns=stopping_patterns, **kwargs)
