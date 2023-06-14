import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM


# Pretrained bloom models
bloom_models_mapping = {
    'bloom-560M': 'bigscience/bloom-560m',
    'bloom-1.7B': 'bigscience/bloom-1b7',
    'bloom-3B': 'bigscience/bloom-3b',
    'bloom-7.1B':'bigscience/bloom-7b1',
    'bloom': 'bigscience/bloom',
}

# Pretrained Dialo-GPT models
dialo_gpt_models_mapping = {
    'DialoGPT-small': 'microsoft/DialoGPT-small',
    'DialoGPT-medium': 'microsoft/DialoGPT-medium',
    'DialoGPT-large': 'microsoft/DialoGPT-large',
}

# Pretrained GPT-2 models
gpt2_mapping = {
    'gpt2-small':'gpt2-small',
    'gpt2-medium': 'gpt2-medium',
    'gpt2-large': 'gpt2-large',
    'gpt2-xl': 'gpt2-xl',
}

# Pretrained GPT-J and GPT-Neo models
gpt_j_and_neo_models_mapping = {
    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
    'gpt-neo-125M': 'EleutherAI/gpt-neo-125m',
    'gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-neoX-20B': 'EleutherAI/gpt-neox-20b',
}

# Pretrained OPT models
opt_models_mapping = {
    'opt-125M': 'facebook/opt-125m',
    'opt-350M': 'facebook/opt-350m',
    'opt-1.3B': 'facebook/opt-1.3b',
    'opt-2.7B': 'facebook/opt-2.7b',
    'opt-6.7B': 'facebook/opt-6.7b',
    'opt-13B': 'facebook/opt-13b',
    'opt-30B': 'facebook/opt-30b',
    'opt-66B': 'facebook/opt-66b',
}

# Pretrained BART models
bart_models_mapping = {
    'bart-base': 'facebook/bart-base',
    'bart-large': 'facebook/bart-large'
}

# Pretrained RoBERTa models
roberta_models_mapping = {
    'roberta': 'roberta-base'
}

# Summarize all pretrained models
all_models_mapping = {
    **bloom_models_mapping,
    **dialo_gpt_models_mapping,
    **gpt2_mapping,
    **gpt_j_and_neo_models_mapping,
    **opt_models_mapping,
    **bart_models_mapping,
    **roberta_models_mapping
}

# Summarize all supported model names
AUTHORIZED_MODELS = list(all_models_mapping.keys())


def set_all_seeds(seed: int):
    """Set seed for all random number generators (random, numpy and torch).

    Parameters
    ----------
    seed : int
        The seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(model_name: str) -> AutoModelForCausalLM:
    """Load one of the supported pretrained model.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    AutoModel
        The model.
    """

    if model_name not in AUTHORIZED_MODELS:
        raise(ValueError(f'The model name must be one of {*AUTHORIZED_MODELS,}.'))
    
    model = AutoModelForCausalLM.from_pretrained(all_models_mapping[model_name], device_map='auto', torch_dtype='auto')
    model.eval()

    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a pretrained tokenizer corresponding to one of the supported models.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    AutoTokenizer
        The tokenizer.
    """

    if model_name not in AUTHORIZED_MODELS:
        raise(ValueError(f'The model name must be one of {*AUTHORIZED_MODELS,}.'))
    
    tokenizer = AutoTokenizer.from_pretrained(all_models_mapping[model_name])

    return tokenizer

    
def generate_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 60,
                  do_sample: bool = True, top_k: int = 100, top_p: float = 0.92, temperature: float = 0.9,
                  num_return_sequences: int = 1, seed: int | None = None) -> list[str]:
    """Generate text according to `prompt` using the `model` and `tokenizer` specified.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The model used to generate the text.
    tokenizer : AutoTokenizer
        The tokenizer to use to process the input and output text.
    prompt : str
        The prompt to the model.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.
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
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    list[str]
        List containing all `num_return_sequences` sequences generated.
    """
    
    if seed is not None:
        set_all_seeds(seed)

    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k,
                                top_p=top_p, temperature=temperature, num_return_sequences=num_return_sequences)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)   

    return generated_text
    

def load_and_generate_text(model_name: str, prompt: str, max_new_tokens: int = 60, do_sample: bool = True,
                  top_k: int = 100, top_p: float = 0.92, temperature: float = 0.9,
                    num_return_sequences: int = 1, seed: int | None = None) -> list[str]:
    """Load a model and its tokenizer and generate text according to `prompt`.

    Parameters
    ----------
    model_name : str
        The model used to generate the text.
    prompt : str
        The prompt to the model.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.
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
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    list[str]
        List containing all `num_return_sequences` sequences generated.
    """

    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)

    return generate_text(model, tokenizer, prompt, max_new_tokens, do_sample, top_k,
                          top_p, temperature, num_return_sequences, seed)