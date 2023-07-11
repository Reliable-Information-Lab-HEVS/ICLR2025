import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from tokenizers.processors import TemplateProcessing
import warnings
import re


# Pretrained bloom models
BLOOM_MODELS_MAPPING = {
    'bloom-560M': 'bigscience/bloom-560m',
    'bloom-1.7B': 'bigscience/bloom-1b7',
    'bloom-3B': 'bigscience/bloom-3b',
    'bloom-7.1B':'bigscience/bloom-7b1',
    'bloom-176B': 'bigscience/bloom',
}

# Pretrained Dialo-GPT models
DIALO_GPT_MODELS_MAPPING = {
    'dialo-gpt-small': 'microsoft/DialoGPT-small',
    'dialo-gpt-medium': 'microsoft/DialoGPT-medium',
    'dialo-gpt-large': 'microsoft/DialoGPT-large',
}

# Pretrained StableLM models
STABLE_LM_MODELS_MAPPING = {
    'stable-lm-3B': 'StabilityAI/stablelm-base-alpha-3b',
    'stable-lm-7B': 'StabilityAI/stablelm-base-alpha-7b',
}

# Pretrained StarCoder models
STAR_CODER_MODELS_MAPPING = {
    'star-coder-base': 'bigcode/starcoderbase',
    # Star-coder-base further trained on Python
    'star-coder': 'bigcode/starcoder',
    # Star-coder-based further trained on English data
    'star-coder-plus': 'bigcode/starcoderplus'
}

# Pretrained Star-chat models
STAR_CHAT_MODELS_MAPPING = {
    'star-chat-alpha': 'HuggingFaceH4/starchat-alpha',
    'star-chat-beta': 'HuggingFaceH4/starchat-beta'
}

# Pretrained GPT-2 models
GPT2_MODELS_MAPPING = {
    'gpt2-medium': 'gpt2-medium',
    'gpt2-large': 'gpt2-large',
    'gpt2-xl': 'gpt2-xl',
}

# Pretrained GPT-J and GPT-Neo models
GPT_J_AND_NEO_MODELS_MAPPING = {
    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
    'gpt-neo-125M': 'EleutherAI/gpt-neo-125m',
    'gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-neoX-20B': 'EleutherAI/gpt-neox-20b',
}

# Pretrained OPT models
OPT_MODELS_MAPPING = {
    'opt-125M': 'facebook/opt-125m',
    'opt-350M': 'facebook/opt-350m',
    'opt-1.3B': 'facebook/opt-1.3b',
    'opt-2.7B': 'facebook/opt-2.7b',
    'opt-6.7B': 'facebook/opt-6.7b',
    'opt-13B': 'facebook/opt-13b',
    'opt-30B': 'facebook/opt-30b',
    'opt-66B': 'facebook/opt-66b',
}

# Pretrained CodeGEN models
CODEGEN_MODELS_MAPPING = {
    'codegen-350M': 'Salesforce/codegen-350M-mono',
    'codegen-2B': 'Salesforce/codegen-2B-mono',
    'codegen-6B': 'Salesforce/codegen-6B-mono',
    'codegen-16B': 'Salesforce/codegen-16B-mono',
}

# Pretrained CodeGEN2 models
CODEGEN2_MODELS_MAPPING = {
    'codegen2-1B': 'Salesforce/codegen2-1B',
    'codegen2-3.7B': 'Salesforce/codegen2-3_7B',
    'codegen2-7B': 'Salesforce/codegen2-7B',
    'codegen2-16B': 'Salesforce/codegen2-16B',
    'codegen25-7B': 'Salesforce/codegen25-7B-mono',
    'codegen25-7B-instruct': 'Salesforce/codegen25-7b-instruct',
}

# Pretrained Vicuna models
VICUNA_MODELS_MAPPING = {
    'vicuna-7B': 'lmsys/vicuna-7b-v1.3',
    'vicuna-13B': 'lmsys/vicuna-13b-v1.3',
}

# Pretrained BERT models
BERT_MODELS_MAPPING = {
    'bert-base-uncased': 'bert-base-uncased',
    'bert-large-uncased': 'bert-large-uncased',
    'bert-base-cased': 'bert-base-cased',
    'bert-large-cased': 'bert-large-cased',
}

# Pretrained RoBERTa models
ROBERTA_MODELS_MAPPING = {
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large',
}

# Pretrained BART models
BART_MODELS_MAPPING = {
    'bart-base': 'facebook/bart-base',
    'bart-large': 'facebook/bart-large',
}

# Pretrained T5 models
T5_MODELS_MAPPING = {
    't5-small': 't5-small',
    't5-base': 't5-base',
    't5-large': 't5-large',
    't5-3B': 't5-3b',
    't5-11B': 't5-11b',
}

# Pretrained FLAN-T5 models
FLAN_T5_MODELS_MAPPING = {
    'flan-t5-small': 'google/flan-t5-small',
    'flan-t5-base': 'google/flan-t5-base',
    'flan-t5-large': 'google/flan-t5-large',
    'flan-t5-xl': 'google/flan-t5-xl',
    'flan-t5-xxl': 'google/flan-t5-xxl',
}

# Decoder-based models
DECODER_MODELS_MAPPING = {
    **BLOOM_MODELS_MAPPING,
    **DIALO_GPT_MODELS_MAPPING,
    **STABLE_LM_MODELS_MAPPING,
    **STAR_CODER_MODELS_MAPPING,
    **STAR_CHAT_MODELS_MAPPING,
    **GPT2_MODELS_MAPPING,
    **GPT_J_AND_NEO_MODELS_MAPPING,
    **OPT_MODELS_MAPPING,
    **CODEGEN_MODELS_MAPPING,
    **CODEGEN2_MODELS_MAPPING,
    **VICUNA_MODELS_MAPPING,
}

# Encoder-based models
ENCODER_MODELS_MAPPING = {
    **BERT_MODELS_MAPPING,
    **ROBERTA_MODELS_MAPPING,
}

# Full transformer-based (encoder + decoder) models
TRANSFORMER_MODELS_MAPPING = {
    **BART_MODELS_MAPPING,
    **T5_MODELS_MAPPING,
    **FLAN_T5_MODELS_MAPPING,
}

# All models mapping
ALL_MODELS_MAPPING = {
    **DECODER_MODELS_MAPPING,
    **ENCODER_MODELS_MAPPING, 
    **TRANSFORMER_MODELS_MAPPING,
}

# Summarize all supported model names
AUTHORIZED_MODELS = list(ALL_MODELS_MAPPING.keys())



def load_model(model_name: str, quantization: bool = False, device_map: str | None = None) -> PreTrainedModel:
    """Load one of the supported pretrained model.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization : bool, optional
        Whether to load the model in 8 bits mode to save memory, by default False.
    device_map : str | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model will be put on a single GPU if relatively small, else split using 'auto'.

    Returns
    -------
    PreTrainedModel
        The model.
    """

    if model_name not in AUTHORIZED_MODELS:
        raise(ValueError(f'The model name must be one of {*AUTHORIZED_MODELS,}.'))
    
    # Automatically find the best device_map depending on the model size
    if device_map is None:
    
        # The following regex match any digits possibly separated with a dot ('.') which is immeditely
        # followed by a 'B' or 'M' to capture the model size following our model name convention. Parenthesis 
        # allow to capture given groups of the regex thanks to the match object .group() method.
        pattern = r'([0-9]+(?:\.[0-9]+)?)([BM])'

        match = re.search(pattern, model_name)
        if match:
            matched_number = match.group(1)
            matched_letter = match.group(2)
            # Model size in billion (B) of parameters
            model_size = float(matched_number) if matched_letter == 'B' else float(matched_number)/1e3
            device_map = 'balanced_low_0' if model_size > 7 else 'sequential'
        elif 'gpt2' in model_name or 'dialo-gpt' in model_name:
            device_map = 'sequential'
        else:
            device_map = 'balanced_low_0'
        
    # Override quantization if we don't have access to GPUs
    if not torch.cuda.is_available() and quantization:
        quantization = False
        warnings.warn('There are no GPUs available. The model will NOT be loaded in 8 bits mode.', RuntimeWarning)

    # Provide dtype='auto' if we do not quantize the models
    dtype = torch.float16 if quantization else 'auto'
    
    # Initiate different model types depending on architecture
    if model_name in DECODER_MODELS_MAPPING.keys():
        model = AutoModelForCausalLM.from_pretrained(DECODER_MODELS_MAPPING[model_name], device_map=device_map,
                                                    torch_dtype=dtype, load_in_8bit=quantization, low_cpu_mem_usage=True)
    elif model_name in ENCODER_MODELS_MAPPING.keys():
        model = AutoModelForMaskedLM.from_pretrained(ENCODER_MODELS_MAPPING[model_name], device_map=device_map,
                                                    torch_dtype=dtype, load_in_8bit=quantization, low_cpu_mem_usage=True)
    elif model_name in TRANSFORMER_MODELS_MAPPING.keys():
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSFORMER_MODELS_MAPPING[model_name], device_map=device_map,
                                                      torch_dtype=dtype, load_in_8bit=quantization, low_cpu_mem_usage=True)
        
    model.eval()

    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a pretrained tokenizer corresponding to one of the supported models.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    PreTrainedTokenizerBase
        The tokenizer.
    """

    if model_name not in AUTHORIZED_MODELS:
        raise(ValueError(f'The model name must be one of {*AUTHORIZED_MODELS,}.'))
    
    tokenizer = AutoTokenizer.from_pretrained(ALL_MODELS_MAPPING[model_name])

    # For Dialo-GPT models, update the post-processor to automatically add the eos token at the end
    # We need to sacrifice the ByteLevel processor for that because it is currently not possible to
    # chain post-processors (should only impact the offsets, that we do not care about)
    # if model_name in DIALO_GPT_MODELS_MAPPING.keys():
    #     tokenizer.backend_tokenizer.post_processor = \
    #         TemplateProcessing(single="$0 <|endoftext|>", special_tokens=[("<|endoftext|>", tokenizer.eos_token_id)])

    return tokenizer


def load_model_and_tokenizer(model_name: str, quantization: bool = False,
                             device_map: str = 'auto') -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load both a model and corresponding tokenizer.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization : bool, optional
        Whether to load the model in 8 bits mode to save memory, by default False.
    device_map : str, optional
        The device map to decide how to split the model between available devices, by default 'auto'.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizerBase]
        The model and tokenizer.
    """

    return load_model(model_name, quantization=quantization, device_map=device_map), load_tokenizer(model_name)
