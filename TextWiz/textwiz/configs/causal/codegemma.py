import torch

from .. import _infer_model_sizes

MODELS_MAPPING = {
    'codegemma-7B-instruct': 'google/codegemma-7b-it',
    'codegemma-7B': 'google/codegemma-7b',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'codegemma' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 8000 for model in MODELS_MAPPING.keys()}
