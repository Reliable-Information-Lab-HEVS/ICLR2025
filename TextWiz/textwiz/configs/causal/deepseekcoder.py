import torch

from .. import _infer_model_sizes

MODELS_MAPPING = {
    'deepseek-coder-33B-instruct': 'deepseek-ai/deepseek-coder-33b-instruct',
    'deepseek-coder-33B': 'deepseek-ai/deepseek-coder-33b-base',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'deepseek' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 128000 for model in MODELS_MAPPING.keys()}
