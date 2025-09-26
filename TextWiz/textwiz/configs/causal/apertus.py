import torch

from .. import _infer_model_sizes

# Pretrained llama-3 models
MODELS_MAPPING = {
    'apertus-8B': 'swiss-ai/Apertus-8B-2509',
    'apertus-8B-instruct': 'swiss-ai/Apertus-8B-Instruct-2509',
    'apertus-70B': 'swiss-ai/Apertus-70B-2509',
    'apertus-70B-instruct': 'swiss-ai/Apertus-70B-Instruct-2509',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'apertus' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 65536 for model in MODELS_MAPPING.keys()}
