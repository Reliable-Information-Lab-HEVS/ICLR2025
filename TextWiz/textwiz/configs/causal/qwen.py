import torch

from .. import _infer_model_sizes

# Pretrained llama-3 models
MODELS_MAPPING = {
    'qwen2.5-coder-32B-instruct': 'Qwen/Qwen2.5-Coder-32B-Instruct',
    'qwen2.5-coder-32B': 'Qwen/Qwen2.5-Coder-32B',
    'qwen2.5-coder-7B-instruct': 'Qwen/Qwen2.5-Coder-7B-Instruct',
    'qwen2.5-coder-7B': 'Qwen/Qwen2.5-Coder-7B',
    'qwen3-coder-30B-instruct': 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'qwen' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 32768 for model in MODELS_MAPPING.keys()}
