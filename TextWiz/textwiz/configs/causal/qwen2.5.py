import torch

from .. import _infer_model_sizes

# Pretrained llama-3 models
MODELS_MAPPING = {
    'qwen2.5-coder-32B-Instruct': 'Qwen/Qwen2.5-Coder-32B-Instruct',
    'qwen2.5-coder-32B': 'Qwen/Qwen2.5-Coder-32B',
    'qwen2.5-coder-7B-Instruct': 'Qwen/Qwen2.5-Coder-7B-Instruct',
    'qwen2.5-coder-7B': 'Qwen/Qwen2.5-Coder-7B',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'qwen2.5' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 32768 for model in MODELS_MAPPING.keys()}
