import torch

# Pretrained StarCoder models
MODELS_MAPPING = {
    'star-coder-base': 'bigcode/starcoderbase',
    'star-coder': 'bigcode/starcoder',
    'star-coder-plus': 'bigcode/starcoderplus',
    'star-coder-2-15B': 'bigcode/starcoder2-15b',
    'star-coder-2-7B': 'bigcode/starcoder2-7b',
    'star-coder-2-3B': 'bigcode/starcoder2-3b',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = {model: 15.5 for model in MODELS_MAPPING.keys()}
MODELS_FAMILY = {model: 'star-coder' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {
    'star-coder-base': 8192,
    'star-coder': 8192,
    'star-coder-plus': 8192,
    'star-coder-2-15B': 16384,
    'star-coder-2-7B': 16384,
    'star-coder-2-3B': 16384,
}
MODELS_ADDITIONAL_MODEL_KWARGS = {
    'star-coder-base': {'trust_remote_code': True},
}