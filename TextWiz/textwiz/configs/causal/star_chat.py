import torch

# Pretrained Star-chat models
MODELS_MAPPING = {
    'star-chat-alpha': 'HuggingFaceH4/starchat-alpha',
    'star-chat-beta': 'HuggingFaceH4/starchat-beta',
    'star-chat-2-instruct': 'HuggingFaceH4/starchat2-15b-v0.1',
}
MODELS_DTYPES = {
    'star-chat-alpha': torch.float16,
    'star-chat-beta': torch.bfloat16,
    'star-chat-2-instruct': torch.bfloat16,
}
MODELS_PARAMS = {model: 15.5 for model in MODELS_MAPPING.keys()}
MODELS_FAMILY = {model: 'star-chat' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 8192 for model in MODELS_MAPPING.keys()}