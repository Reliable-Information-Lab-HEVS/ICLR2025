from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import time

import engine


for model in engine.SMALL_MODELS:
    try:
        foo = engine.HFModel(model)
    except Exception as e:
        print(f'Issue with {model}: {type(e).__name__}: {str(e)}')
        pass