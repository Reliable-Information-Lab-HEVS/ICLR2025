import multiprocessing as mp
import time
import os
import torch
from concurrent.futures import ProcessPoolExecutor

# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers import PreTrainedModel
# from transformers import PreTrainedTokenizerBase

# from engine import loader
# from engine import loader2
from helpers import utils


print(f'Has been initialized at the beginning in foo: {torch.cuda.is_initialized()}')

def test():
     print('This is a test')

@utils.duplicate_function_for_gpu_dispatch
def target(name: str, foo, bar = 3):
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(f'Has been initialized at the beginning in func: {torch.cuda.is_initialized()}')
    print(f'Number of gpus seen by torch: {torch.cuda.device_count()}')
    print(f'Has been initialized at the end in func: {torch.cuda.is_initialized()}')
    time.sleep(5)
    print('Done!')
    test()


@utils.duplicate_function_for_gpu_dispatch
def target2(name: str, foo, bar = 3):
    time.sleep(5)
    print('target2')
    test()



LARGE_MODELS = (
    'gpt-neoX-20B',
    # 'opt-30B',
    'llama2-70B',
)


if __name__ == '__main__':
    # num_gpus = torch.cuda.device_count()
    num_gpus = 5

    # model_footprints = []
    # # Estimate number of gpus needed for each model
    # for model in LARGE_MODELS:
    #     quantization = model == 'bloom-176B'
    #     gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization)
    #     model_footprints.append(gpu_needed)

    model_footprints = [2,5]

    args = ([1,2],)
    utils.dispatch_jobs(LARGE_MODELS, model_footprints, num_gpus, target, args)
    