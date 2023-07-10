import os
import gc

import engine
from engine import stopping
from helpers import datasets
from helpers import utils

dataset = datasets.HumanEval()

models = [
    'bloom-560M',
    # 'bloom-1.7B',
    # 'bloom-3B',
    # 'bloom-7.1B',
    # 'bloom',
    # 'stable-lm-3B',
    # 'stable-lm-7B',
    # 'star-coder-base',
    # 'star-coder',
    # 'star-coder-plus',
    # # 'star-chat-alpha',
    # # 'star-chat-beta',
    # 'gpt2-medium',
    # 'gpt2-large',
    # 'gpt2-xl',
    # 'gpt-j-6B',
    # 'gpt-neo-125M',
    # 'gpt-neo-1.3B',
    # 'gpt-neo-2.7B',
    # 'gpt-neoX-20B',
    # 'opt-125M',
    # 'opt-350M',
    # 'opt-1.3B',
    # 'opt-2.7B',
    # 'opt-6.7B',
    # 'opt-13B',
    # 'opt-30B',
    # 'opt-66B',
    # 'codegen-350M',
    # 'codegen-2B',
    # 'codegen-6B',
    # 'codegen-16B',
    # 'codegen2-1B',
    # 'codegen2-3.7B',
    # 'codegen2-7B',
    # 'codegen2-16B',
    # 'codegen25-7B',
    # 'codegen25-7B-instruct',
    # 'vicuna-7B',
    # 'vicuna-13B',
]


# We need to set top_k to 0 to deactivate top-k sampling
human_eval_generation_kwargs = {
    'max_new_tokens': 512,
    'do_sample': True,
    'top_k': 0,
    'top_p': 0.95,
    'num_return_sequences': 200,
    'seed': None,
    'truncate_prompt_from_output': True,
    'stopping_patterns': stopping.CODE_STOP_PATTERNS
}

human_eval_greedy_generation_kwargs = {
    'max_new_tokens': 512,
    'do_sample': False,
    'top_k': 0,
    'top_p': 1.,
    'num_return_sequences': 1,
    'seed': None,
    'truncate_prompt_from_output': True,
    'stopping_patterns': stopping.CODE_STOP_PATTERNS
}

temperatures = [0., 0.2, 0.4, 0.6, 0.8, 1.]

def main():

    for model_name in models:

        # Load in 8 bits for bloom due to model size
        quantization = True if model_name == 'bloom' else False

        model = engine.HFModel(model_name, quantization=quantization)
        folder = os.path.join(utils.RESULTS_FOLDER , 'HumanEval_completions', model_name)

        for temperature in temperatures:

            filename = os.path.join(folder, f'temperature_{temperature}.jsonl')

            for sample in dataset:

                task_id = sample['task_id']
                prompt = sample['prompt']

                # In this case we use greedy decoding (the temperature parameters does not matter anymore
                # so we set it to the default which is 1)
                if temperature == 0:
                    completions = [model(prompt, temperature=1., **human_eval_greedy_generation_kwargs)]
                # In this case we use top-p sampling
                else:
                    completions = model(prompt, temperature=temperature, **human_eval_generation_kwargs)

                # Save the model completions
                results = [{'task_id': task_id, 'completion': completion} for completion in completions]
                utils.save_jsonl(results, filename, append=True)

        del model
        gc.collect()


if __name__ == '__main__':

    main()