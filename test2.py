
import engine
from engine import stopping
from helpers import utils
from helpers import datasets

model_name = 'star-coder'

model = engine.HFModel(model_name)

data = datasets.HumanEval()
prompt = data[4]['prompt']

# prompt = 'def hello_world()\n    """Python function to display hello\n    """\n'
    


out = model(prompt, max_new_tokens=512, do_sample=False, top_k=None, top_p=0.95, num_return_sequences=1,
            stopping_patterns=True)

prompt_infill = "<fim_prefix>" + prompt + "<fim_suffix><fim_middle>"

out_infill = model(prompt_infill, max_new_tokens=512, do_sample=False, top_k=None, top_p=0.95, num_return_sequences=1,
                   stopping_patterns=True)

print('Normal:')
print(f'{out}\n\n\n')
print('infill:')
print(out_infill)
