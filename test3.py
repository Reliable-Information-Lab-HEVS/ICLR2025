import engine
from engine import stopping
from helpers import datasets

model = engine.HFModel('llama2-7B-chat')

dataset = datasets.HumanEvalInstruct()
sample = dataset[0]

instruction = sample['instruction']
context = sample['context']

out1 = model(instruction + context, max_new_tokens=40, do_sample=False, top_k=0, top_p=1., batch_size=1,
             post_process_output=False, truncate_prompt_from_output=False)
out2 = model(instruction.strip() + '\n' + context.strip(), max_new_tokens=40, do_sample=False, top_k=0, top_p=1., batch_size=1,
             post_process_output=False, truncate_prompt_from_output=False)
out3 = model(instruction, model_context=context, max_new_tokens=40, do_sample=False, top_k=0, top_p=1., batch_size=1,
             post_process_output=False, truncate_prompt_from_output=False)
out4 = model(instruction.strip(), model_context=context.strip(), max_new_tokens=40, do_sample=False, top_k=0, top_p=1., batch_size=1,
             post_process_output=False, truncate_prompt_from_output=False)


print(f'Without context and without strip:\n\n{out1}\n\n')
print(f'Without context and with strip:\n\n{out2}\n\n')
print(f'With context and without strip:\n\n{out3}\n\n')
print(f'With context and with strip:\n\n{out4}')

