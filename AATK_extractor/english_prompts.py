import os
import sys
# Add top-level package to the path (only way to import custom module in a script that is not in the root folder)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from helpers import utils

path_to_prompts = os.path.join(utils.DATA_FOLDER, 'AATK_english_prompts.txt')
path_to_extended_prompts = os.path.join(utils.DATA_FOLDER, 'AATK_prompt_variations_chatGPT.txt')

dataset = utils.load_jsonl(os.path.join(utils.DATA_FOLDER, 'AATK.jsonl'))

prompts = [line for line in utils.load_txt(path_to_prompts) if line != '']
extended_prompts = [line for line in utils.load_txt(path_to_extended_prompts) if line != '']

for sample in dataset:
    id = sample['id']
    intent_index = prompts.index(id)
    intent_variations_index = extended_prompts.index(id)

    intent = prompts[intent_index+1]
    intent_variations = extended_prompts[intent_variations_index+1:intent_variations_index+11]

    sample['intent'] = intent
    sample['intent_variations'] = intent_variations


utils.save_jsonl(dataset, os.path.join(utils.DATA_FOLDER, 'AATK_english_chatGPT.jsonl'))