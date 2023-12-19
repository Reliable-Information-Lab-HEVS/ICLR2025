
from TextWiz import textwiz, memory_estimator
from helpers import utils, aatk, datasets

import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

# model = textwiz.HFModel('gpt2-medium')

# input = 'This is a beautiful house'
# # input = memory_estimator.LARGE_TEXT
# # input = ''
# encodings = model.tokenizer(input, return_tensors='pt')

# device = model.input_device
# max_length = model.get_context_size()
# seq_len = encodings.input_ids.size(1)
# stride = 512



# nlls = []
# prev_end_loc = 0
# for begin_loc in tqdm(range(0, seq_len, stride)):
#     end_loc = min(begin_loc + max_length, seq_len)
#     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#     input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#     target_ids = input_ids.clone()
#     target_ids[:, :-trg_len] = -100

#     with torch.no_grad():
#         outputs = model.model(input_ids, labels=target_ids)

#         # loss is calculated using CrossEntropyLoss which averages over valid labels
#         # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
#         # to the left by 1.
#         neg_log_likelihood = outputs.loss

#     nlls.append(neg_log_likelihood)

#     prev_end_loc = end_loc
#     if end_loc == seq_len:
#         break

# ppl = torch.exp(torch.stack(nlls).mean())

# print(f'Original:{ppl}')

# foofoo = model.perplexity(input)
# print(f'Mine:{foofoo}')


model = textwiz.HFModel('zephyr-7B-beta')

data = datasets.AATKEnglish().samples_by_id()

results_filenames = aatk.extract_filenames(dataset='AATK_english', category='results', only_unprocessed=False)

for file in results_filenames:

    res = utils.load_jsonl(file)

    for sample in res:
        record = data[sample['id']]
        variation_id = sample['prompt_id'] if sample['prompt_id'] == 'original' else int(sample['prompt_id'].replace('variation ', ''))
        prompt = record['intent'] if variation_id == 'original' else record['intent_variations'][variation_id]

        ppl = model.perplexity(prompt)
        sample['perplexity'] = ppl

    # save res
    utils.save_jsonl(res, file.replace('AATK_english', 'AATK_perplexity'))