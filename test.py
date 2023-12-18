
from TextWiz import textwiz

import torch
from tqdm import tqdm

model = textwiz.HFModel('zephyr-7B-beta')

input = 'This is a beautiful house'
encodings = model.tokenizer(input, return_tensors='pt')

device = model.input_device
max_length = 8192
seq_len = encodings.input_ids.size(1)
stride = 1


# with torch.no_grad():
#     outputs = model.model(encodings, labels=target_ids)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        print(f'outputs:\n{outputs}')

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())



