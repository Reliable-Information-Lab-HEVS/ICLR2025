
from TextWiz import textwiz, memory_estimator

import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

model = textwiz.HFModel('zephyr-7B-beta')

# input = 'This is a beautiful house'
input = memory_estimator.LARGE_TEXT
# input = ''
encodings = model.tokenizer(input, return_tensors='pt')

device = model.input_device
max_length = 8192
seq_len = encodings.input_ids.size(1)
stride = 1
target_ids = model.tokenizer.encode(input, add_special_tokens=False)


# # with torch.no_grad():
# #     outputs = model.model(encodings, labels=target_ids)

# nlls = []
# prev_end_loc = 0
# for begin_loc in range(0, seq_len, stride):
#     end_loc = min(begin_loc + max_length, seq_len)
#     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#     input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#     target_ids = input_ids.clone()
#     target_ids[:, :-trg_len] = -100
#     print(f'target_shape:{target_ids.shape}')
#     print(f'target:{target_ids}')

#     with torch.no_grad():
#         outputs = model.model(input_ids, labels=target_ids)

#         logits = outputs.logits
#         print(f'logits_shape:\n{logits.shape}')
#         print(f'logits:\n{logits}')

#         foo = model.tokenizer.decode(torch.argmax(logits[:, 0, :]))
#         print(f'FIRST TOKEN:{foo}')

#         loss_func = CrossEntropyLoss()
#         loss = loss_func(logits[])

#         # loss is calculated using CrossEntropyLoss which averages over valid labels
#         # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
#         # to the left by 1.
#         neg_log_likelihood = outputs.loss

#     nlls.append(neg_log_likelihood)

#     prev_end_loc = end_loc
#     if end_loc == seq_len:
#         break

# ppl = torch.exp(torch.stack(nlls).mean())
# print(ppl)


tot = 0

prev_end_loc = 0
for begin_loc in range(0, seq_len, stride):
    end_loc = min(begin_loc + max_length, seq_len)
    # Will always be equal to stride except on last iteration
    target_length = end_loc - prev_end_loc

    # Compute inputs and targets
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()

    # This will mask the (max_length - stride) already processed targets in the loss
    target_ids[:, :-target_length] = -100

    # Remove first target as we cannot compute the probability distribution for the first token of the input.
    # This is not an issue since for first iteration the first token is <EOS>, and it is masked for other iterations
    target_ids = target_ids[:, 1:]
    # Remove batch dimension of size 1
    target_ids = target_ids.squeeze(0)

    tot += torch.sum(target_ids[0, :] != 0).item()

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

assert tot == (seq_len-1)