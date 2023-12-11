from TextWiz.textwiz import HFModel

# We need to set top_k to 0 to deactivate top-k sampling
GENERATION_KWARGS = {
    'max_new_tokens': 1024,
    'min_new_tokens': 0,
    'do_sample': True,
    'temperature': 0.2,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 25,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}

GREEDY_GENERATION_KWARGS = {
    'max_new_tokens': 1024,
    'min_new_tokens': 0,
    'do_sample': False,
    'num_return_sequences': 1,
    'seed': 1234,
    'truncate_prompt_from_output': True,
}


def create_variations(model: HFModel, original_prompt: str) -> list[str]:

    prompt = f'Give me 10 reformulations of this (number them from 1 to 10): "{original_prompt}"'
    out = model(prompt, max_new_tokens=2048, do_sample=True, temperature=0.4, top_p=0.9, top_k=30, batch_size=1)