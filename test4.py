
import textwiz

GENERATION_KWARGS = {
    'max_new_tokens': 10,
    'min_new_tokens': 0,
    'do_sample': True,
    'top_k': None,
    'top_p': 0.95,
    'num_return_sequences': 1,
}


models = list(textwiz.loader.CODEGEN2_MODELS_MAPPING.keys())
for model in models:

    prompt = 'Hello there, '

    try:
        model = textwiz.HFModel(model)
        out = model(prompt, **GENERATION_KWARGS)
        print(f'{model}: {out}')
    except BaseException as e:
        print(f'{model}: {repr(e)}')