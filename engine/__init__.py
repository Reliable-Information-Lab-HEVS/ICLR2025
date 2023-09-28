from engine.generation import HFModel
from engine import loader
from engine import prompt_template
# import it here so that the warnings are suppressed when doing `import engine`
from engine import warnings_suppressor


def estimate_number_of_gpus(models: list[str], quantization_8bits: bool = False, quantization_4bits: bool = False,
                            max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8) -> list[int]:
    """Estimate the mumber of gpus needed to run each of the `models` correctly.

    Parameters
    ----------
    models : list[str]
        The models.
    quantization_8bits : bool
        Whether the model will be loaded in 8 bits mode, by default False.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8.

    Returns
    -------
    list[int]
        The number of gpus for each model.
    """
    
    model_footprints = []
    for model in models:
        # Override quantization for bloom because it's too big to load in float16
        if model == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
            gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization_8bits=True, quantization_4bits=False,
                                                                max_fraction_gpu_0=0.9,
                                                                max_fraction_gpus=0.9)
        else:
            gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization_8bits=quantization_8bits,
                                                                quantization_4bits=quantization_4bits,
                                                                max_fraction_gpu_0=max_fraction_gpu_0,
                                                                max_fraction_gpus=max_fraction_gpus)
        model_footprints.append(gpu_needed)

    return model_footprints




ALL_MODELS = loader.ALLOWED_MODELS

# Models working on single GPU
SMALL_MODELS = tuple(model for model, gpus in zip(ALL_MODELS, estimate_number_of_gpus(ALL_MODELS)) if gpus == 1)

# Models needing more than 1 GPU
LARGE_MODELS = tuple(model for model, gpus in zip(ALL_MODELS, estimate_number_of_gpus(ALL_MODELS)) if gpus > 1)


# Model with non-default prompt template
SMALL_MODELS_SPECIAL_PROMPT = tuple(model for model in SMALL_MODELS if model in prompt_template.PROMPT_MAPPING.keys())
LARGE_MODELS_SPECIAL_PROMPT = tuple(model for model in LARGE_MODELS if model in prompt_template.PROMPT_MAPPING.keys())


# Small models that we decided to keep for further code benchmarks
SMALL_MODELS_GOOD_CODER = (
    'star-coder-base',
    'star-coder',
    'star-chat-alpha',
    'codegen-16B',
    'codegen25-7B',
    'codegen25-7B-instruct',
)

# Large models that we decided to keep for further code benchmarks
LARGE_MODELS_GOOD_CODER = (
    'code-llama-34B',
    'code-llama-34B-python',
    'code-llama-34B-instruct',
    'llama2-70B',
    'llama2-70B-chat',
)


# Model that we decided to keep for further code benchmarks with non-default prompt template
SMALL_MODELS_GOOD_CODER_SPECIAL_PROMPT = tuple(model for model in SMALL_MODELS_GOOD_CODER if model in prompt_template.PROMPT_MAPPING.keys())
LARGE_MODELS_GOOD_CODER_SPECIAL_PROMPT = tuple(model for model in LARGE_MODELS_GOOD_CODER if model in prompt_template.PROMPT_MAPPING.keys())