from engine.generation import HFModel
from engine import loader
# import it here so that the warnings are suppressed when doing `import engine`
from engine import warnings_suppressor


# TODO: maybe find a better file for this function? But should not really be in `utils` as we need `loader`...
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
                                                                max_fraction_gpu_0=max_fraction_gpu_0,
                                                                max_fraction_gpus=max_fraction_gpus)
        else:
            gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization_8bits=quantization_8bits,
                                                                quantization_4bits=quantization_4bits,
                                                                max_fraction_gpu_0=max_fraction_gpu_0,
                                                                max_fraction_gpus=max_fraction_gpus)
        model_footprints.append(gpu_needed)

    return model_footprints