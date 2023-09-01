import re
from typing import Any

from transformers import PreTrainedTokenizerBase
import torch.nn as nn
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun, CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool

from engine import loader
from engine import generation
from engine import code_eval


class HuggingFaceLLM(LLM):

    # Even though defined as class variables, these attributes are instance variables because
    # LLM is a child of BaseModel in pydantic (https://docs.pydantic.dev/latest/) which overrides
    # the class attribute system (except if the type is explicitly defined as ClassVar[...]).
    # Note also that BaseModel provides a default __init__() based on these arguments.
    model: nn.Module
    model_id: str
    tokenizer: PreTrainedTokenizerBase
    quantization: bool = False
    device_map: str = 'auto'
    max_new_tokens: int = 60
    do_sample: bool = True
    top_k: int = 40
    top_p: float = 0.90
    temperature: float = 0.90
    seed: int | None = None
    truncate: bool = True


    # Provides a constructor using the model name
    @classmethod
    def from_name(cls, model: str, max_new_tokens: int = 60, do_sample: bool = True, top_k: int = 40,
                  top_p: float = 0.90, temperature: float = 0.9, seed: int | None = None,
                  quantization: bool = False, device_map: str = 'auto'):
        
        model_instance, tokenizer = loader.load_model_and_tokenizer(model, quantization=quantization,
                                                                    device_map=device_map)
        truncate = True if model in loader.DECODER_MODELS_MAPPING.keys() else False

        return cls(
            model=model_instance,
            model_id=model,
            tokenizer=tokenizer,
            quantization=quantization,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            truncate=truncate)


    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None) -> str:

        text = generation.generate_text(self.model, self.tokenizer, prompt, max_new_tokens=self.max_new_tokens,
                                        do_sample=self.do_sample, top_k=self.top_k, top_p=self.top_p,
                                        temperature=self.temperature, num_return_sequences=1,
                                        seed=self.seed, truncate_prompt_from_output=self.truncate)
        
        # Cut off the text as soon as any stop words occur
        if stop is not None:
            text = re.split("|".join(stop), text)[0]

        return text
    

    @property
    def _identifying_params(self) -> dict[str, Any]:

        params = {
            'model': self.model_id,
            'quantization': self.quantization,
            'device_map': self.device_map,
            'max_new_tokens': self.max_new_tokens,
            'do_sample': self.do_sample,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'temperature': self.temperature,
            'seed': self.seed,
        }

        return params
    

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface_model"
    

 
class Flake8Tool(BaseTool):

    # Even though defined as class variables, these attributes are instance variables because
    # BaseTool is a child of BaseModel in pydantic (https://docs.pydantic.dev/latest/) which overrides
    # the class attribute system (except if the type is explicitly defined as ClassVar[...]).
    # Note also that BaseModel provides a default __init__() based on these arguments.
    name: str = "Flake8"
    description: str = ("Useful for when you need to evaluate code quality and refactore code."
                        " Input: code snippet. Output: code quality evaluation.")

    def _run(self, snippet: str, run_manager: CallbackManagerForToolRun | None = None) -> str:

        return code_eval.evaluate_snippet(snippet)
    

    async def _arun(self, snippet: str, run_manager: AsyncCallbackManagerForToolRun | None = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("flake8 does not support async")
    