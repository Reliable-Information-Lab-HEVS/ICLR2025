"""
This file contains the prompt templates for the models we use, for causal generation. These
templates are especially not meant for conversations with the models, only for prompt completion, without
memory of previous prompts.
"""

from engine.loader import ALLOWED_MODELS

PROMPT_MODES = ('default', 'generation', 'infill', 'chat')

class GenericPromptTemplate(object):

    def __init__(self, mode: str = 'default'):

        if mode not in PROMPT_MODES:
            raise ValueError(f'The mode for creating the prompt must be one of {*PROMPT_MODES,}')
        
        self.mode = mode
        self.default_mode = 'generation'
        self.extra_eos_tokens = []


    def get_prompt(self, prompt: str, suffix: str = '') -> str:

        if self.mode == 'default':
            return self.format_default(prompt, suffix)
        elif self.mode == 'generation':
            return self.format_generation(prompt)
        elif self.mode == 'infill':
            return self.format_infill(prompt, suffix)
        elif self.mode == 'chat':
            return self.format_chat(prompt)
        

    def format_default(self, prompt: str, suffix: str = '') -> str:

        if self.default_mode == 'generation':
            return self.format_generation(prompt)
        elif self.default_mode == 'infill':
            return self.format_infill(prompt, suffix)
        elif self.default_mode == 'chat':
            return self.format_chat(prompt)


    def format_generation(self, prompt: str) -> str:
        return prompt
    

    def format_chat(self, prompt: str) -> str:
        raise RuntimeError(f'Chat mode not supported for {self.__class__.__name__}.')
    

    def format_infill(self, prefix: str, suffix: str = '') -> str:
        raise RuntimeError(f'Infill mode not supported for {self.__class__.__name__}.')
    

    def get_extra_eos(self) -> list[str]:
        return self.extra_eos_tokens
    

class DialoGPTPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.eos_token = '<|endoftext|>'

    def format_chat(self, prompt: str) -> str:

        return prompt + self.eos_token
    


class StarCoderPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'infill'

        self.prefix_token = '<fim_prefix>'
        self.suffix_token = '<fim_suffix>'
        self.middle_token = '<fim_middle>'

    def format_infill(self, prefix: str, suffix: str = '') -> str:

        return self.prefix_token + prefix + self.suffix_token + suffix + self.middle_token
    
    
# Starchat prompt modeling (see https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py)
# See also FastChat (/https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#817) but note that
# it was modified by me (https://github.com/lm-sys/FastChat/pull/2239)
class StarChatPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default', system_prompt: str = ''):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_token = '<|system|>'
        self.system_prompt = system_prompt
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'
        self.sep_token = '<|end|>'
        self.extra_eos_tokens = [self.sep_token]

    
    def format_chat(self, prompt: str) -> str:

        return self.system_token + '\n' + self.system_prompt + self.sep_token + '\n' + self.user_token + '\n' \
            + prompt + self.sep_token + '\n' + self.assistant_token + '\n'
    


class Codegen2PromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        # Keep the default to generation as the infill mode seems to be worse (at least on HumanEval)
        # self.default_mode = 'infill'
        self.default_mode = 'generation'

        self.mask_token = '<mask_1>'
        self.eos_token = '<|endoftext|>'
        self.sep_token = '<sep>'
        self.extra_eos_tokens = ['<eom>']

    def format_infill(self, prefix: str, suffix: str = '') -> str:

        return prefix + self.mask_token + suffix + self.eos_token + self.sep_token + self.mask_token
    

# Vicuna 1.3 prompt modeling (https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py)
class VicunaPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default', system_prompt: str = ''):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_prompt = system_prompt
        self.user_token = 'USER'
        self.assistant_token = 'ASSISTANT'

    
    def format_chat(self, prompt: str) -> str:

        if self.system_prompt == '':
            return self.user_token + ': ' + prompt + ' ' + self.assistant_token + ':'
        else:
            return self.system_prompt + ' ' + self.user_token + ': ' + prompt + ' ' + self.assistant_token + ':'


# Llama2-chat prompt modeling (https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
# See also FastChat (https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L123) but note that
# it was modified by me (https://github.com/lm-sys/FastChat/pull/2239)
class Llama2ChatPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default', system_prompt: str = ''):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_prompt = system_prompt
        self.system_template = '<<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
        self.user_token = '[INST]'
        self.assistant_token = '[/INST]'

    def format_chat(self, prompt: str) -> str:

        # System prompt must be embedded in first user prompt
        embedded_prompt = self.system_template.format(system_prompt=self.system_prompt) + prompt
        # Note that we do not call strip() as meta does in source code, because some prompts explicitly end with '\n'
        return self.user_token + ' ' + embedded_prompt + ' ' + self.assistant_token

    

# Mapping from model name to prompt class name
PROMPT_MAPPING = {
    # DialoGPT
    'dialo-gpt-small': DialoGPTPromptTemplate,
    'dialo-gpt-medium': DialoGPTPromptTemplate,
    'dialo-gpt-large': DialoGPTPromptTemplate,

    # StarCoder
    'star-coder-base': StarCoderPromptTemplate,
    'star-coder': StarCoderPromptTemplate,
    'star-coder-plus': StarCoderPromptTemplate,

    # StarChat
    'star-chat-alpha': StarChatPromptTemplate,
    'star-chat-beta': StarChatPromptTemplate,

    # Codegen2
    'codegen2-1B': Codegen2PromptTemplate,
    'codegen2-3.7B': Codegen2PromptTemplate,
    'codegen2-7B': Codegen2PromptTemplate,
    'codegen2-16B': Codegen2PromptTemplate,
    'codegen25-7B': Codegen2PromptTemplate,
    'codegen25-7B-instruct': Codegen2PromptTemplate,

    # Vicuna (1.3)
    'vicuna-7B': VicunaPromptTemplate,
    'vicuna-13B': VicunaPromptTemplate,

    # Llama2-chat
    'llama2-7B-chat': Llama2ChatPromptTemplate,
    'llama2-13B-chat': Llama2ChatPromptTemplate,
    'llama2-70B-chat': Llama2ChatPromptTemplate,
}


def get_prompt_template(model_name: str, mode: str = 'default', system_prompt: str = '') -> GenericPromptTemplate:
    """Return the prompt template class formating corresponding to `model_name`.

    Parameters
    ----------
    model_name : str
        Name of the current model.
    mode : str, optional
        The generation mode for the model, by default 'default'. Note that changing this value may cause
        issues as not all prompt templates support all modes.
    system_prompt : str, optional
        The system prompt for templates with chat mode that support it, by default ''. Ignored by all non-chat
        templates.

    Returns
    -------
    GenericPromptTemplate
        A prompt template class corresponding to `model_name`.

    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    if mode not in PROMPT_MODES:
        raise ValueError(f'The mode for creating the prompt must be one of {*PROMPT_MODES,}')

    if model_name in PROMPT_MAPPING.keys():
        try:
            prompt = PROMPT_MAPPING[model_name](mode=mode, system_prompt=system_prompt)
        except TypeError:
            prompt = PROMPT_MAPPING[model_name](mode=mode)
    else:
        prompt = GenericPromptTemplate(mode)

    return prompt