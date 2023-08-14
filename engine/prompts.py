
TEMPLATE_MODES = ('default', 'generation', 'infill', 'chat')

class GenericPrompt(object):

    def __init__(self, mode: str = 'default'):

        if mode not in TEMPLATE_MODES:
            raise ValueError(f'The mode for creating the prompt must be one of {*TEMPLATE_MODES,}')
        
        self.mode = mode
        self.default_mode = 'generation'
        self.extra_eos_tokens = []

    def format(self, prompt: str, suffix: str = '') -> str:

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

        return prompt
    
    def format_infill(self, prefix: str, suffix: str = '') -> str:

        raise RuntimeError(f'Infill not supported.')
    
    def extra_eos(self) -> list[str]:

        return self.extra_eos_tokens
    

class DialoGPTPrompt(GenericPrompt):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.eos_token = '<|endoftext|>'

    def format_chat(self, prompt: str) -> str:

        return prompt + self.eos_token
    


class StarCoderPrompt(GenericPrompt):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'infill'

        self.prefix_token = '<fim_prefix>'
        self.suffix_token = '<fim_suffix>'
        self.middle_token = '<fim_middle>'

    def format_infill(self, prefix: str, suffix: str = '') -> str:

        return self.prefix_token + prefix + self.suffix_token + suffix + self.middle_token
    
    
# Starchat prompt modeling (see https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py)
# See also FastChat (/https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#817) but current
# implementation is slightly wrong: it does not format the system prompt correctly (misses a line return)
# see https://github.com/lm-sys/FastChat/issues/2220
class StarChatPrompt(GenericPrompt):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_token = '<|system|>'
        self.system_prompt = ''
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'
        self.sep_token = '<|end|>'
        self.extra_eos_tokens = [self.end_turn_token]

    
    def format_chat(self, prompt: str) -> str:

        return self.system_token + '\n' + self.system_prompt + self.sep_token + '\n' + self.user_token + '\n' + \
            + prompt + self.sep_token + '\n' + self.assistant_token + '\n'
    


class Codegen2Prompt(GenericPrompt):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'infill'

        self.mask_token = '<mask_1>'
        self.eos_token = '<|endoftext|>'
        self.sep_token = '<sep>'
        self.extra_eos_tokens = ['<eom>']

    def format_infill(self, prefix: str, suffix: str = '') -> str:

        return prefix + self.mask_token + suffix + self.eos_token + self.sep_token + self.mask_token
    

# Vicuna 1.3 prompt modeling (https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py)
class VicunaPrompt(GenericPrompt):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_prompt = ''
        self.user_token = 'USER'
        self.assistant_token = 'ASSISTANT'
        self.sep_token = ' '

    
    def format_chat(self, prompt: str) -> str:

        if self.system_prompt == '':
            return self.user_token + ': ' + prompt + self.sep_token + self.assistant_token + ':'
        else:
            return self.system_prompt + self.sep_token + self.user_token + ': ' + prompt + self.sep_token + \
                self.assistant_token + ':'


# Llama2-chat prompt modeling (https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
# See also FastChat (https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L123) but current
# implementation is slightly wrong: it does not add a space between the first prompt and [/INST] token
# see https://github.com/lm-sys/FastChat/issues/2220
class Llama2ChatPrompt(GenericPrompt):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_prompt = ''
        self.system_template = '<<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
        self.user_token = '[INST]'
        self.assistant_token = '[/INST]'
        self.sep_token = ' '

    def format_chat(self, prompt: str) -> str:

        # System prompt must be embedded in first user prompt
        embedded_prompt = self.system_template.format(system_prompt=self.system_prompt) + prompt
        # Note that we do not call strip() as meta does in source code, because some prompts explicitly end with '\n'
        return self.user_token + ' ' + embedded_prompt + self.sep_token + self.assistant_token

    