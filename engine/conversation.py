import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from engine import loader
from helpers import utils


class Conversation(object):
    """Class used to store a conversation with a model."""

    def __init__(self):

        self.user_history_text = []
        self.model_history_text = []


    def __len__(self):
        return len(self.user_history_text)
    
    
    def __iter__(self):
        """Create a generator over (user_input, model_answer) tuples for all turns in the conversation.
        """

        # Generate over copies so that the object in the class cannot change during iteration
        for user_history, model_history in zip(self.user_history_text.copy(), self.model_history_text.copy()):
            yield user_history, model_history
    

    def __str__(self):

        N = len(self)

        if N == 0:
            return "The conversation is empty."
        else:
            out = ''
            for i, (user, model) in enumerate(self):
                out += f'>> User: {user}\n'
                out += f'>> Model: {model}'
                if i < N - 1:
                    out += '\n\n'
            return out
        

    def update_conversation(self, user_input: str, model_output: str):
        """Update the conversation history with a new prompt and answer by a model.

        Parameters
        ----------
        user_input : str
            Prompt to the model.
        model_output : str
            Answer of the model.
        """

        self.user_history_text.append(user_input)
        self.model_history_text.append(model_output)
    

    def erase_conversation(self):
        """Reinitialize the conversation.
        """

        self.user_history_text = []
        self.model_history_text = []
        



def tokenize_for_conversation(tokenizer: PreTrainedTokenizerBase, conversation: Conversation,
                              prompt: str) -> torch.Tensor:
    """Tokenize a `conversation` and a new `prompt` according to `tokenizer`.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
       The tokenizer to use.
    conversation : Conversation
        The conversation object keeping past inputs and answers.
    prompt : str
        The new prompt to the model.

    Returns
    -------
    torch.Tensor
        The input to pass to the model.
    """

    # After a close inspection of source code
    # (https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/pipelines/conversational.py#L18)
    # for conversational pipeline and all tokenizers, this seems to be the accepted way to treat inputs for
    # conversation with a model. This is the DialoGPT way of handling conversation, but is in fact reused by
    # all other tokenizers that we use.

    input_ids = []

    for user_history, model_history in conversation:
        # Tokenize first user input and add eos token
        input_ids.extend(tokenizer.encode(user_history, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            input_ids.append(tokenizer.eos_token_id)

        # Tokenize model response and add eos token
        input_ids.extend(tokenizer.encode(model_history, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            input_ids.append(tokenizer.eos_token_id)

    # Tokenize prompt and add eos token
    input_ids.extend(tokenizer.encode(prompt, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)

    # Truncate conversation if it is larger than model capacity
    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[-tokenizer.model_max_length:]

    # Convert to Pytorch
    input_ids = torch.tensor([input_ids], dtype=torch.int64)

    return input_ids



def generate_conversation(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str,
                          conv_history: Conversation | None, max_new_tokens: int = 60, do_sample: bool = True,
                          top_k: int = 40, top_p: float = 0.90, temperature: float = 0.9,
                          seed: int | None = None, input_device: int | str = 0) -> Conversation:
    """Generate a turn in a conversation with a `model`. To mimic a conversation, we simply append all past
    prompts and model responses to the current prompt.

    Parameters
    ----------
    model : PreTrainedModel
        The model to converse with.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use to process the input and output text.
    prompt : str
        The user input to the conversation.
    conv_history : Conversation | None
        The current conversation state. Pass `None` to start a conversation from scatch.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.
    do_sample : bool, optional
        Whether to introduce randomness in the generation, by default True.
    top_k : int, optional
        How many tokens with max probability to consider for randomness, by default 50.
    top_p : float, optional
        The probability density covering the new tokens to consider for randomness, by default 0.92.
    temperature : float, optional
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness), by default 0.9.
    seed : int | None, optional
        An optional seed to force the generation to be reproducible.
    input_device : int | str, optional
        The device on which to put the inputs, by default 0.


    Returns
    -------
    Conversation
        A Conversation object keeping track of past inputs/responses, as well as the current full model input ids.
    """
    
    if seed is not None:
        utils.set_all_seeds(seed)

    # Check that the history is not empty
    if conv_history is None:
        conv_history = Conversation()

    input = tokenize_for_conversation(tokenizer, conv_history, prompt)
    if torch.cuda.is_available():
        input = input.to(device=input_device)

    # Suppress pad_token_id warning
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Generate model response
    with torch.no_grad():
        output = model.generate(input, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k,
                                top_p=top_p, temperature=temperature, pad_token_id=pad_token_id)
        
    model_answer_ids = output[0, input.shape[-1]:]
    model_answer_text = tokenizer.decode(model_answer_ids, skip_special_tokens=True)
    conv_history.update_conversation(prompt, model_answer_text)

    return conv_history



class HFChatModel(object): 
    """Class encapsulating a HuggingFace model and its tokenizer to generate text in a conversational fashion. 
    """

    def __init__(self, model_name: str, quantization: bool = False, device_map: str | None = None,
                 gpu_rank: int = 0, dtype: torch.dtype | None = None):

        self.model, self.tokenizer = loader.load_model_and_tokenizer(model_name, quantization=quantization,
                                                                     device_map=device_map, gpu_rank=gpu_rank,
                                                                     dtype=dtype)
        self.model_name = model_name
        self.quantization = quantization
        # The model is on multiple devices
        try:
            self.device_map = self.model.hf_device_map
            devices = set(self.device_map.values())
            # remove possible non-gpu devices from the device_map
            devices = {val for val in devices if not isinstance(val, str)}
            self.input_device = min(devices) if len(devices) > 0 else 'cpu'
        # The model is on a single device
        except AttributeError:
            device = next(self.model.parameters()).get_device()
            self.device_map = 'cpu' if device == -1 else f'cuda:{device}'
            self.input_device = 'cpu' if device == -1 else device
        self.dtype = self.model.dtype


    def __repr__(self) -> str:
        return f'HFChatModel({self.model_name}, {self.quantization}, {self.device_map})'
    
    def __str__(self) -> str:
        if self.quantization:
            return f'{self.model_name} model, quantized 8 bits version. Used for conversation.'
        else:
            return f'{self.model_name} model, original (not quantized) version. Used for conversation.'
        

    def __call__(self, prompt: str, conv_history: Conversation | None, max_new_tokens: int = 60,
                 do_sample: bool = True, top_k: int = 40, top_p: float = 0.90, temperature: float = 0.9,
                 seed: int | None = None, gpu_rank: int = 0) -> Conversation:
        
        return generate_conversation(self.model, self.tokenizer, prompt, conv_history,
                                     max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k,
                                     top_p=top_p, temperature=temperature, seed=seed, gpu_rank=gpu_rank)