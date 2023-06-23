import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import loader
import utils


class Conversation(object):
    """Class used to store a conversation with a model."""

    def __init__(self):

        self.user_history_text = []
        self.model_history_text = []
        self.total_conversation_ids = torch.tensor([[]], dtype=torch.int64)

    def __len__(self):
        return len(self.user_history_text)
    
    def __str__(self):

        if len(self.user_history_text) == 0:
            return "The conversation is empty."
        else:
            out = ''
            for i in range(len(self.user_history_text)):
                out += f'>> User: {self.user_history_text[i]}\n'
                out += f'>> Model: {self.model_history_text[i]}\n\n'
            return out
        

    def update_conversation(self, user_input: str, model_output: str, total_output_ids: torch.Tensor):
        """Update the conversation history with a new prompt and answer by a model.

        Parameters
        ----------
        user_input : str
            Prompt to the model.
        model_output : str
            Answer of the model.
        total_output_ids : torch.Tensor
            New token ids representation of the full conversation.
        """

        self.user_history_text.append(user_input)
        self.model_history_text.append(model_output)
        self.total_conversation_ids = total_output_ids


    def concatenate_history_and_prompt(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Create the new input to feed the model consisting of the conversation history and the new prompt.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            The token ids corresponding to the new prompt.

        Returns
        -------
        torch.Tensor
            The total input to feed to the model.
        """

        return torch.cat([self.total_conversation_ids, prompt_ids], dim=-1)
    

    def erase_conversation(self):
        """Reinitialize the conversation.
        """

        self.user_history_text = []
        self.model_history_text = []
        self.total_conversation_ids = torch.tensor([[]], dtype=torch.int64)




def generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str, max_new_tokens: int = 60, do_sample: bool = True,
                  top_k: int = 40, top_p: float = 0.90, temperature: float = 0.9, num_return_sequences: int = 1, seed: int | None = None) -> list[str]:
    """Generate text according to `prompt` using the `model` and `tokenizer` specified.

    Parameters
    ----------
    model : PreTrainedModel
        The model used to generate the text.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use to process the input and output text.
    prompt : str
        The prompt to the model.
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
    num_return_sequences : int, optional
        How many sequences to generate according to the `prompt`, by default 1.
    seed : int | None, optional
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    list[str]
        List containing all `num_return_sequences` sequences generated.
    """
    
    if seed is not None:
        utils.set_all_seeds(seed)

    inputs = tokenizer(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k,
                                top_p=top_p, temperature=temperature, num_return_sequences=num_return_sequences)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)   

    return generated_text



def generate_conversation(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str, conv_history: Conversation | None, max_new_tokens: int = 60,
                          do_sample: bool = True, top_k: int = 40, top_p: float = 0.90, temperature: float = 0.9, seed: int | None = None) -> Conversation:
    """Generate a turn in a conversation with a `model`. To mimic a conversation, we simply append all past prompts and model responses to the current prompt.

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

    # encode the prompt (we only need the input_ids since we are not running inference on batches)
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
    # Concatenate the prompt to the history
    input = conv_history.concatenate_history_and_prompt(prompt_ids)

    # Generate model response
    if torch.cuda.is_available():
        input = input.to('cuda')
    with torch.no_grad():
        output = model.generate(input, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k,
                                top_p=top_p, temperature=temperature)
        
    model_answer_ids = output[0, input.shape[-1]:]
    model_answer_text = tokenizer.decode(model_answer_ids, skip_special_tokens=True)
    conv_history.update_conversation(prompt, model_answer_text, output)

    return conv_history
    
    

def load_and_generate_text(model_name: str, prompt: str, quantization: bool = False, max_new_tokens: int = 60,
                           do_sample: bool = True, top_k: int = 100, top_p: float = 0.92, temperature: float = 0.9,
                           num_return_sequences: int = 1, seed: int | None = None) -> list[str]:
    """Load a model and its tokenizer and generate text according to `prompt`.

    Parameters
    ----------
    model_name : str
        The model used to generate the text.
    prompt : str
        The prompt to the model.
    quantization : bool, optional
        Whether to load the model in 8 bits mode to save memory, by default False.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.
    do_sample : bool, optional
        Whether to introduce randomness in the generation, by default True.
    top_k : int, optional
        How many tokens with max probability to consider for randomness, by default 100.
    top_p : float, optional
        The probability density covering the new tokens to consider for randomness, by default 0.92.
    temperature : float, optional
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness), by default 0.9.
    num_return_sequences : int, optional
        How many sequences to generate according to the `prompt`, by default 1.
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    list[str]
        List containing all `num_return_sequences` sequences generated.
    """

    model, tokenizer = loader.load_model_and_tokenizer(model_name, quantization)

    return generate_text(model, tokenizer, prompt, max_new_tokens, do_sample, top_k,
                          top_p, temperature, num_return_sequences, seed)