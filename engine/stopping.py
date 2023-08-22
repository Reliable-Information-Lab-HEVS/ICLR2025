import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria

from engine.code_parser import PythonParser

# If we reach one of these patterns, it means that the model has finished generating the solution as a 
# function and continues useless generation (basically stop words used in the Codex/HumanEval 
# paper: https://arxiv.org/pdf/2107.03374.pdf). Should only be used when the prompt is a python function definition.
CODE_STOP_PATTERNS = (
    '\nclass',
    '\ndef',
    '\n#',
    '\nif',
    '\nprint',
    '\n@'
)

# Extended code stopping patterns. This is mostly useful for chat models which often output code blocks 
# starting with ">>>" to show examples
EXTENDED_CODE_STOP_PATTERNS = CODE_STOP_PATTERNS + (
    '\n>>>'
)


class TextPatternStopping(StoppingCriteria):

    def __init__(self, prompt_ids_length: int, tokenizer: PreTrainedTokenizerBase,
                 stopping_patterns: list[str] | tuple[str] | None = CODE_STOP_PATTERNS,
                 extra_eos_tokens: list[str] | None = None, parser: PythonParser | None = None):

        super().__init__()
        self.prompt_ids_length = prompt_ids_length
        self.tokenizer = tokenizer
        self.stopping_patterns = stopping_patterns
        self.extra_eos_tokens = extra_eos_tokens
        self.parser = parser
        self.all_patterns = tuple()

        # Add stopping_patterns to the tuple of patterns if there are any
        if self.stopping_patterns is not None and len(self.stopping_patterns) > 0:
            self.all_patterns += tuple(self.stopping_patterns)
        # Add extra eos to the tuple of patterns if there are any
        if self.extra_eos_tokens is not None and len(self.extra_eos_tokens) > 0:
            self.all_patterns += tuple(self.extra_eos_tokens)

        if len(self.all_patterns) == 0:
            raise ValueError('You did not provide any patterns or extra eos tokens upon which to stop generation.')
        
        
    def __repr__(self):
        return f'TextPatternStopping{*self.all_patterns,}'
    
    def __str__(self):
        return f'{*self.all_patterns,}'

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        outputs = input_ids[:, self.prompt_ids_length:]
        generated_sequences = self.tokenizer.batch_decode(outputs)
        if self.parser is not None:
            generated_sequences = [self.parser(sequence) for sequence in generated_sequences]
        done_sequences = []

        for sequence in generated_sequences:
            done = any([pattern in sequence for pattern in self.all_patterns])
            done_sequences.append(done)

        return all(done_sequences)



def post_process_stopping_patterns(prompt_truncated_generated_sequences: list[str],
                                   stopping_patterns: list[str] | tuple[str] | None = CODE_STOP_PATTERNS) -> list[str]:
    """Post-process the outputs of a model to truncate according to a list of patterns upon which we stop
    generation (this is needed because the StoppingCriteria cannot immediately stop the generation of each
    sequence upon meeting a pattern in the case of more than 1 `num_return_sequences`).

    Parameters
    ----------
    prompt_truncated_generated_sequences : list[str]
        Decoded PROMPT-TRUNCATED outputs of a model. Passing the full decoded outputs may induce errors in the logic.
    stopping_patterns : list[str] | tuple[tr], optional
        The list of patterns to use to stop generation, by default CODE_STOP_PATTERNS

    Returns
    -------
    list[str]
        The truncated outputs to meet the criteria of the stopping patterns.
    """

    # If there are no stopping patterns
    if stopping_patterns is None or len(stopping_patterns) == 0:
        return prompt_truncated_generated_sequences

    generated_sequences_curated = []
    
    for sequence in prompt_truncated_generated_sequences:
        
        stop_index = len(sequence)

        # Scan the sequence for each pattern, and return the minimum index such that none of the patterns are
        # in the sequence
        for pattern in stopping_patterns:
            index = sequence.find(pattern)
            if index != -1:
                stop_index = min(stop_index, index)

        curated_sequence = sequence[0:stop_index]
        generated_sequences_curated.append(curated_sequence)

    return generated_sequences_curated



def post_process_extra_eos_tokens(prompt_truncated_outputs: torch.Tensor, pad_token_id: int,
                                  extra_eos_tokens_ids: list[int] | None) -> torch.Tensor:
    """Process the outputs of a model to convert all tokens that were generated after an extra eos to 
    `pad_token_id`. This way, everything after the extra eos will be ignored when calling
    tokenizer.batch_decode(..., skip_special_tokens=True) later.

    NOTE: if the original tokenizer.eos_token is found at some point in the generated sequence, all
    subsequent tokens are set to tokenizer.pad_token automatically so we don't need to add tokenizer.eos_token
    to extra_eos_tokens.

    Parameters
    ----------
    prompt_truncated_outputs : torch.Tensor
        The PROMPT-TRUNCATED output of a model. Passing the full outputs may induce errors in the logic.
    pad_token_id : int
        The id of the pad token.
    extra_eos_tokens_ids : list[int] | None
        The list of extra eos tokens ids.

    Returns
    -------
    torch.Tensor
        The modified output.
    """

    # If there are no extra eos tokens
    if extra_eos_tokens_ids is None or len(extra_eos_tokens_ids) == 0:
        return prompt_truncated_outputs
    
    outputs = prompt_truncated_outputs.clone().detach()

    for i, sequence_ids in enumerate(prompt_truncated_outputs):

        stop_index = len(sequence_ids)

        # Scan the sequence for each eos, and set all subsequent ids to pad_token_id
        for eos_ids in extra_eos_tokens_ids:
            nonzero = torch.nonzero(sequence_ids == eos_ids)
            if len(nonzero) != 0:
                stop_index = min(stop_index, int(nonzero[0][0]))

        # Everything after the first extra eos is set to pad_token
        outputs[i, stop_index:] = pad_token_id

    return outputs



def post_process_sequences(prompt_truncated_outputs: torch.Tensor, tokenizer: PreTrainedTokenizerBase,
                           stopping_patterns: list[str] | tuple[str] | None = CODE_STOP_PATTERNS,
                           extra_eos_tokens: list[str] | None = None) -> list[str]:
    """Apply all steps of post-processing to the prompt-truncated outputs of a model.

    Parameters
    ----------
    prompt_truncated_outputs : torch.Tensor
        The PROMPT-TRUNCATED output of a model. Passing the full outputs may induce errors in the logic.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used by the model.
    stopping_patterns : list[str] | tuple[tr], optional
        The list of patterns to use to stop generation, by default CODE_STOP_PATTERNS
    extra_eos_tokens : list[str] | None, optional
        The list of extra eos tokens, by default None

    Returns
    -------
    list[str]
        The post-processed generated sequences.
    """
    
    extra_eos_tokens_ids = tokenizer.convert_tokens_to_ids(extra_eos_tokens)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Check if we find any of the extra eos tokens. We first look for extra eos in this way so that we
    # can later use tokenizer.batch_decode(..., skip_special_tokens=True), i.e. easily remove all the 
    # special tokens
    
    processed_outputs = post_process_extra_eos_tokens(prompt_truncated_outputs, pad_token_id, extra_eos_tokens_ids)
    prompt_truncated_sequences = tokenizer.batch_decode(processed_outputs, skip_special_tokens=True)
    final_sequences = post_process_stopping_patterns(prompt_truncated_sequences, stopping_patterns)

    return final_sequences