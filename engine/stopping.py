import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import StoppingCriteria

# If we reach one of these patterns, it means that the model has finished generating the solution as a 
# function and continues useless generation (basically stop words used in the Codex/HumanEval 
# paper: https://arxiv.org/pdf/2107.03374.pdf). Should only be used when the prompt is a function definition.
CODE_STOP_PATTERNS = (
    '\nclass',
    '\ndef',
    '\n#',
    '\nif',
    '\nprint',
    '\n@'
)


class TextPatternStopping(StoppingCriteria):

    def __init__(self, prompt_ids_length: int, tokenizer: PreTrainedTokenizerBase,
                 stop_patterns: tuple[str] = CODE_STOP_PATTERNS):

        super().__init__()
        self.prompt_ids_length = prompt_ids_length
        self.tokenizer = tokenizer
        self.stop_patterns = stop_patterns
        # Used to retain which sequences are done being generated for speed if we require a lot of sequences
        # to be generated in a single pass
        # self.memoization = {}


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        outputs = input_ids[:, self.prompt_ids_length:]
        generated_sequences = self.tokenizer.batch_decode(outputs)
        done_sequences = []

        for sequence in generated_sequences:
            done = any([pattern in sequence for pattern in self.stop_patterns])
            done_sequences.append(done)

        return all(done_sequences)



def post_process_sequences(generated_sequences: list[str], prompt:str,
                           stop_patterns: tuple[str] = CODE_STOP_PATTERNS) -> list[str]:
    """Post-process the outputs of a model to truncate according to a list of patterns upon which we stop
    generation (this is needed because the StoppingCriteria cannot immediately stop the generation of each
    sequence upon meeting a pattern in the case of more than 1 `num_return_sequences`).

    Parameters
    ----------
    generated_sequences : list[str]
        Decoded outputs of a model.
    prompt : str
        The prompt used for generation.
    stop_patterns : list[str], optional
        The list of patterns to use to stop generation, by default CODE_STOP_PATTERNS

    Returns
    -------
    list[str]
        The truncated outputs to meet the criteria of the stopping patterns.
    """

    prompt_length = len(prompt)
    generated_sequences_curated = []
    
    for sequence in generated_sequences:

        if sequence.startswith(prompt):
            sequence = sequence[prompt_length:]
            reattach_prompt = True
        else:
            reattach_prompt = False
        
        stop_index = len(sequence)

        # Scan the sequence for each pattern, and return the minimum index such that none of the patterns are
        # in the sequence
        for pattern in stop_patterns:
            index = sequence.find(pattern)
            if index != -1:
                stop_index = min(stop_index, index)

        curated_sequence = prompt + sequence[0:stop_index] if reattach_prompt else sequence[0:stop_index]
        generated_sequences_curated.append(curated_sequence)

    return generated_sequences_curated