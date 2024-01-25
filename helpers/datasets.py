import os
from abc import ABC

from helpers import utils

# TODO: Maybe inherit from a torch Dataset, but for now no need
class SampleDataset(ABC):
    """Base class for dataset consisting of a serie of samples (dictionaries) with an id. We define it as
    an ABC because it cannot be instantiated (path attribute does not exist and is needed for __init__).
    """

    # Should always be overriden in subclasses
    path: str
    id_key: str

    def __init__(self):

        # Load all dataset in memory since it is small
        self.samples = utils.load_jsonl(self.path)

    def __len__(self):

        return len(self.samples)
    
    def __getitem__(self, key: int | slice) -> dict[str, str] | list[dict[str, str]]:

        return self.samples[key]
    
    def __iter__(self):
        """Create a simple generator over the samples.
        """

        for i in range(len(self)):
            yield self[i]

    def samples_by_id(self) -> dict[str, dict]:
        """Maps the task ids to the tasks themselves.
        """

        return {task[self.id_key]: task for task in self}


class HumanEval(SampleDataset):
    """Class representing the HumanEval dataset.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval.jsonl')
    id_key: str = 'task_id'


class HumanEvalInstruct(SampleDataset):
    """Class representing the HumanEval_Instruct dataset.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval_Instruct.jsonl')
    id_key: str = 'task_id'


class HumanEvalPHP(SampleDataset):
    """Class representing the MutiPL-E variation of the HumanEval dataset for the PHP language.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval_php.jsonl')
    id_key: str = 'task_id'


class HumanEvalCPP(SampleDataset):
    """Class representing the MutiPL-E variation of the HumanEval dataset for the C++ language.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval_cpp.jsonl')
    id_key: str = 'task_id'


class HumanEvalRust(SampleDataset):
    """Class representing the MutiPL-E variation of the HumanEval dataset for the Rust language.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval_rs.jsonl')
    id_key: str = 'task_id'


# Dataset mapping from dataset name to actual dataset to use for evaluation
HUMANEVAL_DATASETS_MAPPING = {
    'HumanEval': HumanEval,
    'HumanEvalInstruct': HumanEvalInstruct,
    'HumanEvalPHP': HumanEvalPHP,
    'HumanEvalCPP': HumanEvalCPP,
    'HumanEvalRust': HumanEvalRust,
}

# Mapping from extension to dataset name for MultiPL-E
MULTIPLE_LANGUAGE_MAPPING = {
    'php': 'HumanEvalPHP',
    'cpp': 'HumanEvalCPP',
    'rs': 'HumanEvalRust',
}


class AATK(SampleDataset):
    """Class representing the automatic and Python-only version of the Asleep At The Keyboard (AATK) benchmark.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'AATK.jsonl')
    id_key: str = 'id'


class AATKEnglish(SampleDataset):
    """Class representing the automatic and Python-only version of the Asleep At The Keyboard (AATK) benchmark.
    Also contains the prompts in natural language (english).
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'AATK_english.jsonl')
    id_key: str = 'id'


class AATKEnglishV2(SampleDataset):
    """Class representing the automatic and Python-only version of the Asleep At The Keyboard (AATK) benchmark.
    Also contains the prompts in natural language (english).
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'AATK_english_v2.jsonl')
    id_key: str = 'id'


