import os

from helpers import utils

# TODO: Maybe inherit from a torch Dataset, but for now no need
class HumanEval(object):
    """Class representing the HumanEval dataset.
    """

    def __init__(self):

        self.path = os.path.join(utils.DATA_FOLDER, 'HumanEval.jsonl')
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
        """Maps the task_ids to the tasks themselves.
        """

        return {task['task_id']: task for task in self}
    


class HumanEvalInstruct(HumanEval):
    """Class representing the HumanEval_Instruct dataset.
    """

    def __init__(self):
        # Simply overwrite attributes
        self.path = os.path.join(utils.DATA_FOLDER, 'HumanEval_Instruct.jsonl')
        self.samples = utils.load_jsonl(self.path)


class HumanEvalPHP(HumanEval):
    """Class representing the MutiPL-E variation of the HumanEval dataset for the PHP language.
    """

    def __init__(self):
        # Simply overwrite attributes
        self.path = os.path.join(utils.DATA_FOLDER, 'HumanEval_php.jsonl')
        self.samples = utils.load_jsonl(self.path)



# Dataset mapping from dataset name to actual dataset to use for evaluation
DATASETS_MAPPING = {
    'HumanEval': HumanEval,
    'HumanEvalInstruct': HumanEvalInstruct,
    'HumanEvalPHP': HumanEvalPHP,
}