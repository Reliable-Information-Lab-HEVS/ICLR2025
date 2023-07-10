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
    
    def __getitem__(self, key: int | slice) -> dict[str, str]:

        return self.samples[key]
    
    def __iter__(self):
        """Create a simple generator over the samples.
        """

        for i in range(len(self)):
            yield self[i]