import os
from abc import ABC

#from torch.utils.data import Dataset

from helpers import utils
#from TextWiz.textwiz import GenericConversation

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

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval', 'HumanEval.jsonl')
    id_key: str = 'task_id'


class HumanEvalInstruct(SampleDataset):
    """Class representing the HumanEval_Instruct dataset.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval', 'HumanEval_Instruct.jsonl')
    id_key: str = 'task_id'


class HumanEvalPHP(SampleDataset):
    """Class representing the MutiPL-E variation of the HumanEval dataset for the PHP language.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval', 'HumanEval_php.jsonl')
    id_key: str = 'task_id'


class HumanEvalCPP(SampleDataset):
    """Class representing the MutiPL-E variation of the HumanEval dataset for the C++ language.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval', 'HumanEval_cpp.jsonl')
    id_key: str = 'task_id'


class HumanEvalRust(SampleDataset):
    """Class representing the MutiPL-E variation of the HumanEval dataset for the Rust language.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'HumanEval', 'HumanEval_rs.jsonl')
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

    path: str = os.path.join(utils.DATA_FOLDER, 'AATK', 'AATK.jsonl')
    id_key: str = 'id'


class AATKInstructChatGPT(SampleDataset):
    """Class representing the automatic and Python-only version of the Asleep At The Keyboard (AATK) benchmark.
    Also contains the prompts in natural language (instruct).
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'AATK', 'AATK_instruct_chatGPT.jsonl')
    id_key: str = 'id'


class AATKInstructZephyr(SampleDataset):
    """Class representing the automatic and Python-only version of the Asleep At The Keyboard (AATK) benchmark.
    Also contains the prompts in natural language (instruct).
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'AATK', 'AATK_instruct_zephyr.jsonl')
    id_key: str = 'id'


class AATKInstructLlama3(SampleDataset):
    """Class representing the automatic and Python-only version of the Asleep At The Keyboard (AATK) benchmark.
    Also contains the prompts in natural language (instruct).
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'AATK', 'AATK_instruct_llama3.jsonl')
    id_key: str = 'id'


# class WalliserDeutschDataset(Dataset):

#     def __init__(self, tokenizer, template: GenericConversation | None = None, sample_size: int = 8192):

#         self.path = os.path.join(utils.DATA_FOLDER, 'walliser_deutsch', 'clean_data.txt')
#         self.texts = utils.load_txt(self.path, separator='\n\n\n\n')

#         # Remove whitespaces
#         self.texts = [x.strip() for x in self.texts]

#         # Apply conv template if any
#         if template is not None:
#             for i in range(len(self.texts)):
#                 template.erase_conversation()
#                 template.append_user_message(
#                     ("Please generate a short (100-150 words) news story about an event that "
#                      "took place in the Swiss canton of Valais, in the local language - Walliser Deutsch.")
#                 )
#                 template.append_model_message(self.texts[i])
#                 self.texts[i] = template.get_prompt()

#         # Remove whitespaces once again
#         self.texts = [x.strip() for x in self.texts]

#         # Tokenize
#         tokenized_texts = [tokenizer(text) for text in self.texts]

#         # Split if any text is longer than sample_size
#         self.tokenized_texts = []
#         for sample in tokenized_texts:
#             current_size = len(sample['input_ids'])
#             truncated_samples = [{k: v[i:i+sample_size] for k,v in sample.items()} for i in range(0, current_size, sample_size)]
#             self.tokenized_texts.extend(truncated_samples)
        
#         # Add labels to samples -> This is not needed when using DataCollatorForLanguageModeling as it will
#         # be added automatically
#         # self.tokenized_texts = [{'labels': sample['input_ids'].copy(), **sample} for sample in self.tokenized_texts]

    # def __len__(self) -> int:
    #     return len(self.tokenized_texts)
    
    # def __getitem__(self, index: int | slice) -> dict | list[dict]:
    #     return self.tokenized_texts[index]
            

class CyberSecEvalInstruct(SampleDataset):
    """Class representing the original instruct version of CyberSecEval dataset.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'CyberSecEval', 'instruct.jsonl')
    # Dummy id_key
    id_key: str = 'None'

    def samples_by_id(self) -> dict[str, dict]:
        """Maps the task ids to the tasks themselves.
        """
        return NotImplementedError('`samples_by_id` is not implemented for CyberSecEval datasets.')
    

class CyberSecEvalInstructLlama3(CyberSecEvalInstruct):
    """Class representing the Llama3-reformulated instruct version of CyberSecEval dataset.
    """

    path: str = os.path.join(utils.DATA_FOLDER, 'CyberSecEval', 'instruct_llama3.jsonl')


class CyberSecEvalInstructLlama3_PythonOnly(CyberSecEvalInstructLlama3):
    """Class representing the Llama3-reformulated instruct version of CyberSecEval dataset, with only the Python
    problems.
    """

    def __init__(self):
        super().__init__()
        self.samples = [sample for sample in self.samples if sample['language'] == 'python']
    