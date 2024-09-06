# LLMs

This is the main repo containing all the work on code generation by LLMs. It is mostly based on my library [TextWiz](https://github.com/Cyrilvallez/TextWiz) for model inference.

## Install

### Clone the repo

As `TextWiz` is a submodule inside this repository, it is not possible to simply clone this repo the usual way. One also has to initialize the submodule with the `--recurse_submodules` flag: 

```sh
git clone https://github.com/Cyrilvallez/LLMs.git --recurse-submodules
```

One also has to pass the flag whenever pulling upstream changes in order to keep the submodule in sync in case it was also modified:

```sh
git pull --recurse-submodules
```

Or you can also run the following command once:

```sh
git config submodule.recurse true
```

And it will automatically pass the flag whenever you simply `git pull` from the remote.

### Download results

Due to the file size, the results are not included in this repository. To download them, run the following:

```sh
cd LLMs
python3 download_results.py
```

and everything will be added in the correct location.

## Python environment

In case you need to install Conda and are on **linux**, you can run

```sh
source config.sh
```

which will install [mini-forge](https://github.com/conda-forge/miniforge), and create the required environment. In case you already have Conda installed, simply run:

```sh
conda env create -f requirements.yaml
```

to create the computing environment.

## Add models to the library

To add models to the library, one first has to add the appropriate `model.py` file in `TextWiz/textwiz/configs/causal` folder. For examples of what the file should contain, please refer to existing files in the repository - the structure is quite sttraightforward (you need to specify model name, default dtype, number of parameters, choose a family name for the model, default context size, and optionally some additional arguments/version requirements).  

Then, if one of the models you added requires a specific prompt/chat template, navigate to `TextWiz/textwiz/templates`, and modify both `conversation_template.py` and `prompt_template.py`. In both files, you will find a dictionary mapping(`CONVERSATION_MAPPING` and `PROMPT_MAPPING` respectively) between model names and prompt/conversation template class. You should add your model names to these mapping (creating appropriate class if necesary, see class examples in each file) to use proper templates.

## Add datasets to the library

All datasets in this repository (except one) are formatted as `jsonl` files, that is files with one `json` record per line. Each of these `json` record represents a sample of the dataset, and all have the same keys. To add a dataset, add (or create) the new `jsonl` file representing the data somewhere in the `data` folder. Then, navigate to the `helpers/datasets.py` file, and add a new class corresponding to your dataset in the following way:

```python
class YourDataset(SampleDataset):
    """New dataset!!
    """

    # This is where you added your `jsonl`file
    path: str = os.path.join(utils.DATA_FOLDER, 'new_dataset.jsonl')
    # If your dataset has a key corresponding to an ID for each sample, add it here (otherwise set it to whatever string, such as "" or "None")
    id_key: str = 'task_id'
```

You can then easily manipulate the dataset through a `YourDataset` instance.