# LLMs

This is the main repo containing all the work on code generation by LLMs. It is mostly based on my library [TextWiz](https://github.com/Cyrilvallez/TextWiz) for model inference.

## Install

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