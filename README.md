# OCCRP Document Classifier

This is a Python library to perform document classification for [OCCRP Aleph](aleph.occrp.org/). It allows to train and test a classifier that can predict the type of a document.

This project was developed during [Data Science for Social Good fellowship](https://sebastian.vollmer.ms/dssg/) in Kaiserslautern, Germany in the summer of 2022. The fellowship is funded by Rhineland Palatinate Technical University at Kaiserslautern-Landau, in cooperation with the German Research Centre for Artificial Intelligence (DFKI GmbH).

# Table of Contents
- [OCCRP Document Classifier](#occrp-document-classifier)
- [Table of Contents](#table-of-contents)
- [Quick Start](#quick-start)
  - [1. Clone the repo](#1-clone-the-repo)
  - [2. Select a root path](#2-select-a-root-path)
  - [3. Create the folder structure using script](#3-create-the-folder-structure-using-script)
  - [4. Install requirements](#4-install-requirements)
  - [5. Run the CLI](#5-run-the-cli)
- [Installation](#installation)
  - [Python CLI](#python-cli)
  - [Docker](#docker)
  - [Docker GPU](#docker-gpu)
- [Usage](#usage)
  - [Prediction](#prediction)
- [Mlflow](#mlflow)
- [TODOs and ENHANCEMENTS](#todos-and-enhancements)
- [Contributors](#contributors)


# Quick Start

## 1. Clone the repo

```bash
git clone <repo-url>
cd <repo-directory>
```

## 2. Select a root path

In [config.py](./src/config.py):

```python
ROOT_PATH = "/data" if IN_DOCKER else "/data/dssg/occrp/data"
```

Replace `/data/dssg/occrp/data` with any path of your local filesystem.

You also need to replace that path in the Docker volumes defined in [run_gpu_model.sh](./run_gpu_model.sh) and in [run_models_in_sequence.sh](./run_models_in_sequence.sh):

```bash
    # replace /data/dssg/occrp/data/ with the selected ROOT_PATH
    -v /data/dssg/occrp/data:/data \
    -v /data/dssg/occrp/data/:/data/dssg/occrp/data/ \
```

## 3. Create the folder structure using script

Once you selected a ROOT_PATH, all the subdirectories necessary to run the project can be created using the [init_data_structure.sh](./init_data_structure.sh) script passing the selected ROOT_PATH as an argument:

```bash
./init_data_structure.sh "<ROOT_PATH>"
```

The repository should contain a `data.zip` file in order to run the init script. Use quotes to pass the root path as an argument, especially if it contains spaces.

Once the script run is finished, running `tree -L 3` in the ROOT_PATH should show the following folder structure:
```
ROOT_PATH
├── input
│   ├── document_classification_clean
│   │   ├── bank-statements
│   │   ├── company-registry
│   │   ├── contracts
│   │   │   ...
│   └── rvl-cdip
│   │   ├── images
│   │   ├── labels
├── logs
├── mlruns
├── output
│   ├── document_classifier
│   ├── feature_extraction
│   └── firstpage_classification
└── processed_clean
    ├── document_classifier
    │   ├── bank-statements
    │   ├── company-registry
    │   ├── contracts
    │   │   ...
    └── firstpage_classifier
        ├── firstpages
        └── middlepages_1233
```
That's it! You just have setted all the necessary data to train models.

## 4. Install requirements

```bash
# install the dependencies
pipenv install

# activate the environment
pipenv shell
```

## 5. Run the CLI

```bash
# check if the CLI is working
python src/main.py --help

```

# Installation

## Python CLI

This projects uses Python 3.8 and [pipenv](https://pypi.org/project/pipenv/) to manage its dependencies. The list of requirements is available in the [Pipfile](./Pipfile). To install the requirements:

```bash
# install the dependencies
pipenv install

# activate the environment
pipenv shell
```

After this, you should be able to run any of the commands available in the [CLI](./docs/CLI.md).

## Docker

The [Dockerfile](./Dockerfile) allows to run the CLI in a Docker container with Python. See the [Docker README](./docs/Docker.md) to find instructions of how to use it.

## Docker GPU

The [gpu.Dockerfile](./gpu.Dockerfile) allows to run the CLI in a Docker container with GPU out of the box using the [NVIDIA drivers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow). See the [GPU README](./docs/GPU.md) to find instructions of how to use it.

# Usage
 
Basic command line usage:

 ```console
python src/main.py [OPTIONS] COMMAND [ARGS]...
```

More information about the commands available can be obtained using `python src/main.py --help` or in the [Command Line Interface README](./docs/CLI.md).

## Prediction

This project comes with default trained classifier models to be used out of the box. To do that, just select a `INPUT_PATH` directory with documents to classify and run:

```bash
python src/main.py predict INPUT_PATH OUTPUT_PATH
```

A json file with the results of the prediction will be saved in  
`OUTPUT_PATH/prediction__%Y_%m_%d_%H_%M_%S.json`

More details can be found in the [FAQ](./docs/FAQ.md)

# Mlflow

Find an experiment in the UI via the hash

If you have a MLflow hash, e.g. from the config, and want to know how to find it in the UI:

1. Navigate to the mlruns directory
2. `find -name <hash>`, returns something like `./1/0a5006859f154daebc7a697d190f7a2`. The first number is the `experiment_id`, 
the second one is the `run_id`. 
3. Navigate to the ML UI by replacing the experiment_id and run_id: `http://127.0.0.1:5000/#/experiments/<experiment_id>/runs/<run_id>`

Find more information about how to use MLflow in the [Cheatsheet](./docs/Cheatsheet.md).

# TODOs and ENHANCEMENTS

We used Visual Studio Code for development and the extensions [ToDo Tree](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree) by Gruntfuggly. We configured it in a way that distinguishes `TODO`s and `ENHANCEMENT`s. `TODO` for us means that important work is still required, `ENHANCEMENT` means that a certain improvement is desired but optional

To make this work as in our settings, install ToDo Tree in VS Code. Add this to your `.vscode/settings.json` (which is not in the repository):

```json
    "todo-tree.highlights.customHighlight": {
        "ENHANCEMENT": {
            "icon": "note",
            "foreground": "black",
            "background": "lightgreen",
            "iconColour": "gray",
        },
    },
    "todo-tree.general.tags": [
        "TODO",
        "ENHANCEMENT",
    ],
```

# Contributors

**Fellows**: José Miguel Cordero, Theresa Henn, Frederik H., and José Sánchez Gómez

**Technical Mentor**: Diego Arenas

**Project Manager**: Michael Brill

**Project Partner**: Organized Crime and Corruption Reporting Project (OCCRP)
