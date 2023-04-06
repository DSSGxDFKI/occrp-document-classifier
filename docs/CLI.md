# `python src/main.py`

**Usage**:

```console
$ python src/main.py [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--log TEXT`: Set logging level [default: INFO]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `predict`: Predict documents classes.
* `train-document-classifier`: Train the multiclass document classifier.
* `train-feature-extraction`: Train the document feature extractor model.
* `train-primarypage-classifier`: Train the primary page binary classifier.
* `train-full-classifier`: Train the multiclass and the binary classifiers.

## `python src/main.py predict`

Predict the documents contained in the INPUT_PATH using the model MODEL_NAME
and outputs the prediction in OUTPUT_PATH in a json format

**Usage**:

```console
$ python src/main.py predict [OPTIONS] INPUT_PATH OUTPUT_PATH
```

**Arguments**:

* `INPUT_PATH`: [required]
* `OUTPUT_PATH`: [required]

**Options**:

* `--model-name TEXT`: [default: EfficientNetB4]
* `--help`: Show this message and exit.

## `python src/main.py train-document-classifier`

Train the document classifier with OCCRP data using the specifications in the config.py file.

**Usage**:

```console
$ python src/main.py train-document-classifier [OPTIONS]
```

**Options**:

* `--model-name TEXT`: The classifier model to be trained  [default: EfficientNetB4]
* `--verbose / --no-verbose`: [default: False]
* `--help`: Show this message and exit.

## `python src/main.py train-feature-extraction`

Train the document feature extractor model with a dataset such as RVL-CDIP.

**Usage**:

```console
$ python src/main.py train-feature-extraction [OPTIONS]
```

**Options**:

* `--model-name TEXT`: The feature extraction model to be trained  [default: EfficientNetB4]
* `--verbose / --no-verbose`: [default: False]
* `--help`: Show this message and exit.

## `python src/main.py train-primarypage-classifier`

Train the primary page classifier.

**Usage**:

```console
$ python src/main.py train-primarypage-classifier [OPTIONS]
```

**Options**:

* `--model-name TEXT`: The classifier model to be trained  [default: EfficientNetB4]
* `--verbose / --no-verbose`: [default: False]
* `--help`: Show this message and exit.

## `python src/main.py train-full-classifier`

Train the multiclass and the binary classifier using the MODEL_NAME architecture.
    

**Usage**:

```console
$ python src/main.py train-full-classifier [OPTIONS]
```

**Options**:

* `--model-name TEXT`: The classifier model to be trained  [default: EfficientNetB4]
* `--verbose / --no-verbose`: [default: False]
* `--help`: Show this message and exit.
