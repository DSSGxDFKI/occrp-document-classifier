# FAQ
## How do I train a document classifier?

### 1. Set up the config

In order to train models, the training data should be located in the right place. That configuration is set in the config file. Please read the [config README](./config.md) to see more details.
### 2. Use the correct parameters

Also in the config file, you need to set the correct parameters for the model you are training. This is done in the dictionary `MODEL_CONFIG_TRANSFER_LEARNING`. 

```python
MODEL_CONFIG_TRANSFER_LEARNING: dict = {
        ...
        "EfficientNetB4": {
            # image size
            "im_size": 227,

            # number of epochs of the training
            "n_epochs": 200,

            # size of the training batch
            "batch_size": 25,

            # convert to black and white images
            "as_gray": False,

            # load previous trained model (optional, not necessary)
            "load_temp_weights_path": None,

            # load a feature extractor model
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/EfficientNetB4_2022_08_12-08_19_20/weights/best_model"
            ),
        },
        ...
}
```

### 2. Run the CLI

You can train a model contained in the `MODEL_CONFIG_TRANSFER_LEARNING` dict calling it by the name in the CLI. For example, to train a multiclass document classifier usin the EfficientNetB4 example configuration, we pass it the name as an argument of the command:
```
python src/main.py train-document-classifier --model-name EfficientNetB4
```

To train a binary primary page classifier, the idea is the same but with a different command:
```
python src/main.py train-primarypage-classifier --model-name EfficientNetB4
```

As both classifier are necessary to perform predictions on new data, we recommend to train both at the same time with this command:
```
python src/main.py train-full-classifier --model-name EfficientNetB4
```

The trained binary classifier will be saved in folder `ROOT_PATH/output/firstpage_classification` and the multiclass classifier will be saved in folder `ROOT_PATH/output/document_classifier`. Also both models can be accessed through MLFlow.

A trained classifier has this folder structure:
```
EfficientNetB4_2022_08_18-14_28_38  # the folder name contains the model name and the training date
├── assessment      : markdown reports for validation and testing, csv of predictions
├── model_inputs
│   ├── train.txt   : training split
│   ├── val.txt     : validation split
│   ├── test.txt    : testing split
│   └── labels.csv  : class labels and their integer index
├── tfexplain       : tfexplain library output
├── weights         : trained keras model
│   ├── best model          
```

## How do I predict using a trained model?

### 1. Register your model
Once you have trained multiclass and binary classifiers, they need to be registered in the config file. For that, you need to add a new key in the `PREDICTION_MODELS` dictionary of the config file with the name of the model, and as value a dictionary containing two keys: `binary_classifier` with the path to the trained binary classifier and `multiclass_classifier` with the path to the trained multiclass classifier. Here is the example of the EfficientNetB4 trained model:

```python
PREDICTION_MODELS = {
    "EfficientNetB4": {
        "binary_classifier": os.path.join(ROOT_PATH, "output/firstpage_classification/EfficientNetB4_binary_final"),
        "multiclass_classifier": os.path.join(ROOT_PATH, "output/document_classifier/EfficientNetB4_multiclass_final")
    },
        ...
}
```

We offer by default three trained models: EfficientNetB0, EfficientNetB4, and EfficientNetB4BW.

### 2. Run the CLI

Once the config is set wit the prediction model, the [CLI](./CLI.md) is ready to be used. To predict the class of documents there is a command `predict` that takes a directory of documents and outputs a JSON file with the prediction.

For example, we may have a folder with two pdf documents
```
/home/john/documents_to_predict
│   doc1.pdf  # three pages
│   doc2.pdf  # one page
```
And we want to predict their class and save the output in /home/john/prediction.

To do that, you cna call the `predict` command:
```bash
python src/main.py predict /home/john/documents_to_predict /home/john/prediction
```
This will save a prediction json with a timestamp in the filename with this format:
```json
# /home/john/prediction/prediction__2022_08_18_12_11_13.json
{
    "/home/john/documents_to_predict/doc1.pdf": ["gazette", "other", "other"],
    "/home/john/documents_to_predict/doc2.pdf": ["company-registry"],
}
Where each list contains the prediction for al-l the pages of the document.
```


## How do I add a new class to the training?

### 1. Add folder in the right directories

If you want to add a new class and train a new classifier, first you need yo have a folder with training images for that class. For example, if you have a folder with emails in JPG:
```
emails
├── email1.jpg
├── email2.jpg
├── email3.jpg
| ...  
```

You can copy that folder to `ROOT_PATH/processed_clean/document_classifier` to add that class to the multiclass classifier.


On the other hand, if you want to train with documents that are not converted to JPG, for example:
```
emails
├── email1.png
├── email2.pdf
├── email3.tiff
| ...  
```

Then you should copy that folder to `ROOT_PATH/input/document_classification_clean` to add that class to the multiclass classifier. It will get automatically converted to JPG in `ROOT_PATH/processed_clean/document_classifier`.

Also you may want to classify this new class as a 'primarypage' or relevant page. To to that, add the JPG files in the folder `ROOT_PATH/processed_clean/firstpage_classifier/firstpages`.


### 2. Set the new class in the config

The `LABELS_FILTER` list in the config file contains all the classes supported. To include a new one, just write the name of the class in the list. It should be the same name of the folder containing the class data.

```python
LABELS_FILTER: List[str] = [
        "bank-statements",
        "company-registry",
        "contracts",
        "court-documents",
        "gazettes",
        "invoices",
        "passport-scan",
        "receipts",
        "shipping-receipts",
        "emails"  # New class
    ]
```
### 3. Training

After the previous steps, you are ready to train a classifier including your new class:
```
python src/main.py train-full-classifier --model-name EfficientNetB4
```

## How do I add a new class model architecture?

### 1. Add a new architecture function
To implement a new classifier neural architecture, you need to create a module in `src/model/architectures` containing a function that returns the desired new architecture (Keras model). Then you need to add that function to the `MODELS` dictionary in the `src/model/architectures/architecture_dictionary.py` module.
 
### 2. Set the parameters for feature extraction training
In order to train the new architecture, you need to add a dictionary in the `MODEL_CONFIG_FEATURE_EXTRACTION` dictionary in the `src/config.py` file containing the training parameters. For example:

```python
MODEL_CONFIG_FEATURE_EXTRACTION: dict = {
    ...
    "NewModel": {
            "im_size": 227,
            "n_epochs": 10,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,
        }
}
```
Now you can train the feature extractor model necessary to train the classifiers:
```bash
python src/main.py train-feature-extraction --model-name NewModel
```

The trained feature extractor can be found in `ROOT_PATH/output/feature_extraction`.

### 3. Set the parameters for classifier training

To train document classifiers based in the new feature extractor, you need to add that config in the `MODEL_CONFIG_TRANSFER_LEARNING` dictionary of the config file. 

```python
MODEL_CONFIG_TRANSFER_LEARNING: dict = {
    ...
    "NewModel": {
            "im_size": 227,
            "n_epochs": 10,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,

            # load a feature extractor model: here use the path of the feature extractor trained in the previous step
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/NewModel__ ..."
            ),
        }
}
```
After this, you should be able to train documents classifiers with your new architecture.

## Where do I find the cheatsheet?

For a list of handy commands see here: [Cheatsheet](./Cheatsheet.md).