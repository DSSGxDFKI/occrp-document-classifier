```
occrp-document-classifier
│   README.md
│   Dockerfile                      : Dockerized project
│   gpu.Dockerfile            : Dockerized project with GPU
│   Pipfile                         : project requirements
│   run_gpu_model.sh                : script to run CLI command in GPU
│   portainer.sh                    : script to launch a Portainer instance
│   run_models_in_sequence.sh       : script to run multiple commands in sequence
|   Cheatsheet.md                   : useful commands
│   ...
│
└───src
│   │   main.py                     : Command Line Interface (CLI)
│   │   config.py                   : settings of the project
│   │
│   └───feature_extraction
│   │   │   feature_extractor.py    : class for feature extractor model
│   │
│   └───document_classification
│   │   │   document_classifier.py  : class for multiclass document classifier model
│   │   │   primarypage_classifier.py : class for binary primary page classifier model
│   │
│   └───model
│   │   │   image_model.py          : Abstract class that defines a machine learning image model
│   │
│   └───prediction                  
│   │   │   data_load.py            : data loading utils
│   │   │   predict.py              : functions to perform prediction in new data (pdfs/images) based on trained classifiers
│   │
│   └───preprocessing
│   │   │   convert_to_img.py       : convert PDF or TIFF pages to JPG
│   │   │   import_data.py          : data import utils
│   │   │   train_val_test_split.py : split the dataset into train, validation and test sets for training
│   │
│   └───utils
│       │   confusion_matrix.py     : generate a confusion 
│       │   gaussian.py             : computes the classifier thresholds using Gaussian fitting
│       │   helper.py               : command line utils
│       │   logger.py               : logging utils
│   
└───tests                           : Unit testing
│   │   ...
│
└───notebooks                       : Proof of concepts of models and features, assessments
│   │   ...
|
└───docs                            : Documentation
│   │   ...
```
