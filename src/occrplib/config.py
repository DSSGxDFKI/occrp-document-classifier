import json
import os
from typing import List

import mlflow

from pydantic import BaseSettings
import absl.logging


class Settings(BaseSettings):
    def __init__(self) -> None:
        super().__init__()

    IN_DOCKER = os.environ.get("IN_DOCKER", False)

    # Root path for all data input/output of the project
    ROOT_PATH = "/data" if IN_DOCKER else "data"

    # RVL-CDIP dataset paths
    RVL_LABELS: str = os.path.join(ROOT_PATH, "input/rvl-cdip/labels/")
    RVL_IMAGES: str = os.path.join(ROOT_PATH, "input/rvl-cdip/images/")

    # Raw (PDF, TIFF, PNG, JPG) documents for multiclass document classifier.
    # The documents in this path will be converted to JPG in INPUT_IMAGES_TRANSFER_LEARNING path
    # It should contain folders with the classes to be trained on
    RAW_DOCUMENTS_TRANSFER_LEARNING: str = os.path.join(ROOT_PATH, "input", "document_classification_clean")

    # JPG files to train the multiclas document classifier, divided in class folders
    INPUT_IMAGES_TRANSFER_LEARNING: str = os.path.join(ROOT_PATH, "processed_clean", "document_classifier")

    # JPG files to train the binary primarypage classifier, divided in class folders
    INPUT_IMAGES_PRIMARYPAGE_CLASSIFICATION: str = os.path.join(
        ROOT_PATH, "processed_clean", "firstpage_classifier"
    )  # NOQA E501

    # # Paths output
    DATA_PATH_LOGS: str = os.path.join(ROOT_PATH, "logs/")
    MLFLOW_LOGS = os.path.join(ROOT_PATH, "mlruns")
    mlflow.set_tracking_uri("file://" + MLFLOW_LOGS)

    # Path to output trained feature extractors
    OUTPUT_FEATURE_SELECTION: str = os.path.join(ROOT_PATH, "output/feature_extraction/")

    # Path to output trained documents classifiers
    OUTPUT_TRANSFER_LEARNING: str = os.path.join(ROOT_PATH, "output/document_classifier/")

    # Path to output trained primarypage classifiers
    OUTPUT_PRIMARYPAGE_CLASSIFICATION: str = os.path.join(ROOT_PATH, "output/firstpage_classification/")

    # CPU or GPU configuration
    PROCESSOR_SETTINGS: dict = {
        "n_CPU": 16,  # Number of CPUs to be used
        "GPU_id": None,  # GPU_id = 0 or 1 or None. When None, CPU is used
        "GPU_mb_memory": 38_000,  # Amount of GPU memory in MB
    }

    DISABLE_FEATURE_EXTRACTOR = True

    MODEL_CONFIG_FEATURE_EXTRACTION: dict = {
        "VGG16": {"im_size": 224, "n_epochs": 10, "batch_size": 32, "as_gray": False, "load_temp_weights_path": None},
        "AlexNet": {"im_size": 227, "n_epochs": 10, "batch_size": 25, "as_gray": False, "load_temp_weights_path": None},
        "AlexNetBW": {"im_size": 227, "n_epochs": 10, "batch_size": 25, "as_gray": True, "load_temp_weights_path": None},
        "ResNet50": {"im_size": 227, "n_epochs": 10, "batch_size": 25, "as_gray": False, "load_temp_weights_path": None},
        "EfficientNetB0": {
            "im_size": 227,
            "n_epochs": 10,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,
        },
        "EfficientNetB0BW": {
            "im_size": 227,
            "n_epochs": 10,
            "batch_size": 25,
            "as_gray": True,
            "load_temp_weights_path": None,
        },
        "EfficientNetB4": {
            "im_size": 227,
            "n_epochs": 10,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,
        },
        "EfficientNetB4BW": {
            "im_size": 227,
            "n_epochs": 10,
            "batch_size": 25,
            "as_gray": True,
            "load_temp_weights_path": None,
        },
        "EfficientNetB7": {
            "im_size": 227,
            "n_epochs": 10,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": "/data/output/feature_extraction/EfficientNetB7_2022_08_10-23_01_11/weights/epoch07",
        },
        "regularization_settings": {
            "dropout_param": 0.0,
            "regularization_param": 0.0,
            "cnn_layer_filters_value": 128,
        },
    }

    MODEL_CONFIG_TRANSFER_LEARNING: dict = {
        "EfficientNetB0": {
            "im_size": 227,
            "n_epochs": 200,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/EfficientNetB0_2022_08_10-18_54_40_plain_fulltrained/weights/epoch10"
            ),
        },
        "EfficientNetB4": {
            "im_size": 227,
            "n_epochs": 200,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/EfficientNetB4_2022_08_12-08_19_20/weights/best_model"
            ),
        },
        "EfficientNetB4BW": {
            "im_size": 227,
            "n_epochs": 200,
            "batch_size": 25,
            "as_gray": True,
            "load_temp_weights_path": None,
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/EfficientNetB4BW_2022_08_15-12_32_47/weights/epoch07"
            ),
        },
    }

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
    ]

    LABELS_FILTER_PRIMARYPAGES: List[str] = [
        "firstpages",
        "middlepages_1233",
    ]

    # To ignore some verbose warnings during training
    # https://stackoverflow.com/a/69107989/15972591
    absl.logging.set_verbosity(absl.logging.ERROR)

    # Prediction
    PREDICTION_MODELS = {
        "EfficientNetB4": {
            "binary_classifier": os.path.join(ROOT_PATH, "output/document_classifier/EfficientNetB4_binary_final"),
            "multiclass_classifier": os.path.join(ROOT_PATH, "output/document_classifier/EfficientNetB4_multiclass_final"),
        },
        "EfficientNetB4BW": {
            "binary_classifier": os.path.join(ROOT_PATH, "output/document_classifier/EfficientNetB4BW_binary_final"),
            "multiclass_classifier": os.path.join(ROOT_PATH, "output/document_classifier/EfficientNetB4BW_multiclass_final"),
        },
        "EfficientNetB0": {
            "binary_classifier": os.path.join(ROOT_PATH, "output/document_classifier/EfficientNetB0_binary_final"),
            "multiclass_classifier": os.path.join(ROOT_PATH, "output/document_classifier/EfficientNetB0_multiclass_final"),
        },
    }


settings = Settings()


def export_settings_to_json(export_path: str, settings: Settings = settings) -> None:
    """Export the current settings into a json file

    Args:
        export_path (str): path to save the settings
    """

    with open(export_path, "w", encoding="utf-8") as export_file:
        json.dump(vars(settings), export_file, ensure_ascii=False, indent=4)
    mlflow.log_artifact(export_path)
