import os
import pathlib
from typing import List
import pytest
import mlflow
from pytest import TempPathFactory
from pydantic import BaseSettings
import absl.logging
import pandas as pd

from occrplib.document_classification.document_classifier import DocumentClassifier
from occrplib.feature_extraction.feature_extractor import FeatureExtractor
from occrplib.model.image_model import ImageModel


@pytest.fixture(scope="session")
def tmp_path_session(tmp_path_factory: TempPathFactory) -> pathlib.Path:
    tmp_path_session = tmp_path_factory.mktemp("temporary")
    print(f"Make temp path: {tmp_path_session}")
    mlflow.set_tracking_uri(tmp_path_session)
    mlflow.set_experiment("pytest")
    return tmp_path_session


# Image Model
# TODO need to test other model architectures
# TODO need to test GPU = None vs. GPU = 1
@pytest.fixture(scope="session")
def image_model(tmp_path_session: pathlib.Path) -> DocumentClassifier:
    model_name = "AlexNet"

    document_classifier = ImageModel(
        model_name="AlexNet",
        output_path=tmp_path_session,
        input_images_path="",
        # n_epochs=5,
        # batch_size=25,
        # load_temp_weights_path="",
        # GPU_mb_memory=38_000,
        # dropout_param=0.0,
        **settings.MODEL_CONFIG_TRANSFER_LEARNING[model_name],
        **settings.PROCESSOR_SETTINGS,
    )
    document_classifier.n_labels = 5
    document_classifier.model_input_path = tmp_path_session
    return document_classifier


# Document Classifiers
# TODO need to test other model architectures
@pytest.fixture(scope="session")
def document_classifier(tmp_path_session: pathlib.Path) -> DocumentClassifier:
    document_classifier = DocumentClassifier(model_name="AlexNet")
    document_classifier.n_labels = 5
    document_classifier.model_input_path = tmp_path_session
    return document_classifier


@pytest.fixture(scope="session")
def document_classifier1(document_classifier: DocumentClassifier, tmp_path_session: pathlib.Path) -> DocumentClassifier:
    """Enable category others is True."""

    document_classifier.ENABLE_CATEGORY_OTHERS = True
    return document_classifier


@pytest.fixture(scope="session")
def document_classifier2(document_classifier: DocumentClassifier, tmp_path_session: pathlib.Path) -> DocumentClassifier:
    """Enable category others is False."""

    document_classifier.ENABLE_CATEGORY_OTHERS = False
    return document_classifier


# Feature Extractors
@pytest.fixture(scope="session")
def feature_extractor(tmp_path_session: pathlib.Path) -> FeatureExtractor:
    feature_extractor = FeatureExtractor(model_name="AlexNet")
    feature_extractor.n_labels = 5
    feature_extractor.model_input_path = tmp_path_session
    return feature_extractor


# Feature Extractors
@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    train_df: pd.DataFrame = pd.read_csv("tests/test_data/train_df")
    return train_df


# config
class Settings(BaseSettings):
    def __init__(self) -> None:
        super().__init__()

    IN_DOCKER = False

    ROOT_PATH = "/data" if IN_DOCKER else "/data/dssg/occrp/data"

    RVL_LABELS: str = os.path.join(ROOT_PATH, "input/rvl-cdip/labels/")
    RVL_IMAGES: str = os.path.join(ROOT_PATH, "input/rvl-cdip/images/")
    # RVL_LABELS_CSV: str = os.path.join(RVL_LABELS, "labels.csv")

    RAW_DOCUMENTS_TRANSFER_LEARNING: str = os.path.join(ROOT_PATH, "input", "document_classification_clean")
    INPUT_IMAGES_TRANSFER_LEARNING: str = os.path.join(ROOT_PATH, "processed_clean")
    INPUT_IMAGES_FIRSTPAGE_CLASSIFICATION: str = os.path.join(
        ROOT_PATH, "processed_firstpages_vs_middle_pages", "processed_clean"
    )  # NOQA E501
    WEB_TEST_DATA_PATH = os.path.join(ROOT_PATH, "testing_data/web")

    # Weights of the feature extraction model.
    # MODEL_FEATURE_EXTRACTION: str = os.path.join(ROOT_PATH, "output/feature_extraction/AlexNet_2022_08_09-20_52_48_dropout0.1_fulltrained/weights/epoch10")  # NOQA E501

    CLASSIFIER_TRAIN_TEST_SPLIT = os.path.join(
        ROOT_PATH, "output/document_classifier/AlexNet_2022_08_12-13_11_53/model_inputs"
    )

    # # Paths output
    DATA_PATH_LOGS: str = os.path.join(ROOT_PATH, "logs/")
    MLFLOW_LOGS = os.path.join(ROOT_PATH, "mlruns")
    mlflow.set_tracking_uri("file://" + MLFLOW_LOGS)

    OUTPUT_FEATURE_SELECTION: str = os.path.join(ROOT_PATH, "output/feature_extraction/")
    OUTPUT_TRANSFER_LEARNING: str = os.path.join(ROOT_PATH, "output/document_classifier/")

    ENABLE_CATEGORY_OTHERS = False

    PROCESSOR_SETTINGS: dict = {
        # "processor_type": "GPU",   # GPU or CPU
        "n_CPU": 16,  # From 0 to 16.
        "GPU_id": None if IN_DOCKER else None,  # GPU_id = 0 or 1 or None
        "GPU_mb_memory": 38_000,  # Amount of GPU memory in MB
    }

    MODEL_CONFIG_FEATURE_EXTRACTION: dict = {
        "VGG16": {"im_size": 224, "n_epochs": 10, "batch_size": 32, "as_gray": False, "load_temp_weights_path": None},
        "AlexNet": {"im_size": 227, "n_epochs": 10, "batch_size": 25, "as_gray": False, "load_temp_weights_path": None},
        "AlexNetBW": {"im_size": 227, "n_epochs": 10, "batch_size": 25, "as_gray": True, "load_temp_weights_path": None},
        "ResNet50": {
            "im_size": 227,
            "n_epochs": 10,
            "batch_size": 25,
            # "n_temp_save": 20,
            "as_gray": False,
            "load_temp_weights_path": None,
        },
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
            "as_gray": True,
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
            "n_epochs": 3,
            "batch_size": 25,
            "n_temp_save": 20,
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
        "VGG16": {
            "im_size": 224,
            "n_epochs": 200,
            "batch_size": 32,
            "as_gray": False,
            "load_temp_weights_path": None,
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/VGG16_2022_08_10-09_31_18_plain_fulltrained/weights/epoch08"
            ),
        },
        "AlexNet": {
            "im_size": 227,
            "n_epochs": 200,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/AlexNet_2022_08_10-07_48_48_plain_fulltrained/weights/epoch04"
            ),
        },
        "AlexNetDropout": {
            "im_size": 227,
            "n_epochs": 200,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/AlexNet_2022_08_09-20_52_48_dropout0.1_fulltrained/weights/epoch10"
            ),
        },
        "AlexNetBW": {
            "im_size": 227,
            "n_epochs": 200,
            "batch_size": 25,
            "as_gray": True,
            "load_temp_weights_path": None,
            "model_feature_extractor": "",
        },
        "ResNet50": {
            "im_size": 227,
            "n_epochs": 200,
            "batch_size": 25,
            "as_gray": False,
            "load_temp_weights_path": None,
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/ResNet50_2022_08_10-14_03_21_plain_semitrained/weights/epoch05"
            ),
        },
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
        "EfficientNetB7": {
            "im_size": 227,
            "n_epochs": 200,
            "batch_size": 25,
            # "n_temp_save": 20,
            "as_gray": False,
            "load_temp_weights_path": None,
            "model_feature_extractor": os.path.join(
                ROOT_PATH, "output/feature_extraction/EfficientNetB7_2022_08_10-23_01_11_plain_semitrained/weights/epoch07"
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
        # "middle-page",
        "passport-scan",
        "receipts",
        "shipping-receipts",
        # "transcripts",
    ]

    LABELS_FILTER_PRIMARYPAGES: List[str] = [
        "firstpages",
        "middlepages_1233",
    ]

    # To ignore some verbose warnings during training
    # https://stackoverflow.com/a/69107989/15972591
    absl.logging.set_verbosity(absl.logging.ERROR)

    ENABLE_CATEGORY_OTHERS = True
    WEB_TEST_DATA_PATH = os.path.join(ROOT_PATH, "testing_data/web")


settings = Settings()
