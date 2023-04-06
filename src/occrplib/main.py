import datetime
import json
import logging
import os

import typer

from occrplib.feature_extraction.feature_extractor import FeatureExtractor
from occrplib.document_classification.document_classifier import DocumentClassifier
from occrplib.document_classification.primarypage_classifier import PrimaryPageClassifier
from occrplib.prediction.predict import predict_from_directory
from occrplib.utils.logger import LOGGING_FILENAME, LOGGING_FORMAT, LOGGING_LEVELS

logger = logging.getLogger(__name__)
cli = typer.Typer()


@cli.command()
def train_feature_extraction(
    model_name: str = typer.Option("EfficientNetB4", help="The feature extraction model to be trained"),
    verbose: bool = False,
) -> None:
    """
    Train the document feature extractor model with a dataset such as RVL-CDIP.
    """
    feature_extractor = FeatureExtractor(model_name, verbose)
    feature_extractor.mlflow_train_pipeline()


@cli.command()
def train_document_classifier(
    model_name: str = typer.Option("EfficientNetB4", help="The classifier model to be trained"),
    verbose: bool = False,
    yes_to_all_user_input: bool = False,
) -> None:
    """
    Train the document classifier with OCCRP data using the specifications in the config.py file.
    """
    document_classifier = DocumentClassifier(model_name, verbose, yes_to_all_user_input)
    document_classifier.mlflow_train_pipeline()


@cli.command()
def train_primarypage_classifier(
    model_name: str = typer.Option("EfficientNetB4", help="The classifier model to be trained"),
    verbose: bool = False,
    yes_to_all_user_input: bool = False,
) -> None:
    """
    Train the primary page classifier.
    """
    primarypage_classifier = PrimaryPageClassifier(model_name, verbose, yes_to_all_user_input)
    primarypage_classifier.mlflow_train_pipeline()


@cli.command()
def train_full_classifier(
    model_name: str = typer.Option("EfficientNetB4", help="The classifier model to be trained"),
    verbose: bool = False,
) -> None:
    """Train the multiclass and the binary classifier using the MODEL_NAME architecture."""
    logger.info("Training multiclass document classifier")
    document_classifier = DocumentClassifier(model_name, verbose)
    document_classifier.mlflow_train_pipeline()

    logger.info("Training binary primary page classifier")
    primarypage_classifier = PrimaryPageClassifier(model_name, verbose)
    primarypage_classifier.mlflow_train_pipeline()


@cli.command()
def predict(input_path: str, output_path: str, model_name: str = "EfficientNetB4") -> None:
    """Predict the documents contained in the INPUT_PATH using the model MODEL_NAME
    and outputs the prediction in OUTPUT_PATH in a json format
    """
    prediction = predict_from_directory(input_path, model_name)
    prediction_filename = datetime.datetime.now().strftime(r"prediction__%Y_%m_%d_%H_%M_%S.json")
    with open(os.path.join(output_path, prediction_filename), "w") as prediction_file:
        json.dump(prediction, prediction_file, ensure_ascii=False, indent=4)


@cli.callback()
def main(log: str = typer.Option("INFO", help="Logging level")) -> None:
    """Train and predict a document classifier."""

    logging.basicConfig(
        format=LOGGING_FORMAT,
        level=LOGGING_LEVELS[log.upper()],
        handlers=[logging.FileHandler(LOGGING_FILENAME), logging.StreamHandler()],
    )


if __name__ == "__main__":
    cli()
