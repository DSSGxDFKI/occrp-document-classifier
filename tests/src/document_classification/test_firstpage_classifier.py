import pathlib
import pytest
from keras.models import Sequential
import mlflow

from occrplib.document_classification.primarypage_classifier import PrimaryPageClassifier
import conftest  # NOQA F501


@pytest.fixture(scope="session")
def firstpage_classifier(tmp_path_session: pathlib.Path) -> PrimaryPageClassifier:
    firstpage_classifier = PrimaryPageClassifier(model_name="AlexNet")
    mlflow.set_tracking_uri(tmp_path_session)
    mlflow.set_experiment("pytest")
    firstpage_classifier.n_labels = 5
    firstpage_classifier.model_input_path = tmp_path_session
    return firstpage_classifier


################################################################


def test_load_transfer_learning_model_type(firstpage_classifier: PrimaryPageClassifier) -> None:
    firstpage_classifier.load_transfer_learning_model()
    assert type(firstpage_classifier.model) == Sequential


def test_load_transfer_learning_model_activation_and_loss(firstpage_classifier: PrimaryPageClassifier) -> None:
    firstpage_classifier.load_transfer_learning_model()
    assert firstpage_classifier.model.loss == "categorical_crossentropy"
    assert firstpage_classifier.model.layers[-1].activation.__name__ == "softmax"
    assert firstpage_classifier.model.layers[-1].name == "output_transfer_learning"


def test_load_transfer_learning_model_layers(firstpage_classifier: PrimaryPageClassifier) -> None:
    firstpage_classifier.load_transfer_learning_model()
    assert len(firstpage_classifier.model.layers) == 21  # should have 21
