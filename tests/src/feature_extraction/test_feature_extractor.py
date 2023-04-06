# TODO testing
from keras.models import Sequential, Functional
import pathlib

from occrplib.feature_extraction.feature_extractor import FeatureExtractor
from occrplib.document_classification.document_classifier import DocumentClassifier
import conftest  # NOQA F501


def test_get_data_iterator(feature_extractor: FeatureExtractor) -> None:
    # already tested in parent class
    NotImplementedError


def test_setup_model(feature_extractor: FeatureExtractor) -> None:
    feature_extractor.setup_model()
    assert type(feature_extractor.model) == Sequential or type(feature_extractor.model) == Functional


def test_load_temp_weights(
    feature_extractor: FeatureExtractor, document_classifier: DocumentClassifier, tmp_path_session: pathlib.Path
) -> None:
    # with no path
    feature_extractor.load_weights_path = None
    feature_extractor.load_temp_weights()
    if feature_extractor.load_weights_path is not None:
        assert type(feature_extractor.model) == Sequential or type(feature_extractor.model) == Functional

    # with correct path
    feature_extractor.load_weights_path = document_classifier.load_temp_weights_path
    feature_extractor.load_temp_weights()
    if feature_extractor.load_weights_path is not None:
        assert type(feature_extractor.model) == Sequential or type(feature_extractor.model) == Functional
