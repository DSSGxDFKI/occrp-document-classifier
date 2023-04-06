import os
import pathlib
import pytest
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator

from occrplib.document_classification.document_classifier import DocumentClassifier
import conftest  # NOQA F501


def test_load_transfer_learning_model_type(document_classifier1: DocumentClassifier) -> None:
    document_classifier1.load_transfer_learning_model()
    assert type(document_classifier1.model) == Sequential


def test_load_transfer_learning_model_activation_and_loss(document_classifier1: DocumentClassifier) -> None:
    document_classifier1.load_transfer_learning_model()
    assert document_classifier1.model.loss == "binary_crossentropy"
    assert document_classifier1.model.layers[-1].activation.__name__ == "sigmoid"


def test_load_transfer_learning_model_activation_and_loss2(document_classifier2: DocumentClassifier) -> None:
    document_classifier2.load_transfer_learning_model()
    assert document_classifier2.model.loss == "categorical_crossentropy"
    assert document_classifier2.model.layers[-1].activation.__name__ == "softmax"


def test_split_train_val_test_labels(document_classifier1: DocumentClassifier) -> None:
    document_classifier1.split_train_val_test_labels()
    for df in [document_classifier1.train_df, document_classifier1.val_df, document_classifier1.test_df]:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        for column_name in [
            "path",
            "class",
            "filename",
            "page_number",
            "index",
            "label",
            "directory",
            "doc type",
            "doc type str",
        ]:
            assert column_name in document_classifier1.val_df


def test_save_train_test_split(tmp_path_session: pathlib.Path, document_classifier1: DocumentClassifier) -> None:
    document_classifier1.split_train_val_test_labels()
    document_classifier1.save_train_test_split()
    for filename, df in [
        ("train.txt", document_classifier1.train_df),
        ("val.txt", document_classifier1.val_df),
        ("test.txt", document_classifier1.test_df),
    ]:
        assert os.path.isfile(os.path.join(tmp_path_session, filename)) is True


def test_set_train_test_generators(document_classifier: DocumentClassifier) -> None:
    document_classifier.set_train_test_generators()
    assert type(document_classifier.original_datagen) == ImageDataGenerator
    assert type(document_classifier.data_gen_x7) == ImageDataGenerator


def test_training_loop(document_classifier: DocumentClassifier) -> None:
    document_classifier.split_train_val_test_labels()
    document_classifier.set_train_val_test_data_iterators()
    document_classifier.load_transfer_learning_model()

    document_classifier.training_loop()

    document_classifier.saved_hist


def test_run_tfexplain(document_classifier1: DocumentClassifier) -> None:
    # TODO testing this function once we refactored it as a function instead of method
    pass


if __name__ == "__main__":
    pytest.main(["--disable-pytest-warnings"])
