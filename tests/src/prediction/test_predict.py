import os
import shutil
from pathlib import Path
from typing import Any, Generator, List, Tuple

import pytest

from occrplib.document_classification.document_classifier import DocumentClassifier
from occrplib.document_classification.primarypage_classifier import PrimaryPageClassifier
from occrplib.prediction.predict import (
    load_classifiers,
    predict_documents,
    predict_from_directory,
    predict_image,
    predict_images,
    predict_pdf,
    predict_pdfs,
    predict_tiff,
    predict_tiffs,
)


@pytest.fixture(params=["EfficientNetB0", "EfficientNetB4"])
def model_name(request: Any) -> str:
    return request.param


@pytest.fixture
def jpg_image_path() -> List[str]:
    return [
        "tests/test_data/prediction/jpg/image_1.jpg",
        "tests/test_data/prediction/jpg/image_2.jpg",
        "tests/test_data/prediction/jpg/image_3.jpg",
    ]


@pytest.fixture
def pdf_path_pagecount() -> List[Tuple[str, int]]:
    return [
        ("tests/test_data/prediction/pdf/normal_pdf.pdf", 32),
        ("tests/test_data/prediction/pdf/twopager_georgian.pdf", 2),
        ("tests/test_data/prediction/pdf/very_large_file.pdf", 132),
    ]


@pytest.fixture
def tiff_path_pagecount() -> List[Tuple[str, int]]:
    return [
        ("tests/test_data/prediction/tiff/multi-page-tiff.tiff", 17),
        ("tests/test_data/prediction/tiff/multi-page-tiff2.tiff", 25),
    ]


@pytest.fixture
def document_path_pagecount(
    jpg_image_path: List[str], pdf_path_pagecount: List[Tuple[str, int]], tiff_path_pagecount: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    return [(path, 1) for path in jpg_image_path] + pdf_path_pagecount + tiff_path_pagecount


@pytest.fixture
def documents_folder_path_pagecounts(
    tmp_path: Path, document_path_pagecount: List[Tuple[str, int]]
) -> Generator[Tuple[str, List[Tuple[str, int]]], None, None]:
    documents_path = tmp_path / "docs"
    documents_path.mkdir()
    for file_path, _ in document_path_pagecount:
        shutil.copy(file_path, documents_path)
    yield str(documents_path.resolve()), document_path_pagecount
    shutil.rmtree(documents_path)


def assert_list_of_strings(str_list: List[Any]) -> None:
    for x in str_list:
        assert type(x) == str


def test_load_classifiers(model_name: str) -> None:
    binary_clf, multiclass_clf = load_classifiers(model_name)

    # check types
    assert isinstance(binary_clf, PrimaryPageClassifier)
    assert isinstance(multiclass_clf, DocumentClassifier)

    # check if the classifier are trained
    assert binary_clf.model is not None
    assert multiclass_clf.model is not None


def test_predict_images(model_name: str, jpg_image_path: List[str]) -> None:
    batch_prediction = predict_images(jpg_image_path, model_name)
    assert_list_of_strings(batch_prediction)

    individual_prediction = [predict_image(image, model_name) for image in jpg_image_path]
    assert_list_of_strings(individual_prediction)

    assert batch_prediction == individual_prediction


def test_predict_pdfs(model_name: str, pdf_path_pagecount: List[Tuple[str, int]]) -> None:
    predictions = predict_pdfs([path for path, _ in pdf_path_pagecount], model_name)
    assert [len(p) for p in predictions] == [n_pages for _, n_pages in pdf_path_pagecount]
    for pred in predictions:
        assert_list_of_strings(pred)

    invididual_predictions = [predict_pdf(path, model_name) for path, _ in pdf_path_pagecount]
    assert invididual_predictions == predictions


def test_predict_tiffs(model_name: str, tiff_path_pagecount: List[Tuple[str, int]]) -> None:
    predictions = predict_tiffs([path for path, _ in tiff_path_pagecount], model_name)
    assert [len(p) for p in predictions] == [n_pages for _, n_pages in tiff_path_pagecount]
    for pred in predictions:
        assert_list_of_strings(pred)

    invididual_predictions = [predict_tiff(path, model_name) for path, _ in tiff_path_pagecount]
    assert invididual_predictions == predictions


def test_predict_documents(model_name: str, document_path_pagecount: List[Tuple[str, int]]) -> None:
    predictions = predict_documents([path for path, _ in document_path_pagecount], model_name)
    assert [len(p) for p in predictions] == [n_pages for _, n_pages in document_path_pagecount]
    for pred in predictions:
        assert_list_of_strings(pred)


def test_predict_from_directory(
    model_name: str, documents_folder_path_pagecounts: Tuple[str, List[Tuple[str, int]]]
) -> None:
    directory, documents_path_pagecount = documents_folder_path_pagecounts

    page_counts = {os.path.basename(file_path): page_count for file_path, page_count in documents_path_pagecount}

    prediction = predict_from_directory(directory, model_name)
    for doc_path, page_predictions in prediction.items():
        assert len(page_predictions) == page_counts[os.path.basename(doc_path)]
        assert_list_of_strings(page_predictions)
