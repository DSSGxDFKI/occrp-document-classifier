import logging
import os
import tempfile
from typing import Callable, Dict, List, Tuple

import pandas as pd
from pdf2image.exceptions import PDFPageCountError

from occrplib.config import settings
from occrplib.document_classification.document_classifier import DocumentClassifier
from occrplib.document_classification.primarypage_classifier import PrimaryPageClassifier
from occrplib.preprocessing.convert_to_img import convert_pdf_to_jpg, convert_tif_to_jpg
from occrplib.preprocessing.train_val_test_split import get_data_files_df

logger = logging.getLogger(__name__)


def load_classifiers(architecture_name: str) -> Tuple[PrimaryPageClassifier, DocumentClassifier]:
    """Load the binary and multiclass document classifiers defined in the config file using
    an architecture name

    Args:
        architecture_name (str): string name of the classifier, should be defined as a key
            in the PREDICTION_MODELS dictionary in the config file

    Returns:
        Tuple[PrimaryPageClassifier, DocumentClassifier]: trained classifiers
    """
    binary_classifier = PrimaryPageClassifier.load_from_path(
        architecture_name, settings.PREDICTION_MODELS[architecture_name]["binary_classifier"]
    )
    document_classifier = DocumentClassifier.load_from_path(
        architecture_name, settings.PREDICTION_MODELS[architecture_name]["multiclass_classifier"]
    )
    return binary_classifier, document_classifier


def predict_images(images_path: List[str], architecture_name: str) -> List[str]:
    """Predict the class of multiple jpg images

    Args:
        images_path (List[str]): list with absolute paths of images to predict
        architecture_name (str): classifier name to be used in the prediction

    Returns:
        List[str]: list of predictions (string labels)
    """
    # load models
    binary_classifier, document_classifier = load_classifiers(architecture_name)

    # convert images to dataframe
    df = pd.DataFrame({"file_path": images_path})

    # predict using binary
    binary_prediction_df = binary_classifier.predict_from_df(df)
    df = pd.concat([df, binary_prediction_df[["predicted_label"]]], axis=1)

    # filter firstpage
    firstpage_prediction_df = df[df["predicted_label"] == "firstpages"].reset_index(drop=True)

    # predict using multiclass classifier
    if len(firstpage_prediction_df) > 0:
        document_type_prediction_df = document_classifier.predict_from_df(firstpage_prediction_df[["file_path"]])
        firstpage_prediction_df = pd.concat(
            [firstpage_prediction_df[["file_path"]], document_type_prediction_df[["predicted_label"]]], axis=1
        )

    # concatenate all results
    df = df[["file_path"]].merge(firstpage_prediction_df, on="file_path", how="left").fillna("other")

    # return list of labels predicted
    return df["predicted_label"].to_list()


def predict_image(image_path: str, architecture_name: str) -> str:
    """Predict a single jpg image class

    Args:
        image_path (str): path to an image
        architecture_name (str): classifier name to be used in the prediction

    Returns:
        str: class of the image
    """
    return predict_images([image_path], architecture_name)[0]


def predict_documents_with_conversion(
    file_paths: List[str], architecture_name: str, convert_to_jpg: Callable[[str, str, bool], None]
) -> List[List[str]]:
    """Predicts the class of a list of documents that requires a conversion to PDF

    Args:
        file_paths (List[str]): absolute paths to documents
        architecture_name (str): classifier to be used in prediction
        convert_to_jpg (Callable[[str, str, bool], None]): conversion function

    Returns:
        List[List[str]]: list of prediction for each document. Each prediction
            is a list of the predicted label per page
    """
    # convert documents to images using temp directory
    with tempfile.TemporaryDirectory() as tmpdir_path:
        for file_path in file_paths:
            try:
                convert_to_jpg(file_path, tmpdir_path, False)
            except PDFPageCountError as e:
                logger.error(f"Error when trying to convert file <{file_path}>: {repr(e)}")
        pages_df = get_data_files_df(tmpdir_path)
        pages_df["page"] = pages_df["filename"].str.extract(r"(\d+).jpg").astype("int")
        pages_df = pages_df.sort_values("page")

        # predict images
        pages_df["predicted_label"] = predict_images(pages_df["path"].to_list(), architecture_name)

        # rearrange into list of list with predicted labels per page
        def get_raw_basename(file_path):
            return os.path.splitext(os.path.basename(file_path))[0]

        return [
            pages_df[pages_df["class"] == get_raw_basename(file_path)]["predicted_label"].to_list()
            for file_path in file_paths
        ]


def predict_pdfs(pdf_paths: List[str], architecture_name: str) -> List[List[str]]:
    """Predict the class of multiple pdf files

    Args:
        pdf_paths (List[str]): list with absolute paths to pdf files
        architecture_name (str): classifier to be used in prediction

    Returns:
        List[List[str]]: list of predictions for each page for each document
    """
    return predict_documents_with_conversion(pdf_paths, architecture_name, convert_pdf_to_jpg)


def predict_pdf(pdf_path: str, architecture_name: str) -> List[str]:
    """Predict the class of a single pdf document

    Args:
        pdf_path (str): path to pdf file
        architecture_name (str): classifier to be used for prediction

    Returns:
        List[str]: predicted class for each page of the pdf
    """
    return predict_pdfs([pdf_path], architecture_name)[0]


def predict_tiffs(tiff_paths: List[str], architecture_name: str) -> List[List[str]]:
    """Predict the class of multiple tiff files

    Args:
        tiff_paths (List[str]): list of tiff paths
        architecture_name (str): classifier to be used in prediction

    Returns:
        List[List[str]]: prediction for each page of each tiff document
    """
    return predict_documents_with_conversion(tiff_paths, architecture_name, convert_tif_to_jpg)


def predict_tiff(tiff_path: str, architecture_name: str) -> List[str]:
    """Predict the class of a single tiff document

    Args:
        tiff_path (str): path to tiff path
        architecture_name (str): classifier to be used in prediction

    Returns:
        List[str]: prediction for each tiff page
    """
    return predict_tiffs([tiff_path], architecture_name)[0]


def predict_documents(file_paths: List[str], architecture_name: str) -> List[List[str]]:
    """Predict the class of multiple documents. Supported format are:
        .jpg, .jpeg, .tiff, .tif, .pdf

    Args:
        file_paths (List[str]): paths of documents to predict
        architecture_name (str): classifier to be used in prediction

    Returns:
        List[List[str]]: prediction for each page of each document
    """
    # get files types
    df = pd.DataFrame({"file_path": file_paths})
    df["extension"] = df["file_path"].apply(lambda path: os.path.splitext(path)[-1].lower())

    predictions = []
    # predict images
    images = df[df["extension"].isin([".jpg", ".jpeg"])]["file_path"].to_list()
    if images:
        images_prediction_df = pd.DataFrame(
            {"file_path": images, "prediction": list(map(lambda p: [p], predict_images(images, architecture_name)))}
        )
        predictions.append(images_prediction_df)

    # predict tiff
    tiffs = df[df["extension"].isin([".tiff", ".tif"])]["file_path"].to_list()
    if tiffs:
        tiffs_prediction_df = pd.DataFrame({"file_path": tiffs, "prediction": predict_tiffs(tiffs, architecture_name)})
        predictions.append(tiffs_prediction_df)

    # predict pdf
    pdfs = df[df["extension"] == ".pdf"]["file_path"].to_list()
    if pdfs:
        pdfs_prediction_df = pd.DataFrame({"file_path": pdfs, "prediction": predict_pdfs(pdfs, architecture_name)})
        predictions.append(pdfs_prediction_df)

    # merge all prediction, else return blank string
    prediction_df = pd.concat(predictions)
    df = df.merge(prediction_df, how="left", on="file_path").fillna("")
    return [[""] if p == "" else p for p in df["prediction"]]


def predict_from_directory(directory_path: str, architecture_name: str) -> Dict[str, List[str]]:
    """Predict the class of the documents contained in a directory

    Args:
        directory_path (str): path to directory
        architecture_name (str): classifier to be used in prediction

    Raises:
        ValueError: when the directory path doesn't exists

    Returns:
        Dict[str, List[str]]: dictionary with the absolute path of each document as key and a list
            with the prediction of each page of the document as value
    """
    # directory to list of file paths
    if not os.path.isdir(directory_path):
        raise ValueError("directory_path should be a path to an existing directory")

    file_paths = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if os.path.splitext(filename)[-1] in [".jpg", ".jpeg", ".tiff", ".tif", ".pdf"]:
                file_paths.append(os.path.join(root, filename))

    return dict(zip(file_paths, predict_documents(file_paths, architecture_name)))
