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


def predict_images_probs(
    images_path: List[str], binary_classifier: PrimaryPageClassifier, document_classifier: DocumentClassifier
) -> List[dict]:
    """Predicts multiple images and returns a list with a dictionary containing the prediction and the
        class probabilities for each image

    Args:
        images_path (List[str]): list of images paths to predict
        binary_classifier (PrimaryPageClassifier): primary page classifier
        document_classifier (DocumentClassifier): document type classifier

    Returns:
        List[dict]: prediction and probabilities of supported classes for each image.
        Has the same length of images_path
    """
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
        firstpage_prediction_df = pd.concat([firstpage_prediction_df[["file_path"]], document_type_prediction_df], axis=1)

    # concatenate all results
    df = df[["file_path"]].merge(firstpage_prediction_df, on="file_path", how="left").fillna(0)

    # return list of labels predicted
    # return document_type_prediction_df.iloc[:, :-1].to_dict("records")
    return (
        df[settings.LABELS_FILTER + ["predicted_label"]]
        .rename(columns={"predicted_label": "__predicted"})
        .to_dict("records")
    )


def predict_image_probs(
    image_path: str, binary_classifier: PrimaryPageClassifier, document_classifier: DocumentClassifier
) -> dict:
    """Predict a single jpg image class

    Args:
        image_path (str): path to an image
        binary_classifier (PrimaryPageClassifier): primary page classifier
        document_classifier (DocumentClassifier): document type classifier

    Returns:
        dict: prediction and probabilities of classes
    """
    return predict_images_probs([image_path], binary_classifier, document_classifier)[0]


def predict_documents_probs_with_conversion(
    file_paths: List[str],
    binary_classifier: PrimaryPageClassifier,
    document_classifier: DocumentClassifier,
    convert_to_jpg: Callable[[str, str, bool], None],
) -> List[List[dict]]:
    """Predicts the class of a list of documents that requires a conversion to PDF

    Args:
        file_paths (List[str]): absolute paths to documents
        binary_classifier (PrimaryPageClassifier): primary page classifier
        document_classifier (DocumentClassifier): document type classifier
        convert_to_jpg (Callable[[str, str, bool], None]): conversion function

    Returns:
        List[List[dict]]: list of prediction and probabilities for each document. Each prediction
            is a list of the dictionary with probabilities and predicted class per page
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
        pages_df["predicted_label"] = predict_images_probs(
            pages_df["path"].to_list(), binary_classifier, document_classifier
        )

        # rearrange into list of list with predicted labels per page
        def get_raw_basename(file_path):
            return os.path.splitext(os.path.basename(file_path))[0]

        return [
            pages_df[pages_df["class"] == get_raw_basename(file_path)]["predicted_label"].to_list()
            for file_path in file_paths
        ]


def predict_pdfs_probs(
    pdf_paths: List[str], binary_classifier: PrimaryPageClassifier, document_classifier: DocumentClassifier
) -> List[List[dict]]:
    """Predict the class of multiple pdf files

    Args:
        pdf_paths (List[str]): list with absolute paths to pdf files
        binary_classifier (PrimaryPageClassifier): primary page classifier
        document_classifier (DocumentClassifier): document type classifier

    Returns:
        List[List[dict]]: list of predictions for each page for each document
    """
    return predict_documents_probs_with_conversion(pdf_paths, binary_classifier, document_classifier, convert_pdf_to_jpg)


def predict_pdf_probs(
    pdf_path: str, binary_classifier: PrimaryPageClassifier, document_classifier: DocumentClassifier
) -> List[dict]:
    """Predict the class of a single pdf document

    Args:
        pdf_path (str): path to pdf file
        binary_classifier (PrimaryPageClassifier): primary page classifier
        document_classifier (DocumentClassifier): document type classifier

    Returns:
        List[dict]: predicted class and probabilities for each page of the pdf
    """
    return predict_pdfs_probs([pdf_path], binary_classifier, document_classifier)[0]


def predict_tiffs_probs(
    tiff_paths: List[str], binary_classifier: PrimaryPageClassifier, document_classifier: DocumentClassifier
) -> List[List[dict]]:
    """Predict the class of multiple tiff files

    Args:
        tiff_paths (List[str]): list of tiff paths
        binary_classifier (PrimaryPageClassifier): primary page classifier
        document_classifier (DocumentClassifier): document type classifier

    Returns:
        List[List[dict]]: prediction for each page of each tiff document
    """
    return predict_documents_probs_with_conversion(tiff_paths, binary_classifier, document_classifier, convert_tif_to_jpg)


def predict_tiff_probs(
    tiff_path: str, binary_classifier: PrimaryPageClassifier, document_classifier: DocumentClassifier
) -> List[dict]:
    """Predict the class of a single tiff document

    Args:
        tiff_path (str): path to tiff path
        binary_classifier (PrimaryPageClassifier): primary page classifier
        document_classifier (DocumentClassifier): document type classifier

    Returns:
        List[dict]: prediction for each tiff page
    """
    return predict_tiffs_probs([tiff_path], binary_classifier, document_classifier)[0]


def predict_documents_probs(
    file_paths: List[str], binary_classifier: PrimaryPageClassifier, document_classifier: DocumentClassifier
) -> List[List[dict]]:
    """Predict the class of multiple documents. Supported format are:
        .jpg, .jpeg, .tiff, .tif, .pdf

    Args:
        file_paths (List[str]): paths of documents to predict
        binary_classifier (PrimaryPageClassifier): primary page classifier
        document_classifier (DocumentClassifier): document type classifier

    Returns:
        List[List[dict]]: prediction for each page of each document
    """
    # get files types
    df = pd.DataFrame({"file_path": file_paths})
    df["extension"] = df["file_path"].apply(lambda path: os.path.splitext(path)[-1].lower())

    predictions = []
    # predict images
    images = df[df["extension"].isin([".jpg", ".jpeg"])]["file_path"].to_list()
    if images:
        images_prediction_df = pd.DataFrame(
            {
                "file_path": images,
                "prediction": list(map(lambda p: [p], predict_images_probs(images, binary_classifier, document_classifier))),
            }
        )
        predictions.append(images_prediction_df)

    # predict tiff
    tiffs = df[df["extension"].isin([".tiff", ".tif"])]["file_path"].to_list()
    if tiffs:
        tiffs_prediction_df = pd.DataFrame(
            {"file_path": tiffs, "prediction": predict_tiffs_probs(tiffs, binary_classifier, document_classifier)}
        )
        predictions.append(tiffs_prediction_df)

    # predict pdf
    pdfs = df[df["extension"] == ".pdf"]["file_path"].to_list()
    if pdfs:
        pdfs_prediction_df = pd.DataFrame(
            {"file_path": pdfs, "prediction": predict_pdfs_probs(pdfs, binary_classifier, document_classifier)}
        )
        predictions.append(pdfs_prediction_df)

    # merge all prediction, else return blank string
    prediction_df = pd.concat(predictions)
    df = df.merge(prediction_df, how="left", on="file_path").fillna("")
    return [[""] if p == "" else p for p in df["prediction"]]
