import logging
import os

import cv2
import numpy as np
import pandas as pd

from typing import Tuple, List

logger = logging.getLogger(__name__)


def import_images(
    images_dir: str, names: pd.DataFrame, im_size: int, verbose: bool = False, as_gray: bool = False
) -> Tuple[np.ndarray, List[int], List]:
    """Given file names, create reduced dataset containing images of interest/ labels of interest.

    Args:
        images_dir (str): directory containing the files to be imported.
        names (pd.DataFrame): dataframe containing a column with the image files to import, and a column with the document
        page type label.

        im_size (int): size of the images to import.
        verbose (bool, optional): Defaults to False.
        as_gray (bool, optional): If as_gray = True, it imports the image with a single grayscale channel.
                                    Otherwise, it imports images it in 3-channel full color image. Defaults to False.

    Returns:
        images_scaled (np.ndarray): image tensor with images imported. Numbers are scaled from 0-255 to 0-1.
        labels (List[int]): list of document-page type label corresponding to each loaded page.
        faulty (List): list of indexes corresponding to faulty files that could not be imported.
    """

    n_subset: int = names.shape[0]

    images: List[np.ndarray] = []
    labels: List[int] = []
    faulty: List[int] = []
    faulty_counter: int = 0
    file_counter: int = 0

    as_color: int = 0
    if not as_gray:
        as_color = 1

    if verbose:
        print(f"The shape of names table is: {names.shape}")
        print("Importing dataset ...")
    # In this loop, we extract all document images,
    # and resize them to 227x227x3 (color image)
    for i in range(n_subset):
        data_path = os.path.join(images_dir, names["file_path"][i])
        try:
            im: np.ndarray = cv2.imread(data_path, as_color)
            img: np.ndarray = np.array(im)  # reading image as array  # ENHANCEMENT isn't this already an ndarray?
            img_resized: np.ndarray = cv2.resize(img, (im_size, im_size))
            images.append(img_resized)
            labels.append(int(names["true_index"][i]))
        except error as e:
            # Save the positions of damaged files:
            faulty_counter = faulty_counter + 1
            faulty.append(file_counter)
            logger.error(f"Faulty file in {data_path}, error: {e}")
        file_counter = file_counter + 1

    if verbose:
        print(f"There are a total of {faulty_counter} faulty files in the data.")

    # Turn into np-array and normalize.
    images_scaled: np.ndarray = np.array(images).astype("float32") / 255.0
    if as_gray:
        images_scaled = np.expand_dims(images_scaled, axis=3)
    if verbose:
        print(f"Size of image tensor is: {images_scaled.shape}")

    return images_scaled, labels, faulty
