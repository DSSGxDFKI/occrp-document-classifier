import logging
import os

import cv2
from cv2 import error
import numpy as np
import pandas as pd

from typing import Tuple, List

logger = logging.getLogger(__name__)


def import_images(
    images_dir: str, names: pd.DataFrame, im_size: int, verbose: bool = False, as_gray: bool = False
) -> Tuple[np.ndarray, List[int], List]:
    """Given file names, create reduced dataset containing images of interest/ labels of interest.
    Receives:
        Directory of images, names of images relative paths, image size

    Returns:
        np.array with image tensor, image labels, list of faulty files.
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
    # and resize them to the image size (227x227x3 for color images)
    for i in range(n_subset):
        data_path = os.path.join(images_dir, names["directory"][i])
        try:
            im: np.ndarray = cv2.imread(data_path, as_color)
            img: np.ndarray = np.array(im)  # reading image as array  # isn't this already an ndarray?
            img_resized: np.ndarray = cv2.resize(img, (im_size, im_size))
            images.append(img_resized)
            labels.append(int(names["doc type"][i]))
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
