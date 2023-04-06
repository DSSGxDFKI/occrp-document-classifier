from collections import defaultdict
import logging
import os.path
import shutil
import subprocess
from typing import Tuple

import filetype
from pdf2image import convert_from_path
from PIL import Image, ImageSequence
import tqdm

logger = logging.getLogger(__name__)


image_size = (400, 400)  # x and y


def convert_pdf_to_jpg(input_path_to_file: str, output_path: str, only_first_page: bool) -> None:
    """Converts a single pdf into jpgs, one jpg for each page of the pdf.

    Args:
        input_path_to_file (str):input path where the document in pdf format to be converted is located.
        output_path (str): output path where to save all document pages in by-page jpg format images.
        only_first_page (bool): if True, only saves and converts the primary page of the document.

    Raises:
        ValueError: "Wrong input"
    """

    if not (check_inputs(input_path_to_file, output_path)):
        raise ValueError("Wrong input")

    pages = convert_from_path(input_path_to_file, dpi=100, size=image_size, fmt="jpeg", thread_count=7)

    process_until_page: int = len(pages)
    if only_first_page:
        process_until_page = 1

    # for each page of document save it as jpg to folder in output directory
    for i, page in enumerate(pages[0:process_until_page], 1):
        pdf_filename = os.path.splitext(os.path.basename(input_path_to_file))[0]  # without file extension

        # create folder once
        if i == 1:
            if not os.path.exists(os.path.join(output_path, pdf_filename)):
                os.makedirs(os.path.join(output_path, pdf_filename))

        # save jpg
        jpg_filename = str(i) + ".jpg"
        page.save(os.path.join(output_path, pdf_filename, jpg_filename), "JPEG")
        # print("save to", os.path.join(output_path, pdf_filename, jpg_filename))
    logger.debug(f"save to {os.path.join(output_path, pdf_filename)}")


def convert_tif_to_jpg(input_path_to_file: str, output_path: str, only_first_page: bool) -> None:
    """Converts a single tif file into by-page jpg image files.

    Args:
        input_path_to_file (str): input path where the document in tif format to be converted is located.
        output_path (str): output path where to save all document pages in by-page jpg format images.
        only_first_page (bool): if True, only saves and converts the primary page of the document.

    Raises:
        ValueError: "Wrong input"
    """

    if not (check_inputs(input_path_to_file, output_path)):
        raise ValueError("Wrong input:")

    tif = Image.open(input_path_to_file)
    tif_filename = os.path.splitext(os.path.basename(input_path_to_file))[0]

    process_until_page: int = len(list(enumerate(ImageSequence.Iterator(tif), 1)))
    if only_first_page:
        process_until_page = 1

    for i, page in list(enumerate(ImageSequence.Iterator(tif), 1))[0:process_until_page]:
        jpg_filename = str(i) + ".jpg"

        # create folder once
        if i == 1:
            if not os.path.exists(os.path.join(output_path, tif_filename)):
                os.makedirs(os.path.join(output_path, tif_filename))

        page = page.convert("RGB")
        page.thumbnail(image_size)
        page.save(os.path.join(output_path, tif_filename, jpg_filename))
    logger.debug(f"save to {os.path.join(output_path, tif_filename)}")


def convert_docx_to_pdf(input_path_to_file: str, output_path: str, only_first_page: bool) -> None:
    """Converts a file with the extension doc or docx to jpg. Requires libreoffice to be installed and available.

    Args:
        input_path_to_file (str): path to the doc/docx file
        output_path (str): path to the directory where the jpgs shall be saved.
    """
    if not (check_inputs(input_path_to_file, output_path)):
        logger.debug("Wrong input for {input_path_to_file}")
        raise ValueError("Wrong input")

    if not (check_inputs(input_path_to_file, output_path)):
        logger.debug("Wrong input for {input_path_to_file}")
        raise ValueError("Wrong input")

    # Check if libreoffice is installed
    libreoffice_installed: str = subprocess.run("libreoffice; echo $?", capture_output=True, shell=True).stdout.decode()
    libreoffice_installed = libreoffice_installed.strip()

    if libreoffice_installed != "0":  # TODO this check does often not work, error code is often also 0 if conversion fails
        raise ImportError(
            f"Libreoffice is not available. Make sure it is installed to convert {input_path_to_file}. \
            It returned error code {libreoffice_installed}."
        )

    # ENHANCEMENT
    # we are copying the data from the input directory to the output folder directory
    # this is unnecessary and should be changed
    temp_folder = os.path.join(output_path, "temp")
    temp_doc_file = os.path.join(temp_folder, os.path.basename(input_path_to_file))
    if not (os.path.exists(temp_folder)):
        os.makedirs(temp_folder)
    shutil.copy(input_path_to_file, os.path.join(temp_folder, os.path.basename(input_path_to_file)))

    # convert the copied doc file to pdf
    command = subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", temp_doc_file, "--outdir", temp_folder])
    if command.returncode != 0:  # it seems this check is not really useful, always returns 0 even if conversion failed
        logger.debug(f"Libreoffice didn't work to convert {input_path_to_file}")
        raise ValueError("Libreoffice didn't work")

    temp_pdf_file: str = os.path.splitext(temp_doc_file)[0] + ".pdf"

    # pass pdf to jpg conversion function
    convert_pdf_to_jpg(temp_pdf_file, output_path, only_first_page)

    # remove the temp doc and pdf
    os.remove(temp_doc_file)
    os.remove(temp_pdf_file)


def convert_png_to_jpg(input_path_to_file: str, output_path: str) -> None:
    """Converts a PNG file to a JPG file."""

    if not (check_inputs(input_path_to_file, output_path)):
        return

    png: Image = Image.open(input_path_to_file)

    png = png.convert("RGB")
    # png = png.thumbnail(image_size)
    png_filename: str = os.path.splitext(os.path.basename(input_path_to_file))[0]

    # we create a folder for each file even if it has only one page
    # because we check the existence of these folders when the skip_converted_files is True
    parent_folder = os.path.join(output_path, png_filename)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    output_full_path = os.path.join(parent_folder, "1.jpg")

    png.save(output_full_path)
    logger.debug(f"save to {os.path.join(output_full_path)}")


def copy_jpg_to_jpg(input_path_to_file: str, output_path: str) -> None:
    """Conversion not necessary, just copy jpg to output folder.

    Args:
        input_path_to_file (str): input path where the document in jpg format is located.
        output_path (str): output path where to copy the jpg imageS.
        only_first_page (bool): if True, only saves and converts the primary page.
    """

    if not (check_inputs(input_path_to_file, output_path)):
        return

    raw_filename, extension = os.path.splitext(os.path.basename(input_path_to_file))

    # we create a folder for each file even if it has only one page
    # because we check the existence of these folders when the skip_converted_files is True
    parent_folder = os.path.join(output_path, raw_filename)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    output_full_path = os.path.join(parent_folder, "1.jpg")

    shutil.copy(input_path_to_file, output_full_path)
    logger.debug(f"save to {os.path.join(output_full_path)}")


def convert_all_to_jpg(
    input_path_to_dir: str, output_path_root: str, skip_converted_files: bool = True, only_first_page: bool = False
) -> None:
    """Looks for all .pdf, .tif, .jpg, .png, .doc, .docx in a directory and its subdirectories and converts them to jpgs.

    Args:
        input_path_to_dir (str): input path where all documents to be converted are located.
        output_path_root (str): output path where to save all document pages in by-page jpg format images.
        skip_converted_files (bool, optional): _description_. Defaults to True.
        only_first_page (bool, optional): if True, only saves and converts the primary page of each document. Defaults to False.

    Raises:
        ValueError: "Input directory is not a directory"
        ValueError: "File type not supported"
    """  # NOQA E501

    extensions: Tuple[str, ...] = ("pdf", "tif", "jpg", "doc", "docx", "png")
    converted_files = 0
    skipped_files = 0
    failed_files = 0
    converted_type_count: dict = defaultdict(lambda: 0)

    # check if input_path_to_dir is working
    if not (os.path.isdir(input_path_to_dir)):
        logger.error(f"{input_path_to_dir} is not a directory.")
        raise ValueError(f"{input_path_to_dir} is not a directory.")

    # loop over all files in all subdirectories
    for dirpath, dirnames, filenames in tqdm.tqdm(list(os.walk(os.path.join(input_path_to_dir)))):
        for filename in tqdm.tqdm(filenames, leave=False):
            input_file = os.path.join(dirpath, filename)  # path to file
            input_file_extension: str

            raw_filename, extension = os.path.splitext(filename)

            # check if in extension list
            if filename.lower().endswith(extensions):
                input_file_extension = extension.lower()

            # or extensions has more than 4 characters (something like 40bde8e2bf50eae532e2bdf9623cac38c9ff5)
            elif len(extension) > 4:
                # guess or get file extension
                input_file_extension_guess = filetype.guess(input_file)
                if input_file_extension_guess is not None:
                    input_file_extension = "." + input_file_extension_guess.extension
                else:
                    # if not guessed
                    logger.debug(f"could not guess file extension, skip {input_file}")
                    continue
            else:
                logger.debug(f"file type not supported, skip {input_file}")
                continue

            # check if target directory for this file exists, otherwise create
            relative_output_path = os.path.relpath(dirpath, input_path_to_dir)
            if relative_output_path == ".":
                relative_output_path = ""  # it caused a bug in the docx conversion when a dot was in the path
            abs_output_path = os.path.join(output_path_root, relative_output_path)

            if not (os.path.exists(abs_output_path)):
                os.makedirs(abs_output_path)
            else:
                if skip_converted_files and os.path.isfile(os.path.join(abs_output_path, raw_filename, "1.jpg")):
                    skipped_files += 1
                    logger.debug("skipped file")
                    continue

            # convert to jpg
            try:
                if input_file_extension == ".pdf":
                    convert_pdf_to_jpg(input_file, abs_output_path, only_first_page)
                elif input_file_extension == ".tif":
                    convert_tif_to_jpg(input_file, abs_output_path, only_first_page)
                elif input_file_extension == ".doc" or input_file_extension == ".docx":
                    convert_docx_to_pdf(input_file, abs_output_path, only_first_page)
                elif input_file_extension == ".jpg":
                    copy_jpg_to_jpg(input_file, abs_output_path)
                elif input_file_extension == ".png":
                    convert_png_to_jpg(input_file, abs_output_path)
                else:
                    logger.debug(f"File type '{input_file_extension}' not supported: {input_file}")
                    raise ValueError("File type not supported")
                converted_type_count[input_file_extension] += 1
                converted_files += 1
            except Exception as e:
                logger.debug(f"Error with file {input_file}: {repr(e)}")
                failed_files += 1

    logger.info(f"Converted files: {converted_files}. Files with error: {failed_files}. Skipped files: {skipped_files}.")

    for extension in extensions:
        if converted_type_count[extension] > 0:
            logger.info(f"{extension[1:]} files converted: {converted_type_count[extension]}")


def check_inputs(input_path_to_file: str, output_path: str) -> bool:
    """ Check if input and output files are working \
       Return true if they do, otherwise false.
    Args:
        input_path_to_file (str):  input path where the document to be converted is located.
        output_path (str): path to the directory where the jpgs shall be saved.

    Returns:
        bool: True if the input file exists and the output folder exists. Otherwise, it returns False.
    """

    """"""
    # ENHANCEMENT check_inputs()
    # we check for the existence of folders in check_inputs() and in the conversion functions, that's doubled. Smart?
    # However,the single functions might also have to contain the check to run them individually. Also, they can be
    # useful when converting twice, from docx to pdf to jpg. So not clear yet if this needs refactoring.


    # check if input_path is a file
    if not (os.path.isfile(input_path_to_file)):
        return False

    # check if output_path is a directory
    if not (os.path.isdir(output_path)):
        return False

    return True
