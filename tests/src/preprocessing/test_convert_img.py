import os
import shutil
import pathlib

import pytest

from occrplib.preprocessing.convert_to_img import (
    convert_all_to_jpg,
    convert_docx_to_pdf,
    convert_pdf_to_jpg,
    convert_tif_to_jpg,
    check_inputs,
    copy_jpg_to_jpg,
    convert_png_to_jpg,
)


# fixtures


@pytest.fixture
def pdf_path() -> str:
    return "tests/test_data/test_pdf.pdf"


@pytest.fixture
def capital_pdf_path() -> str:
    return "tests/test_data/NORMAL_PDF.PDF"


@pytest.fixture
def corrupt_pdf_path() -> str:
    return "tests/test_data/corrupt_pdf.pdf"


@pytest.fixture
def large_pdf_path() -> str:
    return "tests/test_data/very_large_file.pdf"


@pytest.fixture
def nonexisting_pdf_path(tmp_path: pathlib.Path) -> str:
    output_path = tmp_path / "idontexist.pdf"
    return str(output_path.resolve())


@pytest.fixture
def output_path(tmp_path: pathlib.Path) -> str:
    output_path = tmp_path / "test_folder"
    output_path.mkdir()
    return str(output_path.resolve())


@pytest.fixture
def nonexisting_output_path(tmp_path: pathlib.Path) -> str:
    output_path = tmp_path / "idontexist"
    return str(output_path.resolve())


@pytest.fixture
def tiff_path() -> str:
    return "tests/test_data/test_tiff.tif"


@pytest.fixture
def multipage_tiff_path() -> str:
    return "tests/test_data/multipage_tiff.tiff"


@pytest.fixture
def doc_path() -> str:
    return "tests/test_data/word1.doc"


@pytest.fixture
def jpg_path() -> str:
    return "tests/test_data/test_jpg.jpg"


@pytest.fixture
def png_path() -> str:
    return "tests/test_data/PNG-file.png"


@pytest.fixture
def corrupt_jpg_path() -> str:
    return "tests/test_data/corrupt_jpg.jpg"


@pytest.fixture
def pdf_tif_jpg_folder_path(tmp_path: pathlib.Path, pdf_path: str, tiff_path: str, jpg_path: str) -> str:
    output_path = tmp_path / "pdf_tif_jpg"
    output_path.mkdir()
    for file_path in [pdf_path, tiff_path, jpg_path]:
        shutil.copy(file_path, output_path)
    return str(output_path.resolve())


# auxiliary functions
def assert_filename_folder_created(file_path: str, output_path: str) -> str:
    """Asserts if the output folder contains a folder with the name of the specified file
    (without extension)

    Args:
        file_path (str): file, the filename should be the folder name contained in output
        output_path (str): directory that should contain the filename folder

    Returns:
        str: path of filename folder
    """
    filename, _ = os.path.splitext(os.path.basename(file_path))
    created_folder = os.path.join(output_path, filename)
    assert os.path.isdir(created_folder)
    return created_folder


def assert_converted_number_of_pages(directory: str, n_pages: int) -> None:
    """Asserts if a directory contain a specific number of jpg pages inside

    Args:
        directory (str): directory to check
        n_pages (int): number of pages that should be contained
    """
    number_of_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    assert number_of_files == n_pages
    for page in range(1, n_pages + 1):
        assert os.path.isfile(os.path.join(directory, f"{page}.jpg"))


# Convert PDF to JPG
def test_convert_pdf_to_jpg(pdf_path: str, output_path: str) -> None:
    convert_pdf_to_jpg(pdf_path, output_path, only_first_page=False)
    created_folder = assert_filename_folder_created(pdf_path, output_path)
    assert_converted_number_of_pages(created_folder, 2)


def test_convert_capital_pdf_to_jpg(capital_pdf_path: str, output_path: str) -> None:
    convert_pdf_to_jpg(capital_pdf_path, output_path, only_first_page=False)
    created_folder = assert_filename_folder_created(capital_pdf_path, output_path)
    assert_converted_number_of_pages(created_folder, 32)


def test_tiff_in_convert_pdf_to_jpg(tiff_path: str, output_path: str) -> None:
    with pytest.raises(Exception):
        convert_pdf_to_jpg(tiff_path, output_path, only_first_page=False)


def test_convert_corrupt_pdf_to_jpg(corrupt_pdf_path: str, output_path: str) -> None:
    with pytest.raises(Exception):
        convert_pdf_to_jpg(corrupt_pdf_path, output_path, only_first_page=False)


def test_output_path_doesnt_exists_in_convert_pdf_to_jpg(pdf_path: str, nonexisting_output_path: str) -> None:
    with pytest.raises(Exception):
        convert_pdf_to_jpg(pdf_path, nonexisting_output_path, only_first_page=False)


def test_convert_large_pdf_to_jpg(large_pdf_path: str, output_path: str) -> None:
    convert_pdf_to_jpg(large_pdf_path, output_path, only_first_page=False)
    created_folder = assert_filename_folder_created(large_pdf_path, output_path)
    assert_converted_number_of_pages(created_folder, 132)


# Convert TIF to JPG
def test_convert_tif_to_jpg(tiff_path: str, output_path: str) -> None:
    convert_tif_to_jpg(tiff_path, output_path, only_first_page=False)
    created_folder = assert_filename_folder_created(tiff_path, output_path)
    assert_converted_number_of_pages(created_folder, 1)


def test_convert_multipage_tif_to_jpg(multipage_tiff_path: str, output_path: str) -> None:
    convert_tif_to_jpg(multipage_tiff_path, output_path, only_first_page=False)
    created_folder = assert_filename_folder_created(multipage_tiff_path, output_path)
    assert_converted_number_of_pages(created_folder, 17)


def test_output_path_doesnt_exists_in_convert_tif_to_jpg(tiff_path: str, nonexisting_output_path: str) -> None:
    with pytest.raises(Exception):
        convert_tif_to_jpg(tiff_path, nonexisting_output_path)


# Convert DOCX to JPG
def test_convert_docx_to_pdf(doc_path: str, output_path: str) -> None:
    # is expected to fail outside of docker container where libreoffice runs
    convert_docx_to_pdf(doc_path, output_path, only_first_page=False)
    created_folder = assert_filename_folder_created(doc_path, output_path)
    assert_converted_number_of_pages(created_folder, 9)


# Copy JPG to JPG
def test_copy_jpg_to_jpg(jpg_path: str, output_path: str) -> None:
    copy_jpg_to_jpg(jpg_path, output_path)
    created_folder = assert_filename_folder_created(jpg_path, output_path)
    assert_converted_number_of_pages(created_folder, 1)


# Copy JPG to JPG
def test_copy_corrupt_jpg_to_jpg(corrupt_jpg_path: str, output_path: str) -> None:
    with pytest.raises(Exception):
        copy_jpg_to_jpg(corrupt_jpg_path, output_path)


# Convert PNG to JPG
def test_convert_png_to_jpg(png_path: str, output_path: str) -> None:
    convert_png_to_jpg(png_path, output_path)
    created_folder = assert_filename_folder_created(png_path, output_path)
    assert_converted_number_of_pages(created_folder, 1)


# Convert all to JPG
def test_convert_all_to_jpg(pdf_tif_jpg_folder_path: str, output_path: str) -> None:
    convert_all_to_jpg(pdf_tif_jpg_folder_path, output_path)

    for filename in os.listdir(pdf_tif_jpg_folder_path):
        _, extension = os.path.splitext(os.path.basename(filename))
        file_path = os.path.join(pdf_tif_jpg_folder_path, filename)
        if os.path.isfile(file_path):
            if extension == ".jpg":
                assert os.path.isfile(os.path.join(output_path, os.path.basename(file_path)))
            else:
                assert_filename_folder_created(file_path, output_path)


# Check inputs
def test_check_inputs_existing_paths(pdf_path: str, output_path: str) -> None:
    assert check_inputs(pdf_path, output_path)


def test_check_inputs_nonexisting_paths(
    pdf_path: str, output_path: str, nonexisting_pdf_path: str, nonexisting_output_path: str
) -> None:
    assert not check_inputs(nonexisting_pdf_path, output_path)
    assert not check_inputs(pdf_path, nonexisting_output_path)
    assert not check_inputs(nonexisting_pdf_path, nonexisting_output_path)


if __name__ == "__main__":
    pytest.main(["--disable-pytest-warnings"])
