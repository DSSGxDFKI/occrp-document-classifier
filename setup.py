from setuptools import find_packages, setup
import sys
import platform
import warnings

requirements = [
    "keras",
    "Keras-Preprocessing",
    "imbalanced-learn",
    "filetype",
    "matplotlib",
    "mlflow",
    "numpy",
    "pandas",
    "pdf2image",
    "pydantic",
    "pytest",
    "scikit-learn",
    "scipy",
    "tqdm",
    "typer",
    "tf-explain",
    "opencv-python",
]
if (sys.platform == "darwin") and (platform.processor() == "arm"):
    # TF package for Apple Silicon (M1/M2 processor),
    warnings.warn(
        "This package requires tensorflow. To install it in Apple Silicon. "
        "See https://developer.apple.com/metal/tensorflow-plugin/"
    )
    requirements.append("tensorflow-macos")
else:
    requirements.append("tensorflow")

setup(
    name="occrp-document-classifier",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements
    # py_modules=["config"]
)
