import logging
import os
from typing import Optional

import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator
from occrplib.config import export_settings_to_json, settings
from occrplib.model.image_model import ImageModel

from occrplib.model.architectures.architecture_dictionary import MODELS
from occrplib.utils.logger import LOGGING_FILENAME, LOGGING_FORMAT, LOGGING_LEVELS

logger = logging.getLogger(__name__)


class FeatureExtractor(ImageModel):

    """Class for the big feature extractor model.

    Uses a large data set such as RVL-CDIP.
    """

    # ------------------------------------------------------- #
    # ------------------ Initialization:
    # ------------------------------------------------------- #

    def __init__(self, model_name: str, verbose: bool = False) -> None:
        """Function that initializes the parameters of an object of the class FeatureExtractor.
            It uses the initialization function of the class imageModel, with the specific parameters
            in the config.py file associated with the feature extraction training task.
        Args:
            model_name (str): string containing the model name matching the dictionary of models provided. This
            parameter is usually provided via CLI.
            verbose (bool, optional): Defaults to False.
        """
        super().__init__(
            model_name=model_name,
            verbose=verbose,
            train_test_split_path=settings.RVL_LABELS,
            input_images_path=settings.RVL_IMAGES,
            output_path=settings.OUTPUT_FEATURE_SELECTION,
            **settings.MODEL_CONFIG_FEATURE_EXTRACTION[model_name],
            **settings.PROCESSOR_SETTINGS,
            **settings.MODEL_CONFIG_FEATURE_EXTRACTION["regularization_settings"],
        )

        self.SETTINGS_JSON_PATH = os.path.join(self.model_input_path, "settings.json")
        self.mlflow_experiment_name = "Document Classification"

    # ------------------------------------------------------- #
    # ------------------ Model training:
    # ------------------------------------------------------- #

    def setup_model(self) -> None:
        """Function that creates a neural-network object with the architecture using
        the self.model_name parameter. If the parameter "load_temp_weights_path"
        is None in the config.py file for the requested model, the weights of the
        model are randomly initialized. If "load_temp_weights_path" contains the
        path to a semi-trained model, the function will load the weights contained
        in the saved model to resume training.
        """
        if self.load_temp_weights_path is not None:
            self.model = tf.keras.models.load_model(self.load_temp_weights_path)
        else:
            try:
                self.model = MODELS[self.model_name](
                    verbose=False,
                    im_size=self.im_size,
                    dropout_param=self.dropout_param,
                    regularization_param=self.regularization_param,
                )
            except KeyError:
                print(f"Model {self.model_name} not found. Make sure it is added to src/feature_extraction/models.py")
                logger.error(f"Model {self.model_name} not found")
                exit()

    def train_model(self) -> None:
        """Function train_model instance for the class FeatureExtractor. The placeholder for this function
        can be found in the parent class image_model. It performs the sequence of steps necessary for training
        a feature extraction model: create folders, load training/validating/testing split, initialize model,
        train model with data and save assessment outputs. This pre-training stage is defaulted to be performed
        on the RVL-CDIP dataset.
        """
        self.create_folders()
        # self.load_labels_from_csv(self.RVL_LABELS_CSV)
        self.split_train_val_test_labels()
        self.set_train_val_test_data_iterators()
        self.setup_model()
        self.training_loop()

        self.save_accuracy_loss()
        self.save_in_sample_validations()

        export_settings_to_json(self.SETTINGS_JSON_PATH)

        logger.info(self.model.evaluate(self.test_iterator))

    # ------------------------------------------------------- #
    # ------------------ Importing data:
    # ------------------------------------------------------- #

    def get_data_iterator(
        self,
        generator: ImageDataGenerator,
        df: pd.DataFrame,
        shuffle: bool = False,
        seed: int = 42,
        relative_paths: bool = True,
        x_col: Optional[str] = "file_path",
        y_col: Optional[str] = "true_str_index",
    ) -> DataFrameIterator:
        """Function that defines the data iterators for the training of the feature extraction.
            Due to the size of the RVL-CDIP dataset, it is not necessary to perform data augmentation.
            The default of this function is not to use data augmentation.
        Args:
            generator (ImageDataGenerator): _description_
            df (pd.DataFrame): dataframe containing the paths and labels of the documents to import.
            shuffle (bool, optional): set True to shuffle incoming data. Defaults to False.
            seed (int, optional): changes seed for random shuffling of documents. Defaults to 42.
            relative_paths (bool, optional): if paths in df are relative, set relative_paths = True. Defaults to True.

        Returns:
            DataFrameIterator: an object that iterates over the images provided in the df file.
            Serves to import images in batches efficiently.
        """
        return super().get_data_iterator(generator, df, shuffle, seed, relative_paths, x_col=x_col, y_col=y_col)


if __name__ == "__main__":
    logging.basicConfig(
        format=LOGGING_FORMAT,
        level=LOGGING_LEVELS["INFO"],
        handlers=[logging.FileHandler(LOGGING_FILENAME), logging.StreamHandler()],
    )
    feature_extractor = FeatureExtractor("AlexNet", verbose=True)
    feature_extractor.mlflow_train_pipeline()
