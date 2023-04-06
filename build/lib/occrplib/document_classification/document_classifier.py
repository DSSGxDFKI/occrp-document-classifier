from __future__ import annotations
import json
import logging
import os
from typing import List, Optional

import keras.optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.engine.functional import Functional
from tf_explain.core.grad_cam import GradCAM
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf

from occrplib.config import settings
from occrplib.model.image_model import ImageModel
from occrplib.preprocessing.convert_to_img import convert_all_to_jpg
from occrplib.preprocessing.train_val_test_split import get_split_and_labels, compare_input_with_selected_labels
from occrplib.preprocessing.import_data import import_images
from occrplib.utils.gaussian import get_classes_thresholds

logger = logging.getLogger(__name__)


class DocumentClassifier(ImageModel):
    """Document classifier transfer learning model. Is build upon the feature extraction model."""

    # ------------------------------------------------------- #
    # ------------------ Initialization:
    # ------------------------------------------------------- #

    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        yes_to_all_user_input: bool = False,
        training_augmentation: bool = True,
        input_images_path: str = settings.INPUT_IMAGES_TRANSFER_LEARNING,
        train_test_split_path: str = "",
        scale: int = 3,
        output_path: str = settings.OUTPUT_TRANSFER_LEARNING
    ) -> None:
        """Function that initializes the parameters of an object of the class DocumentClassifier.
            It uses the initialization function of the class imageModel, with the specific parameters
            in the config.py file associated with the document-classification training task.
        Args:
            model_name (str): string containing the model name matching the dictionary of models provided.
            This parameter is usually provided via CLI.

            verbose (bool, optional): Defaults to False.
            training_augmentation (bool, optional): If training_augmentation = True, it performs
            data augmentation on training set. Defaults to True.
            input_images_path (_type_, optional): path to directory containing the documents to
            be used for training. Defaults to settings.INPUT_IMAGES_TRANSFER_LEARNING.
            train_test_split_path (str, optional): path to directory with the labels.txt, train.txt,
            val.txt and test.txt files. Defaults to "".
            scale (int, optional): Number of standard deviations used by the probability threshold
            method used in prediction. Defaults to 3.

        Raises:
            ValueError: "Couldn't find the folder where the weights of the feature extractions are, exiting."
            ValueError: "Couldn't open the folder where the weights of the feature extractions are, exiting.
            Check permissions."
        """

        super().__init__(
            model_name=model_name,
            verbose=verbose,
            yes_to_all_user_input=yes_to_all_user_input,
            input_images_path=input_images_path,
            output_path=output_path,
            train_test_split_path=train_test_split_path,
            **settings.MODEL_CONFIG_TRANSFER_LEARNING[model_name],
            **settings.PROCESSOR_SETTINGS,
            **settings.MODEL_CONFIG_FEATURE_EXTRACTION["regularization_settings"],
        )
        self.RAW_DOCUMENTS_TRANSFER_LEARNING = settings.RAW_DOCUMENTS_TRANSFER_LEARNING
        self.LABELS_FILTER: List[str] = settings.LABELS_FILTER
        self.ENABLE_CATEGORY_OTHERS = True
        self.CLASS_THRESHOLDS_PATH = os.path.join(self.model_input_path, "class_thresholds.json")
        self.train_only_with_first_pages = True
        self.mlflow_experiment_name = "Document Classification"
        self.scale = scale
        self.model: Optional[Sequential] = None
        self.training_augmentation = training_augmentation
        self.disable_feature_extractor = settings.DISABLE_FEATURE_EXTRACTOR
        
        if not self.disable_feature_extractor:
            if not os.path.exists(self.model_feature_extractor) and not os.path.exists(
                mlflow.get_tracking_uri().split("file://")[-1]
            ):
                raise ValueError("Couldn't find the folder where the weights of the feature extractions are, exiting.")
            elif not os.access(self.model_feature_extractor, os.R_OK) and not os.access(
                mlflow.get_tracking_uri().split("file://")[-1], os.R_OK
            ):
                raise ValueError(
                    "Couldn't open the folder where the weights of the feature extractions are, exiting. Check permissions."
                )

        self.training_augmentation = training_augmentation

    def mlflow_log_parameters(self) -> None:
        """Function that initializes the key identifier parameters of a Document Classifier training session
        into mlflow.
        """
        super().mlflow_log_parameters()
        mlflow.log_param("path_feature_extraction_model", self.model_feature_extractor)
        mlflow.log_param("LABELS_FILTER", self.LABELS_FILTER)
        mlflow.log_param("training_augmentation", self.training_augmentation)
        mlflow.log_param("ENABLE_CATEGORY_OTHERS", self.ENABLE_CATEGORY_OTHERS)
        mlflow.log_param("input_images_path", self.input_images_path)
        mlflow.log_param("train_only_with_first_pages", self.train_only_with_first_pages)
        mlflow.log_param("n_std", self.scale)

    def add_others_to_labels(self) -> None:
        """Function that adds the "others" label to our list of labels."""
        self.labels_df: pd.DataFrame = pd.concat(
            [self.labels_df, pd.DataFrame({"index": [len(self.labels_df)], "label": ["other"]})]
        ).reset_index(drop=True)

        self.process_labels()

    # ------------------------------------------------------- #
    # ------------------ Model training:
    # ------------------------------------------------------- #

    def load_transfer_learning_model(
        self,
        activation: str = "softmax",
        loss: str = "categorical_crossentropy",
        optimizer: Optional[keras.optimizers.optimizer_v2.adam.Adam] = tf.keras.optimizers.Adam(learning_rate=0.0001),
    ) -> None:
        """Function that loads the feature extraction model provided in the
            class variable "model_feature_extractor", eliminates the original
            output layer, and adds a new output layer corresponding to the
            document classification task. Then, an optimizer is selected for
            the training and the model is compiled.

        Args:
            activation (str, optional): activation function used in the output layer. Defaults to "softmax".
            loss (str, optional): loss function used by the model. Defaults to "categorical_crossentropy".
            optimizer (_type_, optional): optimizer used by the training procedure. Defaults to tf.keras.optimizers.Adam(learning_rate=0.0001).
        """  # NOQA E501

        logger.info("Loading feature extraction model")

        if self.ENABLE_CATEGORY_OTHERS:
            activation = "sigmoid"
            loss = "binary_crossentropy"

        feature_extraction_model = tf.keras.models.load_model(self.model_feature_extractor)

        if self.model_name in ["ResNet50", "EfficientNetB0", "EfficientNetB4", "EfficientNetB4BW", "EfficientNetB7"]:
            transfer_model: Sequential = Sequential()
            for layer in feature_extraction_model.layers[0:-1]:
                transfer_model.add(layer)
        else:
            transfer_model = feature_extraction_model
            transfer_model.pop()

        transfer_model.add(Dense(self.n_labels, activation=activation, name="output_transfer_learning"))

        transfer_model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        if self.verbose:
            transfer_model.summary()

        self.model = transfer_model

    def train_model(self, skip_conversion: bool = True) -> None:
        """Function train_model instance for the class DocumentClassifier. The placeholder for this function
        can be found in the parent class image_model. It performs the sequence of steps necessary for training
        a document classification model: create folders, convert all unconverted files into per-page image format,
        create training/validating/testing split, load weights from feature extraction and setup model,
        train model with data and save assessment outputs.
        """

        self.create_folders()
        if not skip_conversion:
            self.convert_images()
        self.split_train_val_test_labels()
        self.set_train_val_test_data_iterators()
        self.load_transfer_learning_model()
        self.training_loop()

        if self.ENABLE_CATEGORY_OTHERS:
            self.generate_classes_thresholds(save=True)
            self.add_others_to_labels()

        self.save_accuracy_loss()
        self.save_in_sample_validations()

        self.save_model_inputs()
        self.run_tfexplain()
        self.clean_up()

    def get_callbacks(self, save_best_only: bool = True, early_stopping: bool = True) -> list:
        """Generates the Keras callbacks used for saving models throughout the training process.
            We allow for two functionalities: save only the best model/save models for all epochs,
            and perform early stopping vs continue training until completion.

        Args:
            save_best_only (bool, optional): If True, saves only the model with the lowest validation loss.
            Defaults to True.
            early_stopping (bool, optional): If True, it trains stops the training process if 3 consecutive
            epochs do not provide improvement in the validation loss. Defaults to True.

        Returns:
            list: a list containing the callbacks to be used by the model.fit method in the training_loop() method.
        """

        return super().get_callbacks(save_best_only, early_stopping)

    # ------------------------------------------------------- #
    # ------------------ Robust prediction with thresholds:
    # ------------------------------------------------------- #

    def generate_classes_thresholds(self, save: bool = False) -> None:
        """Function that generates the probability thresholds used for
            our robust prediction framework.
        Args:
            save (bool, optional): If save = True, it saves the thresholds to a json dictionary. Defaults to False.
        """

        prediction_df = self.predict_from_df(self.train_df, predict_col=False)
        prediction_df["true_label"] = self.train_df["true_label"]
        train_prediction_dict = {}
        for label in self.text_labels:
            train_prediction_dict[label] = prediction_df[prediction_df["true_label"] == label][label]

        self.class_thresholds = get_classes_thresholds(train_prediction_dict, scale=self.scale)

        if save:
            self.export_thresholds()

    def export_thresholds(self) -> None:
        """Function that saves a threshold dictionary in a json file."""

        with open(self.CLASS_THRESHOLDS_PATH, "w") as thresholds_file:
            json.dump(self.class_thresholds, thresholds_file, indent=4)
        mlflow.log_artifact(self.CLASS_THRESHOLDS_PATH)

    def import_thresholds(self, thresholds_path: str) -> None:
        """Function that imports the thresholds used for robust document type
            prediction from a given json file.

        Args:
            thresholds_path (str): path to a json file containing a threshold dictionary.
        """
        with open(thresholds_path, "r") as thresholds_file:
            self.class_thresholds = json.load(thresholds_file)

    # ----------------------------------------INPUT_IMAGES_TRANSFER_LEARNING--------------- #
    # ------------------ Importing data:
    # ------------------------------------------------------- #

    def convert_images(self, skip_converted_files: bool = True) -> None:
        """Function that converts the documents in the RAW_DOCUMENTS_TRANSFER_LEARNING
            directory into a sequence of per-page jpg files. The resulting image files are
            saved in the directory INPUT_IMAGES_TRANSFER_LEARNING defined in config.py.

        Args:
            skip_converted_files (bool, optional): If skip_converted_files = True, it only converts files that
            are not in format. Defaults to True.
        """

        if self.RAW_DOCUMENTS_TRANSFER_LEARNING:
            logger.info("Converting documents into images")
            convert_all_to_jpg(
                self.RAW_DOCUMENTS_TRANSFER_LEARNING,
                self.input_images_path,
                skip_converted_files=skip_converted_files,
            )
            logger.info("All converted!")
        else:
            logger.info("Raw documents could not be found, skipping conversion step")

    def split_train_val_test_labels(self) -> None:
        """Function that creates the dataframes train_df, test_df and val_df, which contain the
            paths to the training, testing and validating datasets. If a path is provided in
            the variable train_test_split_path, it loads this split from files. Otherwise, a new
            random split is created.

        Returns:
            _type_: no meaningful return is provided.
        """

        if self.yes_to_all_user_input is False:
            compare_input_with_selected_labels(self.input_images_path, self.LABELS_FILTER)

        logger.info("Splitting the input dataset into train, validation and test sets")

        if self.train_test_split_path:
            logger.info("Loading previously generated train-val-test split")
            return super().split_train_val_test_labels()

        (self.train_df, self.val_df, self.test_df, self.labels_df) = get_split_and_labels(
            self.input_images_path, self.LABELS_FILTER, only_first_page=self.train_only_with_first_pages
        )
        try:
            if self.test_df_path:
                self.test_df = pd.read_csv(self.test_df_path, sep=" ", names=["file_path", "true_str_index"], dtype=str)
        except AttributeError as e:
            logger.error(f"Error in reading the test_df from disk: {e}")
            pass

        # for df in [self.train_df, self.val_df, self.test_df]:
        #     df["doc type str"] = df["doc type"].astype("str")

        self.process_labels()

        n_train = len(self.train_df)
        n_val = len(self.val_df)
        n_test = len(self.test_df)

        logger.info(f"Train size: {n_train}. Validation size: {n_val}. Test sample size: {n_test}")

    def set_train_test_generators(self) -> None:
        """Function that initializes the data generators used by the training function.
        The generator "original_datagen" is setup to perform direct data import with no
        modification. The generator "data_gen_x7" can admit either direct import or
        import with augmentation, depending on the class parameter training_augmentation.
        """
        self.original_datagen = self.get_data_generator()
        self.data_gen_x7 = self.get_data_generator(augmentation=self.training_augmentation)

    def save_train_test_split(self) -> None:
        """Function that saves in "train.txt", "val.txt" and "val.txt" the training/validation/testing
        split dataframes.
        """
        for filename, df in [("train.txt", self.train_df), ("val.txt", self.val_df), ("test.txt", self.test_df)]:
            path = os.path.join(self.model_input_path, filename)

            # Save to disk
            df[["file_path", "true_label"]].to_csv(
                path,
                sep=" ",
                header=False,
                index=False,
            )

            # Save to MLflow
            mlflow.log_artifact(path, artifact_path="split")

    # ------------------------------------------------------- #
    # ------------------ Model assessment:
    # ------------------------------------------------------- #

    def save_model_inputs(self) -> None:
        """Function that saves the data inputs associated with the training process. More specifically,
        we save in "train.txt", "val.txt" and "val.txt" the training/validation/testing
        split dataframes. Also saves in "label.csv" the encoding of the text labels into integer
        numbers.
        """
        self.save_train_test_split()
        self.save_labels()

    def save_labels(self) -> None:
        """Function that saves in "label.csv" the encoding of the text labels into integer
        numbers.
        """
        labels_path = os.path.join(self.model_input_path, "labels.csv")
        self.labels_df.to_csv(labels_path, index=False)
        mlflow.log_artifact(labels_path)

    def run_tfexplain(self) -> None:
        """This function automatically generates visualization of the "regions of interest" that
        the model uses in each image to classify a given page as a class. By default, warm
        colors correspond to high attention/importance regions, and cold colors correspond to
        low attention/importance regions. In particular, we use the visualization method GradCAM,
        which is part of the library tf-explain on Python. To see further documentation of regions
        of interest and attention for visual neural networks, we recommend visiting the tf-explain
        package documentation.
        """

        # ENHANCEMENT tfexplain as function
        # Instead of having run_tfexplain as a fixed part of the pipeline, this method
        # should be a function which take any kind of images and any model and creates
        # the tfexplain figures.

        logger.info("Starting with tfexplain. ")
        explainer = GradCAM()
        tfexplain_nr_images: int = 12

        def tfexplain_dataset(dataframe: pd.DataFrame, afix: str) -> None:
            for label in self.LABELS_FILTER:
                print(f"Starting to process {label} with tfexplain for {afix} dataset")

                # subset by label, reindex
                df_by_label = dataframe[dataframe["true_label"] == label]
                df_by_label = df_by_label.reset_index()

                # saving these tables for future reference:
                df_by_label.to_csv(os.path.join(self.tfexplain_path, f"{afix}_{label}.csv"), sep=",")

                # plot
                def create_and_save_tfexplain_unexplained_overview_plot(
                    afix: str,
                    label: str,
                    save_path: str,
                    df_by_label: pd.DataFrame,
                    tfexplain_nr_images: int = 12,
                ) -> None:

                    figure_unexplained, axis = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False)
                    figure_unexplained.set_figwidth(15)
                    figure_unexplained.set_figheight(12)

                    for i in range(tfexplain_nr_images):
                        if i < df_by_label.shape[0]:
                            img = mpimg.imread(df_by_label["path"][i])

                            x_ = i % 3
                            y_ = i % 4

                            caption_unexplained: str = df_by_label["path"][i].split("/")[-2]
                            caption_unexplained = "\n".join(
                                caption_unexplained[i: i + 20] for i in range(0, len(caption_unexplained), 20)
                            )
                            caption_unexplained = (
                                caption_unexplained[0:39] + "..." if len(caption_unexplained) > 40 else caption_unexplained
                            )

                            axis[x_, y_].set_title(caption_unexplained)
                            axis[x_, y_].imshow(img)

                    plt.savefig(os.path.join(save_path, f"tfexplain_{afix}_{label}_unexplained.png"))
                    mlflow.log_artifact(
                        local_path=os.path.join(save_path, f"tfexplain_{afix}_{label}_unexplained.png"),
                        artifact_path="tfexplain",
                    )
                    plt.close()

                create_and_save_tfexplain_unexplained_overview_plot(
                    afix=afix,
                    label=label,
                    tfexplain_nr_images=tfexplain_nr_images,
                    save_path=self.tfexplain_path,
                    df_by_label=df_by_label,
                )

                # load images
                # ENHANCEMENT refactor
                # This should be refactored. Above we use the DataGenerator, should be used here as well.
                # Afterwards, the function import_images can be deleted. 
                images, labels, faulty = import_images(images_dir="", names=df_by_label, im_size=self.im_size, verbose=False)

                if np.array(labels).shape[0] == 0:  # if folder is empty
                    logger.info(f"Skipped tfexplain for {label} because data folder was empty.")
                    continue

                # create ndarray, normalise images
                labels_ndarray: np.ndarray = np.array(labels).reshape(-1, 1)
                images_normalized = tf.image.per_image_standardization(images)
                images_normalized = np.array(images_normalized)

                # pick layer name for tfexplain
                layer_names: list = []
                layer_name: str
                model_for_tfexplain: Sequential = self.model
                if self.model is not None:
                    for layer in self.model.layers:
                        # check if a layer is actually a Functional object containing more layers, unpack it
                        # this happens with our EfficientNet architecture
                        if type(layer) == Functional:
                            model_for_tfexplain = self.model.layers[0]
                            for inside_layer in layer.layers:
                                layer_names.append(inside_layer.name)
                                if "conv2d_" in inside_layer.name:
                                    layer_name = inside_layer.name  # pick either the last conv2d one
                        else:
                            layer_names.append(layer.name)
                            if "conv2d_" in layer.name:
                                layer_name = layer.name  # pick either the last conv2d one
                if layer_names.count("conv2d_last") == 1:
                    layer_name = "conv2d_last"  # or the one with the name conv2d_last
                if layer_names.count("top_conv") == 1:
                    layer_name = "top_conv"  # or the one with the name top_conv

                max_n_overview_plots: int = 1  # increase this to get more plots
                n_subplots_per_plot: int = 12  # better to not touch this
                n_overview_plots: int = min(round(len(df_by_label) / n_subplots_per_plot), max_n_overview_plots)

                for overview_plot in range(n_overview_plots):
                    # Create overview figure for several images
                    figure_explained, axis = plt.subplots(nrows=3, ncols=4)
                    figure_explained.set_figwidth(15)
                    figure_explained.set_figheight(12)

                    # tf explain
                    for i in range(tfexplain_nr_images):
                        select_img_number: int = i + overview_plot * n_subplots_per_plot
                        if select_img_number < images_normalized.shape[0]:
                            explain_train = explainer.explain(
                                validation_data=(
                                    images_normalized[[select_img_number], ...],
                                    labels_ndarray[[select_img_number]],
                                ),
                                model=model_for_tfexplain,
                                class_index=1,
                                layer_name=layer_name,
                            )

                            x_ = i % 3
                            y_ = i % 4

                            caption: str = df_by_label["path"][i].split("/")[-2]
                            caption = "\n".join(caption[i: i + 20] for i in range(0, len(caption), 20))
                            caption = caption[0:39] + "..." if len(caption) > 40 else caption

                            axis[x_, y_].set_title(caption)
                            axis[x_, y_].imshow(explain_train)

                    # Save overview figure with several images
                    plt.savefig(os.path.join(self.tfexplain_path, f"tfexplain_{afix}_{label}_explained{overview_plot}.png"))
                    logger.info(f"created tf_explain overview graph for {label}, {afix} data")
                    mlflow.log_artifact(
                        local_path=os.path.join(
                            self.tfexplain_path, f"tfexplain_{afix}_{label}_explained{overview_plot}.png"
                        ),
                        artifact_path="tfexplain",
                    )
                    plt.close()

        tfexplain_dataset(self.train_df, afix="train")
        tfexplain_dataset(self.val_df, afix="val")

    # ------------------------------------------------------- #
    # ------------------ Model loading:
    # ------------------------------------------------------- #

    def load_trained_classifier_from_mlflow(self, run_id: str) -> None:
        """Function that loads the trained document classification model from an
            mlflow session outputs.

        Args:
            run_id (str): The mlflow id corresponding to the model of interest.

        Raises:
            ValueError: "You have a model!"
        """
        labels_csv_path = os.path.join(
            (mlflow.get_run(run_id).info.artifact_uri.split("file://")[-1]), "labels.csv"
        )
        if self.model is None:
            self.model = mlflow.keras.load_model(f"runs:/{run_id}/model")
            self.load_labels_from_csv(labels_csv_path)
        else:
            raise ValueError("You have a model!")

    def load_trained_classifier_from_path(self, model_path: str, labels_csv_path: str) -> None:
        """Function that loads the document classification model directly
            from a path where it is saved.

        Args:
            model_path (str):  path to classifier model weights.
            labels_csv_path (str): path to csv with label encoding.

        Raises:
            ValueError: when there is a model already loaded
        """

        if self.model is None:
            self.model = tf.keras.models.load_model(model_path)
            self.load_labels_from_csv(labels_csv_path)
        else:
            raise ValueError("You have a model!")

    @classmethod
    def load_from_path(cls, model_name: str, model_path_or_id: str) -> DocumentClassifier:
        """Function that loads a trained model from either mlflow logs or from
            direct path.

        Args:
            model_name (str): _description_
            model_path_or_id (str): _description_

        Returns:
            DocumentClassifier: _description_
        """
        document_classifier = cls(model_name=model_name)
        if os.path.isdir(model_path_or_id):  # load model from dir
            document_classifier.load_trained_classifier_from_path(
                os.path.join(model_path_or_id, "weights", "best_model"),
                os.path.join(model_path_or_id, "model_inputs", "labels.csv"),
            )
            thresholds_path = os.path.join(model_path_or_id, "model_inputs", "class_thresholds.json")
        else:  # load model from MLflow
            document_classifier.load_trained_classifier_from_mlflow(model_path_or_id)
            thresholds_path = os.path.join(
                (mlflow.get_run("cf37dad071c542ef8738ad9c8c1e24b6").info.artifact_uri.split("file://")[-1]),
                "class_thresholds.json",
            )
        # set generators
        document_classifier.set_train_test_generators()

        # load thresholds if they exist

        if os.path.isfile(thresholds_path):
            document_classifier.import_thresholds(thresholds_path)
            document_classifier.ENABLE_CATEGORY_OTHERS = True
        else:
            document_classifier.ENABLE_CATEGORY_OTHERS = False

        return document_classifier
