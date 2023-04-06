import logging
import os
import subprocess
import textwrap
import time
from subprocess import CompletedProcess
from typing import Dict, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_score, recall_score

from occrplib.config import settings
from occrplib.utils.confusion_matrix import generate_confusion_matrix
from occrplib.utils.logger import LOGGING_FILENAME

logger = logging.getLogger(__name__)


class ImageModel:
    """Generic image deep learning classifier"""

    file_path_col = "file_path"
    true_label_col = "true_label"
    true_index_col = "true_index"
    true_str_index_col = "true_str_index"
    predicted_label_col = "predicted_label"
    predicted_index_col = "predicted_index"
    predicted_str_index_col = "predicted_str_index"

    # ------------------------------------------------------- #
    # ------------------ Initialization:
    # ------------------------------------------------------- #

    def __init__(
        self,
        model_name: str,
        output_path: str,
        n_epochs: int,
        im_size: int,
        batch_size: int,
        load_temp_weights_path: str,
        GPU_mb_memory: Optional[int],
        input_images_path: str,
        dropout_param: float = 0,
        regularization_param: float = 0,
        cnn_layer_filters_value: int = 128,
        as_gray: bool = False,
        GPU_id: Optional[int] = None,
        n_CPU: int = 1,
        verbose: bool = False,
        yes_to_all_user_input: bool = False,
        model_id: str = None,
        train_test_split_path: str = "",
        model_feature_extractor: str = "",
    ) -> None:
        """This function initializes all the basis parameters of the ImageModel class.
            The values for these basis parameters are located in the config.py file.

        Args:
            model_name (str): model name as provided in the CLI.
            output_path (str): output path as provided in the config.py file.
            n_epochs (int): number of epochs for training as provided in the config.py file.
            im_size (int): image size as provided in the config.py file
            batch_size (int): size of training batches as provided in the config.py file
            load_temp_weights_path (str): path of a partially trained model to continue model training. If None,
            training starts with random initialization.
            GPU_mb_memory (Optional[int]): ammount of memory to be allocated from the GPU to the training process as
            provided in the config.py file.
            input_images_path (str): path to the directory containing all training images as provided in the config.py file.
            dropout_param (float, optional): probability of dropout layers provided in config.py.
            regularization_param (float, optional): Size of regularization in model layers. The default of zero is
            recommended. Defaults to 0.
            cnn_layer_filters_value (int, optional): Number of filters in the last convolutional layer of a model. Defaults to 128.
            as_gray (bool, optional): If as_gray= True, it imports a grayscale version of images. Otherwise, it used the 3-channel color version. Defaults to False.
            GPU_id (Optional[int], optional): If GPU_id = None, it defaults to using the number of requested CPUs. Otherwise, the GPU of the corresponding ID is allocated for processing. Defaults to None.
            n_CPU (int, optional): If GPU_id = None, the training/prediction is assigned this number of CPUs. Defaults to 1.
            verbose (bool, optional): Defaults to False.
            yes_to_all_user_input (bool, optional): Controls whether user input after starting a command is required or if it will be skipped.
            model_id (str, optional): Defaults to None.
            train_test_split_path (str, optional): path that contains the train/val/test txt files. Defaults to "".
            model_feature_extractor (str, optional): Imports the feature extractor model indicated in the config.py file. Defaults to "".
        """  # noqa E501
        self.model_name = model_name
        self.model: Sequential
        if model_id is None:
            self.model_id: str = time.strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S")
        else:
            self.model_id = model_id

        self.verbose = verbose

        if not settings.IN_DOCKER:
            # if not in docker, user can choose this parameter
            self.yes_to_all_user_input = yes_to_all_user_input
        else:
            # if in docker, always yes to all user inputs, because not user input is possible
            self.yes_to_all_user_input = True

        # root input
        self.input_images_path = input_images_path
        self.train_test_split_path = train_test_split_path

        # output where everything get saved
        self.output_path = output_path
        self.model_path = os.path.join(self.output_path, self.model_id)
        self.model_feature_extractor = model_feature_extractor

        self.weights_path: str = os.path.join(self.model_path, "weights")
        self.model_input_path: str = os.path.join(self.model_path, "model_inputs")
        self.assessment_path: str = os.path.join(self.model_path, "assessment")
        self.tfexplain_path: str = os.path.join(self.model_path, "tfexplain")

        # train val test labels split
        if self.train_test_split_path:
            self.train_df_path = os.path.join(self.train_test_split_path, "train.txt")
            self.val_df_path = os.path.join(self.train_test_split_path, "val.txt")
            self.test_df_path = os.path.join(self.train_test_split_path, "test.txt")
            self.labels_df_path = os.path.join(self.train_test_split_path, "labels.csv")

        # parameters
        self.im_size = im_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.load_temp_weights_path = load_temp_weights_path
        self.as_gray = as_gray
        self.color_mode = "grayscale" if self.as_gray else "rgb"

        self.n_CPU = n_CPU
        self.GPU_id = GPU_id
        self.GPU_mb_memory = GPU_mb_memory

        self.dropout_param = dropout_param
        self.regularization_param = regularization_param
        self.cnn_layer_filters_value = cnn_layer_filters_value

        self.class_thresholds: Dict[str, float] = {}
        self.mlflow_experiment_name: str = ""

        self.set_resources()

    def mlflow_train_pipeline(self) -> None:
        """Function that both initializes mlflow, and saves parameters of a training model into
            the mlflow pipeline.
        Raises:
            e: _description_
        """
        mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)
        mlflow.tensorflow.autolog()
        # os.chmod(mlflow.get_tracking_uri().split("file://")[-1], 0o0777)
        # self.clean_up()

        with mlflow.start_run() as run:
            try:
                self.mlflow_log_parameters()
                self.train_model()
            except Exception as e:
                logger.exception("The following error occurred when trying to run the" f"training pipeline: {repr(e)}")
                raise e
            finally:
                mlflow.log_artifact(LOGGING_FILENAME)
        logger.info(f"Finished run: {run.info}")

    def mlflow_log_parameters(self) -> None:
        """Function that initializes the key identifier parameters of a model training session
        into mlflow.
        """
        mlflow.log_param("model_name", self.model_name)
        mlflow.log_param("model_id", self.model_id)
        mlflow.log_param("as_gray", self.as_gray)
        mlflow.log_param("model_path", self.model_path)
        mlflow.log_param("input_images_path", self.input_images_path)

    def create_folders(self) -> None:
        """Creates the necessary folders to save the model output
        Raises:
            ValueError: checks if the images input path exists
        """

        if not os.path.isdir(self.input_images_path):
            raise ValueError("Images input path should exist")

        folders_to_create = [
            self.output_path,
            self.model_path,
            self.weights_path,
            self.model_input_path,
            self.assessment_path,
            self.tfexplain_path,
        ]
        for folder in folders_to_create:
            os.makedirs(folder, exist_ok=True)

    def set_resources(self) -> None:
        """Set CPU or GPU for Tensorflow processing
        Raises:
            ValueError:  if the GPU id doesn't exist
        """

        if self.GPU_id is None:
            logger.info("Setting CPU")
            # tf.config.threading.set_intra_op_parallelism_threads(self.n_CPU)
            # tf.config.threading.set_inter_op_parallelism_threads(self.n_CPU)
        else:
            logger.info("Setting GPU")
            gpus = tf.config.list_physical_devices("GPU")
            if self.GPU_id < len(gpus):
                tf.config.set_visible_devices(gpus[self.GPU_id], device_type="GPU")
                tf.config.set_logical_device_configuration(
                    tf.config.get_visible_devices("GPU")[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=self.GPU_mb_memory)],
                )
            else:
                raise ValueError(f"GPU id {self.GPU_id} doesn't exist")

    def split_train_val_test_labels(self) -> None:
        """Imports the training/testing/validating split used by our training process,
        contained in the train.txt, val.txt and test.txt files.
        """
        self.load_labels_from_csv(self.labels_df_path)

        self.train_df = pd.read_csv(self.train_df_path, sep=" ", names=["file_path", "true_str_index"], dtype=str)
        self.val_df = pd.read_csv(self.val_df_path, sep=" ", names=["file_path", "true_str_index"], dtype=str)
        self.test_df = pd.read_csv(self.test_df_path, sep=" ", names=["file_path", "true_str_index"], dtype=str)

        for df in [self.train_df, self.val_df, self.test_df]:
            df["true_index"] = df["true_str_index"].astype("int")
            df["true_label"] = df["true_index"].map(self.labels_dict)

    def process_labels(self) -> None:
        """Function that, given a label encoding dataframe, creates parameters of
        interest such as number of labels and different label formats.
        """
        self.n_labels = len(self.labels_df)
        self.int_labels = self.labels_df["index"].to_list()
        self.int_labels_str = list(map(str, self.int_labels))
        self.text_labels = self.labels_df["label"].to_list()
        self.labels_dict: dict = self.labels_df.set_index("index")["label"].to_dict()
        logger.info(f"Labels found: {self.text_labels}")

    # ------------------------------------------------------- #
    # ------------------ Model training:
    # ------------------------------------------------------- #

    def train_model(self) -> None:
        """Method placeholder. The implementation of this method is done in the sub-classes
            FeatureExtractor, PrimaryPageClassifier and DocumentClassifier.
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def get_callbacks(self, save_best_only: bool = False, early_stopping: bool = True) -> list:
        """Generates the Keras callbacks used for saving models throughout the training process.
            We allow for two functionalities: save only the best model/save models for all epochs,
            and perform early stopping vs continue training until completion.

        Args:
            save_best_only (bool, optional): save best model from all epochs trained. Defaults to False.
            early_stopping (bool, optional): turn on early stopping, model will stop training when
                val_loss stops decreasing. Defaults to True.

        Returns:
            list: list with callback objects
        """
        callbacks = []

        output_folder = "best_model" if save_best_only else "epoch{epoch:02d}"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.weights_path, output_folder),
            save_weights_only=False,
            save_best_only=save_best_only,
            save_freq="epoch",
            verbose=1,
        )
        callbacks.append(checkpoint_callback)

        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3))

        return callbacks

    def training_loop(self) -> None:
        """Function that performs the training of the chosen model. It incorporates callbacks
        to save the model architecture and weights, and saves 4 assessment measurements:
        both training and validating losses and accuracies.
        """
        self.saved_hist = pd.DataFrame(None, columns=["loss", "accuracy", "val_loss", "val_accuracy"])

        hist = self.model.fit(
            self.train_iterator,
            epochs=self.n_epochs,
            validation_data=self.val_iterator,
            validation_freq=1,
            callbacks=self.get_callbacks(),
        )

        histpd = pd.DataFrame(hist.history)
        self.saved_hist = pd.concat([self.saved_hist, histpd])

    # ------------------------------------------------------- #
    # ------------------ Importing data:
    # ------------------------------------------------------- #

    def get_data_iterator(
        self,
        generator: ImageDataGenerator,
        df: pd.DataFrame,
        shuffle: bool = False,
        seed: int = 42,
        relative_paths: bool = False,
        x_col: Optional[str] = "file_path",
        y_col: Optional[str] = "true_str_index",
    ) -> DataFrameIterator:
        """Function that creates an iterator that systematically imports images into
            memory in batches for training and prediction.
        Args:
            generator (ImageDataGenerator): a generator technique to import the data. Such generators are defined in get_data_generator().
            df (pd.DataFrame): dataframe containing the paths and labels of the documents to import.
            shuffle (bool, optional): set True to shuffle incoming data. Defaults to False.
            seed (int, optional): changes seed for random shuffling of documents. Defaults to 42.
            relative_paths (bool, optional): if paths in df are relative, set relative_paths = True. Defaults to False.
            x_col (str, optional): name of column in df corresponding to file paths. Defaults to "file_path".
            y_col (str, optional): name of column in df corresponding to document page label. Defaults to "true_str_index".

        Returns:
            DataFrameIterator: an object that iterates over the images provided in the df file.
            Serves to import images in batches efficiently.
        """  # NOQA E501
        directory = self.input_images_path if relative_paths else None

        return generator.flow_from_dataframe(
            directory=directory,
            dataframe=df,
            x_col=x_col,
            y_col=y_col,
            shuffle=shuffle,
            seed=seed,
            target_size=(self.im_size, self.im_size),
            batch_size=self.batch_size,
            class_mode=None if y_col is None else "categorical",
            color_mode=self.color_mode,
            classes=list(map(str, range(self.n_labels))),
        )

    def get_data_generator(self, augmentation: bool = False) -> ImageDataGenerator:
        """Function that establishes the importation method of the data.
            There are two options: direct and augmented. When the importation
            is direct, no changes are done to the original image. When importation
            has augmentation, it performs random visual modifications to the
            original images such as: rotation, zoom, brightness, etc.
        Args:
            augmentation (bool, optional): If augmentation = True, performs augmentation by random modification to the original images. Defaults to False.

        Returns:
            ImageDataGenerator: _description_
        """  # NOQA E501
        if augmentation:
            return ImageDataGenerator(
                rescale=1 / 255,
                samplewise_center=True,
                samplewise_std_normalization=True,
                rotation_range=15,
                shear_range=10,
                zoom_range=0.1,
                fill_mode="nearest",
                channel_shift_range=0.9,
                height_shift_range=0.05,
                width_shift_range=0.1,
                brightness_range=[0.5, 1.5],
            )
        return ImageDataGenerator(rescale=1 / 255, samplewise_center=True, samplewise_std_normalization=True)

    def set_train_test_generators(self) -> None:
        """Function that defines the data generator objects for both direct
        importation and augmented importation of the image data.
        """
        self.original_datagen = self.get_data_generator()
        self.data_gen_x7 = self.get_data_generator()

    def set_train_val_test_data_iterators(self) -> None:
        """Function that initializes the data iterators for the training,
        validating and testing datasets.
        """
        self.set_train_test_generators()

        self.train_iterator: DataFrameIterator = self.get_data_iterator(self.data_gen_x7, self.train_df, True)

        self.val_iterator: DataFrameIterator = self.get_data_iterator(
            self.original_datagen,
            self.val_df,
        )

        self.test_iterator: DataFrameIterator = self.get_data_iterator(
            self.original_datagen,
            self.test_df,
        )

    # ------------------------------------------------------- #
    # ------------------ Model assessment:
    # ------------------------------------------------------- #

    def save_accuracy_loss(self) -> None:
        """For a given training session, this function saves the evolution of
        the training/validating accuracy throughout the training of the model.
        This is saved in the assessment folder of the model training output
        folder.
        """

        # Save accuracy/loss evolution:
        path = os.path.join(self.assessment_path, "training_metrics.csv")
        self.saved_hist.to_csv(path, index=False)
        mlflow.log_artifact(path)

    def save_false_predictions(self, prediction_df: pd.DataFrame, name: str) -> None:
        """Creates a csv file with information about those images which have been incorrectly classified.
        Args:
            prediction_df (_type_): DataFrame with columns label1, label2 etc., true-label, predicted, path
            name (_type_): Affix for filename.
        """
        # save csv file
        path_false_predictions = os.path.join(self.assessment_path, f"{name}_false_predictions.csv")
        false_prediction_df: pd.DataFrame = prediction_df[
            prediction_df["true_label"] != prediction_df[self.predicted_label_col]
        ]
        false_prediction_df = false_prediction_df.sample(frac=1).reset_index(drop=True)  # shuffle and reset index
        false_prediction_df.to_csv(path_false_predictions, index=False)
        mlflow.log_artifact(path_false_predictions, artifact_path="false_predictions")

        # plot and save overview figure
        max_n_overview_plots: int = 3  # increase this to get more plots
        n_subplots_per_plot: int = 12  # better to not touch this
        n_overview_plots: int = min(round(len(false_prediction_df) / n_subplots_per_plot), max_n_overview_plots)
        for overview_plot in range(n_overview_plots):
            figure_unexplained, axis = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False)
            figure_unexplained.set_figwidth(15)
            figure_unexplained.set_figheight(12)

            for i in range(n_subplots_per_plot):
                select_img_number: int = i + overview_plot * n_subplots_per_plot
                if select_img_number < len(false_prediction_df):
                    img = mpimg.imread(
                        os.path.join(self.input_images_path, false_prediction_df["file_path"][select_img_number])
                    )

                    x_ = i % 3
                    y_ = i % 4

                    caption: str = false_prediction_df["file_path"][select_img_number].split("/")[-2]
                    caption = "\n".join(caption[i : i + 20] for i in range(0, len(caption), 20))
                    caption = caption[0:19] + "..." if len(caption) > 20 else caption

                    axis[x_, y_].set_title(
                        f"{caption}\n"
                        f"predicted: {false_prediction_df[self.predicted_label_col][select_img_number]}\n"
                        f"true label: {false_prediction_df['true_label'][select_img_number]}"
                    )

                    axis[x_, y_].imshow(img)

            plt.savefig(os.path.join(self.assessment_path, f"{name}_false_predictions{overview_plot}.png"))
            mlflow.log_artifact(
                local_path=os.path.join(self.assessment_path, f"{name}_false_predictions{overview_plot}.png"),
                artifact_path="false_predictions",
            )
            plt.close()

    def save_validations(self, df: pd.DataFrame, name: str) -> None:
        """Given a dataFrame containing the paths to the document pages to be predicted
            by the model, it saves both a report of findings and information
            about incorrect predictions of the model.
        Args:
            df (pd.DataFrame): _description_
            name (str): _description_
        """
        df = df.reset_index(drop=True)
        prediction_df: pd.DataFrame = self.predict_from_df(df, save=True, name=name)
        prediction_df["true_label"] = df["true_label"]
        prediction_df["file_path"] = df["file_path"]
        self.save_test_report_from_df(prediction_df, name)
        self.save_false_predictions(prediction_df, name)

    def save_in_sample_validations(self) -> None:
        """Function that automatically generates the report of findings/ information about
        incorrect predictions for the training, validating and testing datasets.
        """
        for df, name in [(self.train_df, "train"), (self.val_df, "val"), (self.test_df, "test")]:
            logger.info(f"Saving in-sample {name} results...")
            self.save_validations(df, name)

    def save_test_report_from_df(
        self, prediction_df: pd.DataFrame, name: str, true_col: str = "true_label", pred_col: str = "predicted_label"
    ) -> None:
        """Function that saves a markdown report with performance metrics
            of classification
        Args:
            prediction_df (pd.DataFrame): table with the predictions for each document page.
            name (str): name given to the saved metrics.
            true_col (str, optional): column containing the true label. Defaults to "true_label".
            pred_col (str, optional):column containing the predicted label. Defaults to "predicted_label".
        """
        overall_accuracy, metrics_per_class_df, confusion_matrix_df, overall_ap = self.test_metrics_from_df(
            prediction_df, true_col, pred_col
        )
        markdown_report = self.performance_metrics_to_markdown(
            overall_accuracy, metrics_per_class_df, confusion_matrix_df, overall_ap
        )
        if name in ["train", "test"]:
            mlflow.log_metric(f"{name}_accuracy", overall_accuracy)
            mlflow.log_metric(f"{name}_average_precision", overall_ap)
        elif name == "page_2":
            mlflow.log_metric("page_2_accuracy", overall_accuracy)

        report_path = os.path.join(self.assessment_path, f"{name}_report.md")
        with open(report_path, "w") as report_file:
            report_file.write(markdown_report)
        mlflow.log_artifact(report_path)

    def test_metrics_from_df(
        self, prediction_df: pd.DataFrame, true_col: str = "true_label", pred_col: str = "predicted_label"
    ) -> Tuple[float, pd.DataFrame, pd.DataFrame, float]:
        """Function that calculates the overall accuracy, confusion matrix,
            accuracy/precision per label and average precision. It takes as
            inputs the prediction_df dataFrame generated by predict_from_df().
        Args:
            prediction_df (pd.DataFrame): dataframe containing the prediction probabilities generated by predict_from_df().
            true_col (str, optional): name of the column in prediction_df containing the true document-page label. Defaults to "true_label".
            pred_col (str, optional): name of the column in prediction_df containing the predicted document-page label. Defaults to "predicted_label".

        Raises:
            ValueError: "true_col and pred_col should be columns of df"_

        Returns:
            overall_accuracy (float): percentage of correctly classified document pages from the total.
            metrics_per_class_df (pd.DataFrame): dataframe containing per-class accuracy and precision.
            confusion_matrix_df (pd.DataFrame): confusion matrix associated with given predictions.
            overall_ap_ (float): area below the precision-recall curve.
        """  # NOQA E501
        if (true_col not in prediction_df.columns) or (pred_col not in prediction_df.columns):
            raise ValueError("true_col and pred_col should be columns of df")

        encoder_label = {v: k for k, v in self.labels_dict.items()}
        # labels_list = [self.labels_dict[i] for i in range(len(self.labels_dict))]
        y_true, y_pred = prediction_df[true_col], prediction_df[pred_col]

        overall_accuracy = accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=self.text_labels)
        accuracy_classes = cm.diagonal() / cm.sum(axis=1)
        precision_classes = precision_score(y_true, y_pred, average=None, labels=self.text_labels)
        recall_classes = recall_score(y_true, y_pred, average=None, labels=self.text_labels)
        average_precision_classes = [
            average_precision_score((y_true == col).astype("int"), prediction_df[col])
            if col in prediction_df.columns
            else np.nan
            for col in self.text_labels
        ]
        metrics_per_class_df = pd.DataFrame(
            {
                "accuracy": accuracy_classes,
                "precision": precision_classes,
                "recall": recall_classes,
                "average_precision": average_precision_classes,
            },
            index=self.text_labels,
        )

        overall_ap = np.nanmean(
            [average_precision_classes[i] for i, label in enumerate(self.text_labels) if label not in ["other"]]
        )

        confusion_matrix_df = generate_confusion_matrix(
            y_true.map(encoder_label), y_pred.map(encoder_label), self.labels_dict
        )
        return overall_accuracy, metrics_per_class_df, confusion_matrix_df, overall_ap

    def performance_metrics_to_markdown(
        self,
        overall_accuracy: float,
        metrics_per_class_df: pd.DataFrame,
        confusion_matrix_df: pd.DataFrame,
        overall_ap: float,
    ) -> str:
        """Function that generates an markdown-format python object containing a
            report of performance metrics of interest.
        Args:
            overall_accuracy (float): variable containing the overall accuracy of the model, between 0 and 1.
            metrics_per_class_df (pd.DataFrame):
            confusion_matrix_df (pd.DataFrame): _description_
            overall_ap (float): variable containing the average precision of the model, between 0 and 1.
                                The average precision is obtained as area below the precision-recall curve.

        Returns:
            str: string variable containing the markdown report to be saved.
        """
        markdown_report = textwrap.dedent(
            f"""\
            # Test report
            
            Overall accuracy: {100*overall_accuracy:.2f}%

            Overall average precision (PR-AUC): {100*overall_ap:.2f}%

            ## Metrics per class
            
            {metrics_per_class_df.to_markdown()}

            ## Confusion matrix

            {confusion_matrix_df.to_markdown()}

            """
        )

        return "\n".join([line.strip() for line in markdown_report.splitlines()])

    # ------------------------------------------------------- #
    # ------------------ Prediction:
    # ------------------------------------------------------- #

    def load_labels_from_csv(self, labels_csv_path: str) -> None:
        """Function that loads the encoding correspondence between text labels for
            document pages and integers. This label encoding can be found in
            is saved in a csv file in the model_inputs folder of the training model
            output folder.
        Args:
            labels_csv_path (str): path to the csv file containing the label encodings.
        """
        self.labels_df = pd.read_csv(labels_csv_path)
        self.labels_df = self.labels_df[self.labels_df["label"] != "other"]
        self.process_labels()

    def decide_predicted_label(self, prediction_df: pd.DataFrame) -> pd.Series:
        """Using the output probability matrix from the classifier,
            determine which should be the label of each page image.
        Args:
            prediction_df (pd.DataFrame): table with class probabilities (output of classification)
            diff_threshold (Optional[float], optional): threshold for difference between max probability
                and the next. Defaults to None.
        Returns:
            pd.Series: predicted labels
        """
        prediction_labels_df = prediction_df[[col for col in prediction_df.columns if col in self.text_labels]]
        if self.class_thresholds:

            def get_label(row: pd.Series) -> str:
                for label, value in row.sort_values(ascending=False).iteritems():
                    if value > self.class_thresholds[str(label)]:
                        return str(label)
                return "other"

            return pd.Series(prediction_labels_df.apply(get_label, axis=1))
        return prediction_labels_df.idxmax(axis=1)

    def predict_from_df(
        self, df: pd.DataFrame, predict_col: bool = True, save: bool = False, name: str = None
    ) -> pd.DataFrame:
        """Function that receives a list of document-page image paths, and calculates the
            model predictions for each of the provided documents.

        Args:
            df (pd.DataFrame): dataFrame containing the paths to the document pages to be predicted by the model.
            predict_col (bool, optional): If predict_col = True, it adds a column with the predicted label. Defaults to True.
            save (bool, optional): if save = True, saves the outputs in the assessment folder. Defaults to False.
            name (str, optional): name of the file containing the predictions. Defaults to None.

        Raises:
            ValueError: "if save=True, you need a name"

        Returns:
            pd.DataFrame: DataFrame containing, for each document, the likelihood of belonging to each of the document types.
        """
        df_gen = self.get_data_iterator(self.original_datagen, df, y_col=None)

        prediction = self.model.predict(df_gen)
        prediction_df = pd.DataFrame(prediction)
        prediction_df.columns = prediction_df.columns.map(self.labels_dict)

        if predict_col:
            prediction_df["predicted_label"] = self.decide_predicted_label(prediction_df)

        if save:
            if name is None:
                raise ValueError("if save=True, you need a name")

            # save predictions
            path = os.path.join(self.assessment_path, f"{name}_predictions.csv")
            prediction_df.to_csv(path, index=False)
            mlflow.log_artifact(path)

        return prediction_df

    # ------------------------------------------------------- #
    # ------------------ Clean-up:
    # ------------------------------------------------------- #

    def clean_up(self) -> None:
        """Function that changes permissions for the files created."""
        # delete logs, models in temp folder etc?

        grant_permissions: CompletedProcess = subprocess.run(["chmod", "777", "-Rf", settings.ROOT_PATH])
        grant_permissions_returncode: int = grant_permissions.returncode
        if grant_permissions_returncode != 0 and grant_permissions_returncode != 1:
            logger.info(
                f"The command {print(*grant_permissions.args)} returned with the unusal code {grant_permissions_returncode} \
                when trying to grant permissions!"
            )
