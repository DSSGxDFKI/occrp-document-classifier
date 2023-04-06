# TODO testing
import mlflow
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, DataFrameIterator

import conftest  # NOQA F501
from occrplib.model.image_model import ImageModel
from occrplib.document_classification.document_classifier import DocumentClassifier
import pandas as pd


def test_mlflow_log_parameters(image_model: ImageModel) -> None:
    image_model.mlflow_log_parameters()
    run_id: str = mlflow.active_run()._info._run_id
    params: dict = mlflow.get_run(run_id).data._params  # strangely mlflow.active_run().data._params doesn't work

    # check if the following five params have been logged
    assert set(params.keys()) == set(["model_name", "model_id", "as_gray", "model_path", "input_images_path"])


def test_create_folders(image_model: ImageModel) -> None:
    # image_model.create_folders()

    # folders_to_create = [
    #     image_model.output_path,
    #     image_model.model_path,
    #     image_model.weights_path,
    #     image_model.model_input_path,
    #     image_model.assessment_path,
    #     image_model.tfexplain_path,
    # ]
    # for folder in folders_to_create:
    #     assert os.path.isdir(folder)

    # no point in testing this
    NotImplementedError


def test_set_resources(image_model: ImageModel) -> None:
    image_model.set_resources()
    if image_model.GPU_id is None:
        assert tf.config.threading.get_inter_op_parallelism_threads() == image_model.n_CPU
        assert tf.config.threading.get_intra_op_parallelism_threads() == image_model.n_CPU
    else:
        tf.config.get_visible_devices()[0].device_type == "GPU"
        # we could test the other line of that function too


def test_split_train_val_test_labels() -> None:
    # already tested in test_document_classifier.py
    assert True


def test_get_callbacks(image_model: ImageModel) -> None:
    callbacks = image_model.get_callbacks(early_stopping=False)
    assert len(callbacks) == 1

    callbacks_early_stopping = image_model.get_callbacks(early_stopping=True)
    assert len(callbacks_early_stopping) == 2


def test_training_loop(document_classifier: DocumentClassifier) -> None:
    # already tested in test_document_classifier.py
    assert True
    NotImplementedError


def test_get_data_generator(image_model: ImageModel) -> None:
    generator_augmented = image_model.get_data_generator(augmentation=True)
    assert type(generator_augmented) == ImageDataGenerator

    if (
        generator_augmented.rotation_range == 0
        and generator_augmented.shear_range == 0.0
        and generator_augmented.zoom_range == [1.0, 1.0]
        and generator_augmented.fill_mode == "nearest"
        and generator_augmented.channel_shift_range == 0.0
        and generator_augmented.height_shift_range == 0.0
        and generator_augmented.width_shift_range == 0.0
    ):
        assert False  # because some of these should be different to have augmentation

    generator_not_augmented = image_model.get_data_generator(augmentation=False)
    assert type(generator_not_augmented) == ImageDataGenerator

    assert generator_not_augmented.rotation_range == 0
    assert generator_not_augmented.shear_range == 0.0
    assert generator_not_augmented.zoom_range == [1.0, 1.0]
    assert generator_not_augmented.fill_mode == "nearest"
    assert generator_not_augmented.channel_shift_range == 0.0
    assert generator_not_augmented.height_shift_range == 0.0
    assert generator_not_augmented.width_shift_range == 0.0


def test_get_data_iterator(image_model: ImageModel) -> None:
    image_model.set_train_test_generators()

    train_df = pd.read_csv("tests/test_data/train_df")
    image_model.n_labels = len(sorted(list(train_df["class"].unique())))  # = 9

    # this function always breaks in the test even though it works in the actual run with the same data
    # error is that train_df["doc type str"] should not be int, but it in the run it is also int
    iterator: DataFrameIterator = image_model.get_data_iterator(image_model.data_gen_x7, train_df, True)
    iterator2 = image_model.get_data_iterator(image_model.original_datagen, train_df, True)

    assert isinstance(iterator, DataFrameIterator)
    assert isinstance(iterator2, DataFrameIterator)


def test_set_train_test_generators(image_model: ImageModel) -> None:
    image_model.set_train_test_generators()
    assert type(image_model.data_gen_x7) == ImageDataGenerator

    if all(
        [
            image_model.data_gen_x7.rotation_range == 0,
            image_model.data_gen_x7.shear_range == 0.0,
            image_model.data_gen_x7.zoom_range == [1.0, 1.0],
            image_model.data_gen_x7.fill_mode == "nearest",
            image_model.data_gen_x7.channel_shift_range == 0.0,
            image_model.data_gen_x7.height_shift_range == 0.0,
            image_model.data_gen_x7.width_shift_range == 0.0,
        ]
    ):
        assert False  # because some of these should be different to have augmentation


def test_set_train_val_test_data_iterators(image_model: ImageModel, document_classifier: DocumentClassifier) -> None:
    image_model.set_train_test_generators()
    document_classifier.split_train_val_test_labels()

    document_classifier.set_train_val_test_data_iterators()

    isinstance(document_classifier.train_gen, DataFrameIterator)
    isinstance(document_classifier.val_gen, DataFrameIterator)
    isinstance(document_classifier.test_gen, DataFrameIterator)


def test_decide_predicted_label(document_classifier: DocumentClassifier) -> None:
    # very difficult if not impossible to test this with the image model alone, therefore with doc classifier

    document_classifier.split_train_val_test_labels()
    document_classifier.set_train_test_generators()
    document_classifier.load_transfer_learning_model()
    prediction_df = document_classifier.predict_from_df(document_classifier.test_df)

    predicted_labels = document_classifier.decide_predicted_label(prediction_df)
    isinstance(predicted_labels, pd.Series)


def test_predict_from_df(document_classifier: DocumentClassifier) -> None:
    document_classifier.split_train_val_test_labels()
    document_classifier.set_train_test_generators()
    document_classifier.load_transfer_learning_model()
    prediction_df = document_classifier.predict_from_df(document_classifier.test_df, predict_col=False)

    isinstance(prediction_df, pd.DataFrame)

    for column in set(["true-label", "path"] + document_classifier.LABELS_FILTER):
        assert column in set(prediction_df.columns)

    prediction_df_incl_predicted = document_classifier.predict_from_df(document_classifier.test_df, predict_col=True)
    assert "predicted" in set(prediction_df_incl_predicted.columns)

    # some tests on the shape of the prediction_df would be neat
    # li
    assert (
        prediction_df.shape[0]
        == document_classifier.model.predict(
            document_classifier.get_data_iterator(document_classifier.original_datagen, document_classifier.test_df)
        ).shape[0]
    )


def test_save_false_predictions(image_model: ImageModel) -> None:
    # Our code is so messed up that we cannot test this
    # The functions have to refactored to to this

    # image_model.save_false_predictions()

    # name = "train"
    # overview_plot = 0
    # os.path.isfile(os.path.join(image_model.assessment_path, f"{name}_false_predictions{overview_plot}.png"))
    assert False


def test_test_metrics_from_df(document_classifier: DocumentClassifier) -> None:
    document_classifier.split_train_val_test_labels()
    document_classifier.set_train_test_generators()
    document_classifier.load_transfer_learning_model()
    test_prediction_df = document_classifier.predict_from_df(document_classifier.test_df)
    test_prediction_df["predicted_threshold"] = document_classifier.decide_predicted_label(
        test_prediction_df, diff_threshold=document_classifier.class_thresholds
    )

    test_prediction_df = test_prediction_df.dropna(subset=["true-label"])  # not sure if this makes sense but otherwise error

    overall_accuracy, metrics_per_class_df, cm_df, _ = document_classifier.test_metrics_from_df(
        test_prediction_df, true_col="true-label", pred_col="predicted_threshold"
    )
    assert isinstance(overall_accuracy, float)
    assert isinstance(metrics_per_class_df, pd.DataFrame)
    assert set(["accuracy", "precision", "recall", "average_precision"]) == set(metrics_per_class_df.columns)
    assert all(metrics_per_class_df.loc[metrics_per_class_df["accuracy"].notna(), "average_precision"]) > 0
    assert all(metrics_per_class_df.loc[metrics_per_class_df["accuracy"].notna(), "average_precision"]) <= 1
    assert all(cm_df.columns == document_classifier.LABELS_FILTER)


def test_performance_metrics_to_markdown(document_classifier: DocumentClassifier) -> None:
    document_classifier.split_train_val_test_labels()
    document_classifier.set_train_test_generators()
    document_classifier.load_transfer_learning_model()
    test_prediction_df = document_classifier.predict_from_df(document_classifier.test_df)
    test_prediction_df["predicted_threshold"] = document_classifier.decide_predicted_label(
        test_prediction_df, diff_threshold=document_classifier.class_thresholds
    )
    test_prediction_df = test_prediction_df.dropna(subset=["true-label"])  # not sure if this makes sense but otherwise error

    overall_accuracy, metrics_per_class_df, confusion_matrix_df, overall_ap = document_classifier.test_metrics_from_df(
        test_prediction_df, true_col="true-label", pred_col="predicted_threshold"
    )

    markdown = document_classifier.performance_metrics_to_markdown(
        overall_accuracy, metrics_per_class_df, confusion_matrix_df, overall_ap
    )

    isinstance(markdown, str)
