# TODO testing
import pickle
from occrplib.utils.gaussian import get_classes_thresholds


def test_fit_gaussian_model() -> None:
    # tested indirectly by test_get_classes_thresholds()
    NotImplementedError


def test_get_gaussian_parameters_per_class() -> None:
    # tested indirectly by test_get_classes_thresholds()
    NotImplementedError


def test_get_classes_thresholds() -> None:
    with open("tests/test_data/train_prediction_dict.pkl", "rb") as f:
        loaded_dict: dict = pickle.load(f)
    scale: int = 3

    thresholds = get_classes_thresholds(loaded_dict, scale)
    for value in thresholds.values():
        assert value >= 0.5
        assert value <= 1.0

    # all the thresholds.values() in this example are = 0.5
    # we should save better example data and write more meaningful tests

    # for key in loaded_dict.keys():
    #     for inner_key in loaded_dict[key].keys():
    #         loaded_dict[key][inner_key] = 1 - loaded_dict[key][inner_key]

    # with open('tests/test_data/train_prediction_dict.pkl', 'wb') as f:
    #   pickle.dump(train_prediction_dict, f)
