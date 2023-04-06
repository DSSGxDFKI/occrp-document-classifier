# TODO testing

import os
import pathlib
import json

from occrplib.config import export_settings_to_json
import conftest  # NOQA F501


def test_export_settings_to_json(tmp_path_session: pathlib.Path) -> None:
    json_file = os.path.join(tmp_path_session, "settings.json")
    export_settings_to_json(json_file, conftest.settings)
    assert os.path.isfile(json_file)

    # check if data were saved correctly into json
    with open(json_file, "r") as file:
        json_data = json.load(file)

    for key in [
        "INPUT_IMAGES_TRANSFER_LEARNING",
        "PROCESSOR_SETTINGS",
        "LABELS_FILTER",
        "MLFLOW_LOGS",
        "ENABLE_CATEGORY_OTHERS",
    ]:
        assert key in list(json_data.keys())

    assert isinstance(json_data["ROOT_PATH"], str)
    assert len(json_data["ROOT_PATH"]) > 0

    assert isinstance(json_data["MODEL_CONFIG_FEATURE_EXTRACTION"]["EfficientNetB4"]["im_size"], int)
    assert json_data["MODEL_CONFIG_FEATURE_EXTRACTION"]["EfficientNetB4"]["im_size"] > 0
