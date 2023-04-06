# TODO testing
import random
import pandas as pd
import numpy as np

from occrplib.utils.confusion_matrix import generate_confusion_matrix


def test_generate_confusion_matrix() -> None:
    labels: list = [
        "interesting_documents",
        "boring_documents",
        "exciting_documents",
        "mysterious_documents",
        "secret_documents",
        "scientifically_proven_wrong_documents",
        "top_secret_documents",
    ]

    labels_dict: dict = {v: k for (v, k) in enumerate(labels)}
    encoder_label = {v: k for k, v in enumerate(labels)}

    y_true: pd.Series = pd.Series()
    y_pred: pd.Series = pd.Series()
    for i in range(400):
        ran_int: int = random.randint(0, len(labels))
        apend_y_true = labels[ran_int - 1]
        y_true = y_true.append(pd.Series(apend_y_true))

        if random.random() > 0.2:
            append_y_pred = apend_y_true
        else:
            if ran_int != len(labels):
                append_y_pred = labels[ran_int + 1 - 1]
            else:
                append_y_pred = labels[ran_int - 1 - 1]

        y_pred = y_pred.append(pd.Series(append_y_pred))

    confusion_matrix = generate_confusion_matrix(y_true.map(encoder_label), y_pred.map(encoder_label), labels_dict)

    for label in labels:
        assert label in confusion_matrix.columns
        assert confusion_matrix[label].dtype == np.int64

        # columns and rows have the correct sums
        sum(confusion_matrix.iloc[encoder_label[label]]) == len(y_true[y_true == label])
        sum(confusion_matrix[label]) == len(y_pred[y_pred == label])
