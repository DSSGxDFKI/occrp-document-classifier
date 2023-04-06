import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, Iterable


def generate_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], labels: Dict[int, str]) -> pd.DataFrame:
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels=list(labels.keys()),
        )
    )
    confusion_matrix_df.columns = confusion_matrix_df.columns.map(labels)
    confusion_matrix_df.columns.name = "predicted"

    confusion_matrix_df.index = confusion_matrix_df.index.map(labels)
    confusion_matrix_df.index.name = "real"

    return confusion_matrix_df
