from occrplib.preprocessing.train_val_test_split import custom_train_val_test_split
import pandas as pd
import pytest


@pytest.fixture
def iris_df() -> pd.DataFrame:
    return pd.read_csv("tests/test_data/iris.csv")


def test_custom_train_val_test_split(iris_df: pd.DataFrame) -> None:
    train_df, val_df, test_df = custom_train_val_test_split(iris_df, class_col="variety")

    # test no row is missing after splitting
    assert len(iris_df) == (len(train_df) + len(val_df) + len(test_df))
    assert len(pd.concat([iris_df, train_df, val_df, test_df]).drop_duplicates(keep=False)) == 0


def test_custom_train_val_test_split_random_seed(iris_df: pd.DataFrame) -> None:
    train_df_v1, val_df_v1, test_df_v1 = custom_train_val_test_split(iris_df, class_col="variety", random_seed=99)
    train_df_v2, val_df_v2, test_df_v2 = custom_train_val_test_split(iris_df, class_col="variety", random_seed=99)

    assert train_df_v1.equals(train_df_v2)
    assert val_df_v1.equals(val_df_v2)
    assert test_df_v1.equals(test_df_v2)


def test_custom_train_val_test_split_oversampling(iris_df: pd.DataFrame) -> None:
    train_df, val_df, test_df = custom_train_val_test_split(iris_df, class_col="variety", random_seed=77)
    (
        train_df_oversampling,
        val_df_oversampling,
        test_df_oversampling,
    ) = custom_train_val_test_split(iris_df, class_col="variety", random_seed=77)

    majority_label_count = train_df["variety"].value_counts().max()
    for label in train_df["variety"].unique():
        assert majority_label_count == train_df_oversampling["variety"].value_counts()[label]

    assert val_df.equals(val_df_oversampling)
    assert test_df.equals(test_df_oversampling)


def test_custom_train_val_test_split_undersampling(iris_df: pd.DataFrame) -> None:
    train_df, val_df, test_df = custom_train_val_test_split(iris_df, class_col="variety", random_seed=77)
    (
        train_df_undersampling,
        val_df_oversampling,
        test_df_oversampling,
    ) = custom_train_val_test_split(iris_df, class_col="variety", random_seed=77)

    minority_label_count = train_df["variety"].value_counts().min()
    for label in train_df["variety"].unique():
        assert minority_label_count == train_df_undersampling["variety"].value_counts()[label]

    assert val_df.equals(val_df_oversampling)
    assert test_df.equals(test_df_oversampling)


def test_train_val_test_disjoint(iris_df: pd.DataFrame) -> None:
    train_df, val_df, test_df = custom_train_val_test_split(iris_df.drop_duplicates(), class_col="variety")
    assert len(pd.concat([train_df, val_df, test_df]).drop_duplicates(keep=False)) == (
        len(train_df) + len(val_df) + len(test_df)
    )
