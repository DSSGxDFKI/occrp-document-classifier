import os
from typing import List, Tuple

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from occrplib.utils.helper import query_yes_no


def custom_train_val_test_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    val_size: float = 0.1,
    sampling: str = "normal",
    random_seed: int = 100,
    class_col: str = "class",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a dataframe and returns test, validation and testing dataframes

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: test, validation and testing dataframes
    """
    assert test_size + val_size <= 1

    n_samples = len(df)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)

    train_and_val_df, test_df = train_test_split(df, test_size=n_test, random_state=random_seed, stratify=df[class_col])

    train_df, val_df = train_test_split(
        train_and_val_df,
        test_size=n_val,
        random_state=random_seed,
        stratify=train_and_val_df[class_col],
    )

    if sampling == "over":
        ros = RandomOverSampler(random_state=random_seed)
        train_df, _ = ros.fit_resample(train_df, train_df[class_col])
    elif sampling == "under":
        rus = RandomUnderSampler(random_state=random_seed)
        train_df, _ = rus.fit_resample(train_df, train_df[class_col])

    # for df in [train_df, val_df, test_df]:
    #     df["file_path"] = df["path"]
    #     df["true_label"] = df["class"]

    return train_df, val_df, test_df


def get_data_files_df(directory_path: str, select_labels: List[str] = None) -> pd.DataFrame:
    """Parses a directory structure and returns a dataframe with all the files of type JPG

    Args:
        directory_path (str): folder to parse

    Returns:
        pd.DataFrame: files contained in the parsed directory structure
    """
    data = []
    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if (select_labels and (label not in select_labels)) or (not os.path.isdir(label_path)):
            continue
        for root, dirs, files in os.walk(label_path):
            for filename in files:
                if filename.split(".")[-1].lower() == "jpg":
                    data.append(
                        {
                            "path": os.path.join(root, filename),
                            "class": label,
                            "filename": filename,
                        }
                    )
    return pd.DataFrame(data)


def get_labels_from_data(data_df: pd.DataFrame) -> pd.DataFrame:
    return (
        data_df["class"]
        .str.lower()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
        .to_frame(name="label")
        .reset_index()
    )


def get_split_and_labels(
    datasource_path: str,
    selected_labels: List[str] = None,
    only_first_page: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Takes a path to a directory with images, splits them into train, val and test and
    returns three dataframes with paths and labels.

    Args:
        datasource_path (str): Path to directory with images.
        selected_labels (List[str], optional): _description_. Defaults to None.
        only_first_page (bool, optional): If this is True, only the "1.jpg" files will be processed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: _description_
    """
    data_df = get_data_files_df(datasource_path, select_labels=selected_labels)
    labels_df = get_labels_from_data(data_df)
    data_df["page_number"] = data_df["filename"].str.extract(r"(\d+).jpg")

    if only_first_page:
        # remove non-first-pages
        data_df = data_df[data_df["page_number"] == "1"]

    data_df = data_df.merge(labels_df, how="left", left_on="class", right_on="label")
    data_df["file_path"] = data_df["path"]
    data_df["true_label"] = data_df["label"]
    data_df["true_index"] = data_df["index"]
    data_df["true_str_index"] = data_df["true_index"].astype("str")

    # split to train, val, test
    train_df, val_df, test_df = custom_train_val_test_split(data_df)

    for df in [train_df, val_df, test_df]:
        df.reset_index(drop=True)

    return train_df, val_df, test_df, labels_df


def compare_input_with_selected_labels(datasource_path: str, selected_labels: List) -> None:
    """Function that compares the labels used for training with the labels present with the
        datasource path.

    Args:
        datasource_path (str): The directory containing the processed data, ready for training.
        selected_labels (List): The user selected labels, defined in the config.
    """

    actual_folders: list = os.listdir(datasource_path)

    not_found_labels: set = set(selected_labels) - set(actual_folders)
    if len(not_found_labels) > 0:
        print(f"You selected labels which were not found in the data folder ({datasource_path}):")
        print(*not_found_labels, sep="\n")
        if query_yes_no("Do you want to continue?") is False:
            exit()

    not_selected_labels: set = set(actual_folders) - set(selected_labels)
    if len(not_selected_labels) > 0:
        print(f"You did not select the following label(s) which are in the data folder ({datasource_path}):")
        print(*not_selected_labels, sep="\n")
        if query_yes_no("Do you want to continue?") is False:
            exit()
