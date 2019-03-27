"""
Test read file json
"""
import json
from preprocess import header, fname, extract_data
import pandas as pd


def get_info(directory: str):
    """

    :param directory:
    :return:
    """
    # Read json file
    with open(directory, 'r') as file:
        info = json.load(file)

    dataframe = pd.read_csv(fname, names=header)
    x_train, y_train, x_test, y_test = extract_data(raw_data=dataframe,
                                                    window_size=info.get("window_size"),
                                                    attribute=info.get("attribute"),
                                                    train_percentage=info.get("train_ratio"))
    return x_train, y_train, x_test, y_test, info.get("window_size"), info.get("attribute")
