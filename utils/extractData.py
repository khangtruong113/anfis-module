import json

import numpy as np
import pandas as pd

FILE_DIRECTORY = "dataset/dru_10_minutes.csv"
HEADER = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
          "meanCPUUsage", "canonical memory usage", "AssignMem",
          "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
          "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
          "max_disk_io_time", "cpi", "mai",
          "sampling_portion", "agg_type", "sampled_cpu_usage"]


def get_information(config: str, directory: str):
    """

    :param directory:
    :param config:
    :return:
    """
    # Read json file
    with open(config, 'r') as file:
        info = json.load(file)

    dataframe = pd.read_csv(directory, names=HEADER)
    x_train, y_train, x_test, y_test = extract_data(raw_data=dataframe,
                                                    window_size=info.get("window_size"),
                                                    attribute=info.get("attribute"),
                                                    train_percentage=info.get("train_ratio"))
    return x_train, y_train, x_test, y_test, info.get("window_size"), info.get("attribute")


def extract_data(raw_data: pd.DataFrame, window_size: int, attribute: str, train_percentage: float) -> object:
    """
    :rtype: object
    """
    # data
    data = np.asarray(generate_to_data(raw_data, window_size, attribute))
    train_size = int(data.shape[0] * train_percentage)

    # Training data
    tmp_x_train = np.asarray(data[:train_size, :-1])
    x_train_ = np.float32(np.reshape(tmp_x_train, [tmp_x_train.shape[0], 1, tmp_x_train.shape[1]]))

    tmp_y_train = np.asarray(data[:train_size, -1])

    y_train_ = np.float32(np.reshape(tmp_y_train, [tmp_y_train.shape[0], 1]))
    # Test data
    tmp_x_test = np.asarray(data[train_size:, :-1])
    tmp_y_test = np.asarray(data[train_size:, -1])

    x_test_ = np.float32(np.reshape(tmp_x_test, [tmp_x_test.shape[0], 1, tmp_x_test.shape[1]]))
    y_test_ = np.float32(np.reshape(tmp_y_test, [tmp_y_test.shape[0], 1]))
    return x_train_, y_train_, x_test_, y_test_


# Ham generate du lieu tu file ra data ma ANFIS co the train duoc
def generate_to_data(ss, window_size, attribute) -> list:
    window_size += 1
    d = np.asarray(ss[attribute])
    temp_data = []
    for i in np.arange(d.shape[0] - window_size):
        temp = []
        for j in np.arange(window_size):
            temp.append(d[i + j])
        temp_data.append(temp)
    return temp_data


def get_time_series(directory: str, attribute: str) -> np.ndarray:
    """
    :param attribute:
    :param directory:
    :return:
    """
    # Getting time series on dataset
    dataframe = pd.read_csv(directory, names=HEADER)
    return np.asarray(dataframe[attribute])
