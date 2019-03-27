import numpy as np

# Ten file duoc dua vao ANFIS network
fname = "dataset/data_resource_usage_10Minutes_6176858948.csv"

# Cac Header trong file
header = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
          "meanCPUUsage", "canonical memory usage", "AssignMem",
          "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
          "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
          "max_disk_io_time", "cpi", "mai",
          "sampling_portion", "agg_type", "sampled_cpu_usage"]


def extract_data(raw_data, window_size, attribute, train_percentage):
    """

    :rtype: object
    """
    # data
    data = np.asarray(gen_to_data(raw_data, window_size, attribute))
    train_size = int(data.shape[0] * train_percentage)

    # Training data
    tmp_x_train = np.asarray(data[:train_size, :-1])
    x_train_ = np.reshape(tmp_x_train, [tmp_x_train.shape[0], 1, tmp_x_train.shape[1]])

    tmp_y_train = np.asarray(data[:train_size, -1])

    y_train_ = np.reshape(tmp_y_train, [tmp_y_train.shape[0], 1])
    # Test data
    tmp_x_test = np.asarray(data[train_size:, :-1])
    tmp_y_test = np.asarray(data[train_size:, -1])

    x_test_ = np.reshape(tmp_x_test, [tmp_x_test.shape[0], 1, tmp_x_test.shape[1]])
    y_test_ = np.reshape(tmp_y_test, [tmp_y_test.shape[0], 1])
    return x_train_, y_train_, x_test_, y_test_


# Ham generate du lieu tu file ra data ma ANFIS co the train duoc
def gen_to_data(ss, window_size, attribute):
    window_size += 1
    d = np.asarray(ss[attribute])
    temp_data = []
    for i in np.arange(d.shape[0] - window_size):
        temp = []
        for j in np.arange(window_size):
            temp.append(d[i + j])
        temp_data.append(temp)
    return temp_data
