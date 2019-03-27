#!/usr/bin/env python

import old_models
import pandas as pd
from utils import extract_data


# Network features
WINDOW_SIZE = 5
RULE_NUMBER = 40
ATTRIBUTE = 'meanCPUUsage'
p_para_shape = [WINDOW_SIZE, RULE_NUMBER]
TRAIN_PERCENTAGE = 0.8
BATCH_SIZE = 10
EPOCH = 30
LEARNING_RATE = 1e-4

# Cac tham so lien quan den giai thuat SA
NEIGHBOR_NUMBER = 10
REDUCE_FACTOR = 0.95
TEMPERATURE_INIT = 100

# Ten file duoc dua vao ANFIS network
fname = "dataset/data_resource_usage_10Minutes_6176858948.csv"

# Cac Header trong file
header = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
          "meanCPUUsage", "canonical memory usage", "AssignMem",
          "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
          "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
          "max_disk_io_time", "cpi", "mai",
          "sampling_portion", "agg_type", "sampled_cpu_usage"]


def main():
    # Khai bao thong tin ve ten file dua vao anfis
    df = pd.read_csv(fname, names=header)
    # Trich xuat va chia tach cac tap tu file dau vao
    x_train, y_train, x_test, y_test = extract_data(df, window_size=WINDOW_SIZE,
                                                    attribute=ATTRIBUTE, train_percentage=TRAIN_PERCENTAGE)
    # Khai bao ANFIS network
    anf_model = old_models.ANFIS(window_size=WINDOW_SIZE, rule_number=RULE_NUMBER)

    # Bat dau huan luyen mang anfis
    anf_model.hybrid_sa_training(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                 batch_size=BATCH_SIZE, epoch=EPOCH, rate=LEARNING_RATE, neighbor_number=NEIGHBOR_NUMBER,
                                 reduce_factor=REDUCE_FACTOR, temp_init=TEMPERATURE_INIT)


if __name__ == '__main__':
    main()
