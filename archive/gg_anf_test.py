#!/usr/bin/env python
import numpy as np
import pandas as pd
from archive import anf as anfis
import matplotlib.pyplot as plt
from utils import loss_function
from datetime import datetime
# Cac hang so
TRAIN_PERCENTAGE = 0.8
WINDOW_SIZE = 2
RULE_NUMBER = 2
ATTRIBUTE = 'meanCPUUsage'
EPOCH = 30
# Khai bao file
# Ten file
fname = "dataset/data_resource_usage_10Minutes_6176858948.csv"
# Cac Header trong file
header = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
          "meanCPUUsage", "canonical memory usage", "AssignMem",
          "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
          "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
          "max_disk_io_time", "cpi", "mai",
          "sampling_portion", "agg_type", "sampled_cpu_usage"]

# Lay du lieu tu file csv
df = pd.read_csv(fname, names=header)
mean_cpu_usage = df[ATTRIBUTE]


# Ham generate du lieu tu file ra data ma ANFIS co the train duoc
def gen_to_data(ss, window_size, attribute):
    window_size += 1
    d = np.asarray(ss[attribute])
    temp_data = []
    for i in np.arange(d.shape[0] - window_size):
        temp = []
        for j in np.arange(window_size):
            temp.append(d[i+j])
        temp_data.append(temp)
    return temp_data


def logging(ws: int, nr: int, trs, ts, ep, error, attribute: str):
    with open('logs/result.txt', 'a') as f:
        f.write('---> ' + str(datetime.now()) + '<---' + '\n')
        f.write('\t' + '[+] Arttibute\t\t: ' + attribute + '\n')
        f.write('\t' + '[+] Test size\t\t: ' + str(ts) + '\n')
        f.write('\t' + '[+] Training size\t: ' + str(trs) + '\n')
        f.write('\t' + '[+] Window size\t\t: ' + str(ws) + '\n')
        f.write('\t' + '[+] Rule number\t\t: ' + str(nr) + '\n')
        f.write('\t' + '[+] Epoch\t\t: ' + str(ep) + '\n')
        f.write('\t' + '[+] RMSE\t\t: ' + str(error) + '\n\n')


def main():
    data = np.asarray(gen_to_data(df, WINDOW_SIZE, 'meanCPUUsage'))
    train_size = int(data.shape[0]*TRAIN_PERCENTAGE)
    test_size = data.shape[0] - train_size

    y_test = data[train_size:, -1]
    # Training data
    x = data[:train_size, :-1]
    y = data[:train_size, -1]

    # Test data
    x_test = data[train_size:, :-1]

    # Dua du lieu vao ben trong va train
    mean1, mean2, sigma1, sigma2 = 25.0, 40.0, 15.0, 20.0
    a = anfis.ANFIS(x, y, 'gauss', RULE_NUMBER, epoch=EPOCH)
    a.fix_p_para(mean1, mean2, sigma1, sigma2)
    a.hybridTraining()

    # In ra gia tri test

    test_error = np.sqrt(loss_function(a.predict(x_test), y_test))
    print('test RMSE: ', test_error)
    logging(WINDOW_SIZE, RULE_NUMBER, train_size, test_size, EPOCH,
            test_error, ATTRIBUTE)

    # Xuat ra hinh anh so sanh voi du lieu thuc te
    x_axis = np.arange(0, test_size, 1)
    plt.title('Google cluter timeseries: ' + str(ATTRIBUTE))
    plt.plot(x_axis, a.predict(x_test), label='predict')
    plt.plot(x_axis, y_test, label='actual')
    plt.legend()
    plt.savefig('figures/' + str(RULE_NUMBER) + '_'+str(EPOCH)
                + '_' + str(WINDOW_SIZE) + '.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
