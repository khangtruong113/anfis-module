#!/usr/bin/env python
import numpy as np
from skfuzzy import gaussmf, gbellmf, sigmf
import random
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
"""
    Cac ham phu tro cua ANFIS duoc de tai day
    ...
"""
func_dict = {'gaussmf': gaussmf, 'gbellmf': gbellmf, 'sigmf': sigmf}


# random uniform
def rd(x, y):
    return random.uniform(x, y)


# premise parameter
def frame_parameter(rule_number: int, window_size: int,
                    mean1=20000, mean2=25000, sigma1=10000, sigma2=15000):
    x = [[['gaussmf', {'mean': rd(mean1, mean2), 'sigma': rd(sigma1, sigma2)}]
          for _ in np.arange(window_size)] for _ in np.arange(rule_number)]
    return np.asarray(x)


def consequence_parameter(rule_number: int, window_size: int):
    return np.ones((rule_number, window_size+1), dtype=float)


# Dau ra cua lop dau tien
# Lop thuc hien tinh toan do mo qua cac tap mo cho truoc
def first_layer(x: np.ndarray, fp: np.ndarray):
    ws, rn = fp.shape[1], fp.shape[0]
    temp = [[func_dict[fp[i][j][0]](x[j], **fp[i][j][1])
            for j in np.arange(ws)]
            for i in np.arange(rn)]
    return np.asarray(temp)


def loss_function(x, y):
    return mse(x, y)


# Dau ra cua lop thu 2
# Lop thuc hien tinh toan cac luat tu cac tap mo
def second_layer(ofl: np.ndarray):
    ws, rn = ofl.shape[1], ofl.shape[0]
    temp = np.ones(rn, dtype=float)
    for i in np.arange(rn):
        for j in np.arange(ws):
            temp[i] *= ofl[i][j]
    return temp


# Dau ra cua lop thu 3
def third_layer(osl: np.ndarray):
    return osl / osl.sum()


# Dau ra cua lop thu 4
def fouth_layer(otl: np.ndarray, x: np.ndarray, cp: np.ndarray):
    mat = np.append(x, 1)
    temp = [otl[i]*np.dot(mat, cp[i]) for i in np.arange(cp.shape[0])]
    return np.asarray(temp)


# Dau ra cuoi
def fifth_layer(ofl: np.ndarray):
    return sum(ofl)


def show_image(input_list: list):
    plt.plot(np.arange(1, len(input_list) + 1), input_list)
    plt.title('Training loss by epoch')
    plt.ylabel('Train loss')
    plt.xlabel('epoch')
    plt.axis([0, (len(input_list) + 1), 0, (max(input_list) + 1)])
    plt.show()



