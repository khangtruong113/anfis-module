#!/usr/bin/env python

# Just disables the warning, doesn't enable AVX/FMA
import os

import pandas as pd

import remodel as models
from preprocess import extract_data, fname, header

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Network features
WINDOW_SIZE = 20
RULE_NUMBER = 5
ATTRIBUTE = 'meanCPUUsage'
TRAIN_PERCENTAGE = 0.8
EPOCH = 20000
LEARNING_RATE = 1e-2


def main():
    # Khai bao thong tin ve ten file dua vao anfis
    df = pd.read_csv(fname, names=header)
    # Trich xuat va chia tach cac tap tu file dau vao
    x_train, y_train, x_test, y_test = extract_data(df, window_size=WINDOW_SIZE,
                                                    attribute=ATTRIBUTE,
                                                    train_percentage=TRAIN_PERCENTAGE)
    # Khai bao ANFIS network
    anfis_model = models.ANFIS(window_size=WINDOW_SIZE, rule_number=RULE_NUMBER)
    save_path = f"./results/originals/anfis_rule_number_{RULE_NUMBER}.cpkt"
    anfis_model.sa1_train(x_train=x_train, y_train=y_train,
                          epoch=EPOCH, rate=LEARNING_RATE,
                          save_path=save_path,
                          tracking_loss=True)
    anfis_model.loss(x_train, y_train, load_path=save_path)
    # new_anf_model.figure(x_test=x_test, y_test=y_test, load_path=save_path)
    # anfis_model.predict(input_data=x_test, load_path=save_path)
    # anfis_model.mse(x=x_test, y=y_test, load_path=save_path)


if __name__ == '__main__':
    main()
