#!/usr/bin/env python
import numpy as np
from utils import frame_parameter, consequence_parameter, first_layer, \
                  third_layer, second_layer, fouth_layer, fifth_layer
import time


# Lop chua mo hinh ANFIS
class ANFIS:

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 mf: str, rule_number: int, epoch: int):
        self.X = X  # Training_input
        self.Y = Y  # Training_output
        self.mf = mf  # Xac dinh tap mo su dung
        self.epoch = epoch
        self.training_size = X.shape[0]
        self.rule_number = rule_number  # So luat trong mang ANFIS
        if (X.shape[0] != Y.shape[0]):
            print('Size error, check training i/oput')
            exit(0)
        try:
            self.window_size = X.shape[1]
        except IndexError as err:
            print('Training input must be 3-d array: ', err)
            exit(0)
        self.p_para = frame_parameter(self.rule_number,
                                      self.window_size)
        self.c_para = consequence_parameter(self.rule_number, self.window_size)

    def summary(self):  # Tong hop mo hinh mang ANFIS
        print('ANFIS summary')
        print('Training size: ', self.X.shape[0])
        print('Rule number  : ', self.rule_number)
        print('Window size  : ', self.window_size)

    def half_first(self, x: np.ndarray):
        layer1 = first_layer(x, self.p_para)
        layer2 = second_layer(layer1)
        return third_layer(layer2)

    def half_last(self, hf, x):
        layer4 = fouth_layer(hf, x, self.c_para)
        return fifth_layer(layer4)

    def f_single(self, x):
        hf = self.half_first(x)
        wf = fouth_layer(hf, x, self.c_para)
        return wf

    def f_(self, x: np.ndarray):
        return np.asarray([self.f_single(x[i])
                           for i in np.arange(self.training_size)])

    def w_(self, x: np.ndarray):
        return np.asarray([self.half_first(x[i])
                           for i in np.arange(self.training_size)])

    # Phuong thuc du doan theo dau vao voi tham so trong mo hinh
    def output_single(self, x: np.ndarray):
        hf = self.half_first(x)  # w_
        wf = fouth_layer(hf, x, self.c_para)  # f_
        hl = fifth_layer(wf)  # output_single_chuan
        return hl

    def genValSingle(self, x: np.ndarray):
        hf_single = self.half_first(x)  # w_single
        wf_single = fouth_layer(hf_single, x, self.c_para)  # f_single
        hl_single = fifth_layer(wf_single)  # output_single_chuan
        return hf_single, wf_single, hl_single

    def genVal(self, x: np.ndarray):
        w = []
        f = []
        o = []
        for i in np.arange(x.shape[0]):
            ws, fs, os = self.genValSingle(x[i])
            w.append(ws)
            f.append(fs)
            o.append(os)
        return w, f, o

    # Su dung de tinh ra mot chuoi cac gia tri du doan tu 1 mang cho truoc
    # Su dung de tinh loss function va in ra man hinh
    def output(self, inp_value):
        return np.asarray([self.output_single(inp_value[i])
                           for i in np.arange(self.training_size)])

    # Dung cho tap test
    def predict(self, x: np.ndarray):
        return np.asarray([self.output_single(x[i])
                           for i in np.arange(x.shape[0])])

    # loss_function su dung de lam ham muc tieu trong GD
    def lossFunction(self):
        predict_value = self.output(self.X)
        actual_value = self.Y
        return ((predict_value - actual_value)**2).mean(axis=0)

    # Dung de chinh sua tham so khoi tao tuy thuoc vao du lieu train
    def fix_p_para(self, mean1, mean2, sigma1, sigma2):
        self.p_para = frame_parameter(self.rule_number,
                                      self.window_size, mean1,
                                      mean2, sigma1, sigma2)

    def lse(self):
        # Khai bao
        y_ = np.array(self.Y)[np.newaxis].T
        a = np.ones((self.training_size,
                    (self.window_size+1) * self.rule_number), dtype=float)
        w = np.asarray([self.half_first(self.X[i])
                        for i in np.arange(self.training_size)])
        # Bat dau tien hanh linear regression
        for i in np.arange(self.training_size):
            for j in np.arange(self.rule_number):
                for k in np.arange(self.window_size):
                    a[i][j*(self.window_size+1)+k] = w[i][j]*self.X[i][k]
                    a[i][j*(self.window_size+1)+self.window_size] = w[i][j]
        c = np.dot(np.linalg.pinv(a), y_)
        self.c_para = np.reshape(c, self.c_para.shape)

    # Dao ham ham loi ( lay ham Gauss)
    # Tien hanh dao ham cho tat ca cac truong hop
    def derivError(self, mf='gauss', variable='mean'):
        temp = np.zeros(self.p_para.shape)
        y = self.Y
        x = self.X
        # f = self.f_(self.X)
        # w = self.w_(self.X)
        w, f, d = self.genVal(self.X)
        for k in np.arange(self.training_size):
            for j in np.arange(self.rule_number):
                half_delta = (y[k] - d[k]) * (d[k] - f[k][j]) * w[k][j]
                for i in np.arange(self.window_size):
                    # sigma
                    sigma = self.p_para[j][i][1]['sigma']
                    mean = self.p_para[j][i][1]['mean']
                    temp[j][i][0] += half_delta * ((x[k][i] - sigma)) / \
                        (mean**2)
                    # mean
                    temp[j][i][1] += half_delta * ((x[k][i] - sigma)**2) /\
                        (mean**3)
        return temp

    # Su dung GD
    def gd(self, epochs=1, eta=0.01, k=0.9):
        derivE = self.derivError('gauss', 'mean')
        #  Xu ly voi cac tham so mean
        for i in np.arange(self.rule_number):
            for j in np.arange(self.window_size):
                self.p_para[i][j][1]['mean'] -= eta*derivE[i][j][0]
                self.p_para[i][j][1]['sigma'] -= eta*derivE[i][j][1]

    # Su dung giai thuat hon hop
    def hybridTraining(self):
        print("Starting training ...")
        loop = 0
        counter_gd = 0  # Bien dem so lan thuc hien gd trong 1 epoch
        max_gd = 20  # So thuc hien gd trong 1 epoch
        while (loop < self.epoch):
            timer = time.time()
            self.lse()
            while(counter_gd < max_gd):
                self.gd()
                counter_gd += 1
            counter_gd = 0
            print('Loop: \t', loop, '/', self.epoch, '\tTime: \t',
                  time.time() - timer)
            loop += 1
        print("Training completed!")
        self.summary()
