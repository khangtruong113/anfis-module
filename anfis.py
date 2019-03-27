from datetime import datetime
# from random import uniform as random
from scipy.constants import Boltzmann
from random import uniform as random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from scipy.constants import Boltzmann
import pandas as pd
from sklearn.metrics import mean_absolute_error

plt.switch_backend('agg')


# Save figure for 1-dimension data
def save_figures(data, label,
                 x_label, y_label,
                 path,
                 title=None
                 ):
    """
    :param data:
    :param label:
    :param x_label:
    :param y_label:
    :param path:
    :type title: object
    :return None:
    """
    array = np.asarray(data)
    x = np.arange(1, array.shape[0] + 1)
    plt.title(title)
    plt.plot(x, array, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(path)
    plt.close()


# Initialize variables as Premise Parameter
def premise_parameter(para_shape,
                      min_mu=10.0, max_mu=15.0,
                      min_sigma=5.0, max_sigma=10.0):
    """

    :param para_shape:
    :param min_mu:
    :param max_mu:
    :param min_sigma:
    :param max_sigma:
    :return:
    # """
    para_init = {
        'mu': tf.Variable(tf.random_uniform(para_shape, minval=min_mu, maxval=max_mu)),
        'sigma': tf.Variable(tf.random_uniform(para_shape, minval=min_sigma, maxval=max_sigma))
    }
    return para_init


# Adding timestamp for writer
def writer(context):
    print(f"{datetime.now()}: {context}")


# Initialize variables as Consequence Parameter
def consequence_parameter(para_shape):
    para_init = tf.random_normal(para_shape)
    return tf.Variable(para_init)


# Lay mot lan can cua ma tran chua tham so
def neighbor(x):
    delta = tf.random_normal(shape=x.get_shape(), mean=0.0, stddev=0.001 * tf.reduce_mean(x))
    x = x + delta
    return x


class ANFIS:
    def __init__(self,
                 rule_number=5, window_size=20, name=None):
        """
        :rtype:
        :param name: Ten bien the anfis
        :param rule_number: So luat trong mo hinh mang Takagi-Sugeno
        :param window_size: Kich thuoc lay mau cho input dau vao
        """
        self.name = name
        self.rule_number = rule_number
        self.window_size = window_size
        self.premise_shape = [rule_number, window_size]
        self.consequence_shape_weights = [window_size, rule_number]
        self.consequence_shape_bias = [1, rule_number]
        self.w_fuzz = premise_parameter(self.premise_shape)
        self.weights = consequence_parameter(self.consequence_shape_weights)
        self.bias = consequence_parameter(self.consequence_shape_bias)

    def output(self, x: np.ndarray):
        """
        Show list of outputs from list of inputs
        :param x:
        :return output:
        """
        # Reshape
        with tf.name_scope("reshape"):
            x_input = tf.tile(x, [1, self.rule_number, 1])

        # Fuzzification Layer
        with tf.name_scope('layer_1'):
            fuzzy_sets = tf.exp(- tf.divide(tf.square(tf.subtract(x_input, self.w_fuzz['mu']) / 2.0),
                                            tf.square(self.w_fuzz['sigma'])))
        # Rule-set Layer
        with tf.name_scope('layer_2'):
            fuzzy_rules = tf.reduce_prod(fuzzy_sets, axis=2)

        # Normalization Layer
        with tf.name_scope('layer_3'):
            sum_fuzzy_rules = tf.reduce_sum(fuzzy_rules, axis=1)
            normalized_fuzzy_rules = tf.divide(fuzzy_rules, tf.reshape(sum_fuzzy_rules, (-1, 1)))

        # Defuzzification Layer and Output Layer
        with tf.name_scope('layer_4_5'):
            f = tf.add(tf.matmul(tf.reshape(x, (-1, self.window_size)), self.weights), self.bias)
            output = tf.reduce_sum(tf.multiply(normalized_fuzzy_rules, f), axis=1)

        return tf.reshape(output, (-1, 1))

    # Traning ANFIS model
    def train(self,
              x_train, y_train,
              x_test, y_test,
              epoch: int = 10000, rate=1e-2,
              tracking_loss=False,
              load_path=None, save_path=None,
              tracking_path: str = None
              ):
        """
        Original training, only GD per epoch
        :param x_test: object
        :param y_test: object
        :type tracking_path: object
        :param x_train: Inputs to train
        :param y_train: Labels to train
        :param epoch: Number epoch to train
        :param rate: Learning rate to train
        :param tracking_loss: Flags if you wanna track loss values
        :param load_path: Flags if you wanna load model from specified path
        :param save_path: Cpecified path you wanna save model
        :return:
        """
        # Creating Placeholder
        writer("=========================")
        writer("====ORIGINAL-TRAINING====")
        writer("Creating Placeholder ... ")
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        writer("Creating cost and optimizer")
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Check tracking_loss flags
        track_list = np.empty((0,))

        # Something for check best point
        minimun_mse = 20000

        # Initializing session
        with tf.Session() as sess:

            # Check Model path Loading
            if load_path is not None:
                saver.restore(sess, load_path)

            # Start training
            sess.run(tf.global_variables_initializer())
            writer("Starting train ... ")
            for e in range(1, epoch + 1):
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})
                point = sess.run(cost, feed_dict={x: x_test, y: y_test})
                if point < minimun_mse:
                    minimun_mse = point
                    if save_path is not None:
                        saver.save(sess, save_path)

                writer(f"{e}: {c}")
                # Appened new loss value to track_list
                if tracking_loss:
                    track_list = np.append(track_list, c)

            # Check save_path

        # Saving figures
        track_fig_path = f"{tracking_path}/tracking.svg"
        track_data_path = f"{tracking_path}/tracking.csv"
        track_fig_title = f"{self.name} : Rule number  = {self.rule_number}"
        writer(f"Saving tracking figures to {track_fig_path} ")
        save_figures(data=track_list, label="Loss_function",
                     x_label='epoch', y_label='MSE',
                     path=track_fig_path,
                     title=track_fig_title,
                     )
        # writer(track_data_path)

        pd.DataFrame(track_list).to_csv(path_or_buf=track_data_path, header=None)
        # return self.loss(x_test, y_test, save_path)
        return minimun_mse

    def sa1_train(self,
                  x_train, y_train,
                  x_test, y_test,
                  epoch=10000, rate=1e-2,
                  tracking_loss=False,
                  load_path=None, save_path=None,
                  tracking_path=None,
                  neighbor_number=10, reduce_factor=0.95,
                  temp_init=100
                  ):
        """
        On epoch: GD -> SA
        :type tracking_path: object
        :param neighbor_number:
        :param temp_init:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param epoch:
        :param rate:
        :param tracking_loss:
        :param load_path:
        :param reduce_factor:
        :param save_path:
        :return:
        """
        writer("=========================")
        writer("=======SA1-TRAINING======")
        # Creating Placeholder
        writer("Creating Placeholder ... ")
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        writer("Creating cost and optimizer")
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Check tracking_loss flags
        track_list = np.empty((0,))

        # Something for check best point
        minimun_mse = 20000

        # Initializing session
        with tf.Session() as sess:

            # Check Model path Loading
            if load_path is not None:
                saver.restore(sess, load_path)

            # Start training
            sess.run(tf.global_variables_initializer())
            writer("Starting train ... ")
            for e in range(1, epoch + 1):
                # GD phase for all parameters
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                point = sess.run(cost, feed_dict={x: x_test, y: y_test})

                # SA phase for all parameters
                previous_parameters = self.w_fuzz, self.weights, self.bias
                temp = temp_init
                f0 = sess.run(cost, feed_dict={x: x_train, y: y_train})

                for n in range(neighbor_number):
                    sess.run(self.w_fuzz['mu'].assign(neighbor(self.w_fuzz['mu'])))
                    sess.run(self.w_fuzz['sigma'].assign(neighbor(self.w_fuzz['sigma'])))
                    sess.run(self.weights.assign(neighbor(self.weights)))
                    sess.run(self.bias.assign(neighbor(self.bias)))
                    f = sess.run(cost, feed_dict={x: x_train, y: y_train})
                    if f < f0:
                        f_new = f
                        previous_parameters = self.w_fuzz, self.weights, self.bias
                    else:
                        df = f - f0
                        r = random(0, 1)
                        if r > np.exp(-df / Boltzmann / temp):
                            f_new = f
                            previous_parameters = self.w_fuzz, self.weights, self.bias
                        else:
                            f_new = f0
                            self.w_fuzz, self.weights, self.bias = previous_parameters
                    f0 = f_new
                    writer(f"{e}, {n}")
                    temp = reduce_factor * temp

                # Appened new loss value to track_list
                if tracking_loss:
                    track_list = np.append(track_list, f0)
                if point < minimun_mse:
                    minimun_mse = point
                    if save_path is not None:
                        saver.save(sess, save_path)
                # writer(f"{e}: {c}")

            # Check save_path
            if save_path is not None:
                saver.save(sess, save_path)
        # Saving figures
        track_fig_path = f"{tracking_path}/tracking.svg"
        track_data_path = f"{tracking_path}/tracking.csv"
        track_fig_title = f"{self.name} : Rule number  = {self.rule_number}"
        writer(f"Saving tracking figures to {track_fig_path} ")
        save_figures(data=track_list, label="Loss_function",
                     x_label='epoch', y_label='MSE',
                     path=track_fig_path,
                     title=track_fig_title,
                     )

        pd.DataFrame(track_list).to_csv(path_or_buf=track_data_path, header=None)
        # return self.loss(x_test, y_test, save_path)
        return minimun_mse

    def sa2_train(self,
                  x_train, y_train,
                  x_test, y_test,
                  epoch=10000, rate=1e-2,
                  tracking_loss=False,
                  load_path=None, save_path=None,
                  tracking_path=None,
                  neighbor_number=10, reduce_factor=0.95,
                  temp_init=100
                  ):
        """
        On epoch: GD All parameters -> SA consequence parameters
        :type tracking_path: object
        :param neighbor_number:
        :param temp_init:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param epoch:
        :param rate:
        :param tracking_loss:
        :param load_path:
        :param reduce_factor:
        :param save_path:
        :return:
        """
        writer("=========================")
        writer("=======SA2-TRAINING======")
        # Creating Placeholder
        writer("Creating Placeholder ... ")
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        writer("Creating cost and optimizer")
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Check tracking_loss flags
        track_list = np.empty((0,))

        # Something for check best point
        minimun_mse = 20000

        # Initializing session
        with tf.Session() as sess:

            # Check Model path Loading
            if load_path is not None:
                saver.restore(sess, load_path)

            # Start training
            sess.run(tf.global_variables_initializer())
            writer("Starting train ... ")
            for e in range(1, epoch + 1):
                # GD phase for all parameters
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})
                point = sess.run(cost, feed_dict={x: x_test, y: y_test})

                # SA phase for all parameters
                previous_parameters = self.w_fuzz, self.weights, self.bias
                temp = temp_init
                f0 = sess.run(cost, feed_dict={x: x_train, y: y_train})

                for n in range(neighbor_number):
                    sess.run(self.weights.assign(neighbor(self.weights)))
                    sess.run(self.bias.assign(neighbor(self.bias)))
                    f = sess.run(cost, feed_dict={x: x_train, y: y_train})
                    if f < f0:
                        f_new = f
                        previous_parameters = self.w_fuzz, self.weights, self.bias
                    else:
                        df = f - f0
                        r = random(0, 1)
                        if r > np.exp(-df / Boltzmann / temp):
                            f_new = f
                            previous_parameters = self.w_fuzz, self.weights, self.bias
                        else:
                            f_new = f0
                            self.w_fuzz, self.weights, self.bias = previous_parameters
                    f0 = f_new
                    temp = reduce_factor * temp

                # Appened new loss value to track_list
                if tracking_loss:
                    track_list = np.append(track_list, c)
                if point < minimun_mse:
                    minimun_mse = point
                    if save_path is not None:
                        saver.save(sess, save_path)
                writer(f"{e}: {c}")

            # Check save_path
            if save_path is not None:
                saver.save(sess, save_path)
        # Saving figures
        track_fig_path = f"{tracking_path}/tracking.svg"
        track_data_path = f"{tracking_path}/tracking.csv"
        track_fig_title = f"{self.name} : Rule number  = {self.rule_number}"
        writer(f"Saving tracking figures to {track_fig_path} ")
        save_figures(data=track_list, label="Loss_function",
                     x_label='epoch', y_label='MSE',
                     path=track_fig_path,
                     title=track_fig_title,
                     )

        pd.DataFrame(track_list).to_csv(path_or_buf=track_data_path, header=None)
        # return self.loss(x_test, y_test, save_path)
        return minimun_mse

    def sa3_train(self,
                  x_train, y_train,
                  x_test, y_test,
                  epoch=10000, rate=1e-2,
                  tracking_loss=False,
                  load_path=None, save_path=None,
                  tracking_path=None,
                  neighbor_number=10, reduce_factor=0.95,
                  temp_init=100
                  ):
        """
        On epoch: GD All parameters -> SA premise parameters
        :type tracking_path: object
        :param neighbor_number:
        :param temp_init:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param epoch:
        :param rate:
        :param tracking_loss:
        :param load_path:
        :param reduce_factor:
        :param save_path:
        :return:
        """
        writer("=========================")
        writer("=======SA3-TRAINING======")
        # Creating Placeholder
        writer("Creating Placeholder ... ")
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        writer("Creating cost and optimizer")
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Check tracking_loss flags
        track_list = np.empty((0,))

        # Something for check best point
        minimun_mse = 20000

        # Initializing session
        with tf.Session() as sess:

            # Check Model path Loading
            if load_path is not None:
                saver.restore(sess, load_path)

            # Start training
            sess.run(tf.global_variables_initializer())
            writer("Starting train ... ")
            for e in range(1, epoch + 1):
                # GD phase for all parameters
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})
                point = sess.run(cost, feed_dict={x: x_test, y: y_test})

                # SA phase for all parameters
                previous_parameters = self.w_fuzz, self.weights, self.bias
                temp = temp_init
                f0 = sess.run(cost, feed_dict={x: x_train, y: y_train})

                for n in range(neighbor_number):
                    sess.run(self.w_fuzz['mu'].assign(neighbor(self.w_fuzz['mu'])))
                    sess.run(self.w_fuzz['sigma'].assign(neighbor(self.w_fuzz['sigma'])))

                    f = sess.run(cost, feed_dict={x: x_train, y: y_train})
                    if f < f0:
                        f_new = f
                        previous_parameters = self.w_fuzz, self.weights, self.bias
                    else:
                        df = f - f0
                        r = random(0, 1)
                        if r > np.exp(-df / Boltzmann / temp):
                            f_new = f
                            previous_parameters = self.w_fuzz, self.weights, self.bias
                        else:
                            f_new = f0
                            self.w_fuzz, self.weights, self.bias = previous_parameters
                    f0 = f_new
                    temp = reduce_factor * temp

                # Appened new loss value to track_list
                if tracking_loss:
                    track_list = np.append(track_list, c)
                if point < minimun_mse:
                    minimun_mse = point
                    if save_path is not None:
                        saver.save(sess, save_path)
                writer(f"{e}: {c}")

            # Check save_path
            if save_path is not None:
                saver.save(sess, save_path)
        # Saving figures
        track_fig_path = f"{tracking_path}/tracking.svg"
        track_data_path = f"{tracking_path}/tracking.csv"
        track_fig_title = f"{self.name} : Rule number  = {self.rule_number}"
        writer(f"Saving tracking figures to {track_fig_path} ")
        save_figures(data=track_list, label="Loss_function",
                     x_label='epoch', y_label='MSE',
                     path=track_fig_path,
                     title=track_fig_title,
                     )

        pd.DataFrame(track_list).to_csv(path_or_buf=track_data_path, header=None)
        # return self.loss(x_test, y_test, save_path)
        return minimun_mse

    # Compute loss from input data and compare output with labels
    def loss(self,
             x_test, y_test,
             load_path=None):
        """

        :param x_test:
        :param y_test:
        :return:
        :param load_path: Load model from this path
        :return: loss function
        """
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        cost_mse = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Check Model path Loading
            sess.run(tf.global_variables_initializer())
            if load_path is not None:
                saver.restore(sess, load_path)
            # op = sess.run(self.output(x_), feed_dict={x_: x})
            mse = sess.run(cost_mse, feed_dict={x: x_test, y: y_test})
            output = sess.run(self.output(x), feed_dict={x: x_test})
            mae = mean_absolute_error(output, y_test)
        return mse, mae

    def compare_figures(self,
                        x_test, y_test,
                        load_path=None,
                        fig_path=None,
                        attribute=None):
        """tracking_path
        :param attribute:
        :param fig_path:
        :param x_test:
        :param y_test:
        :param load_path:
        :return:
        """
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        # cost = tf.reduce_mean(tf.squared_difference(self.output(x), y_test))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Check Model path Loading
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, load_path)
            predict = sess.run(self.output(x), feed_dict={x: x_test})
        x_axis = np.arange(0, x_test.shape[0])

        plt.plot(x_axis, y_test, label='Actual')
        plt.plot(x_axis, predict, label='Predict')
        plt.xlabel('Time')
        plt.ylabel(attribute)
        plt.legend()
        plt.savefig(fig_path)
        plt.close()
