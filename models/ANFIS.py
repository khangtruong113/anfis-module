import numpy as np
import tensorflow as tf
from pandas import DataFrame

from models.GenericModels import GenericModels
from utils.anfisExternal import consequence_parameters, premise_parameters
from utils.figures import Figures


class ANFIS(GenericModels):
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
        self.w_fuzz = premise_parameters(self.premise_shape)
        self.weights = consequence_parameters(self.consequence_shape_weights)
        self.bias = consequence_parameters(self.consequence_shape_bias)

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

    def train(self,
              x_train, y_train,
              x_test, y_test,
              epoch: int = 10000, rate=1e-2,
              tracking_loss=None,
              save_path=None,
              load_path=None,
              batch_size: int = 50,
              tracking_path: str = None) -> None:
        """
        saver = tf.train.Saver()

        :type batch_size: int
        :param load_path:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param epoch:
        :param rate:
        :param tracking_loss:
        :param save_path:
        :param tracking_path:
        :return:
        """
        x: tf.placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Checking tracking_loss flags
        tracking_list = np.empty((0,))

        # Checking best point
        minimum_mse: float = 20000

        # Initializing session
        with tf.Session() as sess:
            if load_path is not None:
                saver.restore(sess, load_path)

            # Starting training
            sess.run(tf.global_variables_initializer())

            for e in range(1, epoch + 1):
                # Shuffle training data
                shuffle = np.random.permutation(np.arange(len(y_train)))
                x_train = x_train[shuffle]
                y_train = y_train[shuffle]
                # Based-batch training
                for i in np.arange(0, len(y_train) // batch_size):
                    start = i * batch_size
                    batch_x = x_train[start:start + batch_size]
                    batch_y = y_train[start:start + batch_size]
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                    if save_path is not None:
                        saver.save(sess, save_path)
                point = sess.run(cost, feed_dict={x: x_test, y: y_test})
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})
                print(f"Epoch : ${e} - Point: ${point}")
                # Appending new loss values to track_list
                if tracking_loss:
                    tracking_list = np.append(tracking_list, c)
        # Saving figures
        tracking_fig_path = f"{tracking_path}/tracking.svg"
        tracking_fig_title = f"{self.name}: Rule number : {self.rule_number}"
        Figures.track(data=tracking_list, data_label="Loss function",
                      first_label='epoch', second_label='MSE',
                      path=tracking_fig_path,
                      title=tracking_fig_title)

        # Saving tracking data
        tracking_data_path = f"{tracking_path}/tracking.csv"
        DataFrame(tracking_list).to_csv(path_or_buf=tracking_data_path, header=None)
