import numpy as np
import tensorflow as tf

from models.GenericModels import GenericModels
from utils.anfisExternal import consequence_parameters, premise_parameters
from utils.figures import Figures
from pandas import DataFrame

class ANFIS(GenericModels):
    def __init__(self, rule_number=5, window_size=20, name=None):
        """
        :param rule_number:
        :param window_size:
        :param name:
        """
        super().__init__()
        self.name = name
        self.rule_number = rule_number
        self.window_size = window_size
        self.premise_shape = [rule_number, window_size]
        self.consequence_shape_weights = [window_size, rule_number]
        self.consequence_shape_bias = [1, window_size]
        self.fuzzy_weights = premise_parameters(self.premise_shape)
        self.defuzzy_weights = consequence_parameters(self.consequence_shape_weights)
        self.defuzzy_bias = consequence_parameters(self.consequence_shape_bias)

    @staticmethod
    def output(self, x: tf.placeholder):
        """
        Show list of outputs from list of inputs
        :param self:
        :param x:
        :return:
        """
        # Reshape
        with tf.name_scope("reshape"):
            x_input = tf.tile(x, [1, self.rule_number, 1])

        # Fuzzification Layer
        with tf.name_scope('Fuzzification_layer'):
            fuzzy_sets = tf.exp(- tf.divide(tf.square(tf.subtract(x_input, self.fuzzy_weights['mu']) / 2.0),
                                            tf.square(self.fuzzy_weights['sigma'])))
        # Rule-set Layer
        with tf.name_scope('Rule_set_layer'):
            fuzzy_rules = tf.reduce_prod(fuzzy_sets, axis=2)

        # Normalization Layer
        with tf.name_scope('Normalization_layer'):
            sum_fuzzy_rules = tf.reduce_sum(fuzzy_rules, axis=1)
            normalized_fuzzy_rules = tf.divide(fuzzy_rules, tf.reshape(sum_fuzzy_rules, (-1, 1)))

        # Defuzzification Layer and Output Layer
        with tf.name_scope('Defuzzification_layer'):
            f = tf.add(tf.matmul(tf.reshape(x, (-1, self.window_size)), self.defuzzy_weights), self.defuzzy_bias)
            output = tf.reduce_sum(tf.multiply(normalized_fuzzy_rules, f), axis=1)

        return tf.reshape(output, (-1, 1))

    def train(self,
              x_train, y_train,
              x_test, y_test,
              epoch: int = 10000, rate=1e-2,
              tracking_loss=None,
              save_path=None,
              load_path=None,
              tracking_path: str = None) -> None:
        """
        saver = tf.train.Saver()

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
        cost = tf.reduce_mean(tf.squared_difference(self.output(self, x), y))
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
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})
                point = sess.run(cost, feed_dict={x: x_test, y: y_test})
                if point < minimum_mse:
                    minimum_mse = point
                    if save_path is not None:
                        saver.save(sess, save_path)

                # Appending new loss values to track_list
                if tracking_loss:
                    tracking_list = np.append(tracking_list, c)
        # Saving figures
        tracking_fig_path = f"{tracking_path}/tracking.svg"
        tracking_data_path = f"{tracking_list}/tracking.csv"
        tracking_fig_title = f"{self.name}: Rule number : {self.rule_number}"
        Figures.track(data=tracking_list, data_label="Loss function",
                      first_label='epoch', second_label='MSE',
                      path=tracking_fig_path,
                      title=tracking_fig_title)
        DataFrame(tracking_list).to_csv(path_or_buf=tracking_data_path, header=None)
