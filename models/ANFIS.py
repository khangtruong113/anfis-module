import os

import numpy as np
import tensorflow as tf
from pandas import DataFrame

from models.GenericModels import GenericModels
from utils.anfisExternal import consequence_parameters, premise_parameters
from utils.figures import Figures
from utils.reports import Reports
from utils.simulatedAnnealing import neighbor, sa_random, boltzmann_constants

DEFAULT_DATASET = 'Google Cloud Resource Usages'


class ANFIS(GenericModels):
    def __init__(self,
                 rule_number=5, window_size=20, name="ANFIS"):
        """
        :rtype:
        :param name: Ten bien the anfis
        :param rule_number: So luat trong mo hinh mang Takagi-Sugeno
        :param window_size: Kich thuoc lay mau cho input dau vao
        """
        super(ANFIS, self).__init__()
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
              rate=1e-2, epoch: int = 100,
              tracking_loss=None,
              batch_size: int = 50) -> None:
        """
        saver = tf.train.Saver()

        :type batch_size: int
        :param x_train:
        :param y_train:
        :param epoch:
        :param rate:
        :param tracking_loss:
        :return:
        """
        self.name = "Original ANFIS"

        print("==== TRAINING PHASE ====")
        print("INFOMATION:")
        print(f"Name: {self.name}")
        print(f"Rule number: {self.rule_number}")
        print(f"Window size: {self.window_size}")
        print(f"Epoch: {epoch}")
        print(f"Start training ...")

        x: tf.placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Saving directory
        saving_dir: str = f"metadata/models/originANFIS/rl{self.rule_number}ws{self.window_size}"
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        saving_path: str = f"{saving_dir}/model.h5"

        # Checking tracking_loss flags
        tracking_list = np.empty((0,))

        # Initializing session
        with tf.Session() as sess:

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
                # Loss value
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})
                # Appending new loss values to track_list
                if tracking_loss:
                    tracking_list = np.append(tracking_list, c)
            print(f"Saving model to {saving_path} ...")
            saver.save(sess, save_path=saving_path)

        # Saving figures
        tracking_dir = f"results/originANFIS/rl{self.rule_number}ws{self.window_size}/tracks"
        if not os.path.exists(tracking_dir):
            os.makedirs(tracking_dir)
        tracking_fig_path = f"{tracking_dir}/track.svg"
        tracking_data_path = f"{tracking_dir}/track.csv"
        tracking_fig_title = f"{self.name}: Rule number : {self.rule_number} Window size : {self.window_size}"
        print(f"Saving tracking figures to {tracking_fig_path}")
        Figures.track(data=tracking_list, data_label="Loss function",
                      first_label='epoch', second_label='MSE',
                      path=tracking_fig_path,
                      title=tracking_fig_title)

        # Saving tracking data
        print(f"Saving tracking list to {tracking_data_path} ...")
        DataFrame(tracking_list).to_csv(path_or_buf=tracking_data_path, header=None)
        print("Training completed!")

    def sa1_train(self,
                  x_train, y_train,
                  epoch: int = 10000, rate=1e-2,
                  tracking_loss=None,
                  neighbor_number=10,
                  reduce_factor=0.95,
                  temp_init=100,
                  batch_size: int = 50
                  ):
        """
                On epoch: GD -> SA
                :param batch_size:
                :param neighbor_number:
                :param temp_init:
                :param x_train:
                :param y_train:
                :param epoch:
                :param rate:
                :param tracking_loss:
                :param reduce_factor:
                :return:
                """
        self.name = "SA1-ANFIS"
        print("==== TRAINING PHASE ====")
        print("INFOMATION:")
        print(f"Name: {self.name}")
        print(f"Rule number: {self.rule_number}")
        print(f"Window size: {self.window_size}")
        print(f"Neighbor number: {neighbor_number}")
        print(f"Reduce factor: {reduce_factor}")
        print(f"Temperature initialized value: {temp_init}")
        print(f"Epoch: {epoch}")
        print(f"Start training ...")

        # Creating Placeholder
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Saving diretory
        saving_dir: str = f"metadata/models/sa1ANFIS/rl{self.rule_number}ws{self.window_size}"
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        saving_path: str = f"{saving_dir}/model.h5"

        # Check tracking_loss flags
        tracking_list = np.empty((0,))

        # Initializing session
        with tf.Session() as sess:

            # Start training
            sess.run(tf.global_variables_initializer())
            for e in range(1, epoch + 1):
                # GD phase for all parameters

                # Shuffle training data
                shuffle = np.random.permutation(np.arange(len(y_train)))
                x_train = x_train[shuffle]
                y_train = y_train[shuffle]
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                for i in np.arange(0, len(y_train) // batch_size):

                    start = i * batch_size
                    batch_x = x_train[start:start + batch_size]
                    batch_y = y_train[start:start + batch_size]
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

                # SA phase for all parameters
                previous_parameters = self.w_fuzz, self.weights, self.bias
                temp = temp_init
                f0 = sess.run(cost, feed_dict={x: x_train, y: y_train})

                for n in range(neighbor_number):
                    sess.run(self.w_fuzz['mu'].assign(neighbor(self.w_fuzz['mu'])))
                    sess.run(self.w_fuzz['sigma'].assign(neighbor(self.w_fuzz['sigma'])))
                    sess.run(self.weights.assign(neighbor(self.weights)))
                    sess.run(self.bias.assign(neighbor(self.bias)))
                    f = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    if f < f0:
                        f_new = f
                        previous_parameters = self.w_fuzz, self.weights, self.bias
                    else:
                        df = f - f0
                        r = sa_random(0, 1)
                        if r > np.exp(-df / boltzmann_constants() / temp):
                            f_new = f
                            previous_parameters = self.w_fuzz, self.weights, self.bias
                        else:
                            f_new = f0
                            self.w_fuzz, self.weights, self.bias = previous_parameters
                    f0 = f_new
                    temp = reduce_factor * temp
                # Loss value
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})

                # Appened new loss value to track_list
                if tracking_loss:
                    tracking_list = np.append(tracking_list, c)

            print(f"Saving model to {saving_path} ...")
            saver.save(sess, save_path=saving_path)

        # Saving figures
        tracking_dir = f"results/sa1ANFIS/rl{self.rule_number}ws{self.window_size}/tracks"

        if not os.path.exists(tracking_dir):
            os.makedirs(tracking_dir)

        tracking_fig_path = f"{tracking_dir}/track.svg"
        tracking_data_path = f"{tracking_dir}/track.csv"
        tracking_fig_title = f"{self.name}: Rule number : {self.rule_number} Window size : {self.window_size}"
        print(f"Saving tracking figures to {tracking_fig_path}")
        Figures.track(data=tracking_list, data_label="Loss function",
                      first_label='epoch', second_label='MSE',
                      path=tracking_fig_path,
                      title=tracking_fig_title)

        # Saving tracking data
        print(f"Saving tracking list to {tracking_data_path} ...")
        DataFrame(tracking_list).to_csv(path_or_buf=tracking_data_path, header=None)
        print("Training completed!")

    def sa2_train(self,
                  x_train, y_train,
                  epoch: int = 10000, rate=1e-2,
                  tracking_loss=None,
                  neighbor_number=10,
                  reduce_factor=0.95,
                  temp_init=100,
                  batch_size: int = 50,
                  ):
        """
                On epoch: GD All parameters -> SA consequence parameters
                :param batch_size:
                :param neighbor_number:
                :param temp_init:
                :param x_train:
                :param y_train:
                :param epoch:
                :param rate:
                :param tracking_loss:
                :param reduce_factor:
                :return:
                """

        self.name = "SA2-ANFIS"
        print("==== TRAINING PHASE ====")
        print("INFOMATION:")
        print(f"Name: {self.name}")
        print(f"Rule number: {self.rule_number}")
        print(f"Window size: {self.window_size}")
        print(f"Neighbor number: {neighbor_number}")
        print(f"Reduce factor: {reduce_factor}")
        print(f"Temperature initialized value: {temp_init}")
        print(f"Epoch: {epoch}")
        print(f"Start training ...")

        # Creating Placeholder
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Saving diretory
        saving_dir: str = f"metadata/models/sa2ANFIS/rl{self.rule_number}ws{self.window_size}"
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        saving_path: str = f"{saving_dir}/model.h5"

        # Check tracking_loss flags
        tracking_list = np.empty((0,))

        # Initializing session
        with tf.Session() as sess:

            # Start training
            sess.run(tf.global_variables_initializer())
            for e in range(1, epoch + 1):
                # GD phase for all parameters
                # Shuffle training data
                shuffle = np.random.permutation(np.arange(len(y_train)))
                x_train = x_train[shuffle]
                y_train = y_train[shuffle]
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                for i in np.arange(0, len(y_train) // batch_size):

                    start = i * batch_size
                    batch_x = x_train[start:start + batch_size]
                    batch_y = y_train[start:start + batch_size]
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

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
                        r = sa_random(0, 1)
                        if r > np.exp(-df / boltzmann_constants() / temp):
                            f_new = f
                            previous_parameters = self.w_fuzz, self.weights, self.bias
                        else:
                            f_new = f0
                            self.w_fuzz, self.weights, self.bias = previous_parameters
                    f0 = f_new
                    temp = reduce_factor * temp

                # Loss value
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})

                # Appened new loss value to track_list
                if tracking_loss:
                    tracking_list = np.append(tracking_list, c)

            print(f"Saving model to {saving_path} ...")
            saver.save(sess, save_path=saving_path)

        # Saving figures
        tracking_dir = f"results/sa2ANFIS/rl{self.rule_number}ws{self.window_size}/tracks"
        tracking_fig_path = f"{tracking_dir}/track.svg"
        tracking_data_path = f"{tracking_dir}/track.csv"
        tracking_fig_title = f"{self.name}: Rule number : {self.rule_number} Window size : {self.window_size}"
        print(f"Saving tracking figures to {tracking_fig_path}")
        Figures.track(data=tracking_list, data_label="Loss function",
                      first_label='epoch', second_label='MSE',
                      path=tracking_fig_path,
                      title=tracking_fig_title)

        # Saving tracking data
        print(f"Saving tracking list to {tracking_data_path} ...")
        DataFrame(tracking_list).to_csv(path_or_buf=tracking_data_path, header=None)
        print("Training completed!")

    def sa3_train(self,
                  x_train, y_train,
                  epoch: int = 10000, rate=1e-2,
                  tracking_loss=None,
                  neighbor_number=10,
                  reduce_factor=0.95,
                  temp_init=100,
                  batch_size: int = 50,
                  ):
        """
                On epoch: GD All parameters -> SA premise parameters
                :param batch_size:
                :param neighbor_number:
                :param temp_init:
                :param x_train:
                :param y_train:
                :param epoch:
                :param rate:
                :param tracking_loss:
                :param reduce_factor:
                :return:
                """
        self.name = "SA3-ANFIS"
        print("==== TRAINING PHASE ====")
        print("INFOMATION:")
        print(f"Name: {self.name}")
        print(f"Rule number: {self.rule_number}")
        print(f"Window size: {self.window_size}")
        print(f"Neighbor number: {neighbor_number}")
        print(f"Reduce factor: {reduce_factor}")
        print(f"Temperature initialized value: {temp_init}")
        print(f"Epoch: {epoch}")
        print(f"Start training ...")

        # Creating Placeholder
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating cost and optimizer
        cost = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

        saver = tf.train.Saver()

        # Saving diretory
        saving_dir: str = f"metadata/models/sa3ANFIS/rl{self.rule_number}ws{self.window_size}"
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        saving_path: str = f"{saving_dir}/model.h5"

        # Check tracking_loss flags
        tracking_list = np.empty((0,))

        # Initializing session
        with tf.Session() as sess:

            # Start training
            sess.run(tf.global_variables_initializer())
            for e in range(1, epoch + 1):
                # GD phase for all parameters
                # Shuffle training data
                shuffle = np.random.permutation(np.arange(len(y_train)))
                x_train = x_train[shuffle]
                y_train = y_train[shuffle]
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})
                for i in np.arange(0, len(y_train) // batch_size):

                    start = i * batch_size
                    batch_x = x_train[start:start + batch_size]
                    batch_y = y_train[start:start + batch_size]
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

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
                        r = sa_random(0, 1)
                        if r > np.exp(-df / boltzmann_constants() / temp):
                            f_new = f
                            previous_parameters = self.w_fuzz, self.weights, self.bias
                        else:
                            f_new = f0
                            self.w_fuzz, self.weights, self.bias = previous_parameters
                    f0 = f_new
                    temp = reduce_factor * temp

                # Loss value
                c = sess.run(cost, feed_dict={x: x_train, y: y_train})

                # Appened new loss value to track_list
                if tracking_loss:
                    tracking_list = np.append(tracking_list, c)

            print(f"Saving model to {saving_path} ...")
            saver.save(sess, save_path=saving_path)

            # Saving figures
            tracking_dir = f"results/sa2ANFIS/rl{self.rule_number}ws{self.window_size}/tracks"

            if not os.path.exists(tracking_dir):
                os.makedirs(tracking_dir)

            tracking_fig_path = f"{tracking_dir}/track.svg"
            tracking_data_path = f"{tracking_dir}/track.csv"
            tracking_fig_title = f"{self.name}: Rule number : {self.rule_number} Window size : {self.window_size}"
            print(f"Saving tracking figures to {tracking_fig_path}")
            Figures.track(data=tracking_list, data_label="Loss function",
                          first_label='epoch', second_label='MSE',
                          path=tracking_fig_path,
                          title=tracking_fig_title)

            # Saving tracking data
            print(f"Saving tracking list to {tracking_data_path} ...")
            DataFrame(tracking_list).to_csv(path_or_buf=tracking_data_path, header=None)
            print("Training completed!")

    def test(self, x_test, y_test, name: str):
        print("==== TESTING PHASE ====")
        # Creating Placeholder
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1, self.window_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Creating predict
        predicted_tensors = self.output(x)
        mse_tensors = tf.reduce_mean(tf.squared_difference(self.output(x), y))
        saver = tf.train.Saver()

        metadata_dir = f"metadata/models/{name}/rl{self.rule_number}ws{self.window_size}"
        print(self.rule_number)
        print(self.window_size)
        load_path = f"{metadata_dir}/model.h5"

        # Saving dir
        result_dir = f"results/{name}/rl{self.rule_number}ws{self.window_size}/test"
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        # Restoring models
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, load_path)
            predict_values: np.ndarray = sess.run(predicted_tensors, feed_dict={x: x_test})
            print(predict_values.shape)

            # Calculating MSE
            mse_point = sess.run(mse_tensors, feed_dict={x: x_test, y: y_test})
            print(mse_point)
        # Saving test data
        compare_test_data = np.concatenate((predict_values, y_test), axis=1)
        DataFrame(compare_test_data).to_csv(path_or_buf=f"{result_dir}/data.csv", header=["predict", "actual"])
        # Saving compare figures
        compare_figures_path = f"{result_dir}/results.svg"
        compare_figures_title = f"{self.name} test results: Rule number = {self.rule_number} , " \
            f"Window size = {self.window_size}"
        Figures.compare_test_figures(predict=predict_values,
                                     actual=y_test,
                                     predict_label='Predicted',
                                     actual_label='Actual',
                                     path=compare_figures_path,
                                     title=compare_figures_title,
                                     ratio=0.3
                                     )
        # Saving reports
        print(f"Saving reports ...")
        Reports.origin_anfis(name=self.name,
                             rule_number=self.rule_number,
                             window_size=self.window_size,
                             mse=float(mse_point),
                             dataset=DEFAULT_DATASET,
                             path=f"{result_dir}/reports.json"
                             )
