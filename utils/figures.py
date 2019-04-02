import matplotlib.pyplot as plt
import numpy as np


class Figures:
    @staticmethod
    def track(
            data: np.ndarray, data_label: str,
            first_label: str, second_label: str,
            path: str,
            title=None) -> None:
        """
        :param data:
        :param data_label:
        :param first_label:
        :param second_label:
        :param path:
        :param title:
        :return:
        """
        array = np.asarray(data)
        first_data = np.arange(1, array.shape[0] + 1)
        plt.title(title)
        plt.plot(first_data, array, label=data_label)
        plt.xlabel(first_label)
        plt.ylabel(second_label)
        plt.legend()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def compare_test_figures(predict: np.ndarray, actual: np.ndarray,
                             predict_label: str, actual_label: str,
                             path: str,
                             title: str,
                             ratio: 1.0):
        sampling_length = round(len(predict)*ratio)
        predict = predict[:sampling_length]
        actual = actual[:sampling_length]
        plt.plot(predict, label=predict_label)
        plt.plot(actual, label=actual_label)
        plt.title(title)
        plt.legend()
        plt.savefig(path)
        plt.close()
