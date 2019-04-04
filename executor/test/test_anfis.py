from models.ANFIS import ANFIS
from utils.extractData import get_information

DATA_CONFIG_DIRECTORY = 'data_config.json'
DATASET_DIRECTORY = 'dataset/dru_10_minutes.csv'


def test():
    x_train, y_train, x_test, y_test, window_size, attribute = get_information(DATA_CONFIG_DIRECTORY, DATASET_DIRECTORY)

    model = ANFIS(rule_number=10)

    model.test(x_test, y_test)
