from models.ANFIS import ANFIS
from utils.extractData import get_information

DATA_CONFIG_DIRECTORY = 'data_config.json'
DATASET_DIRECTORY = 'dataset/dru_10_minutes.csv'
RULE_NUMBER_LIST = [5, 10, 15, 20]


def test():
    x_train, y_train, x_test, y_test, window_size, attribute = get_information(DATA_CONFIG_DIRECTORY, DATASET_DIRECTORY)

    # Origin ANFIS
    for item in RULE_NUMBER_LIST:
        model = ANFIS(rule_number=item)
        model.test(x_test, y_test, name='originANFIS')
