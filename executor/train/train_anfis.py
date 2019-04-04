from models.ANFIS import ANFIS
from utils.extractData import get_information

DATA_CONFIG_DIRECTORY = 'data_config.json'
DATASET_DIRECTORY = 'dataset/dru_10_minutes.csv'
RULE_NUMBER_LIST = [5, 10, 15, 20]


def train():
    x_train, y_train, x_test, y_test, window_size, attribute = get_information(DATA_CONFIG_DIRECTORY, DATASET_DIRECTORY)
    print("===> ORIGIN ANFIS TRAIN ")
    # Origin ANFIS
    for item in RULE_NUMBER_LIST:
        model = ANFIS(rule_number=item, name='originANFIS')
        model.sa1_train(x_train, y_train,
                        batch_size=200, epoch=100,
                        tracking_loss=True)

    print("===> SA1 ANFIS TRAIN ")
    # SA1 ANFIS
    for item in RULE_NUMBER_LIST:
        model = ANFIS(rule_number=item, name='sa1ANFIS')
        model.sa1_train(x_train, y_train,
                        batch_size=200, epoch=100,
                        tracking_loss=True)

    print("===> SA2 ANFIS TRAIN ")
    # SA2 ANFIS
    for item in RULE_NUMBER_LIST:
        model = ANFIS(rule_number=item, name='sa2ANFIS')
        model.sa2_train(x_train, y_train,
                        batch_size=200, epoch=100,
                        tracking_loss=True)

    print("===> SA3 ANFIS TRAIN ")
    # SA3 ANFIS
    for item in RULE_NUMBER_LIST:
        model = ANFIS(rule_number=item, name='sa3ANFIS')
        model.sa3_train(x_train, y_train,
                        batch_size=200, epoch=100,
                        tracking_loss=True)
