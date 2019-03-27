#!/usr/bin/env python

# Just disables the warning, doesn't enable AVX/FMA
import os
from info import get_info
import json
import anfis as models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Network features
DATASET_DIRECTORY = 'data_config.json'
CONFIGURATION_DIRECTORY = 'model_config.json'


def run_scenario(dataset_dicrectory,
                 configuration_directory,
                 model_type):
    """
    :param dataset_dicrectory:
    :param configuration_directory:
    :param model_type:
    """
    # Extract dataset from raw data and config_dataset JSON file
    x_train, y_train, x_test, y_test, window_size, attribute = get_info(dataset_dicrectory)

    mse = None
    with open(configuration_directory, 'r') as file:
        config_list = json.load(file)
    for config_element in config_list:
        model_name = config_element.get('name')
        if model_type == model_name:
            details = config_element.get('details')
            save_path = details.get('path')
            tracking_path = details.get('tracking_path')
            rule_number = details.get('rule_number')
            result_path = details.get('result_path')
            learning_rate = details.get('learning_rate')
            fig_path = details.get('test_fig_path')
            epoch = details.get('epoch')
            anfis_model = models.ANFIS(window_size=window_size, rule_number=rule_number,
                                       name=model_name)
            if model_name == 'Original ANFIS':
                mse = anfis_model.train(x_train=x_train, y_train=y_train,
                                        x_test=x_test, y_test=y_test,
                                        epoch=epoch, rate=learning_rate,
                                        save_path=save_path,
                                        tracking_loss=True,
                                        tracking_path=tracking_path)
            if model_name == 'SA1 ANFIS':
                neighbor_number = details.get('neighbor_number')
                mse = anfis_model.sa1_train(x_train=x_train, y_train=y_train,
                                            x_test=x_test, y_test=y_test,
                                            epoch=epoch, rate=learning_rate,
                                            neighbor_number=neighbor_number,
                                            save_path=save_path,
                                            tracking_loss=True,
                                            tracking_path=tracking_path)
            if model_name == 'SA2 ANFIS':
                neighbor_number = details.get('neighbor_number')
                mse = anfis_model.sa2_train(x_train=x_train, y_train=y_train,
                                            x_test=x_test, y_test=y_test,
                                            epoch=epoch, rate=learning_rate,
                                            neighbor_number=neighbor_number,
                                            save_path=save_path,
                                            tracking_loss=True,
                                            tracking_path=tracking_path)
            if model_name == 'SA3 ANFIS':
                neighbor_number = details.get('neighbor_number')
                mse = anfis_model.sa3_train(x_train=x_train, y_train=y_train,
                                            x_test=x_test, y_test=y_test,
                                            epoch=epoch, rate=learning_rate,
                                            neighbor_number=neighbor_number,
                                            save_path=save_path,
                                            tracking_loss=True,
                                            tracking_path=tracking_path)

            anfis_model.compare_figures(x_test, y_test, load_path=save_path, fig_path=fig_path,
                                        attribute=attribute)
            result = {'name': model_name,
                      'details':
                          {
                              'attribute': attribute,
                              'rule_number': rule_number,
                              'epoch': epoch,
                              'learning_rate': learning_rate,
                          },
                      'result':
                          {
                              'mse': float(mse)
                          }}
            with open(result_path, 'w') as path:
                json.dump(result, path)


def main():
    run_scenario(dataset_dicrectory=DATASET_DIRECTORY, configuration_directory=CONFIGURATION_DIRECTORY)


if __name__ == '__main__':
    main()
