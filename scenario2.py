from handle_scenario import run_scenario

DATASET_DIRECTORY = 'data_config.json'
CONFIGURATION_DIRECTORY = 'model_config.json'
MODEL_TYPE = "SA2 ANFIS"

run_scenario(dataset_dicrectory=DATASET_DIRECTORY, configuration_directory=CONFIGURATION_DIRECTORY,
             model_type=MODEL_TYPE)