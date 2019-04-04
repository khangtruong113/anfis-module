### ANFIS model

## Prerequisites
* OS: Ubuntu 18.04 or 16.04
* Software: conda (lastest version)

## Preparing environments
1. Go to project directory and onstalling conda environments
```
conda env create --name anfis-module -f=environments.yml
```
2. Activate environments and use after this:
```
source activate anfis-module
```

## Cores
1. Training ANFIS models
```
python train_anfis.py
```
2. Testing and writing to reports ANFIS models:
```
python test_and_report_anfis.py
```

## Outputs
1. Models
* Path: ```metadata/models/<model_name>/rl<rule_number>ws<window_size>/models.h5```
2. Tracking
* Figure path: ```results/<model_name>/rl<rule_number>ws<window_size>/tracks/track.svg```
* Data path: ```results/<model_name>/rl<rule_number>ws<window_size>/tracks/track.csv```
3. Test
* Figure path: ```results/<model_name>/rl<rule_number>ws<window_size>/test/results.svg```
* Data path: ```results/<model_name>/rl<rule_number>ws<window_size>/test/data.csv```
4. Reports 
* Reports path: ```results/<model_name>/rl<rule_number>ws<window_size>/test/reports.json```
