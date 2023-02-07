# Imports

# import utility modules
import pandas as pd
import numpy as np
import configparser
import os

from joblib import dump, load

# Read config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
os.chdir(config['PATH']['ROOT_DIR'])

# Read data
data = load(config['PATH']['DATA_DIR'] + '/microarray-data-dict.lib')

colnames = ['Samples', 'Features', 'Classes']

data_dict = {}
for key, v in data.items():
    samples = 0
    output = []
    y_full = []
    for data_name, data_cur in v['none'].items():
        
        # Count samples in X_train, X_test, y_train, y_test, later on divide by 2
        samples = samples + data_cur.shape[0]

        # Count unique classes in y_train, y_test
        if data_name == 'y_train' or data_name == 'y_test':
            output = output + sorted(data_cur[0].unique())
            y_full = y_full + data_cur
    print(y_full)

    # Divide samples by 2, because we have counted samples in both X and y because of nature of data file
    samples = samples / 2

    # Count features, same for X_train and X_test
    features = v['none']['X_train'].shape[1]

    # Count unique classes
    classes = len(np.unique(output))
    

    # Add to dictionary for later conversion to dataframe
    data_dict[key] = [int(samples), features, classes]


df = pd.DataFrame.from_dict(data_dict, columns=colnames, orient= 'index')
print(df.to_latex())