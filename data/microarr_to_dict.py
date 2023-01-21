#!/usr/bin/env python
from IPython import get_ipython
import zipfile
import os
import tarfile

# import utility modules
import pandas as pd
import numpy as np
import configparser
import os
import tarfile

# import sweetviz

# helper functions
# from helpers.helper_functions import transform_data, add_actuals
# from helpers.helper_classes import AddFeatureNames

# sklearn
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import set_config

# Joblib
from joblib import dump, load

# feature_engine
from feature_engine.selection import DropFeatures, DropConstantFeatures, DropDuplicateFeatures

# Progress bar
from tqdm import tqdm

def get_PCA(n_components = 20):
    return Pipeline([
    # Imputing missing values
        ('imputer', SimpleImputer(missing_values=np.nan, strategy = 'mean')),

    # # Step 0:
    #     # Drop constant and duplicate features
        ('drop_constant', DropConstantFeatures(tol=0.98)),

    # # Step 1:
    #     # Apply scaling to data as it is a requirement for the variance maximization procedure of PCA
        ('scaler', StandardScaler()),
    # Step 2:
        # Apply PCA
        ('pca', PCA(n_components=n_components, random_state=SEED)),
    ])

def get_SPCA(n_components = 20):
    return Pipeline([
    
    # Imputing missing values
        ('imputer', SimpleImputer(missing_values=np.nan, strategy = 'mean')),

    # # Step 0:
    #     # Drop constant and duplicate features
        ('drop_constant', DropConstantFeatures(tol=0.98)),

    # Step 1:
        # Apply scaling to data as it is a requirement for the variance maximization procedure of PCA
        ('scaler', StandardScaler()),
    # Step 2:
        # Apply PCA
        ('spca', SparsePCA(n_components=n_components, random_state=SEED, alpha=1, max_iter=400, n_jobs = -1)),
    ])

def get_none_pipeline():
    return Pipeline([
        # Imputing missing values
        ('imputer', SimpleImputer(missing_values=np.nan, strategy = 'mean')),

        # Step 0:
            # Drop constant and duplicate features
        ('drop_constant', DropConstantFeatures(tol=0.98)),

        # Step 1:
            # Scale features as most methods utilized benefit from scaling s.t. no one feature dominates.
        ('scaler', StandardScaler()),
    ])

def iteratively_transform(data_names, fname, seed = 2023):

    if os.path.isfile(fname):
        return -2

    # construct data dictionary.
    data_full = {}

    for i, name in enumerate(tqdm((data_names))):
        data_full[name] = {}
        X_cur = pd.read_csv(config['PATH']['MICR_CSV'] + '/' + name + '_inputs.csv', header = None)
        y_cur = pd.read_csv(config['PATH']['MICR_CSV'] + '/' + name + '_outputs.csv', header = None)
        X_train, X_test, y_train, y_test = train_test_split(X_cur, y_cur, test_size = 0.33, random_state=seed)
        
        for key in ['none','pca']:
            match key:
                case 'none': pipe = get_none_pipeline()
                case 'pca': pipe = get_PCA()
                case 'spca': pipe = get_SPCA()

            data_full[name][key] = {}
            data_full[name][key]['X_train'] = pipe.fit_transform(X_train)
            data_full[name][key]['X_test'] = pipe.transform(X_test)
            data_full[name][key]['y_train'] = y_train
            data_full[name][key]['y_test'] = y_test
    
    dump(data_full, fname)
    return data_full

if __name__ == "__main__":
    # Read config
    config = configparser.ConfigParser()
    config.read('../src/config.ini')
    os.chdir(config['PATH']['ROOT_DIR'])
    set_config(transform_output='pandas')


    # Set constants
    SEED = config.getint('PARAMS', 'SEED')
    N_COMPONENTS = config.getint('PARAMS', 'N_COMPONENTS')

    # Obtain dataset names
    dnames, dset = os.listdir(config['PATH']['MICR_CSV']), set()
    for fname in dnames: dset.add(fname[:fname.find('_')])
    data_full_fname = config['PATH']['DATA_DIR'] + '/micro_dict.lib'


    data_full = iteratively_transform(dset, data_full_fname, SEED)
    if data_full == -2:
        print("data object is already present, skipped transformations...")
