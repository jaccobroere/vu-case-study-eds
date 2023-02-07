##############################################################################################################
################   1. Imports
##############################################################################################################

# import utility modules
import pandas as pd
import numpy as np
import configparser
import os
from joblib import dump, load
import datetime as dt
from tqdm import tqdm
import json


# import sweetviz
import matplotlib.pyplot as plt

# import optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

# helper functions
from helpers.helper_functions import transform_data, add_actuals, get_pca_pipeline, get_model
from helpers.helper_classes import AddFeatureNames, GeneSPCA, EnetSPCA
from helpers.config.hyperparameters import OptunaOptimzation
from helpers.config.hyperparameters import PCA_LGBM_CFG, SPCA_LGBM_CFG, GSPCA_LGBM_CFG, PCA_LR_CFG, SPCA_LR_CFG, GSPCA_LR_CFG


# sklearn
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve, RocCurveDisplay, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# LightGBM
from lightgbm import LGBMClassifier

# feature_engine
from feature_engine.selection import DropFeatures, DropConstantFeatures, DropDuplicateFeatures

##############################################################################################################	
################   2. Read config
##############################################################################################################

# Read config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

os.chdir(config['PATH']['ROOT_DIR'])

DATA_DIR = config['PATH']['DATA_DIR']
DATASETS = json.loads(config.get('PARAMS', 'DATASETS'))
PIPE_DIR = config["LOGGING"]["PIPE_DIR"]
LOG_DIR = config["LOGGING"]["LOG_DIR"]

# Load data library
data = load(DATA_DIR + '/microarray-data-dict.lib')
fitted_pipelines = load(os.path.join(PIPE_DIR, 'fitted_pipelines.lib'))

##############################################################################################################
################   3. Create tables
##############################################################################################################

multicolumn = pd.MultiIndex.from_product([['PCA', 'SPCA', 'GSPCA'], ['LGBM', 'LR']])
res = pd.DataFrame(index=DATASETS, columns=multicolumn)

for dataset in DATASETS:
    X_train, X_test = (
            data[dataset]["none"]["X_train"],
            data[dataset]["none"]["X_test"],
    )
    y_train, y_test = (
            data[dataset]["none"]["y_train"],
            data[dataset]["none"]["y_test"],
    )
    
    pipes = fitted_pipelines[dataset]
    
    for name, pipe in pipes.items():
        pca_name, model_name = name.split("_")
        
        score = pipe.score(X_test, y_test)
        res.loc[dataset, (pca_name, model_name)] = score

res.to_latex(os.path.join(LOG_DIR, 'latex_tables', 'classification_results.tex'))