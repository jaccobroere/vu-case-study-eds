import pandas as pd
import numpy as np
import configparser
import os
import tarfile
from helpers.helper_functions import (
    transform_data,
    add_actuals,
    get_pca_pipeline,
    alpha_setter,
)
import json
from helpers.helper_classes import AddFeatureNames
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import set_config
from joblib import dump, load

# feature_engine
from feature_engine.selection import (
    DropFeatures,
    DropConstantFeatures,
    DropDuplicateFeatures,
)

# Progress bar
from tqdm import tqdm

# %%
# Read config.ini file
config = configparser.ConfigParser()
config.read("config.ini")
os.chdir(config["PATH"]["ROOT_DIR"])

SEED = config.getint("PARAMS", "SEED")
N_COMPONENTS = config.getint("PARAMS", "N_COMPONENTS")
DATASETS = json.loads(config.get("PARAMS", "DATASETS"))

# %%
set_config(transform_output="pandas")


def get_PCA(n_components=15):
    return Pipeline(
        [
            # Imputing missing values
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            # # Step 0:
            #     # Drop constant and duplicate features
            ("drop_constant", DropConstantFeatures(tol=0.98)),
            # # Step 1:
            #     # Apply scaling to data as it is a requirement for the variance maximization procedure of PCA
            ("scaler", StandardScaler()),
            # Step 2:
            # Apply PCA
            ("pca", get_pca_pipeline(method="pca", n_components=n_components)),
        ]
    )


def get_SPCA(n_components=15):
    return Pipeline(
        [
            # Imputing missing values
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            # # Step 0:
            #     # Drop constant and duplicate features
            ("drop_constant", DropConstantFeatures(tol=0.98)),
            # Step 1:
            # Apply scaling to data as it is a requirement for the variance maximization procedure of PCA
            ("scaler", StandardScaler()),
            # Step 2:
            # Apply PCA
            (
                "spca",
                get_pca_pipeline(method="spca", n_components=n_components, alpha=0.01),
            ),
        ]
    )


def get_GSPCA(n_components=15, dataset: str = None):
    alpha = alpha_setter(dataset)
    return Pipeline(
        [
            # Imputing missing values
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            # # Step 0:
            #     # Drop constant and duplicate features
            ("drop_constant", DropConstantFeatures(tol=0.98)),
            # Step 1:
            # Apply scaling to data as it is a requirement for the variance maximization procedure of PCA
            ("scaler", StandardScaler()),
            # Step 2:
            # Apply PCA
            (
                "spca",
                get_pca_pipeline(
                    method="gspca", n_components=n_components, alpha=alpha
                ),
            ),
        ]
    )


def get_none_pipeline():
    return Pipeline(
        [
            # Imputing missing values
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            # Step 0:
            # Drop constant and duplicate features
            ("drop_constant", DropConstantFeatures(tol=0.98)),
            # Step 1:
            # Scale features as most methods utilized benefit from scaling s.t. no one feature dominates.
            ("scaler", StandardScaler()),
        ]
    )


dnames = DATASETS
data_full = {}
data_full_fname = config["PATH"]["DATA_DIR"] + "/small_data_dict.lib"

for name in dnames:
    print(name)
    data_full[name] = {}
    X_cur = pd.read_csv(
        config["PATH"]["MICR_CSV"] + "/" + name + "_inputs.csv", header=None
    )
    y_cur = pd.read_csv(
        config["PATH"]["MICR_CSV"] + "/" + name + "_outputs.csv", header=None
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_cur, y_cur, test_size=0.33, random_state=SEED
    )

    for key in ["pca", "spca", "gspca"]:
        print(key)
        match key:
            case "pca":
                pipe = get_PCA()
            case "spca":
                pipe = get_SPCA()
            case "gspca":
                pipe = get_GSPCA(dataset=name)

        data_full[name][key] = {}
        data_full[name][key]["X_train"] = pipe.fit_transform(X_train)
        data_full[name][key]["X_test"] = pipe.transform(X_test)
        data_full[name][key]["y_train"] = y_train
        data_full[name][key]["y_test"] = y_test

dump(data_full, data_full_fname)
