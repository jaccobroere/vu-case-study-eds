import pandas as pd
import configparser
import os
from joblib import dump, load
import datetime as dt
import json
import copy
from helpers.helper_functions import (
    get_pca_pipeline,
    get_model,
)
from helpers.config.hyperparameters import (
    PCA_LGBM_CFG,
    SPCA_LGBM_CFG,
    GSPCA_LGBM_CFG,
    PCA_LR_CFG,
    SPCA_LR_CFG,
    GSPCA_LR_CFG,
)


# Read config.ini file
config = configparser.ConfigParser()
config.read("config.ini")

os.chdir(config["PATH"]["ROOT_DIR"])

OPTUNA_DIR = config["LOGGING"]["OPTUNA_DIR"]
DATA_DIR = config["PATH"]["DATA_DIR"]
DATASETS = json.loads(config.get("PARAMS", "DATASETS"))
PIPE_DIR = config["LOGGING"]["PIPE_DIR"]


def parse_best_params_from_csv(path) -> dict:
    df = pd.read_csv(path)
    params = {
        "_".join(col.split("_")[1:]): df.loc[df.value.argmax(), col]
        for col in df.columns
        if "params" in col
    }

    return params


def parse_name_from_csv(path) -> str:
    return "_".join(path.split("/")[-1].split("_")[2:4])


def init_hyperparameter_configs():
    hyperparameter_configs = {
        "PCA_LGBM": PCA_LGBM_CFG(),
        "SPCA_LGBM": SPCA_LGBM_CFG(),
        "GSPCA_LGBM": GSPCA_LGBM_CFG(),
        "PCA_LR": PCA_LR_CFG(),
        "SPCA_LR": SPCA_LR_CFG(),
        "GSPCA_LR": GSPCA_LR_CFG(),
    }
    return hyperparameter_configs


if __name__ == "__main__":
    data = load(DATA_DIR + "/microarray-data-dict.lib")

    hyperparameter_configs = init_hyperparameter_configs()
    fitted_pipelines = dict.fromkeys(DATASETS, {})
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")

    for dataset in ["chin"]:
        X_train, X_test = (
            data[dataset]["none"]["X_train"],
            data[dataset]["none"]["X_test"],
        )
        y_train, y_test = (
            data[dataset]["none"]["y_train"],
            data[dataset]["none"]["y_test"],
        )

        for file in os.listdir(os.path.join(OPTUNA_DIR, dataset)):
            if not file.endswith(".csv"):
                continue

            path = os.path.join(OPTUNA_DIR, dataset, file)

            name = parse_name_from_csv(path)
            best_params = parse_best_params_from_csv(path)
            cfg = hyperparameter_configs[name]

            pipe = Pipeline(
                [
                    (
                        "pca",
                        get_pca_pipeline(
                            cfg.get_params()["pca"]["method"], **best_params
                        ),
                    ),
                    (
                        "model",
                        get_model(
                            cfg.get_model(static=True),
                            **cfg.get_params()["static"],
                            **best_params,
                        ),
                    ),
                ]
            )

            pipe.fit(X_train, y_train)

            # Save pipeline in dictionary
            fitted_pipelines[dataset][name] = copy.deepcopy(pipe)

    # dump to joblib
    dump(fitted_pipelines, os.path.join(PIPE_DIR, f"{timestamp}_fitted-pipelines.lib"))
