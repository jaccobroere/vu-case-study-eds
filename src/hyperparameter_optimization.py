import configparser
import os
from joblib import load
import json
import datetime as dt
from tqdm import tqdm
import optuna
import logging
from helpers.config.hyperparameters import OptunaOptimzation
from helpers.config.hyperparameters import (
    PCA_LGBM_CFG,
    SPCA_LGBM_CFG,
    GSPCA_LGBM_CFG,
    PCA_LR_CFG,
    SPCA_LR_CFG,
    GSPCA_LR_CFG,
)
from helpers.helper_functions import alpha_setter

# Read config.ini file
config = configparser.ConfigParser()
config.read("config.ini")

os.chdir(config["PATH"]["ROOT_DIR"])

OPTUNA_DIR = config["LOGGING"]["OPTUNA_DIR"]
DATA_DIR = config["PATH"]["DATA_DIR"]
DATASETS = json.loads(config.get("PARAMS", "DATASETS"))

# Set logger for Optuna
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(
    logging.FileHandler(
        os.path.join(OPTUNA_DIR, f"{TIMESTAMP}_optuna_run.log"), mode="a"
    )
)
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


def init_hyperparameter_configs():
    hyperparameter_configs = {
        # "PCA_LGBM": PCA_LGBM_CFG(),
        # "SPCA_LGBM": SPCA_LGBM_CFG(),
        "GSPCA_LGBM": GSPCA_LGBM_CFG(),
        # "PCA_LR": PCA_LR_CFG(),
        # "SPCA_LR": SPCA_LR_CFG(),
        "GSPCA_LR": GSPCA_LR_CFG(),
    }
    return hyperparameter_configs


def run_all_optimizations(
    X_train, y_train, hyperparameter_configs, dataset, n_trials=50
):
    study_dict = {}

    for name, cfg in tqdm(hyperparameter_configs.items()):
        print(f"Running {name} optimization")

        if name.split("_")[0] == "GSPCA":
            cfg.params["pca"]["alpha"] = alpha_setter(dataset)

        optimizer = OptunaOptimzation(
            X_train,
            y_train,
            n_trials=n_trials,
            hyperparameter_config=cfg,
            name=name,
        )
        optimizer.run()

        # Save study object
        if not os.path.exists(f"{OPTUNA_DIR}{dataset}"):
            os.makedirs(f"{OPTUNA_DIR}{dataset}")

        optimizer.save_study(
            path=f"{OPTUNA_DIR}{dataset}/{TIMESTAMP}_{name}_optuna_run.csv"
        )
        study_dict[name] = optimizer.study
        print(f"Best score: {optimizer.study.best_value}")

    return study_dict


if __name__ == "__main__":
    data = load(DATA_DIR + "/microarray-data-dict.lib")

    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        X_train = data[dataset]["none"]["X_train"]
        y_train = data[dataset]["none"]["y_train"].iloc[:, 0]
        print(f"X_train shape: {X_train.shape}")

        run_all_optimizations(
            X_train, y_train, init_hyperparameter_configs(), dataset, n_trials=50
        )
