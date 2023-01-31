import configparser
import os
from joblib import load
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

# Read config.ini file
config = configparser.ConfigParser()
config.read("config.ini")

os.chdir(config["PATH"]["ROOT_DIR"])

OPTUNA_DIR = config["LOGGING"]["OPTUNA_DIR"]
DATA_DIR = config["PATH"]["DATA_DIR"]
LOG_DIR = config["LOGGING"]["LOG_DIR"]

# Setup logging for optuna and hyperparameter optimization
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
logger.addHandler(
    logging.FileHandler(f"{LOG_DIR}{timestamp}_hyperparameter_optimization.log")
)

# Let optuna log to the same file
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


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


def run_all_optimizations(
    X_train, y_train, hyperparameter_configs, dataset, n_trials=50
):
    study_dict = {}
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")

    for name, cfg in tqdm(hyperparameter_configs.items()):
        print(f"Running {name} optimization")
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
            path=f"{OPTUNA_DIR}{dataset}/{timestamp}_{name}_optuna_run.csv"
        )
        study_dict[name] = optimizer.study

    return study_dict


if __name__ == "__main__":
    data = load(DATA_DIR + "/microarray-data-dict.lib")
    dataset_list = ["chin", "chowdary", "gravier", "west"]

    for dataset in dataset_list:
        print(f"Dataset: {dataset}")
        X_train = data[dataset]["none"]["X_train"]
        y_train = data[dataset]["none"]["y_train"].to_numpy().ravel()
        print(f"X_train shape: {X_train.shape}")

        run_all_optimizations(
            X_train, y_train, init_hyperparameter_configs(), dataset, n_trials=50
        )
