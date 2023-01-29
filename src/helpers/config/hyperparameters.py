import optuna


class HyperparameterConfig:
    def __init__(self, trial: optuna.trial.Trial) -> None:
        self.trial = trial
        self.n_components = 5

    def get_params(self) -> dict:
        params = {
            "n_components": self.n_components,
        }
        return params


class LGBMHyperparameterConfig(HyperparameterConfig):
    def __init__(self, trial: optuna.trial.Trial) -> None:
        self.trial = trial

        self.num_leaves = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 1500),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 200, 10000, step=100
            ),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "subsample": trial.suggest_float("subsample", 0.2, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1),
        }

    def get_params(self) -> dict:
        return self.params


class LRHyperparameterConfig(HyperparameterConfig):
    def __init__(self, trial: optuna.trial.Trial) -> None:
        self.trial = trial
        self.params = {
            "penalty": "elsaticnet",
            "l1_ratio": self.trial.suggest_float("l1_ratio", 0, 1),
            "C": self.trial.suggest_float("C", 0, 10, log=True),
        }

    def get_params(self) -> dict:
        return self.arams
