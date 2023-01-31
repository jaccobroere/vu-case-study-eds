import optuna
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from helpers.helper_functions import get_pca_pipeline
from sklearn.model_selection import cross_val_score


class OptunaOptimzation:
    def __init__(
        self,
        X_train,
        y_train,
        hyperparameter_config,
        n_trials=50,
        name=None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.cfg = hyperparameter_config
        self.name = name

    def _objective(
        self,
        trial,
        X_train,
        y_train,
    ):
        # Set trial and sample parameters
        self.cfg.set_trial(trial=trial)
        self.cfg.init_params()

        # Get parameters
        params = self.cfg.get_params()
        model = self.cfg.get_model()

        # Perform cross validation
        cv_score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=params.get("other").get("cv"),
            n_jobs=params.get("other").get("n_jobs"),
            scoring=params.get("other").get("scoring"),
        )

        return cv_score.mean()

    def run(self):
        self.study = optuna.create_study(
            direction="maximize",
            study_name=self.name,
        )

        # Perform (G)(S)PCA
        params = self.cfg.get_params()
        pca = get_pca_pipeline(**params.get("pca"))
        self.X_train = pca.fit_transform(self.X_train)

        # Optimize model parameters
        self.study.optimize(
            lambda trial: self._objective(
                trial,
                self.X_train,
                self.y_train,
            ),
            n_trials=self.n_trials,
        )

        return self.study

    def save_study(self, path=None):
        if path is None:
            raise ValueError("Path must be specified")

        self.study.trials_dataframe().to_csv(path, index=False)

        return self.study


class HyperparameterConfig:
    def __init__(self, model) -> None:
        self.random_state = 2023
        self.cv = ShuffleSplit(
            n_splits=5, test_size=0.2, random_state=self.random_state
        )
        self.model = model

    def set_trial(self, trial: optuna.trial.Trial) -> None:
        self.trial = trial

    def get_params(self, key=None) -> dict:
        if key:
            return self.params.get(key, None)
        else:
            return self.params

    def get_model(self, static=False) -> object:
        if static:
            return self.model.set_params(**self.params.get("static"))
        else:
            return self.model.set_params(
                **self.params.get("model"), **self.params.get("static")
            )

    def get_trial(self) -> optuna.trial.Trial:
        return self.trial


class PCA_LGBM_CFG(HyperparameterConfig):
    def __init__(self, model=LGBMClassifier()) -> None:
        super().__init__(model=model)
        self.params = {
            "static": {
                "n_jobs": -1,
                "boosting_type": "gbdt",
                "n_estimators": 500,
                "learning_rate": 0.03,
            },
            "other": {
                "cv": self.cv,
                "sampler": optuna.samplers.TPESampler(seed=self.random_state),
                # "pruner": optuna.pruners.MedianPruner(),
                "scoring": "accuracy",
            },
            "pca": {
                "method": "pca",
                "n_components": 15,
            },
        }

    def init_params(self):
        # Check if trial is set
        if self.trial is None:
            raise ValueError("Trial is not set. Please set trial first.")

        # Set parameters
        self.params["model"] = {
            "num_leaves": self.trial.suggest_int("num_leaves", 15, 1500),
            "max_depth": self.trial.suggest_int("max_depth", -1, 15),
            "min_child_samples": self.trial.suggest_int(
                "min_child_samples", 200, 10000, step=100
            ),
            "min_split_gain": self.trial.suggest_float("min_split_gain", 0, 15),
            # "subsample": self.trial.suggest_float("subsample", 0.2, 1),
            # "colsample_bytree": self.trial.suggest_float(
            #     "colsample_bytree", 0.2, 1
            # ),
        }


class SPCA_LGBM_CFG(HyperparameterConfig):
    def __init__(self, model=LGBMClassifier()) -> None:
        super().__init__(model=model)
        self.params = {
            "static": {
                "n_jobs": -1,
                "boosting_type": "gbdt",
                "n_estimators": 500,
                "learning_rate": 0.03,
            },
            "other": {
                "cv": self.cv,
                "sampler": optuna.samplers.TPESampler(seed=self.random_state),
                # "pruner": optuna.pruners.MedianPruner(),
                "scoring": "accuracy",
            },
            "pca": {"method": "spca", "n_components": 15, "alpha": 0.01},
        }

    def init_params(self):
        # Check if trial is set
        if self.trial is None:
            raise ValueError("Trial is not set. Please set trial first.")

        # Set parameters
        self.params["model"] = {
            "num_leaves": self.trial.suggest_int("num_leaves", 15, 1500),
            "max_depth": self.trial.suggest_int("max_depth", -1, 15),
            "min_child_samples": self.trial.suggest_int(
                "min_child_samples", 200, 10000, step=100
            ),
            "min_split_gain": self.trial.suggest_float("min_split_gain", 0, 15),
            # "subsample": self.trial.suggest_float("subsample", 0.2, 1),
            # "colsample_bytree": self.trial.suggest_float(
            #     "colsample_bytree", 0.2, 1
            # ),
        }


class GSPCA_LGBM_CFG(HyperparameterConfig):
    def __init__(self, model=LGBMClassifier()) -> None:
        super().__init__(model=model)
        self.params = {
            "static": {
                "n_jobs": -1,
                "boosting_type": "gbdt",
                "n_estimators": 500,
                "learning_rate": 0.03,
            },
            "other": {
                "cv": self.cv,
                "sampler": optuna.samplers.TPESampler(seed=self.random_state),
                # "pruner": optuna.pruners.MedianPruner(),
                "scoring": "accuracy",
            },
            "pca": {
                "method": "gspca",
                "n_components": 15,
                "alpha": 5,
            },
        }

    def init_params(self):
        # Check if trial is set
        if self.trial is None:
            raise ValueError("Trial is not set. Please set trial first.")

        # Set parameters
        self.params["model"] = {
            "num_leaves": self.trial.suggest_int("num_leaves", 15, 1500),
            "max_depth": self.trial.suggest_int("max_depth", -1, 15),
            "min_child_samples": self.trial.suggest_int(
                "min_child_samples", 200, 10000, step=100
            ),
            "min_split_gain": self.trial.suggest_float("min_split_gain", 0, 15),
            # "subsample": self.trial.suggest_float("subsample", 0.2, 1),
            # "colsample_bytree": self.trial.suggest_float(
            #     "colsample_bytree", 0.2, 1
            # ),
        }


class PCA_LR_CFG(HyperparameterConfig):
    def __init__(self, model=LogisticRegression()) -> None:
        super().__init__(model=model)
        self.params = {
            "static": {
                "solver": "saga",
                "random_state": self.random_state,
                "fit_intercept": True,
                "tol": 1e-4,
                "max_iter": 1000,
                "n_jobs": -1,
                "penalty": "elasticnet",
            },
            "other": {
                "cv": self.cv,
                "sampler": optuna.samplers.TPESampler(seed=self.random_state),
                # "pruner": optuna.pruners.MedianPruner(),
                "scoring": "accuracy",
            },
            "pca": {
                "method": "pca",
                "n_components": 15,
            },
        }

    def init_params(self):
        # Check if trial is set
        if self.trial is None:
            raise ValueError("Trial is not set. Please set trial first.")

        # Set parameters
        self.params["model"] = {
            "l1_ratio": self.trial.suggest_float("l1_ratio", 0, 1),
            "C": self.trial.suggest_float("C", 0.01, 1, log=True),
        }


class SPCA_LR_CFG(HyperparameterConfig):
    def __init__(self, model=LogisticRegression()) -> None:
        super().__init__(model=model)
        self.params = {
            "static": {
                "solver": "saga",
                "random_state": self.random_state,
                "fit_intercept": True,
                "tol": 1e-4,
                "max_iter": 1000,
                "n_jobs": -1,
                "penalty": "elasticnet",
            },
            "other": {
                "cv": self.cv,
                "sampler": optuna.samplers.TPESampler(seed=self.random_state),
                # "pruner": optuna.pruners.MedianPruner(),
                "scoring": "accuracy",
            },
            "pca": {
                "method": "spca",
                "n_components": 15,
                "alpha": 0.1,
            },
        }

    def init_params(self):
        # Check if trial is set
        if self.trial is None:
            raise ValueError("Trial is not set. Please set trial first.")
        # Set parameters
        self.params["model"] = {
            "l1_ratio": self.trial.suggest_float("l1_ratio", 0, 1),
            "C": self.trial.suggest_float("C", 0.01, 1, log=True),
        }


class GSPCA_LR_CFG(HyperparameterConfig):
    def __init__(self, model=LogisticRegression()) -> None:
        super().__init__(model=model)
        self.params = {
            "static": {
                "solver": "saga",
                "random_state": self.random_state,
                "fit_intercept": True,
                "tol": 1e-4,
                "max_iter": 1000,
                "n_jobs": -1,
                "penalty": "elasticnet",
            },
            "other": {
                "cv": self.cv,
                "sampler": optuna.samplers.TPESampler(seed=self.random_state),
                "scoring": "accuracy",
            },
            "pca": {
                "method": "gspca",
                "n_components": 15,
                "alpha": 5,
            },
        }

    def init_params(self):
        # Check if trial is set
        if self.trial is None:
            raise ValueError("Trial is not set. Please set trial first.")

        # Set parameters
        self.params["model"] = {
            "l1_ratio": self.trial.suggest_float("l1_ratio", 0, 1),
            "C": self.trial.suggest_float("C", 0.01, 1, log=True),
        }
