import optuna
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from helpers.helper_functions import get_pca_pipeline
from sklearn.model_selection import cross_val_score


class HyperparameterConfig:
    def __init__(self) -> None:
        self.random_state = 2023
        self.cv = ShuffleSplit(
            n_splits=5, test_size=0.2, random_state=self.random_state
        )

    def set_trial(self, trial: optuna.trial.Trial) -> None:
        self.trial = trial

    def get_params(self) -> dict:
        params = {
            "n_components": self.n_components,
        }
        return params


class LGBMHyperparameterConfig(HyperparameterConfig):
    def __init__(self, trial: optuna.trial.Trial) -> None:
        super().__init__(trial=trial)
        self.trial = trial
        self.model = LGBMClassifier

        # Set parameters
        self.params = {
            "static": {
                "n_jobs": -1,
                "boosting_type": "gbdt",
                "n_estimators": 500,
                "learning_rate": 0.03,
            },
            "model": {
                "num_leaves": self.trial.suggest_int("num_leaves", 15, 1500),
                "max_depth": self.trial.suggest_int("max_depth", -1, 15),
                "min_data_in_leaf": self.trial.suggest_int(
                    "min_data_in_leaf", 200, 10000, step=100
                ),
                "min_gain_to_split": self.trial.suggest_float(
                    "min_gain_to_split", 0, 15
                ),
                "subsample": self.trial.suggest_float("subsample", 0.2, 1),
                "colsample_bytree": self.trial.suggest_float(
                    "colsample_bytree", 0.2, 1
                ),
            },
            "other": {
                "cv": self.cv,
                "sampler": optuna.samplers.TPESampler(seed=self.random_state),
                # "pruner": optuna.pruners.MedianPruner(),
                "scoring": "f1",
            },
            "pca": {
                "method": "pca",
                "n_components": self.trial.suggest_int("n_components", 5, 30),
                # "alpha": self.trial.suggest_float("alpha", 0.1, 1),
            },
        }

    def get_params(self, key=None) -> dict:
        if key:
            return self.params.get(key, None)
        else:
            return self.params

    def get_model(self) -> object:
        instance = self.model(**self.params.get("model"), **self.params.get("static"))
        return instance

    def get_trial(self) -> optuna.trial.Trial:
        return self.trial


class LRHyperparameterConfig(HyperparameterConfig):
    def __init__(
        self,
        model=LogisticRegression,
    ) -> None:
        super().__init__()
        self.model = model()

    def set_trial(self, trial: optuna.trial.Trial) -> None:
        self.trial = trial
        # Set parameters
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
            "model": {
                "l1_ratio": self.trial.suggest_float("l1_ratio", 0, 1),
                "C": self.trial.suggest_float("C", 0.01, 1, log=True),
            },
            "other": {
                "cv": self.cv,
                "sampler": optuna.samplers.TPESampler(seed=self.random_state),
                # "pruner": optuna.pruners.MedianPruner(),
                "scoring": "f1",
            },
            "pca": {
                "method": "pca",
                "n_components": self.trial.suggest_int("n_components", 5, 30),
                # "alpha": self.trial.suggest_float("alpha", 0.01, 1),
            },
        }

    def get_params(self, key=None) -> dict:
        if key:
            return self.params.get(key, None)
        else:
            return self.params

    def get_model(self) -> object:
        self.model.set_params(**self.params.get("model"), **self.params.get("static"))
        return self.model

    def get_trial(self) -> optuna.trial.Trial:
        return self.trial


class OptunaOptimzation:
    def __init__(
        self,
        X_train,
        y_train,
        n_trials=50,
        hyperparameter_config=HyperparameterConfig,
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
        # Get parameters
        self.cfg.set_trial(trial=trial)
        params = self.cfg.get_params()
        model = self.cfg.get_model()

        # Make pipeline
        pca = get_pca_pipeline(**params.get("pca"))
        X = pca.fit_transform(X_train)

        # Perform cross validation
        cv_score = cross_val_score(
            model,
            X,
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

        self.study.optimize(
            lambda trial: self._objective(
                trial,
                self.X_train,
                self.y_train,
            ),
            n_trials=self.n_trials,
        )

        return self.study
