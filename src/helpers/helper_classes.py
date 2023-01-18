from typing import Optional, List
import pandas as pd
from feature_engine.selection.base_selector import BaseSelector


class AddFeatureNames(BaseSelector):
    """Adds the feature names back to the transformed data."""

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        prefix: Optional[str] = "feature_",
    ):
        self.feature_names = feature_names
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit to data, then transform it."""
        if self.feature_names is None:
            self.feature_names = [f"{self.prefix}{i}" for i in range(X.shape[1])]

        if len(self.feature_names) != X.shape[1]:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match "
                f"number of features in feature_names ({len(self.feature_names)})."
            )

        X = pd.DataFrame(X, columns=self.feature_names)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data."""
        X = pd.DataFrame(X, columns=self.feature_names)
        return X
