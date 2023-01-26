from typing import Optional, List
import pandas as pd
import numpy as np
from feature_engine.selection.base_selector import BaseSelector
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm



class Gene_SPCA(BaseEstimator, TransformerMixin):
    
    """ 
        SKLearn compatible transformer implementing the SPCA variant 
        for gene expression data as described in "Sparse Principal Component Analysis" Zou et al (2006)
    """

    def __init__(self, n_comps = 20, max_iter = 2000, tol = 0.001, improve_tol = 0.00001, l1 = 5):
        self.max_iter = max_iter
        self.tol = tol
        self.improve_tol = improve_tol
        self.n_comps = n_comps
        self.l1 = l1
        self.loadings = None
        self.hasFit = False
        self.nonzero = -1
        self.zero = -1
        self.totloadings = -1
    
    def fit(self, X, y = None, verbose = 0):

        self.totloadings = self.n_comps * X.shape[1]

        if verbose: 
            # print("Progress based on max iterations:")
            pbar = tqdm(total = self.max_iter)

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Step 1: Setup first iteration
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        A = Vt.T[:,:self.n_comps]
        B = np.zeros((A[:,0].shape[0], self.n_comps))
        XtX = X.T @ X
        
        #Initialize progress monitors arbitrarily large
        diff, diff_improve = 100, 100 
        iter = 0

        # Loop of step 2 and 3 until convergence / maxiter:
        while iter < self.max_iter and diff > self.tol and diff_improve > self.improve_tol:
            B_old = np.copy(B)
            
            # Update B (step 2*)
            input = A.T @ XtX
            for i in range(self.n_comps):
                B[:, i] = self._soft_threshold(input[i,:], self.l1)
            
            # Monitor change
            diff_old = diff
            diff = self._max_diff(B_old, B)
            diff_improve = np.abs(diff - diff_old)

            # Update A (step 3)
            A_old = A    
            Un, s, Vnt = np.linalg.svd(XtX @ B, full_matrices=False)
            A = Un @ Vnt

            if verbose: pbar.update(1)
            iter = iter + 1
        if verbose: pbar.close()

        # Normalize loadings after loop
        B = self._normalize_mat(B)
        self.loadings = B
        self.nonzero = np.count_nonzero(B)
        self.zero = self.totloadings - self.nonzero
        return self
    
    def transform(self, X, y = None):
        return X @ self.loadings

    # Internal class helper functions
    def _soft_threshold(self, vec, l1):
        temp = np.maximum(0, (np.abs(vec) - l1 / 2))
        return temp * np.sign(vec)

    def _max_diff(self, X1, X2):
        return np.max(np.abs(X1 - X2))

    def _normalize_mat(self, X):
        for i in range(X.shape[1]):
            X[:,i] = X[:,i] / np.maximum(np.linalg.norm(X[:,i]), 1)
        return X 

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
