##############################################################################################################
################ 1. Imports
##############################################################################################################

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from scipy.linalg import sqrtm
from itertools import repeat
from multiprocessing import Pool

##############################################################################################################
################ 2. Classes
##############################################################################################################

class EnetSPCA(BaseEstimator, TransformerMixin):
    """
    SKLearn compatible transformer implementing the SPCA algorithm as described in "Sparse Principal Component Analysis" Zou et al (2006)
    """

    def __init__(self, n_components=20, max_iter=10000, tol=0.00001, alpha = 0.1, l1_ratio = 0.5, use_sklearn = True, n_jobs = 0):
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_jobs = n_jobs
        self.loadings = None
        self.nonzero = -1
        self.zero = -1
        self.totloadings = -1
        self.use_sklearn = use_sklearn
        self.pca_loadings = None

    def fit(self, X, y=None, verbose=0):
        
        n_jobs = self.n_jobs
        # Calculate total number of loadings
        self.totloadings = self.n_components * X.shape[1]
        self.nonzero = self.totloadings

        # Setup progress bar
        if verbose:
            print("Progress based on max iterations:")
            pbar = tqdm(total=self.max_iter)

        # Convert to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values

        ## Step 1: Setup first iteration
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self.pca_loadings = Vt.T[:, :self.n_components]
        A = Vt.T[:, :self.n_components]
        B = np.zeros((A[:, 0].shape[0], self.n_components))
        XtX = X.T @ X
        Sig_root = sqrtm(XtX)
        Sig_root = Sig_root.real

        # ElasticNET() is NOT suitable for alpha = 0, return PCA results
        if self.alpha == 0:
            self.loadings = A
            return self

        # Initialize progress monitors
        diff, diff_nonimprove = 100, 0
        iter = 0

        ## Loop of step 2 and 3 until convergence / maxiter:
        while (
            iter < self.max_iter and diff > self.tol and diff_nonimprove < 3
        ):
            B_old = np.copy(B)

            ## Update B (step 2*)

            # Check if user wants to use sklearn or scipy implementation
            if self.use_sklearn:

                # Setup parallelization if n_jobs != 0
                if n_jobs != 0:

                    if n_jobs == -1:
                        threads = None
                    else:
                        threads = n_jobs

                    # Setup thread pool using a starmap
                    map_arr = list(range(self.n_components))
                    second_arg = A
                    third_arg = Sig_root
                    with Pool(threads) as pool:
                        B = np.array(
                            pool.starmap(
                                self._enet_criterion,
                                zip(map_arr, repeat(second_arg), repeat(third_arg)),
                            )
                        )
                        B = B.T

                else:
                    for i in range(self.n_components):
                        B[:, i] = self._enet_criterion(i, A, Sig_root)



            # Update A (step 3)
            Un, _, Vnt = np.linalg.svd(XtX @ B, full_matrices=False)
            A = Un @ Vnt

            ## Monitor loop progress
            
            # Convergence monitoring
            diff_old = diff
            diff = np.linalg.norm(np.abs(B - B_old))
            if diff_old < diff:
                diff_nonimprove += 1
            
            # Iterations monitoring
            iter = iter + 1
            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        # Normalize loadings after loop
        B = self._normalize_mat(B)
        self.loadings = B
        self.nonzero = np.count_nonzero(B)
        self.zero = self.totloadings - self.nonzero
        return self

    def transform(self, X, y=None):
        if self.alpha == 0:
            return X @ self.pca_loadings
        return X @ self.loadings

    def _max_diff(self, X1, X2):
        return np.max(np.abs(X1 - X2))

    def _normalize_mat(self, X):
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] / np.maximum(np.linalg.norm(X[:, i]), 1)
        return X

    def _enet_criterion(self, i, A, Sig_root):
        return (
            ElasticNet(alpha=self.alpha, l1_ratio = self.l1_ratio, fit_intercept=False, max_iter=14000)
            .fit(Sig_root, Sig_root @ A[:, i])
            .coef_
        )

class GeneSPCA(BaseEstimator, TransformerMixin):

    """
    SKLearn compatible transformer implementing the SPCA variant
    for gene expression data as described in "Sparse Principal Component Analysis" Zou et al (2006)
    """

    def __init__(
        self, n_components=20, max_iter=10000, tol=0.0001, improve_tol=0.00001, alpha=5
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.improve_tol = improve_tol
        self.n_components = n_components
        self.alpha = alpha
        self.loadings = None
        self.hasFit = False
        self.nonzero = -1
        self.zero = -1
        self.totloadings = -1

    def fit(self, X, y=None, verbose=0):

        self.totloadings = self.n_components * X.shape[1]

        if verbose:
            print("Progress based on max iterations:")
            pbar = tqdm(total=self.max_iter)

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Step 1: Setup first iteration
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        A = Vt.T[:, : self.n_components]
        B = np.zeros((A[:, 0].shape[0], self.n_components))
        XtX = X.T @ X

        # Initialize progress monitors arbitrarily large
        diff, diff_improve = 100, 100
        iter = 0

        # Loop of step 2 and 3 until convergence / maxiter:
        while (
            iter < self.max_iter and diff > self.tol and diff_improve > self.improve_tol
        ):
            B_old = np.copy(B)

            # Update B (step 2*)
            input = A.T @ XtX
            for i in range(self.n_components):
                B[:, i] = self._soft_threshold(input[i, :], self.alpha)

            # Monitor change
            diff_old = diff
            diff = self._max_diff(B_old, B)
            diff_improve = np.abs(diff - diff_old)

            # Update A (step 3)
            A_old = A
            Un, s, Vnt = np.linalg.svd(XtX @ B, full_matrices=False)
            A = Un @ Vnt

            if verbose:
                pbar.update(1)
            iter = iter + 1
        if verbose:
            pbar.close()

        # Normalize loadings after loop
        B = self._normalize_mat(B)
        self.loadings = B
        self.nonzero = np.count_nonzero(B)
        self.zero = self.totloadings - self.nonzero
        return self

    def transform(self, X, y=None):
        return X @ self.loadings

    # Applies the soft threshold as described in the paper
    def _soft_threshold(self, vec, l1):
        temp = np.maximum(0, (np.abs(vec) - l1 / 2))
        return temp * np.sign(vec)

    def _max_diff(self, X1, X2):
        return np.max(np.abs(X1 - X2))

    def _normalize_mat(self, X):
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] / np.maximum(np.linalg.norm(X[:, i]), 1)
        return X
