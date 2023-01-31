from typing import Optional, List
import pandas as pd
import numpy as np
from feature_engine.selection.base_selector import BaseSelector
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition._sparse_pca import SparsePCA
from sklearn.decomposition._dict_learning import dict_learning
from sklearn.utils.extmath import svd_flip
from sklearn.linear_model import ElasticNet


from scipy.linalg import sqrtm
from scipy.optimize import minimize

from itertools import repeat

from multiprocessing import Pool


class EnetSPCA(BaseEstimator, TransformerMixin):
    """
    SKLearn compatible transformer implementing the SPCA algorithm as described in "Sparse Principal Component Analysis" Zou et al (2006)
    """

    def __init__(
        self,
        n_components=20,
        max_iter=10000,
        tol=0.00001,
        alpha=0.1,
        l1_ratio=0.5,
        use_sklearn=True,
        n_jobs=0,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loadings = None
        self.nonzero = -1
        self.zero = -1
        self.totloadings = -1
        self.use_sklearn = use_sklearn
        self.n_jobs = n_jobs

    def fit(self, X, y=None, verbose=0):

        # Calculate total number of loadings
        self.totloadings = self.n_components * X.shape[1]

        # Setup progress bar
        if verbose:
            print("Progress based on max iterations:")
            pbar = tqdm(total=self.max_iter)

        # Convert to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values

        ## Step 1: Setup first iteration
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        A = Vt.T[:, : self.n_components]
        B = np.zeros((A[:, 0].shape[0], self.n_components))
        XtX = X.T @ X
        Sig_root = sqrtm(XtX)
        Sig_root = Sig_root.real

        # ElasticNET() is not suitable for alpha = 0, return PCA results
        if self.alpha == 0:
            return A

        # Initialize progress monitors arbitrarily large
        diff, diff_nonimprove = 100, 0
        iter = 0

        ## Loop of step 2 and 3 until convergence / maxiter:
        while iter < self.max_iter and diff > self.tol and diff_nonimprove < 5:
            B_old = np.copy(B)

            ## Update B (step 2*)

            # Check if user wants to use sklearn or scipy implementation
            if self.use_sklearn:

                # Setup parallelization if n_jobs != 0
                if self.n_jobs != 0:

                    if self.n_jobs == -1:
                        threads = None
                    else:
                        threads = self.n_jobs

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
                        B[:, i] = (
                            ElasticNet(
                                alpha=self.alpha, fit_intercept=False, max_iter=10000
                            )
                            .fit(Sig_root, Sig_root @ A[:, i])
                            .coef_
                        )

            else:
                # Scipy implementation, basically not-runnable due to time constraints.
                for i in range(self.n_components):
                    B[:, i] = minimize(
                        self._criterion, np.zeros(A[:, i].shape[0]), args=(XtX, A[:, i])
                    )
                    print(i)

            # Monitor change
            diff_old = diff
            diff = np.linalg.norm(np.abs(B - B_old))
            if diff_old < diff:
                diff_nonimprove += 1

            # print(diff)

            # Update A (step 3)
            # A_old = A
            Un, s, Vnt = np.linalg.svd(XtX @ B, full_matrices=False)
            A = Un @ Vnt

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
        return X @ self.loadings

    def _max_diff(self, X1, X2):
        return np.max(np.abs(X1 - X2))

    def _normalize_mat(self, X):
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] / np.maximum(np.linalg.norm(X[:, i]), 1)
        return X

    def _criterion(self, x, XtX, alpha_j):
        return (
            (alpha_j - x).T @ XtX @ (alpha_j - x)
            + self.alpha * self.l1_ratio * np.linalg.norm(x, 1)
            + 0.5 * self.alpha * (1 - self.l1_ratio) * np.linalg.norm(x, 2)
        )

    def _enet_criterion(self, i, A, Sig_root):
        return (
            ElasticNet(alpha=self.alpha, fit_intercept=False, max_iter=10000)
            .fit(Sig_root, Sig_root @ A[:, i])
            .coef_
        )


class LoadingsSPCA(SparsePCA):
    """
    This class, LoadingsSPCA, is an altered version of the SparsePCA class from
    scikit-learn (sklearn). The main difference is that LoadingsSPCA includes
    an additional attribute, 'loadings', which saves the loadings from the PCA analysis.
    The class uses the same parameters and methods as the sklearn SparsePCA class,
    with the added functionality of saving the loadings for further analysis.

    Parameters
    ----------
    n_components : int or None (default: None)
        Number of sparse components to use. If None, use all the components
    alpha : float (default: 1)
        Sparsity controlling parameter. Higher values lead to sparser solutions
    ridge_alpha : float (default: 0.01)
        Amount of ridge shrinkage to apply in order to improve conditioning when
        calling the transform method
    max_iter : int (default: 1000)
        Maximum number of iterations to perform
    tol : float (default: 1e-8)
        Tolerance for stopping criterion
    method : {'lars', 'cd'} (default: 'lars')
        lars: uses the least angle regression method to solve the lasso problem
        cd: uses the coordinate descent method to compute the Lasso solution
    n_jobs : int or None (default: None)
        Number of parallel jobs to run. None means 1.
        ``-1`` means using all processors.
    U_init : array of shape (n_features, n_components)
        Initial values for the loadings for warm restart scenarios
    V_init : array of shape (n_samples, n_components)
        Initial values for the codes for warm restart scenarios
    verbose : bool (default: False)
        If verbose is True the objective function and sparsity are printed at each
        iteration
    random_state : int, RandomState instance or None (default: None)
        Seed of the pseudo random number generator to use when shuffling the data.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Sparse components extracted from the data.
    error_ : array
        Vector of errors at each iteration
    n_iter_ : int
        Number of iterations run
    loadings_ : array
        The loadings from the PCA analysis

    """

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        ridge_alpha=0.01,
        max_iter=1000,
        tol=1e-8,
        method="lars",
        n_jobs=None,
        U_init=None,
        V_init=None,
        verbose=False,
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            alpha=alpha,
            ridge_alpha=ridge_alpha,
            max_iter=max_iter,
            tol=tol,
            method=method,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            U_init=U_init,
            V_init=V_init,
        )

    def _fit(self, X, n_components, random_state):
        # Transpose U and V for dictionary learning if the have been initialized
        code_init = self.V_init.T if self.V_init is not None else None
        dict_init = self.U_init.T if self.U_init is not None else None

        # Perform dictionary learning to solve the PCA problem with l1 penalty on the components
        code, dictionary, E, self.n_iter_ = dict_learning(
            X.T,
            n_components,
            alpha=self.alpha,
            tol=self.tol,
            max_iter=self.max_iter,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=random_state,
            code_init=code_init,
            dict_init=dict_init,
            return_n_iter=True,
        )

        # flip eigenvectors' sign to enforce deterministic output
        code, dictionary = svd_flip(code, dictionary, u_based_decision=False)
        self.components_ = code.T

        # Normalize the components
        components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm

        # Set attributes
        self.n_components_ = len(self.components_)
        self.error_ = E

        # Save loadings and amount of zero and nonzero elements
        self.loadings = self.components_.T
        self.nonzero = np.count_nonzero(self.loadings)
        self.zero = self.loadings.shape[0] * self.loadings.shape[1] - self.nonzero

        return self


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
            # print("Progress based on max iterations:")
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

            # print(diff)

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

    # Internal class helper functions
    def _soft_threshold(self, vec, l1):
        temp = np.maximum(0, (np.abs(vec) - l1 / 2))
        return temp * np.sign(vec)

    def _max_diff(self, X1, X2):
        return np.max(np.abs(X1 - X2))

    def _normalize_mat(self, X):
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] / np.maximum(np.linalg.norm(X[:, i]), 1)
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

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data."""
        X = pd.DataFrame(X)
        X.columns = self.feature_names
        return X
