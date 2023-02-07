from typing import Optional, List
import pandas as pd
import numpy as np
from feature_engine.selection.base_selector import BaseSelector
from helpers.helper_classes import GeneSPCA, LoadingsSPCA, AddFeatureNames, EnetSPCA
from sklearn.decomposition import PCA, SparsePCA
from sklearn.pipeline import Pipeline


def transform_data(df: pd.DataFrame):
    """Transforms the data to a format that can be used by the model.

    Args:
        df (pd.DataFrame): The raw data.

    Returns:
        pd.DataFrame: The transformed data.
    """
    # Remove "call" columns and transpose data
    cols = [c for c in df.columns if "call" not in c]
    df = df.loc[:, cols].T

    # Set column names and index
    df.columns = df.loc["Gene Accession Number", :]
    df.drop(["Gene Accession Number", "Gene Description"], inplace=True)
    df.index.set_names("patient", inplace=True)
    df.index = pd.to_numeric(df.index)

    return df


def add_actuals(df: pd.DataFrame, actuals: pd.DataFrame, target: str = "cancer"):
    """Adds the actuals to the transformed data.

    Args:
        df (pd.DataFrame): The transformed data.
        actuals (pd.DataFrame): The actuals.

    Returns:
        pd.DataFrame: The transformed data with actuals.
    """
    res = pd.merge(actuals, df, on="patient", how="inner")
    res[target] = (res[target] == "ALL").astype(int)

    # Set index column back to patient
    res.index = res["patient"]
    res.drop("patient", axis=1, inplace=True)

    return res


def get_spca(alpha, n_components = 20, n_jobs = 6):
    return EnetSPCA(alpha = alpha, max_iter = 20, tol = 0.0001, l1_ratio = 0.1, n_components = n_components, n_jobs = 6)


def get_gene_spca(l1, n_components = 20):
    spca_obj = GeneSPCA(max_iter = 10000, tol = 0.0000001, n_components = n_components, alpha = l1)
    return spca_obj

def get_data_pev(X, n_components = 20, verbose = 0, step_size = 0.5, get_transform = get_spca, alpha_arr = None):
    """ 
    Function that returns the explained variance of the first principal component for a single dataset versus 
    the number of non-zero loadings / genes

    Returns:
    - nonzero_columns_arr: array with number of columns with a non-zero influence on the first principal component
    - nonzero_loadings_arr: array with number of non-zero loadings of 'B' matrix
    - PEV_var_arr: array with explained variance of first principal component
    """
    
    # Initialize values for loop
    nz_percentage = 1
    alpha_cur = 0

    # arrays
    nz_loadings = []
    nz_cols = []
    PEV_arr = []

    if alpha_arr:
            for alpha_cur in alpha_arr:
                # Obtain and fit spca object
                spca_cur = get_transform(alpha_cur, n_components = n_components).fit(X, verbose = 1)
                X_spca_cur = spca_cur.transform(X)

                # Obtain PEV: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8636462/
                X_recovered_cur = X_spca_cur @ spca_cur.loadings.T
                PEV = 1 -  np.linalg.norm(X_recovered_cur.values - X.values, ord = 'fro') ** 2 / np.linalg.norm(X.values, ord = 'fro') ** 2

                # Count number of nonzero loadings and columns
                nz_percentage_old = nz_percentage
                nz_percentage = spca_cur.nonzero / spca_cur.totloadings
                if nz_percentage - nz_percentage_old < 0.05:
                    step_size = step_size * 2
                elif nz_percentage - nz_percentage_old > 0.4:
                    step_size = step_size / 2
                
                # Append values to arrays
                zero_rows = sum(np.count_nonzero(spca_cur.loadings[i,:]) == 0 for i in range(spca_cur.loadings.shape[0]))
                nz_cols.append((X.shape[1] - zero_rows) / X.shape[1])
                nz_loadings.append(nz_percentage)
                PEV_arr.append(PEV)

                # Print values when verbose
                if verbose == 1:
                    print("regularization = ", alpha_cur)
                    print("nonzero columns = ", nz_cols[-1])
                    print("nonzero loadings = ", nz_loadings[-1])
                    print("PEV = ", PEV)
                    print("")

    else:
        while nz_percentage > 0.01:

            # Obtain and fit spca object
            spca_cur = get_transform(alpha_cur, n_components = n_components).fit(X)
            X_spca_cur = spca_cur.transform(X)
            _, R_cur = np.linalg.qr(X_spca_cur)

            # Obtain PEV: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8636462/
            X_recovered_cur = X_spca_cur @ spca_cur.loadings.T
            PEV = 1 -  np.linalg.norm(X_recovered_cur.values - X.values, ord = 'fro') ** 2 / np.linalg.norm(X.values, ord = 'fro') ** 2

            # Count number of nonzero loadings and columns
            nz_percentage_old = nz_percentage
            nz_percentage = spca_cur.nonzero / spca_cur.totloadings
            if nz_percentage - nz_percentage_old < 0.05:
                step_size = step_size * 2
            elif nz_percentage - nz_percentage_old > 0.4:
                step_size = step_size / 2
            
            # Append values to arrays
            zero_rows = sum(np.count_nonzero(spca_cur.loadings[i,:]) == 0 for i in range(spca_cur.loadings.shape[0]))
            nz_cols.append((X.shape[1] - zero_rows) / X.shape[1])
            nz_loadings.append(nz_percentage)
            PEV_arr.append(PEV)

            # Print values when verbose
            if verbose == 1:
                print("regularization = ", alpha_cur)
                print("nonzero columns = ", nz_cols[-1])
                print("nonzero loadings = ", nz_loadings[-1])
                print("PEV = ", PEV)
                print("")

            # Update l1_cur
            alpha_cur += step_size
    
    return nz_cols, nz_loadings, PEV_arr

# Bisection search for regularization parameter that sets nonzero loadings to a certain percentage
def get_regularisation_value(X, n_components, percentage_nzero_loadings, get_transform, lower_bound = 0.0001, upper_bound = 1000, verbose = 0, random_state = 2023):
    percent_nz = 0
    alpha_cur = 0
    alpha_lower = lower_bound
    alpha_upper = upper_bound
    
    while abs(percent_nz - percentage_nzero_loadings) > 0.02:
        if upper_bound - alpha_lower < 0.001:
            raise ValueError("Correct alpha likely not in interval")

        alpha_cur = (alpha_lower + alpha_upper) / 2
        cur_transform = get_transform(alpha = alpha_cur, n_components = n_components, random_state = random_state)
        cur_transform.fit(X)
        percent_nz = cur_transform.nonzero / cur_transform.totloadings
        
        if verbose:
            print(f"lower: {alpha_lower}, upper: {alpha_upper}, cur: {alpha_cur}")
            print(f"percentage nonzero: {percent_nz}")
            print('-' * 40)

        if percent_nz > percentage_nzero_loadings:
            alpha_lower = alpha_cur
        else:
            alpha_upper = alpha_cur
    return alpha_cur


def get_pca_pipeline(
    method="pca",
    n_components=5,
    random_state=2023,
    alpha=1,
    n_jobs=-1,
    max_iter=400,
    **kwargs
):
    algorithm = {
        "pca": PCA(n_components=n_components, random_state=random_state),
        "spca": EnetSPCA(
            n_components=n_components,
            alpha=alpha,
            max_iter=max_iter,
            n_jobs=n_jobs,
        ),
        "gspca": GeneSPCA(n_components=n_components, alpha=alpha, max_iter=max_iter),
    }

    return Pipeline(
        [
            ("pca", algorithm[method]),
            ("add_features_names", AddFeatureNames(prefix="cmpnt_")),
        ]
    )


def get_model(model, **kwargs):
    for k, v in kwargs.items():
        try:
            model.set_params(**{k: v})
        except ValueError:
            pass

    return model
