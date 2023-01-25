from typing import Optional, List
import pandas as pd
import numpy as np
from feature_engine.selection.base_selector import BaseSelector
from helpers.helper_classes import Gene_SPCA

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

def get_gene_spca(l1, n_components = 20):
    spca_obj = Gene_SPCA(max_iter = 10000, n_comps = n_components, l1 = l1)
    return spca_obj

#TODO: Possibly install failsave s.t max_iter is not reached, altough this is unlikely
def get_data_pev(X, n_components = 20, verbose = 0):
    """ 
    Function that returns the explained variance of the first principal component for a single dataset versus 
    the number of non-zero loadings / genes

    Returns:
    - nonzero_columns_arr: array with number of columns with a non-zero influence on the first principal component
    - nonzero_loadings_arr: array with number of non-zero loadings of 'B' matrix
    - PEV_var_arr: array with explained variance of first principal component
    """

    # First obtain total variance
    pca = get_gene_spca(0, n_components = n_components)
    _, R = np.linalg.qr(pca.fit_transform(X))
    total_var = sum(R[i][i]**2 for i in range(R.shape[0]))

    # Initialize values for loop
    nonzero_cnt = 999999999
    l1_cur = 0
    nonzero_loadings_arr = []
    nonzero_columns_arr = []
    PEV_var_arr = []

    while nonzero_cnt > 200:

        # Obtain and fit spca object
        spca_cur = get_gene_spca(l1_cur)
        X_spca_cur = spca_cur.fit_transform(X, verbose = 0)

        # Obtain PEV
        _, R_cur = np.linalg.qr(X_spca_cur)
        explained_var_leading = R_cur[0][0]**2
        PEV = explained_var_leading / total_var

        # Count number of nonzero loadings and columns
        zero_rows = sum(np.count_nonzero(spca_cur.loadings[i,:]) == 0 for i in range(spca_cur.loadings.shape[0]))
        nonzero_cnt = X.shape[1] - zero_rows
        
        # Append values to arrays
        nonzero_columns_arr.append(nonzero_cnt)
        nonzero_loadings_arr.append(spca_cur.nonzero)
        PEV_var_arr.append(PEV)

        # Print values when verbose
        if verbose == 1:
            print("l1 = ", l1_cur)
            print("nonzero_cnt = ", nonzero_cnt)
            print("nonzero_loadings = ", spca_cur.nonzero)
            print("PEV = ", PEV)
            print("")

        # Update l1_cur
        l1_cur += 50
    
    return nonzero_columns_arr, nonzero_loadings_arr, PEV_var_arr