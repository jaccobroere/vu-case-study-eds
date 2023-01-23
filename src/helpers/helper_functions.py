from typing import Optional, List
import pandas as pd
from feature_engine.selection.base_selector import BaseSelector


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
    if len(df.loc["Gene Accession Number", :]) == len(df.columns):
        df.columns = df.loc["Gene Accession Number", :]
    else:
        df.columns = df.loc["Gene Accession Number", :].iloc[0]
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
