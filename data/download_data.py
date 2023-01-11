#!/usr/bin/env python
from IPython import get_ipython
import zipfile
import os


def remove(filename):
    if os.path.exists(filename):
        os.remove(filename)


def unzip(filename):
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()


def download(kaggle_query: str):
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not installed, installing now")
        get_ipython().system("pip install kaggle")

    print("Downloading data")
    get_ipython().system(kaggle_query)


if __name__ == "__main__":
    download("kaggle datasets download -d crawford/gene-expression")
    unzip("gene-expression.zip")
