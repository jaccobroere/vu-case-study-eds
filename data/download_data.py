#!/usr/bin/env python
from IPython import get_ipython
import zipfile
import os
import tarfile
import gdown

def remove(filename):
    if os.path.exists(filename):
        os.remove(filename)


def unzip(filename):
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()

def download_gdrive(query: str, fname: str):
    try:
        import gdown
    except ImportError:
        print("gdown package not installed, installing...")
        get_ipython().system("pip install gdown")

    if os.path.exists(fname):
        print("file: " + fname + " already downloaded, skipping")
    else:
        gdown.download(query, fname, quiet=False)


def download(kaggle_query: str):
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not installed, installing now")
        get_ipython().system("pip install kaggle")

    print("Downloading data")
    get_ipython().system(kaggle_query)

def extract_tar_files(dir: str):
    for fname in os.listdir(dir):
        if fname.endswith('.tar.gz'):
            tar = tarfile.open(dir + '/' + fname, "r:gz")
            tar.extractall(dir + '/extract/')
            tar.close() 


if __name__ == "__main__":
    get_ipython().system("git clone https://github.com/kivancguckiran/microarray-data")
    extract_tar_files('microarray-data/csv')
    download_gdrive('https://drive.google.com/uc?id=1LDwIKxKrlbr8q2R2w_2fUZdOq8tn5ar8', 'microarray-data-dict.lib')
    download("kaggle datasets download -d crawford/gene-expression")
    unzip("gene-expression.zip")
 