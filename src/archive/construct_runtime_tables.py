# import utility modules
import pandas as pd
import numpy as np
import configparser
import os
import time

# helper functions
from helpers.helper_classes import Gene_SPCA

# sklearn
from sklearn.decomposition import PCA, SparsePCA

# joblib
from joblib import dump, load

# Read config.ini file
config = configparser.ConfigParser()
config.read('src/config.ini')
os.chdir(config['PATH']['ROOT_DIR'])

# Read parameters
SEED = config.getint('PARAMS', 'SEED')
N_COMPONENTS = config.getint('PARAMS', 'N_COMPONENTS')

# Load in data
data = load(config['PATH']['DATA_DIR'] + '/microarray-data-dict.lib')
print(data)

#########################################
# Relevant transformations
#########################################

    # TODO: make spca and gene spca a fair comparison by making them use 
    # the same number of non-zero loadings

def get_gene_spca(n_components, random_state):
    return Gene_SPCA(n_comps = n_components, l1= 400)

def get_spca(n_components, random_state):
    return SparsePCA(n_components=n_components, random_state=random_state)

def get_pca(n_components, random_state):
    return PCA(n_components=n_components, random_state=random_state)

#########################################
### Config for runtime tables
#########################################

## Which datasets to run

# Golub because original, Christensen because of small dataset, Chin because of large dataset, Nakayama because of large number of classes
# dataset_list = ['golub', 'christensen', 'chin', 'nakayama']

# Easy running datasets
dataset_list = ['sorlie', 'christensen', 'alon']

## Which transformations to run
transforms_dict = {'pca': get_pca, 'gene_spca': get_gene_spca}#, 'spca': get_spca,}

#########################################
#  Loop to construct table of runtimes
#########################################

# Initialize dictionary to store results
timed_results_dict = {}

# Loop through datasets
for data_name in dataset_list:
    
    X = data[data_name]['none']['X_train']
    timed_results_dict[data_name] = {}

    for transform_name, transform_fn in transforms_dict.items():
        print(f'{data_name} {transform_name}')

        # Instantiate transformer
        transformer_cur = transform_fn(N_COMPONENTS, SEED)

        # Time execution of fitting transformer
        timed_result = %timeit -o transformer_cur.fit(X)
        timed_results_dict[data_name][transform_name] = (timed_result.average, timed_result.stdev)

##################################################################
# Reform created dictionary into right format for dataframe
##################################################################
reform = {}

for dname, res_dict in timed_results_dict.items():
    tnames = []
    reform[(dname, 'avg')] = []
    reform[(dname, 'stdev')] = []
    for tname, res in res_dict.items():
        tnames.append(tname)
        reform[(dname, 'avg')].append(res[0])
        reform[(dname, 'stdev')].append(res[1])

# Create dataframe
res_runtimes = pd.DataFrame.from_dict(reform).T
res_runtimes.columns = tnames

print(res_runtimes.to_latex())
