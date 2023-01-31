################################################################################
##### 1.0 IMPORT MODULES
################################################################################
# import utility modules
import pandas as pd
import numpy as np
import configparser
import os
import time

# helper functions
from helpers.helper_classes import Gene_SPCA, EnetSPCA
from helpers.helper_functions import get_regularisation_value

# sklearn
from sklearn.decomposition import PCA

# joblib
from joblib import dump, load

# Read config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
os.chdir(config['PATH']['ROOT_DIR'])

# Read parameters
SEED = config.getint('PARAMS', 'SEED')
# N_COMPONENTS = config.getint('PARAMS', 'N_COMPONENTS')

# Load in data
data = load(config['PATH']['DATA_DIR'] + '/microarray-data-dict.lib')

# Relevant transformations
def get_gene_spca(n_components, random_state, alpha = 10):
    return Gene_SPCA(n_comps = n_components, l1= alpha, tol = 0.001)

def get_spca(n_components, random_state, alpha = 0.001):
    return EnetSPCA(n_comps=n_components, alpha = alpha, tol = 0.001)

def get_pca(n_components, random_state):
    return PCA(n_components=n_components, random_state=random_state)

################################################################################
##### 2.0 Config of plotting script
################################################################################

# Set datasets
datasets = ['chin', 'chowdary', 'gravier', 'west']
N_TIMINGS = 3
N_JOBS = -1
n_components_list = [5]
transforms_dict = {'pca': get_pca, 'spca': get_spca, 'gene_spca': get_gene_spca}


################################################################################
##### 3.0 Obtain results
################################################################################

if __name__ == "__main__":
    results_dict = {}
    lambda_dict = {}
    for n_components in n_components_list:
        for dname in datasets:
            print('-' * 40)
            print(f"Dataset: {dname}, n_components: {n_components}")
            
            X_cur = data[dname]['none']['X_train']
                
            spca_transform = get_spca(n_components = n_components, random_state = SEED, alpha = 0.01)
            spca_fitted = spca_transform.fit(X_cur, n_jobs = N_JOBS)
            spca_nzero_percentage = spca_fitted.nonzero / spca_fitted.totloadings
            print(f"non zero % target: {spca_nzero_percentage}")

            # Find lambda value such that gene_spca has same percentage of nonzero loadings as spca.
            lambda_genespca = get_regularisation_value(X_cur, n_components, spca_nzero_percentage, get_gene_spca, lower_bound = 0, upper_bound = X_cur.shape[1] * 4, verbose = 1)                 
            lambda_dict[(dname, n_components)] = lambda_genespca

            # Time pca
            print(f"Timing pca...")
            results_dict[(dname, 'pca', n_components)] = []
            for i in range(N_TIMINGS):
                cur_pca = get_pca(n_components = n_components, random_state = SEED)
                start = time.time()
                cur_pca.fit(X_cur)
                end = time.time()
                results_dict[(dname, 'pca', n_components)].append(end - start)

            # Time spca
            print(f"Timing spca...")
            results_dict[(dname, 'spca', n_components)] = []
            for i in range(N_TIMINGS):
                cur_spca = get_spca(n_components = n_components, random_state = SEED, alpha = 0.01)
                start = time.time()
                cur_spca.fit(X_cur, n_jobs = N_JOBS)
                end = time.time()
                results_dict[(dname, 'spca', n_components)].append(end - start)

            # Time gene spca
            print(f"Timing gene spca...")
            results_dict[(dname, 'gene_spca', n_components)] = []
            for i in range(N_TIMINGS):
                cur_genespca = get_gene_spca(n_components = n_components, random_state = SEED, alpha = lambda_genespca)
                start = time.time()
                cur_genespca.fit(X_cur)
                end = time.time()
                results_dict[(dname, 'gene_spca', n_components)].append(end - start)

                    
    ################################################################################
    ##### 4.0 Save results to table
    ################################################################################
                
    reform = {}
    for n_components in n_components_list:
        for dname in datasets:
            reform[(dname, 'avg')] = []
            reform[(dname, 'stdev')] = []
            for tname in transforms_dict.keys():
                res_arr = results_dict[(dname, tname, n_components)]
                reform[(dname, 'avg')].append(np.mean(res_arr))
                reform[(dname, 'stdev')].append(np.std(res_arr))

        # Create dataframe
        res_runtimes = pd.DataFrame.from_dict(reform).T
        res_runtimes.columns = transforms_dict.keys()

        # Save to file
        fname = config['LOGGING']['TIME_DIR'] + f"/runtime_table_{n_components}.txt"

        # If exists delete
        if os.path.exists(fname):
            os.remove(fname)

        # Write table to file
        with open(fname, 'a') as f:
            f.write(res_runtimes.to_latex(caption = f"Runtime for {dname} data, {n_components} components", label = f"tab:runtime_{dname}"))
        print(res_runtimes)

    with open(fname, 'a') as f:
        f.write(f"Lambda values: {lambda_dict}")
    dump(results_dict, config['LOGGING']['TIME_DIR'] + f"/runtime_dict.joblib")
