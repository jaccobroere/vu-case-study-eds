##############################################################################
######## Imports
##############################################################################

# import utility modules
import pandas as pd
import numpy as np
import configparser
import os
import numpy as np
import pandas as pd
import latex
from tqdm import tqdm

# Import helpers
from helpers.helper_classes import AddFeatureNames, GeneSPCA, LoadingsSPCA
from helpers.helper_functions import get_data_pev, get_spca, get_gene_spca
from joblib import dump, load

# Import plotting modules
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee'])

##############################################################################
######## Configuration
##############################################################################

# Read config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
os.chdir(config['PATH']['ROOT_DIR'])
plot_dir = config['LOGGING']['FIG_DIR'] + '/pev_plots'


# Read data
data = load(config['PATH']['DATA_DIR'] + '/microarray-data-dict.lib')
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

# Get list of datasets to run on
pev_dataset_list = ['khan', 'alon', 'gravier']
# pev_dataset_list = ['sorlie', 'christensen'] #'chin', 'nakayama']
# pev_dataset_list = ['christensen']

# Read parameters
SEED = config.getint('PARAMS', 'SEED')
N_COMPONENTS = 5
STEP_ALPHA = 0.000001
STEP_L1 = 0.5
VERBOSE = 1

##############################################################################
######## Run and plot
##############################################################################

logging_dict = {}

if __name__ == "__main__":
    for dname in tqdm(pev_dataset_list):
        X = data[dname]['none']['X_train']
        
        print(dname)
        print('-'*40)
        nz_cols_spca, nz_loadings_spca, PEV_arr_spca = get_data_pev(X, n_components = N_COMPONENTS, verbose = VERBOSE, step_size = STEP_ALPHA, get_transform = get_spca, alpha_arr=[0, 0.000001, 0.00001, 0.0001, 0.005, 0.01, 0.02, 0.05, 0.1, 1, 10])
        nz_cols_gspca, nz_loadings_gspca, PEV_arr_gspca = get_data_pev(X, n_components = N_COMPONENTS, verbose = VERBOSE, step_size = STEP_L1, get_transform = get_gene_spca)

        # Plot PEV versus number of non-zero columns
        plt.figure()
        plt.plot(nz_cols_spca, PEV_arr_spca, label = 'SPCA')
        plt.plot(nz_cols_gspca, PEV_arr_gspca, label = 'Gene-SPCA')
        plt.legend()
        plt.xlabel('Percentage of genes with non-zero loadings')
        plt.ylabel('PEV')
        plt.savefig(plot_dir + '/' + dname + '_nzcols_pev.pdf')

        # Plot PEV versus number of non-zero loadings
        # create new plot
        plt.figure()
        plt.plot(nz_loadings_spca, PEV_arr_spca, label = 'SPCA')
        plt.plot(nz_loadings_gspca, PEV_arr_gspca, label = 'Gene-SPCA')
        plt.legend()
        plt.xlabel('Percentage of non-zero loadings')
        plt.ylabel('PEV')
        plt.savefig(plot_dir + '/' + dname + '_nzloadings_pev.pdf')
        
        
        logging_dict[dname] = {'spca': {'nz_cols': nz_cols_spca, 'nz_loadings': nz_loadings_spca, 'PEV_arr': PEV_arr_spca}, 'gspca': {'nz_cols': nz_cols_gspca, 'nz_loadings': nz_loadings_gspca, 'PEV_arr': PEV_arr_gspca}}
        dump(logging_dict, config['LOGGING']['LOG_DIR'] + '/pev_plots/plot_data_' + dname + '.lib')
    dump(logging_dict, config['LOGGING']['LOG_DIR'] + '/pev_plots/plot_data.lib')