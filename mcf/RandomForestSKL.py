"""
Random forest estimation based sklearn and additional features.

# -*- coding: utf-8 -*-
Created on Fri Nov  6 15:59:48 2020

@author: MLechner

Prediction intervals are based on:
    Zhang, Zimmerman, Nettleton, and Nordman (2020): Random Forest Prediction
    Intervals, THE AMERICAN STATISTICIAN, 2020, VOL. 74, NO. 4, 392–406:
    Data Science, https://doi.org/10.1080/00031305.2019.1585288
    (Type III intervals given; symmetric intervals used if skewness of
     distribution is below user specified Cut-off-value of skewness)
Variable importance measures are based on:
    Williamson, Gilbert, Carone, Simon (2020): Nonparametric variable
    importance assessment using machine learning techniques, Biometrics.
    (no crossfitting used, no uncertainty measure given)

Beta version, not yet extensively tested.

Version 0.0.11.

"""
import sys
from multiprocessing import freeze_support
# local modules will be loaded below once path is defined -----------

MODULE_PATH = 'd:/mlechner/py/modules'    # path for local general mcf modules
PFAD = 'd:/mlechner/py/applications'  # Path for data, temporary files, output

# ------------ No change below: loading MCF & general modules --------------
sys.path.append(MODULE_PATH)
sys.path.append(PFAD)
import random_forest_functions as rf
# ----------------- No change above --------------


def main():
    """Do the main programme (needed for MP)."""
    outpfad = PFAD + '/output'
    temppfad = PFAD + '/temp'
    datpfad = PFAD + '/data'

    indata = 'unorderedx_sel1_het2clust_n1000_k12_p21'  # csv for estimation
    preddata = 'unorderedx_sel1_het2clust_n2000_k12_p21'  # csv for effects
    outfiletext = indata + "rf.py.0.0.11"

# Define variables in lists; if not relavant use empty list
    # Identifier
    id_name = ['ID']
    # If no identifier -> it will be added the data that is saved for later use
    # Dependent variable

    y_name = ['y']

    # Features, predictors, independent, confounders variables: ordered
    x_name_ord = [
         'X1', 'x2', 'x3', 'x3', 'x4', 'x5', 'x7', 'X8', 'x9', 'x10', 'X11',
         'x12', 'x13',
         'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'X21', 'x22',
         'x24', 'x25', 'x26', 'x27',  # 'x28', 'x29', 'x30',
         'X11', 'x2'
                ]
    # Features, predictors, independent, confounders variables: unordered
    x_name_unord = ['X34', 'X35',  'x36']
    # Unordered variables will be transformed into dummies

    # If None is specified, default values will be used given in brackets

# Set parameters for estimation
    direct_output_to_file = False        # True: output goes to file (T)
    predictions_for_predfile = None     # True: Predictions for preddata (T)
    predictions_for_trainfile = None    # True: Predictions for indata (T)

    save_forest = None                  # True: Save forest  (F)
    # file name will be same as indat with extension *.pickle

    mp_parallel = None                     # number of parallel processes
    # (# of cores*0.9),  0 or 1: no parallel computations

    boot = None                           # number of trees (1000)

    # data cleaning: True, False (T)
    descriptive_stats = None    # print descript. stats of input + output files
    screen_covariates = None    # (1) (-1): screen covariates (sc)
    check_perfectcorr = None    # if sc=1: if 1, then variables that are
    # perfectly correlated with others will be deleted
    min_dummy_obs = None        # if sc=1: dummy variable with obs in one
    # category smaller than this value will be deleted (-1: 10)
    clean_data_flag = None      # if 1, remove all missings and unnecessary
    # variables from data set

    # Forest Building - tuning parameters (determined by OOB MSE)
    n_min_min = None            # smallest minimum leaf size
    # minimum number of observations in leave (n**0.4/10; at least 2)
    n_min_max = None            # largest minimum leaf size
    # maximum number of observations in leave (sqrt(n)/5; at least 5)
    n_min_grid = None           # number of grid values (3)
    # grid for number of variables drawn at each new split of tree
    m_min_share = None          # minimum share of variables, 0<1, (0.1)
    m_max_share = None          # maximum share of variables, 0<1, (0.66)
    m_grid = None               # number of grid values logarithmically
    # spaced including m_min and m_max (3)
    # Alpha regularity
    alpha_reg_min = None        # smallest alpha, 0 < alpha < 0.4 (0)
    alpha_reg_max = None        # largest alpha  0 < alpha < 0.5 (0.1)
    alpha_reg_grid = None       # number of grid values (3)

    # Other forest parameters
    pred_oob_flag = None        # prediction based on OOB in training samples
    max_depth = None            # Max depth of tree (no limit)
    max_leaf_nodes = None       # Max number of leaf nodes (no limit)

    # Variable importance plots
    variable_importance = None
    # 1: computes variable importance based on permuting every single x
    #            in oob prediction; time consuming (True)

    # Prediction uncertainty
    prediction_uncertainty = None  # add prediction uncertainty measure (True)
    pu_ci_level = None            # 0.5 < 1:confidence level for bounds: (0.90)
    pu_skew_sym = None            # > 0: Absolute value of skewness for using
    #                                    symetric CI-s
# ----------------------NO CHANGE BELOW ---------------------------------------
    fontsize = None            # legend, 1 (very small) to 7 (very large); (2)
    dpi = None                 # > 0: (500)
    ci_level = None            # 0 < 1: confidence level for bounds: (0.90)
    with_output = True
# ---------------------------------------------------------------------------
    rf.randomforest_sk(
        PFAD, outpfad, temppfad, datpfad, indata, preddata, outfiletext,
        id_name, y_name, x_name_ord, x_name_unord,
        predictions_for_predfile, predictions_for_trainfile,
        boot, mp_parallel, save_forest,
        direct_output_to_file, with_output, descriptive_stats,
        screen_covariates, check_perfectcorr, clean_data_flag, min_dummy_obs,
        n_min_grid, n_min_min, n_min_max, m_min_share, m_max_share, m_grid,
        alpha_reg_min, alpha_reg_max, alpha_reg_grid,
        pred_oob_flag, max_depth, max_leaf_nodes,
        variable_importance, prediction_uncertainty,
        fontsize, dpi, ci_level, pu_ci_level, pu_skew_sym)


if __name__ == '__main__':
    freeze_support()
    main()
