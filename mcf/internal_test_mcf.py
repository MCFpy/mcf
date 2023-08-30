"""Created on Wed Apr  1 15:58:30 2020.

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.4.1

-*- coding: utf-8 -*- .

This code should only be internally be used to test many mcf featurs and
capture warnings in some of the external modules.

"""
import pandas as pd
import warnings

from mcf.mcf_functions import ModifiedCausalForest
# from mcf import ModifiedCausalForest

# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = 'c:/mlechner/py_neu/example'
GEN_OUTPATH = APPLIC_PATH + '/output'
DATPATH = APPLIC_PATH + '/data'
TRAINDATA = 'data_y_d_x_1000.csv'
#  Training data must contain outcome, treatment and features.
PREDDATA = 'data_x_1000.csv'
#  Training data must contain features. Treated effects on the treated
#  additionally require treatment information.

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'D0_1_12'   # Name of treatment variable
VAR_Y_NAME = 'y'         # Name of outcome variable
VAR_X_NAME_ORD = ('cont0', 'ord0')  # Using VAR_X_NAME_UNORD is fine as well
VAR_X_NAME_UNORD = ('cat0',)
VAR_X_BALANCE_NAME_ORD = VAR_X_NAME_ORD
VAR_X_BALANCE_NAME_UNORD = VAR_X_NAME_UNORD
VAR_Z_NAME_LIST = ['cont0']
VAR_Z_NAME_ORD = ['ord0']
VAR_Z_NAME_UNORD = ['cat0']
# -----------------------------------------------------------------------------
train_df = pd.read_csv(DATPATH + '/' + TRAINDATA)
pred_df = pd.read_csv(DATPATH + '/' + PREDDATA)
# -----------------------------------------------------------------------------
all_test = True

with warnings.catch_warnings(record=True) as captured_warnings:
    mymcf = ModifiedCausalForest(
        gen_outpath=GEN_OUTPATH,
        var_d_name=VAR_D_NAME, var_y_name=VAR_Y_NAME,
        var_x_name_ord=VAR_X_NAME_ORD, var_x_name_unord=VAR_X_NAME_UNORD,
        var_x_balance_name_ord=VAR_X_BALANCE_NAME_ORD,
        var_x_balance_name_unord=VAR_X_BALANCE_NAME_UNORD,
        var_z_name_list=VAR_Z_NAME_LIST, var_z_name_unord=VAR_Z_NAME_UNORD,
        var_z_name_ord=VAR_Z_NAME_ORD,
        cf_boot=200,  # Only for testing purposes
        lc_yes=all_test, cs_type=all_test,
        fs_yes=all_test, p_bt_yes=all_test, p_iate_se=all_test,
        gen_iate_eff=all_test,
        p_atet=all_test, p_gatet=all_test, p_amgate=all_test, p_bgate=all_test,
        p_gates_smooth=all_test)
    mymcf.train(train_df)
    results = mymcf.predict(pred_df)
    results_with_cluster_id_df = mymcf.analyse(results)
if captured_warnings:
    for warning in captured_warnings:
        print("Captured Warning:", warning.message)
else:
    print('No Warnings recorded.\n')

print('End of testing.\n\nThanks for testing ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF'
      ' \U0001F600 \U0001F600 \U0001F600 \U0001F600 \U0001F600 \U0001F600')
