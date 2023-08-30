"""Created on Wed Apr  1 15:58:30 2020.

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.4.1

-*- coding: utf-8 -*- .

This is an example to show how a minimal specification of the mcf can be
implemented that uses the same data from training and prediction.

"""
import pandas as pd

from mcf.mcf_functions import ModifiedCausalForest
# from mcf import ModifiedCausalForest

# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = 'c:/mlechner/py_neu/example'
DATPATH = APPLIC_PATH + '/data'
TRAINDATA = 'data_y_d_x_1000.csv'
#  Training data must contain outcome, treatment and features.

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'D0_1_12'   # Name of treatment variable
VAR_Y_NAME = 'y'         # Name of outcome variable
VAR_X_NAME_ORD = ('cont0',)  # Using VAR_X_NAME_UNORD is fine as well

# -----------------------------------------------------------------------------
train_df = pd.read_csv(DATPATH + '/' + TRAINDATA)
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD)

tree_df, fill_y_df = mymcf.train(train_df)  # Returns not used here

results = mymcf.predict(fill_y_df)

results_with_cluster_id_df = mymcf.analyse(results)

print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600')
