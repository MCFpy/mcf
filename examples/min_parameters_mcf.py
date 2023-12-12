"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.4.3

This is an example to show how the mcf can be implemented relying completely on
defaults. Note that usually in application it is very likely to be appropriate
to deviate from some of the default specifications.

"""
import os

import pandas as pd

from mcf.mcf_functions import ModifiedCausalForest
# from mcf import ModifiedCausalForest

# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = os.getcwd() + '/example'
DATPATH = APPLIC_PATH + '/data'
TRAINDATA = 'data_y_d_x_1000.csv'
#  Training data must contain outcome, treatment and features.

PREDDATA = 'data_x_1000.csv'
#  Training data must contain features. Treated effects on the treated
#  additionally require treatment information.

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'D0_1_12'   # Name of treatment variable
VAR_Y_NAME = 'y'         # Name of outcome variable
VAR_X_NAME_ORD = ('cont0',)  # Using VAR_X_NAME_UNORD is fine as well

# -----------------------------------------------------------------------------
train_df = pd.read_csv(DATPATH + '/' + TRAINDATA)
pred_df = pd.read_csv(DATPATH + '/' + PREDDATA)
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD)

mymcf.train(train_df)  # Returns not used here

results = mymcf.predict(pred_df)

results_with_cluster_id_df = mymcf.analyse(results)

print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600')
