"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.4.3

This is an example to show how to combine the ModifiedCausalForest class and
the OptimalPolicy class for joint estimation. Please note that there could be
many alternative ways to split sample, including full cross-validation, that
may be superior to the simple split used here.

"""
import os

import pandas as pd

from mcf.mcf_functions import ModifiedCausalForest
from mcf.optpolicy_functions import OptimalPolicy

# from mcf import ModifiedCausalForest
# from mcf import OptimalPolicy

#  In this example we combine mcf estimation and an optimal policy tree in a
#  simple split sample approach.
#  Split alldata_df into 3 random samples of equal number of rows:
#  train_mcf_df: Train mcf.
#  pred_mcf_train_pt_df: Predict IATEs and train policy score.
#  evaluate_pt_df: Evaluate policy score.

#  Step 1: Define data to be used in this example
APPLIC_PATH = os.getcwd() + '/example'
DATPATH = APPLIC_PATH + '/data'
ALLDATA = 'data_y_d_x_4000.csv'
#  Training data must contain outcome, treatment and features.

# ------------------ Reset some parameters  -------------------
VAR_D_NAME = 'D0_1_12'   # Name of treatment variable
VAR_Y_NAME = 'y'         # Name of outcome variable
VAR_X_NAME_ORD = ('cont0', 'cont1', 'cont2',)

VAR_POLSCORE_NAME = ('Y_LC0_un_lc_pot_eff',
                     'Y_LC1_un_lc_pot_eff',
                     'Y_LC2_un_lc_pot_eff')
# --- Parameters --
GEN_IATE_EFF = None
GEN_METHOD = 'policy tree'
PT_DEPTH = 2  # Too small for real application, for demonstration only

# -----------------------------------------------------------------------------
alldata_df = pd.read_csv(DATPATH + '/' + ALLDATA)
alldata_df_shuffled = alldata_df.sample(frac=1, random_state=42)
num_rows = len(alldata_df_shuffled)
rows_per_split = num_rows // 3
train_mcf_df = alldata_df_shuffled.iloc[:rows_per_split]
pred_mcf_train_pt_df = alldata_df_shuffled.iloc[rows_per_split:
                                                2*rows_per_split]
evaluate_pt_df = alldata_df_shuffled.iloc[2*rows_per_split:]
train_mcf_df.reset_index(drop=True, inplace=True)
pred_mcf_train_pt_df.reset_index(drop=True, inplace=True)
evaluate_pt_df.reset_index(drop=True, inplace=True)
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             gen_iate_eff=True,
                             gen_outpath=APPLIC_PATH)
mymcf.train(train_mcf_df)
results = mymcf.predict(pred_mcf_train_pt_df)
mymcf.analyse(results)

results_oos = mymcf.predict(evaluate_pt_df)

data_train_pt = results['iate_data_df']
oos_df = results_oos['iate_data_df']

myoptp = OptimalPolicy(var_d_name=VAR_D_NAME,
                       var_x_ord_name=VAR_X_NAME_ORD,
                       var_polscore_name=VAR_POLSCORE_NAME,
                       pt_depth=PT_DEPTH,
                       gen_outpath=APPLIC_PATH,
                       gen_method=GEN_METHOD)

alloc_train_df = myoptp.solve(data_train_pt, data_title='Training PT data')
results_eva_train = myoptp.evaluate(alloc_train_df, data_train_pt,
                                    data_title='Training PT data')
alloc_eva_df = myoptp.allocate(oos_df, data_title='')
results_eva_train = myoptp.evaluate(alloc_eva_df, oos_df,
                                    data_title='Evaluate PT data')
myoptp.print_time_strings_all_steps()


print('End of computations.\n\nThanks for using ModifiedCausalForest and'
      ' OptimalPolicy. \n\nYours sincerely\nMCF \U0001F600')
