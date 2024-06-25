"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.6.0

This is an example to show how to combine the ModifiedCausalForest class and
the OptimalPolicy class for joint estimation. Please note that there could be
many alternative ways to split sample, including full cross-validation, that
may be superior to the simple split used here.

"""
import os

import math

from mcf.example_data_functions import example_data
from mcf.mcf_functions import ModifiedCausalForest
from mcf.optpolicy_functions import OptimalPolicy
from mcf.reporting import McfOptPolReport


#  In this example we combine mcf estimation and an optimal policy tree in a
#  simple split sample approach.
#  Split alldata_df into 3 random samples of equal number of rows:
#  train_mcf_df: Train mcf.
#  pred_mcf_train_pt_df: Predict IATEs and train policy score.
#  evaluate_pt_df: Evaluate policy score.

#  Step 1: Define data to be used in this example
APPLIC_PATH = os.getcwd() + '/example'

# ---------------------- Generate artificial data ------------------------------

# Parameters to generate artificial data (DataFrame) for this example
TRAIN_OBS = 4000        # Number of observations of training data.
#      For 'best_policy_score': Training data must contain policy scores.
#      For 'policy tree': Training data must contain policy scores and features.
#                         Default is 1000.

alldata_df, _, name_dict = example_data(obs_y_d_x_iate=TRAIN_OBS)

# ------------------ Reset some parameters  -------------------
VAR_D_NAME = 'treat'   # Name of treatment variable
VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2',)

# In this example the policy scores will be outputed from mcf predict method.

# --- Parameters --
GEN_IATE_EFF = None
GEN_METHOD = 'policy tree'
PT_DEPTH_TREE_1 = 2  # Too small for real application, for demonstration only
PT_DEPTH_TREE_2 = 2

# -----------------------------------------------------------------------------
if not os.path.exists(APPLIC_PATH):
    os.makedirs(APPLIC_PATH)

# Get data ready

alldata_df_shuffled = alldata_df.sample(frac=1, random_state=42)
num_rows = len(alldata_df_shuffled)

# Data used for training efficient IATEs
rows_per_split_mcf = math.floor(num_rows * 0.4)
# Data used for obtaining allocation rules
rows_per_split_train = math.floor(num_rows * 0.4)
# Data used for evaluating allocation rules
rows_per_split_eval = num_rows - rows_per_split_mcf - rows_per_split_train
train_mcf_df = alldata_df_shuffled.iloc[:rows_per_split_mcf]
pred_mcf_train_pt_df = alldata_df_shuffled.iloc[rows_per_split_mcf:
                                                2*rows_per_split_train]
evaluate_pt_df = alldata_df_shuffled.iloc[num_rows-rows_per_split_eval:]

# Reset indices
train_mcf_df.reset_index(drop=True, inplace=True)
pred_mcf_train_pt_df.reset_index(drop=True, inplace=True)
evaluate_pt_df.reset_index(drop=True, inplace=True)
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             gen_iate_eff=True,
                             cf_compare_only_to_zero=True,
                             gen_outpath=APPLIC_PATH)
# Train the forest
mymcf.train(train_mcf_df)

# Predict policy scores for data used to train allocation rules
results, _ = mymcf.predict(pred_mcf_train_pt_df)
mymcf.analyse(results)

# Predict policy scores for data used to evaluate allocation rules
results_oos, _ = mymcf.predict(evaluate_pt_df)

data_train_pt = results['iate_data_df']
oos_df = results_oos['iate_data_df']

VAR_POLSCORE_NAME = [VAR_Y_NAME.casefold() + '_lc' + str(i) + '_un_lc_pot_eff'
                     for i in range(3)]
# Names of policy score if outcomes are (automatically) centered in mcf if
# efficient IATEs are computed. If not, change accordingly.

# VAR_POLSCORE_NAME = ('outcome_lc0_un_lc_pot_eff',
#                      'outcome_lc1_un_lc_pot_eff',
#                      'outcome_lc2_un_lc_pot_eff')

myoptp = OptimalPolicy(var_d_name=VAR_D_NAME,
                       var_x_name_ord=VAR_X_NAME_ORD,
                       var_polscore_name=VAR_POLSCORE_NAME,
                       pt_depth_tree_1=PT_DEPTH_TREE_1,
                       pt_depth_tree_2=PT_DEPTH_TREE_2,
                       gen_outpath=APPLIC_PATH,
                       gen_method=GEN_METHOD)

alloc_train_df, _, _ = myoptp.solve(data_train_pt,
                                    data_title='Training PT data')
results_eva_train, _ = myoptp.evaluate(alloc_train_df, data_train_pt,
                                       data_title='Training PT data')
alloc_eva_df, _ = myoptp.allocate(oos_df, data_title='')
results_eva_test, _ = myoptp.evaluate(alloc_eva_df, oos_df,
                                      data_title='Evaluate PT data')
myoptp.print_time_strings_all_steps()
my_report = McfOptPolReport(mcf=mymcf, optpol=myoptp,
                            outputfile='Report_mcf_optpolicy')
my_report.report()


print('End of computations.\n\nThanks for using ModifiedCausalForest and'
      ' OptimalPolicy. \n\nYours sincerely\nMCF \U0001F600')
