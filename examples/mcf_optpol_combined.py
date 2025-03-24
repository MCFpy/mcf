"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.7.2

This is an example to show how to combine the ModifiedCausalForest class and
the OptimalPolicy class for joint estimation. Please note that there could be
many alternative ways to split sample, including full cross-validation, that
may be superior to the simple split used here.

"""
from pathlib import Path
import math

import pandas as pd

from mcf.example_data_functions import example_data
from mcf.mcf_functions import ModifiedCausalForest
from mcf.optpolicy_functions import OptimalPolicy
from mcf.reporting import McfOptPolReport


#  In this example we combine mcf estimation and an optimal policy tree in a
#  simple split-sample approach.
#  (i) Split alldata_df into 3 random samples of equal number of rows:
#  train_mcf_df: Train mcf.
#  pred_mcf_train_pt_df: Predict IATEs and train policy score.
#  evaluate_pt_df: Evaluate policy score.

#  (ii) With the splitted data, we either do cross-fitting to obtain the IATEs
#  or predict the IATEs on a different random sample. The predicted IATEs are
#  subsequently used to train the policy learner.
#  The advantage of crossfitting (CROSSFITTING=True) is that the data is
#  efficiently used to train the policy learner, while computation time can be
#  considerable faster without cross-fitting.

# Get data ready, if crossfitting:
#                       - 80% used for training of mcf and policy learner
#                 if no X-fitting:
#                       - 40% used for training mcf
#                       - 40% used for training the policy learner
#                 20% used for evaluating the performance of the policy learner

#  Step 1: Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example2'

# ---------------------- Generate artificial data ------------------------------

# Parameters to generate artificial data (DataFrame) for this example
TRAIN_OBS = 4000        # Number of observations of training data.
#      For 'best_policy_score': Training data must contain policy scores.
#      For 'policy tree': Training data must contain policy scores and features.
#                         Default is 1000.

alldata_df, _, name_dict = example_data(obs_y_d_x_iate=TRAIN_OBS)

# -------------------------- Define variables ----------------------------------
VAR_D_NAME = 'treat'   # Name of treatment variable
VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2',)
VAR_X_NAME_UNORD = ('x_unord0',)

# In this example the policy scores will be outputed from mcf predict method.

# ------------------ Define parameters for policy learning ---------------------
GEN_METHOD = 'policy tree'
PT_DEPTH_TREE_1 = 2  # Too small for real application, for demonstration only
PT_DEPTH_TREE_2 = 2

# --- Crossfitting ---
CROSSFITTING = True    # Boolean, determines if cross-fitting is used.
# ------------------------------------------------------------------------------
# Check if application path exists, if not create it

if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

# Get data ready, if crossfitting:
#                       - 80% used for training of mcf and policy learner
#                 if no X-fitting:
#                       - 40% used for training mcf
#                       - 40% used for training the policy learner
#                 20% used for evaluating the performance of the policy learner

alldata_df_shuffled = alldata_df.sample(frac=1, random_state=42)
num_rows = len(alldata_df_shuffled)

# ------------------------- Random sample splits -------------------------------
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

# -------------------Training the forest and predicting policy scores-----------
mymcf1 = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                              var_y_name=VAR_Y_NAME,
                              var_x_name_ord=VAR_X_NAME_ORD,
                              var_x_name_unord=VAR_X_NAME_UNORD,
                              gen_iate_eff=True,
                              cf_compare_only_to_zero=True,
                              gen_outpath=APPLIC_PATH / 'mcf_train')
# Train the forest
mymcf1.train(train_mcf_df)

# Predict policy scores (IATEs) for data used to train allocation rules
results1, _ = mymcf1.predict(pred_mcf_train_pt_df)

if CROSSFITTING:
    # --------- reverse role of train_mcf_df and pred_mcf_train_pt_df ----------
    mymcf2 = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                                  var_y_name=VAR_Y_NAME,
                                  var_x_name_ord=VAR_X_NAME_ORD,
                                  var_x_name_unord=VAR_X_NAME_UNORD,
                                  gen_iate_eff=True,
                                  cf_compare_only_to_zero=True,
                                  gen_outpath=APPLIC_PATH / 'mcf_train_x_fit')
    # Train the forest
    mymcf2.train(pred_mcf_train_pt_df)

    # Predict policy scores for data used to train allocation rules
    results2, _ = mymcf2.predict(train_mcf_df)
    results = results1.copy()
    # The ATE can be computed as average of the two ATEs
    results['ate'] = (results1['ate'] + results2['ate']) / 2

    # The IATEs and potential outcomes (which are used as policy scores are
    # predicted for both data sets and put together)
    results['iate_data_df'] = pd.concat(
        (results1['iate_data_df'], results2['iate_data_df']), ignore_index=True
        )
else:
    results = results1

mymcf1.analyse(results)

# Predict policy scores for data used to evaluate allocation rules
if CROSSFITTING:
    # Both forests are used to predict the policy score in the evaluation data
    results_oos1, _ = mymcf1.predict(evaluate_pt_df)
    results_oos2, _ = mymcf2.predict(evaluate_pt_df)

    # Provide ID to deal with common support that might slightly differ for the
    # different forests
    id_name = 'id_mcf'  # No ID provided -> use automatically generated
    #                                      (by mcf training) ID

    # Average policy scores for observations in both common supports
    iate_data_df_1 = results_oos1['iate_data_df'].set_index(id_name)
    iate_data_df_2 = results_oos2['iate_data_df'].set_index(id_name)
    iate_data_df_1a, iate_data_df_2a = iate_data_df_1.align(iate_data_df_2,
                                                            join="inner")
    iate_data_df_sum = iate_data_df_1a.add(iate_data_df_2a, fill_value=0)
    iate_data_df = iate_data_df_sum / 2
    iate_data_df = iate_data_df.reset_index()
    results_oos = results_oos1.copy()
    results_oos['ate'] = (results_oos1['ate'] + results_oos2['ate']) / 2
    results_oos['iate_data_df'] = iate_data_df
else:
    results_oos, _ = mymcf1.predict(evaluate_pt_df)

# ------------------- Train and evaluate the policy tree -----------------------
# Get the data for training and evaluation
data_train_pt = results['iate_data_df']
oos_df = results_oos['iate_data_df']

# Define name of policy scores
# It is recommedated either to use uncentered potential outcomes as policy
# scores (IATEs relative to the zero treatment also make sense, but require to
# provide the IATEs for the comparison of treatment zero against itself,
# i.e. a column of zero's.
VAR_POLSCORE_NAME = [VAR_Y_NAME.casefold() + '_lc' + str(i) + '_un_lc_pot_eff'
                     for i in range(3)]
# VAR_POLSCORE_NAME = ('outcome_lc0_un_lc_pot_eff',
#                      'outcome_lc1_un_lc_pot_eff',
#                      'outcome_lc2_un_lc_pot_eff')

# Names of policy score if outcomes are (automatically) centered in mcf if
# efficient IATEs are computed. If not, change accordingly.

myoptp = OptimalPolicy(var_d_name=VAR_D_NAME,
                       var_x_name_ord=VAR_X_NAME_ORD,
                       var_x_name_unord=VAR_X_NAME_UNORD,
                       var_polscore_name=VAR_POLSCORE_NAME,
                       pt_depth_tree_1=PT_DEPTH_TREE_1,
                       pt_depth_tree_2=PT_DEPTH_TREE_2,
                       gen_outpath=APPLIC_PATH / 'OptPolicy',
                       gen_method=GEN_METHOD)

# Learn the policy tree
alloc_train_df, _, _ = myoptp.solve(data_train_pt,
                                    data_title='Training PT data'
                                    )
# Evaluate the learned policy tree on the training data
results_eva_train, _ = myoptp.evaluate(alloc_train_df, data_train_pt,
                                       data_title='Training PT data'
                                       )
# Allocate the treatments according to the learned tree for the evaluation data
alloc_eva_df, _ = myoptp.allocate(oos_df, data_title='')

# Evaluate the learned policy tree on the evaluation data
results_eva_test, _ = myoptp.evaluate(alloc_eva_df, oos_df,
                                      data_title='Evaluate PT data'
                                      )
myoptp.print_time_strings_all_steps()

if CROSSFITTING:
    print('Crossfitting used.')

my_report = McfOptPolReport(mcf=mymcf1, optpol=myoptp,
                            outputfile='Report_mcf_optpolicy'
                            )
my_report.report()


print('End of computations.\n\nThanks for using ModifiedCausalForest and'
      ' OptimalPolicy (beta). \n\nYours sincerely\nMCF \U0001F600')
