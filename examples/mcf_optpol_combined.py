"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

This is an example to show how to combine the ModifiedCausalForest class and
the OptimalPolicy class for joint estimation. Please note that there could be
many alternative ways to split sample, including full cross-validation, that
may be superior to the simple split used here.

"""
from pathlib import Path
import math

import pandas as pd

from mcf.example_data import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.optpolicy_main import OptimalPolicy
from mcf.reporting import McfOptPolReport


#  This example learns a policy tree from MCF estimates of potential outcomes.
#  Randomly split the observations into three samples:
#  - train_mcf_df: approximately 40% for training the first forest.
#  - pred_mcf_train_pt_df: approximately 40% for learning the policy tree.
#  - evaluate_pt_df: the remaining approximately 20% for evaluating the policy tree.
#
#  With CROSSFITTING=False, the first forest predicts policy scores for the second sample.
#  With CROSSFITTING=True, train a second forest on pred_mcf_train_pt_df and predict scores
#  for train_mcf_df. Combine both sets of predictions to train the policy tree on about 80%
#  of the observations, subject to common-support restrictions.
#  Each policy-training observation receives scores from a forest trained on the other sample.
#
#  The evaluation sample is used neither to train the forests nor to learn the policy tree.
#  With cross-fitting, average the two forests' policy scores for evaluation observations
#  retained on both common supports. Otherwise, use predictions from the first forest.
#  Policy scores are uncentered potential outcomes, obtained from the prediction metadata.

#  Step 1: Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example2'

# ---------------------- Generate artificial data ------------------------------

# Parameters to generate artificial data (DataFrame) for this example
TRAIN_OBS = 4000        # Number of observations of training data.
# These observations are split into forest, policy-learning and evaluation samples.

alldata_df, _, name_dict = example_data(obs_y_d_x_iate=TRAIN_OBS)

# -------------------------- Define variables ----------------------------------
VAR_D_NAME = 'treat'   # Name of treatment variable
VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2',)
VAR_X_NAME_UNORD = ('x_unord0',)

# Policy scores are uncentred potential outcomes returned by predict().

# ------------------ Define parameters for policy learning ---------------------
GEN_METHOD = 'policy_tree'
PT_DEPTH_TREE_1 = 2  # Maximum split depth of the first tree.
PT_DEPTH_TREE_2 = 2  # Maximum additional split depth within each first-stage leaf.
# The two stages are optimized sequentially; the combined tree has at most 16 leaves.

# --- Crossfitting ---
CROSSFITTING = True   # Boolean, determines if cross-fitting is used.
# ------------------------------------------------------------------------------
# Check if application path exists, if not create it

APPLIC_PATH.mkdir(parents=True, exist_ok=True)


# Get data ready, if crossfitting:
#                       - 80% used for training of mcf and policy learner
#                 if no X-fitting:
#                       - 40% used for training mcf
#                       - 40% used for training the policy learner
#                 20% used for evaluating the performance of the policy learner

alldata_df_shuffled = alldata_df.sample(frac=1, random_state=42)
num_rows = len(alldata_df_shuffled)

# ------------------------- Random sample splits -------------------------------
# Data used to train the first forest
rows_per_split_mcf = math.floor(num_rows * 0.4)
# Data used for obtaining allocation rules
rows_per_split_train = math.floor(num_rows * 0.4)
# Data used for evaluating allocation rules
rows_per_split_eval = num_rows - rows_per_split_mcf - rows_per_split_train
train_mcf_df = alldata_df_shuffled.iloc[:rows_per_split_mcf].copy()
pred_mcf_train_pt_df = alldata_df_shuffled.iloc[
    rows_per_split_mcf:rows_per_split_mcf + rows_per_split_train].copy()
evaluate_pt_df = alldata_df_shuffled.iloc[num_rows-rows_per_split_eval:].copy()

# Reset indices
train_mcf_df.reset_index(drop=True, inplace=True)
pred_mcf_train_pt_df.reset_index(drop=True, inplace=True)
evaluate_pt_df.reset_index(drop=True, inplace=True)

# -------------------Training the forest and predicting policy scores-----------
mymcf1 = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                              var_y_name=VAR_Y_NAME,
                              var_x_name_ord=VAR_X_NAME_ORD,
                              var_x_name_unord=VAR_X_NAME_UNORD,
                              gen_ate_eff=True,
                              gen_iate_eff=True,
                              cf_compare_only_to_zero=True,
                              gen_outpath=APPLIC_PATH / 'mcf_train')
# Train the forest
mymcf1.train(train_mcf_df)

# Predict potential-outcome scores for data used to learn allocation rules.
results1 = mymcf1.predict(pred_mcf_train_pt_df)

if CROSSFITTING:
    # --------- reverse role of train_mcf_df and pred_mcf_train_pt_df ----------
    mymcf2 = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                                  var_y_name=VAR_Y_NAME,
                                  var_x_name_ord=VAR_X_NAME_ORD,
                                  var_x_name_unord=VAR_X_NAME_UNORD,
                                  gen_ate_eff=True,
                                  gen_iate_eff=True,
                                  cf_compare_only_to_zero=True,
                                  gen_outpath=APPLIC_PATH / 'mcf_train_x_fit'
                                  )
    # Train the forest
    mymcf2.train(pred_mcf_train_pt_df)

    # Predict policy scores for data used to train allocation rules
    results2 = mymcf2.predict(train_mcf_df)
    # Combine out-of-sample predictions for training the policy learner.
    data_train_pt = pd.concat((results1['iate_data_df'], results2['iate_data_df']),
                               ignore_index=True
                               )
else:
    data_train_pt = results1['iate_data_df'].copy()

# Analyse each forest using its own estimates and standard errors.
mymcf1.analyse(results1)
if CROSSFITTING:
    mymcf2.analyse(results2)

# Use uncentred potential outcomes, in treatment order, as policy scores.
# With default local centering, these columns are supplied for the tree outcome.
# This example has one outcome; keep this selection explicit for multi-outcome models.
VAR_POLSCORE_NAME = results1['iate_names_dic'][0]['names_y_pot_uncenter'].copy()
ID_NAME = name_dict['id_name']

# Predict policy scores for data used to evaluate allocation rules.
results_oos1 = mymcf1.predict(evaluate_pt_df)
scores_1_df = results_oos1['iate_data_df'].set_index(ID_NAME, verify_integrity=True)
scores_1_df = scores_1_df[VAR_POLSCORE_NAME]

if CROSSFITTING:
    results_oos2 = mymcf2.predict(evaluate_pt_df)
    scores_2_df = results_oos2['iate_data_df'].set_index(ID_NAME, verify_integrity=True)
    scores_2_df = scores_2_df[VAR_POLSCORE_NAME]
    common_ids = scores_1_df.index.intersection(scores_2_df.index, sort=False)
    if common_ids.empty:
        raise ValueError('No evaluation observations are on common support for both forests.')
    scores_oos_df = (scores_1_df.loc[common_ids] + scores_2_df.loc[common_ids]) / 2

    # Separate diagnostic: average ATEs over the samples retained by each forest.
    # The policy-score average above uses their common retained observations.
    ate_combined_oos = (results_oos1['ate'] + results_oos2['ate']) / 2
else:
    scores_oos_df = scores_1_df
    ate_combined_oos = results_oos1['ate'].copy()

# Attach the scores to the original observations retained after prediction.
oos_df = evaluate_pt_df.set_index(ID_NAME, verify_integrity=True).loc[scores_oos_df.index].copy()
oos_df[VAR_POLSCORE_NAME] = scores_oos_df
oos_df.reset_index(inplace=True)

# ------------------- Train and evaluate the policy tree -----------------------

myoptp = OptimalPolicy(var_d_name=VAR_D_NAME,
                       var_x_name_ord=VAR_X_NAME_ORD,
                       var_x_name_unord=VAR_X_NAME_UNORD,
                       var_polscore_name=VAR_POLSCORE_NAME,
                       pt_depth_tree_1=PT_DEPTH_TREE_1,
                       pt_depth_tree_2=PT_DEPTH_TREE_2,
                       gen_outpath=APPLIC_PATH / 'OptPolicy',
                       gen_method=GEN_METHOD
                       )

# solve() returns allocations and the retained training rows in matching order.
solve_dict, data_train_pt = myoptp.solve(data_train_pt, data_title='Training PT data')
# Evaluate the learned policy tree on the training data
myoptp.evaluate(solve_dict['allocation_df'], data_train_pt, data_title='Training PT data')
# Allocate the treatments according to the learned tree for the evaluation data
results_alloc = myoptp.allocate(oos_df, data_title='')

# Evaluate the learned policy tree on the evaluation data
myoptp.evaluate(results_alloc['allocation_df'], oos_df, data_title='Evaluate PT data')
myoptp.print_time_strings_all_steps()

# Allocation inference uses the first forest. Rows in both inputs must match by position.

mymcf1.predict_different_allocations(oos_df, results_alloc['allocation_df'])

if CROSSFITTING:
    print('Crossfitting used.')

my_report = McfOptPolReport(mcf=mymcf1, optpol=myoptp, outputpath=APPLIC_PATH,
                             outputfile='Report_mcf_optpolicy')
my_report.report()


print('End of computations.\n\nThanks for using ModifiedCausalForest and'
      ' OptimalPolicy (beta). \n\nYours sincerely\nMCF \U0001F600'
      )
