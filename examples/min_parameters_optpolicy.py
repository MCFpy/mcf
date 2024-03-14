"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Optimal Policy Trees - Python implementation

Please note that this Optimal Policy (Decision) module is experimental.
It is much less tested than the MCF main module.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.5.0

This is an example to show the optimal policy package can be implemented with
a minimum number of specification (it could be even more further simplified
when the same data is used for solving, evaluating, and predicting
allocations; or when only one solving method is used which makes some more
method-specific parameters redundant).

"""
import os

import pandas as pd

from mcf.optpolicy_functions import OptimalPolicy
from mcf.reporting import McfOptPolReport


# ------------- NOT passed to OptimalPolicy -----------------------------------
#  Define data to be used in this example
APPLIC_PATH = os.getcwd() + '/example'
DATPATH = APPLIC_PATH + '/data'
TRAINDATA = 'data_x_ps_1_1000.csv'
#  'best_policy_score': Training data must contain policy scores.
#  'policy tree': Training data must contain policy scores and features.
PREDDATA = 'data_x_ps_2_1000.csv'
#  'best_policy_score': Prediction data must contain policy scores.
#  'policy tree': Prediction data must contain features that are used in
#                 training. Policy scores are not required.

METHODS = ('best_policy_score', 'policy tree',)  # Tuple used to set GEN_METHOD
#  Currently valid methods are: 'best_policy_score', 'policy tree',
#     'policy tree old'

# -------- All what follows are parameters of the ModifiedCausalForest --------
#   Whenever, None is specified, parameter will set to default values.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

VAR_POLSCORE_NAME = ('YLC0_pot', 'YLC1_pot', 'YLC2_Pot', 'YLC3_Pot')
#   Treatment specific variables to measure the value of individual treatments.
#   This is ususally the estimated potential outcome or any other score related

VAR_X_NAME_ORD = ('CONT0',)  # Alternatively specify VAR_X_NAME_UNORD

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

train_df = pd.read_csv(DATPATH + '/' + TRAINDATA)
pred_df = pd.read_csv(DATPATH + '/' + PREDDATA)

for method in METHODS:
    myoptp = OptimalPolicy(gen_method=method,
                           var_polscore_name=VAR_POLSCORE_NAME,
                           var_x_name_ord=VAR_X_NAME_ORD)

    alloc_train_df = myoptp.solve(train_df, data_title=TRAINDATA)

    results_eva_train = myoptp.evaluate(alloc_train_df, train_df,
                                        data_title=TRAINDATA)

    alloc_pred_df = myoptp.allocate(pred_df, data_title=PREDDATA)

    results_eva_pred = myoptp.evaluate(alloc_pred_df, pred_df,
                                       data_title=PREDDATA)
    my_report = McfOptPolReport(
        optpol=myoptp, outputfile='Report_OptP_' + method)
    my_report.report()
print('End of example estimation.\n\nThanks for using OptimalPolicy. \n\nYours'
      ' sincerely\nExperimental OptimalPolicy MCF modul \U0001F600')
