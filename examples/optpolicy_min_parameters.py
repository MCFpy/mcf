"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Optimal Policy Trees - Python implementation

Please note that this Optimal Policy (Decision) module is experimental.
It is much less tested than the MCF main module.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.9.0

This is an example to show the optimal policy package can be implemented with
a minimum number of specification (it could be even more further simplified
when the same data is used for solving, evaluating, and predicting
allocations; or when only one solving method is used which makes some more
method-specific parameters redundant).

"""
from pathlib import Path

from mcf.example_data_functions import example_data
from mcf.optpolicy_main import OptimalPolicy
from mcf.reporting import McfOptPolReport


# ------------- NOT passed to OptimalPolicy -----------------------------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example/output'

training_df, prediction_df, name_dict = example_data()

# ------------- Methods used in Optimal Policy Module --------------------------
METHODS = ('best_policy_score', 'policy_tree', 'bps_classifier',)
#  Tuple used to set GEN_METHOD in this example
#  Currently valid methods are: 'best_policy_score', 'policy_tree',
#  'bps_classifier'

# -------- All what follows are parameters of the OptimalPolicy --------
#   Whenever None is specified, parameter will be set to default values.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

VAR_POLSCORE_NAME = ('y_pot0', 'y_pot1', 'y_pot2')
#   Treatment specific variables to measure the value of individual treatments.
#   This is usually the estimated potential outcome or any other score related

VAR_X_NAME_ORD = ('x_cont0',)  # Alternatively specify VAR_X_NAME_UNORD

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

for method in METHODS:
    myoptp = OptimalPolicy(gen_outpath=APPLIC_PATH,
                           gen_method=method,
                           var_polscore_name=VAR_POLSCORE_NAME,
                           var_x_name_ord=VAR_X_NAME_ORD)

    solve_dict = myoptp.solve(training_df, data_title='training')

    myoptp.evaluate(solve_dict['allocation_df'],
                    training_df,
                    data_title='training'
                    )

    alloc_pred_dict = myoptp.allocate(prediction_df, data_title='prediction')

    myoptp.evaluate(alloc_pred_dict['allocation_df'],
                    prediction_df,
                    data_title='prediction'
                    )
    my_report = McfOptPolReport(
        optpol=myoptp, outputfile='Report_OptP_' + method)
    my_report.report()
print('End of example estimation.\n\nThanks for using OptimalPolicy. \n\nYours'
      ' sincerely\nOptimalPolicy MCF module (beta) \U0001F600')
