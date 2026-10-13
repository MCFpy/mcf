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

Version: 0.11.0

Learn and evaluate allocations with three policy-learning methods.
Policy scores define treatment codes in their column order. Trees and classifiers
also require decision features; best-policy-score allocation needs scores on new data.

"""
from pathlib import Path

from mcf.example_data import example_data
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
#   Omitted keywords use API defaults; None may select automatic settings.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

VAR_POLSCORE_NAME = ('y_pot0', 'y_pot1', 'y_pot2')
# One policy score per treatment, ordered as treatment codes 0, 1, 2.
# Usually estimated potential outcomes; this simulation supplies true scores.

VAR_X_NAME_ORD = ('x_cont0',)  # Ordered features include binary variables.
# Unordered decision variables need at least three integer-coded categories.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

for method in METHODS:
    myoptp = OptimalPolicy(gen_outpath=APPLIC_PATH,
                           gen_method=method,
                           var_polscore_name=VAR_POLSCORE_NAME,
                           var_x_name_ord=VAR_X_NAME_ORD)

    # Preserve the original data for the next method; evaluate only retained rows.
    solve_dict, training_used_df = myoptp.solve(training_df.copy(), data_title='training')

    myoptp.evaluate(solve_dict['allocation_df'],
                    training_used_df,
                    data_title='training'
                    )

    alloc_pred_dict = myoptp.allocate(prediction_df, data_title='prediction')

    myoptp.evaluate(alloc_pred_dict['allocation_df'],
                    prediction_df,
                    data_title='prediction'
                    )
    my_report = McfOptPolReport(optpol=myoptp,
                                 outputpath=alloc_pred_dict['outpath'],
                                 outputfile='Report_OptP_' + method)
    my_report.report()
print('End of example estimation.\n\nThanks for using OptimalPolicy. \n\nYours'
      ' sincerely\nOptimalPolicy MCF module (beta) \U0001F600')
