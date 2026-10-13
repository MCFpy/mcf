"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Optimal Policy Trees - Python implementation

Please note that this Optimal Policy (Decision) module is experimental.
It is less tested than the MCF main module.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

Learn allocations when main treatments contain versions. Policy trees first choose
a main treatment and then its version; the other methods use flattened version scores.
This example covers the version-specific keywords and a small base configuration.

"""
from copy import deepcopy
from pathlib import Path

from mcf.example_data import example_data
from mcf.optpolicy_main import OptimalPolicyVersions
from mcf.reporting import McfOptPolReport


#  Output location and name of text file containing output information
APPLIC_PATH = Path.cwd() / 'example/output'
OUTFILETEXT = 'Versions.0.11.0'    # File for text output
#   Default is 'txtFileWithOutput'. *.txt file extension will be added by the programme

# ---------------- Generate artificial data and use true potential outcomes as policy scores -------
OBS_TRAIN = 1000
OBS_PRED = 1000
D_NO_VERSIONS = 3

training_df, prediction_df, name_dict = example_data(obs_y_d_x_iate=OBS_TRAIN,
                                                     obs_x_iate=OBS_PRED,
                                                     d_no_versions=D_NO_VERSIONS,
                                                     )

# ------------- Methods used in Optimal Policy Module ----------------------------------------------
METHODS = ('policy_tree',
           'best_policy_score',
           'bps_classifier',
           )
#  Tuple used to set GEN_METHOD in this example
#  Currently valid methods are: 'best_policy_score', 'policy_tree', 'bps_classifier'

# ------ The following parameters are keywords of the the OptimalPolicyVersions class---------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
POLICYSCORES_DICT = {'y_pot0': None,
                     'y_pot1': tuple(name for name in name_dict['y_pot_name']
                                     if name.startswith('y_pot1_')),
                     'y_pot2': tuple(name for name in name_dict['y_pot_name']
                                     if name.startswith('y_pot2_')),
                     }
# Keys follow main-treatment order; values list version scores in version order.
# None uses the main-score key as its only version score. Score names must be unique.
# Returned allocations use consecutive codes across all main-treatment/version pairs.

VAR_D_NAME = ('treat', 'treat_version')
# Observed main treatment and version, used in evaluation when available.
# Main codes start at 0 in dictionary order; version codes restart at 0 within
# each main treatment. A main treatment with only one version uses version code 0.

DEPTH_VERSION_TREE = (2, 2, 2)
# Maximum split depth for each version tree (policy_tree only). Default 2.
# Use one value for all main treatments, or one per main treatment in dictionary order.
# Values must be from 2 to 10; each version tree has no second tree stage.

# The keyword PARAMS_OPTPOL is defined in the next section.

# -------- The following parameters are keywords the OptimalPolicy class ---------------------------
# Unspecified OptimalPolicy keywords use their defaults. The wrapper supplies score names.
# Costs refer to main treatments. Share limits/cost multipliers refer to main treatments
# for policy trees, and to flattened versions for the other methods.
VAR_X_NAME_ORD = ('x_cont0',)

# Parameters for the OptimalPolicy class which is used by the OptimalPolicyVersions class
PARAMS_OPTPOL = {'gen_outpath': APPLIC_PATH,        # Also used for the version results
                 'gen_outfiletext': OUTFILETEXT,    # Also used for the version results
                 'gen_method': None,                # To be overwritten in the loop below
                 'var_polscore_name': None,         # This will overwritten by OptimalPolicyVersions
                 'var_x_name_ord': VAR_X_NAME_ORD,
                 '_int_with_output': True,
                 }
# ==================================================================================================
#  Keywords for the OptimalPolicyVersions class used to initialise an instance of the
#  OptimalPolicyVersions class.
params_optpol_version = {'params_optpol': PARAMS_OPTPOL,
                         'policyscores_dict': POLICYSCORES_DICT,
                         'depth_version_tree': DEPTH_VERSION_TREE,
                         'var_d_name': VAR_D_NAME,
                         }
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

# example_data() codes treated versions from 1; OptimalPolicyVersions expects codes from 0.
for data_df in (training_df, prediction_df):
    treated = data_df['treat'] > 0
    data_df.loc[treated, 'treat_version'] -= 1

for method in METHODS:
    # Update parameters with respect to the method used
    params_optpol_version_loop = deepcopy(params_optpol_version)
    params_optpol_version_loop['params_optpol']['gen_method'] = method

    myoptpv = OptimalPolicyVersions(**params_optpol_version_loop)

    # solve() must run first for every method, including best_policy_score.
    solve_dict, training_used_df = myoptpv.solve(training_df.copy(), data_title='training')
    evaluation_train = myoptpv.evaluate(solve_dict['allocation_df'], training_used_df,
                                         data_title='training')

    alloc_pred_dict = myoptpv.allocate(prediction_df, data_title='prediction')
    evaluation_pred = myoptpv.evaluate(alloc_pred_dict['allocation_df'], prediction_df,
                                        data_title='prediction')

    myoptpv.print_time_strings_all_steps()
    # The PDF describes fitted rules. Final version evaluations remain in the dictionaries
    # above and in the separate evaluation text output.
    my_report = McfOptPolReport(optpol=myoptpv, outputpath=evaluation_pred['outpath'],
                                 outputfile='Report_OptP_' + method)
    my_report.report()

print('End of example estimation based on treatment versions.\n\nThanks for using '
      '\nOptimalPolicyVersions. '
      '\n\nYours sincerely\nOptimal Policy MCF module (beta) \U0001F600'
      )
