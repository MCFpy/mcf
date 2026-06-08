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

Version: 0.10.0

This is an example to show the optimal policy package can be implemented accounting for treatment
versions. It features all keywords that are specific to the OptimalPolicyVersion class, but contains
only a very small number of the keyword for the OptimalPolicy class (which is used by the 
OptimalPolicyVersion class under the hat).

"""
from copy import deepcopy
from pathlib import Path

from mcf.example_data import example_data
from mcf.optpolicy_main import OptimalPolicyVersions
from mcf.reporting import McfOptPolReport


#  Output location and name of text file containing output information
APPLIC_PATH = Path.cwd() / 'example/output'
OUTFILETEXT = "Versions.0.10.0"    # File for text output
#   Default is 'txtFileWithOutput'. *.txt file extension will be added by the programme

# ----------------- Generate artifical data and use true potential outcomes as policy scores -------
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

# ------ The following parameters are keywords of the the OptimalPolicyVersion class----------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
POLICYSCORES_DICT = {'y_pot0': None,
                     'y_pot1': ('y_pot1_1', 'y_pot1_2', 'y_pot1_3',),
                     'y_pot2': ('y_pot2_1', 'y_pot2_2', 'y_pot2_3',),
                     }
#   (Ordered) Dictionary: The key is name of the policy score of main treatment, the value is a list
#                         or tuple with the name of the subtreatments

VAR_D_NAME = ('treat', 'treat_version')
#  List List/tuple: 1st element is main treatment, 2nd element is subtreatment.
#  If not None, this information will be used to evaluate the allocations.
#  Default is None.

DEPTH_VERSION_TREE = (2, 2, 2)   # Depth of the tree build for treatment versions
#  for each main treatment. This must either be a list/tuple of integers with length equal of the
#  number of main treatments (and in the same order as the main treatments in policyscores_dict ),
#  or an integer (or None).  Default (or None) is 2.

# The keyword PARAMS_OPTPOL is defined in the next section.

# -------- The following parameters are keywords the OptimalPolicy class ---------------------------
#   Whenever None is specified or keywords are not specified at all, the respective keywords will be
#   set to default values.
#   These keywords are relevant because the OptimalPolicyVersion class uses OptimalPolicy.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
VAR_X_NAME_ORD = ('x_cont0',)

# Parameters for the OptimalPolicy class which is used by the OptimalPolicyVersion class
PARAMS_OPTPOL = {'gen_outpath': APPLIC_PATH,        # Also used for the version results
                 'gen_outfiletext': OUTFILETEXT,    # Also used for the version results
                 'gen_method': None,                # To be overwritten in the loop below
                 'var_polscore_name': None,         # This will overwritten by OptimalPolicyVersion
                 'var_x_name_ord': VAR_X_NAME_ORD,
                 '_int_with_output': True,
                 '_int_report': False,    # Not yet implemented for versions
                 }
# ==================================================================================================
#  Keywords for the OptimalPolicyVersion class used to initialise an instance of the
#  OptimalPolicyVersion class.
params_optpol_version = {'params_optpol': PARAMS_OPTPOL,
                         'policyscores_dict': POLICYSCORES_DICT,
                         'depth_version_tree': DEPTH_VERSION_TREE,
                         'var_d_name': VAR_D_NAME,
                         }
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

for method in METHODS:
    # Update parameters with respect to the method used
    params_optpol_version_loop = deepcopy(params_optpol_version)
    params_optpol_version_loop['params_optpol']['gen_method'] = method

    myoptpv = OptimalPolicyVersions(**params_optpol_version_loop)

    solve_dict, training_df = myoptpv.solve(training_df, data_title='training')
    myoptpv.evaluate(solve_dict['allocation_df'], training_df, data_title='training')

    alloc_pred_dict = myoptpv.allocate(prediction_df, data_title='prediction')
    myoptpv.evaluate(alloc_pred_dict['allocation_df'], prediction_df, data_title='prediction')

    myoptpv.print_time_strings_all_steps()
    my_report = McfOptPolReport(optpol=myoptpv, outputfile='Report_OptP_' + method)
    my_report.report()

print('End of example estimation based on treatment versions.\n\nThanks for using '
      '\nOptimalPolicyVersions. '
      '\n\nYours sincerely\nOptimal Policy MCF module (beta) \U0001F600'
      )
