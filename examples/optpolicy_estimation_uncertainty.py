"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Optimal Policy - Python implementation

Please note that this Optimal Policy (Decision) module is experimental.
It is much less tested than the MCF main module.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.9.0

This is an example showing how to use the adjustments for policy score
uncertainty in the mcf optimal policy module.

"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mcf.example_data_functions import example_data
from mcf.optpolicy_main import OptimalPolicy
from mcf.reporting import McfOptPolReport

# ---- This script is experimental !
#      Be careful, it will run for a long time!


# ------------- NOT passed to OptimalPolicy -----------------------------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example'

# ---------------------- Generate artificial data ------------------------------

# Parameters to generate artificial data (DataFrame) for this example
TRAIN_OBS = 2000        # Number of observations of training data.
#      'best_policy_score': Training data must contain policy scores.
#      'policy_tree', 'bps_classifier': Training data must contain policy scores
#                         and features. Default is 1000.

PRED_OBS = 2000         # Number of observations of prediction data.
#                         Prediction data is used to allocate.
#      'best_policy_score': Prediction data must contain policy scores.
#      'policy_tree', 'bps_classifier': Prediction data must contain the
#                         features that are used in training. Policy scores are
#                         not required. Default is 1000.

# If True, scores are adjusted for fairness
# considerations a là Bearth, Lechner, Mareckova, and Muny (2024). Requires
# the specification of some additional keywords below
# (var_protected_xxx, fair_xxx).

PREDDATA_IS_TRAINDATA = False  # If True, the same data will be used
#                                for training and prediction (in this case,
#                                PRED_OBS is ignored).

NO_FEATURES = 20         # Number of features. Will generate different types of
#                          features (continuous, dummies, ordered, unordered)

NO_TREATMENTS = 3        # Number of treatments

training_df, prediction_df, name_dict = example_data(
    obs_y_d_x_iate=TRAIN_OBS,
    obs_x_iate=PRED_OBS,
    no_features=NO_FEATURES,
    no_treatments=NO_TREATMENTS,
    seed=12345,
    type_of_heterogeneity='WagerAthey',
    descr_stats=True)

if PREDDATA_IS_TRAINDATA:
    prediction_df = training_df

# ------------- Methods used in Optimal Policy Module --------------------------
# METHODS = ('best_policy_score', 'bps_classifier', 'policy_tree',)
# Using 'best_policy_score' requires scores also for the prediction data!
# Therefore, training data is used for the evaluation of 'best_policy_score'.
GEN_METHOD_ALL = ('policy_tree', 'bps_classifier',)

# -------- All what follows are parameters of the OptimalPolicy --------
#   Whenever None is specified, parameter will be set to default values.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
GEN_OUTPATH = Path.cwd() / 'example/outputOPT'  # Directory for output.
#   If it does not exist, it will be created. Default is an *.out directory
#   just below to the directory where the programme is run.

WEIGHTPLOT_PATH = Path.cwd() / 'out'
#   Path to put the plots for the different fairness weighting

GEN_OUTFILETEXT = "OptPolicy_0_8_0_Fairness"  # File for text output
#   Default is 'txtFileWithOutput'.
#   *.txt file extension will be added by the programme

# ---------------- Names of variables used ------------------------------------
VAR_ID_NAME = 'id'     # Name of identifier in data. Default is None.

VAR_D_NAME = 'treat'   # Name of discrete treatment.
#   Needed in training data only if 'changers' (different treatment than
#   observed treatment) are analysed and if allocation is compared to observed
#   allocation.

VAR_POLSCORE_NAME = ('y_pot0', 'y_pot1', 'y_pot2',)
#   Treatment specific variables to measure the value of individual treatments,
#   starting with the first, second, third, ..., treatment (e.g. corresponding
#   to treatment values 0, 1, 2, ...).
#   This is usually the estimated potential outcome or any other score related
#   quantity. If Individualized Average Treatment Effects (e.g., of the effect
#   1-0, 2-0, 3-0, etc. are provided instead, make sure that the reference
#   treatment (in this case treatment 0) is included as a vector of zeros
#   (in this example it corresponds to the IATE of 0-0, which is of course 0
#   for all observations).

VAR_POLSCORE_SE_NAME = ('y_pot0_se', 'y_pot1_se', 'y_pot2_se',)
#   Standard errors of individual policy scores (same order as
#   VAR_POLSCORE_NAME). Default is None.

VAR_POLSCORE_DESC_NAME = (('zero', 'ite1vs0', 'ite2vs0',),
                          (('x_cont0', 'iate1vs0', 'iate2vs0',)))
#   Tuple of tuples. Each tuple of dimension equal to the number of treatments
#   contains treatment specific variables that are used to evaluate the effect
#   of the allocation with respect to those variables. This could be for
#   example policy score not used in training, but which are relevant
#   nevertheless. Default is no variables.

VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2', 'x_ord0', 'x_ord1', 'x_ord2')
#   Ordered variables used to build policy tree (including dummy variables) and
#   classifier. They are also used to characterise the allocation.

VAR_X_NAME_UNORD = ('x_unord0', 'x_unord1', 'x_unord2',)
#   Unordered variables used to build policy tree and classifier.They are also
#    used to characterise the allocation.

VAR_VI_X_NAME = ('x_cont0', 'x_cont1', 'x_cont2',)
#   Names of variables for which variable importance is computed.
#   Default is None.
VAR_VI_TO_DUMMY_NAME = ('x_unord0',)
#   Names of categorical variables for which variable importance is computed.
#   These variables will be broken up into dummies. Default is None.

# ----------------- Allocations adjusted for estimation risk -------------------
# Two methods are explicitly designed to account for estimation risk.
# Generally, the ideas implemented follow the paper by Chernozhukov, Lee, Rosen,
# and Sun (2025), Policy Learning With Confidence, arXiv. However, since several
# approximations are used in the algorithm, the methods will have the direct
# confidence-level-related interpretations suggested by these authors.
#
# To be applicable, it is necessary to provide the standard errors of the
# individualized average treatment effects relative to alternative zero,
# 'VAR_EFFECT_VS_0_SE'.
#
# The first method 'adjust_risk' adjust the policy scores for estimation risk.
# Once the scores are adjusted, standard procedures can be used. The adjustment
# is based on subtracting multiples, k, of the standard error from the
# policy score (or normalized score, ie., the IATE).
#
# The second method 'estimation_error_trade_off' is useful to show the trade-off
# between the expected welfare and the associated estimation risk when
# of allocations based on several values of k.
#
# ESTRISK_VALUE = None
# The is k in the formula 'policy_score - k * standard_error' used to adjust
# the scores for estimation risk. Default (or None) is 1.
#

ESTRISK_VALUE_TUPLE = (0, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,)
#  Values of ESTRISK_VALUE to be used below for comparing different
#  allocations.

# ---------------------- Tree depth of policy tree -----------------------------

PT_DEPTH_TREE_1 = 2   # Depth of 1st optimal tree. Default is 3.
#   In this example, this parameter is set to 2 to speed up computation.

PT_DEPTH_TREE_2 = 2   # Depth of 2nd optimal tree. This set is built within
#   the strata obtained from the leaves of the first tree. If set to 0, a second
#   tree is not built. Default is 1 (together with the default for
#   pt_depth_tree_1 this leads to a (not optimal) total tree of level of 4.
#   In this example, this parameter is set to 2.

#   Note that tree depth is defined such that a depth of 1 implies 2 leaves,
#   a depth of 2 implies 4 leaves, a depth of = 3 implies 8 leaves, etc.

# -------------- Other parameters (costs and restrictions) --------------------
OTHER_MAX_SHARES = (1, 1, 1)  # Maximum share allowed for each treatment.
#   This is a tuple with as many elements as treatments.
#   0 <  OTHER_MAX_SHARES <= 1.  (1,...,1) implies unconstrained optimization
#   (default).

# The following is an example on how the fairness adjustment of the
# OptimalPolicy object can be used.
# For convenience the parameters are collected and passed as a dictionary.
# Of course, they can also be passed as single parameters (or not at all, in
# which case default values are used).
params_org = {
    'gen_outfiletext': GEN_OUTFILETEXT,
    'gen_outpath': GEN_OUTPATH,
    'other_max_shares': OTHER_MAX_SHARES,
    'pt_depth_tree_1': PT_DEPTH_TREE_1,
    'pt_depth_tree_2': PT_DEPTH_TREE_2,
    'var_d_name': VAR_D_NAME,
    'var_polscore_desc_name': VAR_POLSCORE_DESC_NAME,
    'var_polscore_name': VAR_POLSCORE_NAME,
    'var_vi_x_name': VAR_VI_X_NAME,
    'var_vi_to_dummy_name': VAR_VI_TO_DUMMY_NAME,
    'var_x_name_ord': VAR_X_NAME_ORD,
    'var_x_name_unord': VAR_X_NAME_UNORD,
    'var_polscore_se_name': VAR_POLSCORE_SE_NAME,
    }
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

welfare_np = np.zeros((len(ESTRISK_VALUE_TUPLE), len(GEN_METHOD_ALL),))

# 'True' values
pot_pred_true_np = prediction_df[list(VAR_POLSCORE_NAME)].to_numpy()

for mult_idx, muliplier in enumerate(ESTRISK_VALUE_TUPLE):
    params = params_org.copy()
    params['estrisk_value'] = muliplier
    myoptp_risk = OptimalPolicy(**params)
    estrisk = myoptp_risk.estrisk_adjust(training_df.copy(),
                                         data_title='training'
                                         )
    training_risk_df = estrisk['data_estrisk_df']
    polscore_risk_name = myoptp_risk.var_cfg.polscore_name.copy()

    for midx, gen_method in enumerate(GEN_METHOD_ALL):
        print(f'Optp_method: {gen_method}, '
              f'value of adjustment parameter: {muliplier}')
        myoptp = deepcopy(myoptp_risk)
        myoptp.gen_cfg.method = gen_method
        # ----- Training data ----------
        solve_dict = myoptp.solve(training_risk_df.copy(),
                                  data_title='training fair'
                                  )
        myoptp.evaluate(solve_dict['allocation_df'],
                        training_risk_df.copy(),
                        data_title='training risk'
                        )

        results_alloc = myoptp.allocate(prediction_df.copy(),
                                        data_title='prediction'
                                        )
        # Evaluate using prediction data
        myoptp.evaluate(results_alloc['allocation_df'],
                        prediction_df.copy(),
                        data_title='prediction'
                        )
        myoptp.print_time_strings_all_steps()
        my_report = McfOptPolReport(optpol=myoptp,
                                    outputfile='Report_OptP_')
        my_report.report()
        del my_report, myoptp

        # Compute welfare with original policy scores
        new_alloc_np = results_alloc['allocation_df'].to_numpy(
            )[:, 0].reshape(-1)
        row_indices = np.arange(len(new_alloc_np))
        welfare_indiv_np = pot_pred_true_np[row_indices, np.int32(new_alloc_np)]
        welfare_np[mult_idx, midx] = np.mean(welfare_indiv_np)

welfare_norm = welfare_np / welfare_np[0, :]

TITEL = 'Welfare (with adjustments for estimation risk)'
TITEL_PLOT = 'welfare_risk_adjusted_scores'
COLORS_PLOT = ('b', 'g', 'r', 'c', 'm', 'k',)

fig, ax = plt.subplots()
ax.set_title(TITEL)
ax.set_xlabel('Value of adjustment parameter')
ax.set_ylabel('Normalized welfare')

for midx, gen_method in enumerate(GEN_METHOD_ALL):

    ax.plot(ESTRISK_VALUE_TUPLE,
            welfare_norm[:, midx],
            label=gen_method,
            color=COLORS_PLOT[midx],
            )
    ax.legend()

fig.savefig(WEIGHTPLOT_PATH / (TITEL_PLOT + '.jpeg'), format='jpeg')
fig.savefig(WEIGHTPLOT_PATH / (TITEL_PLOT + '.pdf'), format='pdf')
plt.show()
plt.close()

print('End of example estimation.\n\nThanks for using OptimalPolicy. \n\nYours'
      ' sincerely\nExperimental OptimalPolicy MCF module \U0001F600')
