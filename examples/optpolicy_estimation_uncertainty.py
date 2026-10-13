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

Version: 0.11.0

This is an example showing how to use the adjustments for policy score
uncertainty in the mcf optimal policy module.

"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mcf.example_data import example_data
from mcf.optpolicy_main import OptimalPolicy
from mcf.reporting import McfOptPolReport

# Fit a separate policy for every method and risk-adjustment value.


# ------------- NOT passed to OptimalPolicy --------------------------------------------------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example'

# ---------------------- Generate artificial data --------------------------------------------------

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

PREDDATA_IS_TRAINDATA = False  # If True, the same data will be used
#                                for training and prediction (in this case, PRED_OBS is ignored).

NO_FEATURES = 20         # Number of features. Will generate different types of
#                          features (continuous, dummies, ordered, unordered)

NO_TREATMENTS = 3        # Number of treatments

training_df, prediction_df, name_dict = example_data(obs_y_d_x_iate=TRAIN_OBS,
                                                     obs_x_iate=PRED_OBS,
                                                     no_features=NO_FEATURES,
                                                     no_treatments=NO_TREATMENTS,
                                                     seed=12345,
                                                     type_of_heterogeneity='WagerAthey',
                                                     descr_stats=True
                                                     )
if PREDDATA_IS_TRAINDATA:
    prediction_df = training_df.copy()

# ------------- Methods used in Optimal Policy Module ----------------------------------------------
# These methods learn feature-based rules. Best-score allocation additionally needs
# prediction scores and their standard errors if risk-adjusted scores are absent.
GEN_METHOD_ALL = ('policy_tree', 'bps_classifier',)

# -------- All what follows are parameters of the OptimalPolicy ------------------------------------
#   Omitted keywords use API defaults; None may select automatic settings.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
GEN_OUTPATH = Path.cwd() / 'example/outputOPT'  # Directory for output.
# Explicit paths get a method subdirectory. None uses output below the working directory;
# nonempty output directories normally receive a numeric suffix.

WEIGHTPLOT_PATH = Path.cwd() / 'out'
#   Directory for plots comparing different estimation-risk adjustment values.

GEN_OUTFILETEXT = 'OptPolicy_0_11_0_EstimationRisk'  # File for text output
#   Default is 'txtFileWithOutput'.
#   The programme adds the .txt extension.

# Variable names
# --------------
# Optional row identifier. evaluate() still pairs rows by position, not by ID.
VAR_ID_NAME = 'id'

# Observed treatment, used for evaluation against the observed allocation and for switcher
# summaries. Use integer codes 0,...,K-1 in policy-score order.
VAR_D_NAME = 'treat'

# Required policy-score names in treatment order 0,...,K-1. Usually potential outcomes. If using
# effects relative to treatment 0, include a zero score for that reference treatment.
VAR_POLSCORE_NAME = ('y_pot0', 'y_pot1', 'y_pot2',)

# Numeric, finite, nonnegative standard errors, in the same order as the policy scores. Required
# by estrisk_adjust().
VAR_POLSCORE_SE_NAME = ('y_pot0_se', 'y_pot1_se', 'y_pot2_se',)

# Additional evaluation scores. Names are flattened and deduplicated, then grouped into
# consecutive blocks of K treatments. Only complete blocks with all columns present are evaluated.
# Each block must follow policy-score treatment order.
VAR_POLSCORE_DESC_NAME = (('zero', 'ite1vs0', 'ite2vs0',),
                          (('x_cont0', 'iate1vs0', 'iate2vs0',)))

# Ordered decision variables, including binary features, for trees/classifiers and allocation
# descriptions. At least one ordered or unordered feature is required for these learning methods.
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2', 'x_ord0', 'x_ord1', 'x_ord2')

# Unordered decision variables. Use integer-coded categories with at least three distinct values
# in retained training data; put binary features in VAR_X_NAME_ORD.
VAR_X_NAME_UNORD = ('x_unord0', 'x_unord1', 'x_unord2',)

# Features used to predict allocations for variable-importance diagnostics during evaluate().
# Default None.
VAR_VI_X_NAME = ('x_cont0', 'x_cont1', 'x_cont2',)

# Categorical predictors expanded into dummies for variable-importance diagnostics. Default None.
VAR_VI_TO_DUMMY_NAME = ('x_unord0',)

# Estimation-risk adjustment
# adjusted_score = policy_score - k * standard_error, with finite k (default 1).
# Scores and standard errors must be finite, and standard errors nonnegative.
# estrisk_adjust() returns data_estrisk_df and selects its adjusted score columns.
# Repeated calls replace adjustments from the original scores; they do not compound.
# Use a fresh instance for each k here to keep learned policies and reports separate.
# k is a sensitivity parameter, without a direct confidence-level interpretation.

ESTRISK_VALUE_TUPLE = (0, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,)
#  Values of ESTRISK_VALUE to be used below for comparing different
#  allocations.

# Policy-tree depth
# -----------------
# Maximum split depth of the first policy tree. Default 3; this example uses 2. A depth d allows
# at most 2**d leaves; fitted trees can be smaller.
PT_DEPTH_TREE_1 = 2

# Maximum extra split depth within each first-stage leaf. Default 1; 0 disables the second stage.
# The two stages are optimized sequentially. Here 2 + 2 allows at most 4 split levels and 16
# leaves.
PT_DEPTH_TREE_2 = 2

# -------------- Other parameters (costs and restrictions) -----------------------------------------
OTHER_MAX_SHARES = (1, 1, 1)  # Maximum share allowed for each treatment.
#  Maximum allowed share for each treatment, in the order of VAR_POLSCORE_NAME.
#  Specify one finite value per treatment, with 0 <= share <= 1.
#  A value of 0 specifies zero capacity for that treatment.
#  Shares must sum to at least 1; feasibility is also checked using integer observation counts.
#  Default (or None) is (1, ..., 1), corresponding to no share restrictions.

# This example adjusts policy scores for estimation risk before learning allocations.
# Parameters are collected in a dictionary and passed to OptimalPolicy.
# Omitted parameters use their defaults.
params_org = {'gen_outfiletext': GEN_OUTFILETEXT,
              'gen_outpath': GEN_OUTPATH,
              'other_max_shares': OTHER_MAX_SHARES,
              'pt_depth_tree_1': PT_DEPTH_TREE_1,
              'pt_depth_tree_2': PT_DEPTH_TREE_2,
              'var_d_name': VAR_D_NAME,
              'var_id_name': VAR_ID_NAME,
              'var_polscore_desc_name': VAR_POLSCORE_DESC_NAME,
              'var_polscore_name': VAR_POLSCORE_NAME,
              'var_vi_x_name': VAR_VI_X_NAME,
              'var_vi_to_dummy_name': VAR_VI_TO_DUMMY_NAME,
              'var_x_name_ord': VAR_X_NAME_ORD,
              'var_x_name_unord': VAR_X_NAME_UNORD,
              'var_polscore_se_name': VAR_POLSCORE_SE_NAME,
              }
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

welfare_np = np.zeros((len(ESTRISK_VALUE_TUPLE), len(GEN_METHOD_ALL),))

# 'True' values
pot_pred_true_np = prediction_df[list(VAR_POLSCORE_NAME)].to_numpy()

for mult_idx, multiplier in enumerate(ESTRISK_VALUE_TUPLE):
    for midx, gen_method in enumerate(GEN_METHOD_ALL):
        print(f'Optp_method: {gen_method}  Value of adjustment parameter: {multiplier}')
        params = params_org.copy()
        params['gen_method'] = gen_method
        params['estrisk_value'] = multiplier
        params['gen_outpath'] = GEN_OUTPATH / f'estrisk_{multiplier:g}'
        myoptp = OptimalPolicy(**params)
        estrisk = myoptp.estrisk_adjust(training_df.copy(), data_title='training')
        training_risk_df = estrisk['data_estrisk_df']
        # ----- Training data ----------
        solve_dict, training_risk_df = myoptp.solve(training_risk_df.copy(),
                                                    data_title='training risk',
                                                    )
        myoptp.evaluate(solve_dict['allocation_df'], training_risk_df.copy(),
                        data_title='training risk',
                        )
        results_alloc = myoptp.allocate(prediction_df.copy(), data_title='prediction')

        # evaluate() uses original scores via its descriptive-score fallback here.
        # Keep data and allocations in the same row order; IDs do not align them.
        myoptp.evaluate(results_alloc['allocation_df'], prediction_df.copy(),
                        data_title='prediction',
                        )
        myoptp.print_time_strings_all_steps()
        my_report = McfOptPolReport(optpol=myoptp, outputpath=myoptp.gen_cfg.outpath,
                                     outputfile='Report_OptP_' + gen_method)

        my_report.report()
        del my_report, myoptp

        # Compute welfare using original simulated potential outcomes and the same row order.
        new_alloc_np = results_alloc['allocation_df'].to_numpy()[:, 0].reshape(-1)
        row_indices = np.arange(len(new_alloc_np))
        welfare_indiv_np = pot_pred_true_np[row_indices, np.int32(new_alloc_np)]
        welfare_np[mult_idx, midx] = np.mean(welfare_indiv_np)

# Normalize each method by its own k=0 welfare.
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

WEIGHTPLOT_PATH.mkdir(parents=True, exist_ok=True)
fig.savefig(WEIGHTPLOT_PATH / (TITEL_PLOT + '.jpeg'), format='jpeg')
fig.savefig(WEIGHTPLOT_PATH / (TITEL_PLOT + '.pdf'), format='pdf')
plt.show()
plt.close()

print('End of example estimation.\n\nThanks for using OptimalPolicy.'
      ' \n\nYours sincerely\nExperimental OptimalPolicy MCF module \U0001F600'
      )
