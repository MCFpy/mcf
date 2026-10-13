"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Optimal Policy - Python implementation

Please note that this Optimal Policy (Decision) module is experimental.
It is (still) much less tested than the MCF main module.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

This is an example to show how the fairness adjustments can be implemented.
For more details on theory and methods, see Bearth, Lechner, Mareckova, and
Muny, arXiv (2025).

"""
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mcf.example_data import example_data
from mcf.mcf_print_stats import print_f
from mcf.optpolicy_evaluation import dependence_allocation_variables
from mcf.optpolicy_fair_add import reshuffle_share_rows
from mcf.optpolicy_main import OptimalPolicy
from mcf.reporting import McfOptPolReport


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

PREDDATA_IS_TRAINDATA = False  # If True, the same data will be used
#                                for training and prediction (in this case,
#                                PRED_OBS is ignored).

NO_FEATURES = 20         # Number of features. Will generate different types of
#                          features (continuous, dummies, ordered, unordered)

NO_TREATMENTS = 3        # Number of treatments

training_df, prediction_df, name_dict = example_data(obs_y_d_x_iate=TRAIN_OBS,
                                                     obs_x_iate=PRED_OBS,
                                                     no_features=NO_FEATURES,
                                                     no_treatments=NO_TREATMENTS,
                                                     seed=12345,
                                                     type_of_heterogeneity='WagerAthey',
                                                     descr_stats=True,
                                                     correlation_x='high',
                                                     )
if PREDDATA_IS_TRAINDATA:
    prediction_df = training_df.copy()

# ------------- Methods used in Optimal Policy Module --------------------------
GEN_METHOD = 'policy_tree'
# Choose 'best_policy_score', 'policy_tree', or 'bps_classifier'.
# 'best_policy_score' also needs policy scores in the prediction data.

# -------- All what follows are parameters of the OptimalPolicy --------
#   Omitted keywords use API defaults; None may select automatic settings.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
GEN_OUTPATH = Path.cwd() / 'example/outputOPT'  # Directory for output.
# Explicit paths get a method subdirectory. None uses output below the working directory;
# nonempty output directories normally receive a numeric suffix.

WEIGHTPLOT_PATH = Path.cwd() / 'out'
#   Directory for plots comparing the protected-information settings.

GEN_OUTFILETEXT = 'OptPolicy_0_11_0_Fairness'  # File for text output
#   Default is 'txtFileWithOutput'.
#   *.txt file extension will be added by the programme

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

# Additional evaluation scores. Names are flattened and deduplicated, then grouped into
# consecutive blocks of K treatments. Only complete blocks with all columns present are evaluated.
# Each block must follow policy-score treatment order.
VAR_POLSCORE_DESC_NAME = (('zero', 'ite1vs0', 'ite2vs0',),
                          ('x_cont0', 'iate1vs0', 'iate2vs0',)
                          )

# Ordered decision variables, including binary features, for trees/classifiers and allocation
# descriptions. At least one ordered or unordered feature is required for these learning methods.
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2', 'x_ord0', 'x_ord1', 'x_ord2')

# Unordered decision variables. Use integer-coded categories with at least three distinct values
# in retained training data; put binary features in VAR_X_NAME_ORD.
VAR_X_NAME_UNORD = ('x_unord0', 'x_unord1', 'x_unord2',)

# Features used to predict allocations for variable-importance diagnostics during evaluate().
# Default None.
VAR_VI_X_NAME = None

# Categorical predictors expanded into dummies for variable-importance diagnostics. Default None.
VAR_VI_TO_DUMMY_NAME = []

# ---------------------- Method specific parameters ---------------------------
TRAIN_DATA_WELFARE = False  # Use training / prediction data to evaluate welfare

# - - - - - - - - - - - - - - - Fairness - - - - - - - - - - - - - - - - - - - -
#  Use solvefair() to learn an allocation rule with fairness adjustments.
#  Depending on FAIR_ADJUST_TARGET, it adjusts policy scores, decision variables, or both,
#  then learns the allocation rule using the adjusted data.
#  Specify protected variables via VAR_PROTECTED_NAME_ORD and VAR_PROTECTED_NAME_UNORD.
#  The returned dictionary contains the training allocation under 'allocation_df'.

NO_GRID_VALUES_FAIRNESS_PLOTS = 10
# Number of intervals, giving 11 settings from 0 to 1 here. The later experiment
# varies the share of protected-variable rows reshuffled before fairness adjustment.
# It does not average original and fairness-adjusted policy scores.

#  Protected variables: remove their influence on the selected adjustment targets,
#  conditional on the materially relevant variables.

VAR_PROTECTED_NAME_ORD = 'x_cont3'     # ('x_cont3', 'x_ord3',)
VAR_PROTECTED_NAME_UNORD = 'x_unord2'  # ('x_unord2', 'x_unord3',)
#   These variables should NOT be contained in decision variables, i.e.,
#   VAR_X_NAME_ORD and/or VAR_X_NAME_UNORD. If they are included, they will be
#   removed and VAR_X_NAME_ORD and/or VAR_X_NAME_UNORD will be adjusted
#   accordingly. Defaults are None (at least one of them must be specified, if
#   the fairness adjustments are to be used).

#  Materially relevant variables: An effect of the protected variables on the
#  scores is allowed, if captured by these variables (only).
#  These variables may (or may not) be included among the decision variables.
#  These variables must (!) not be included among the protected variables.
VAR_MATERIAL_NAME_ORD = 'x_cont4'      # ('x_cont4', 'x_ord1',)
VAR_MATERIAL_NAME_UNORD = 'x_unord0'   # ('x_unord0', 'x_unord4',)

#   Collect variable used to compute variable importance statistics for allocations
if VAR_VI_X_NAME is None:
    VAR_VI_X_NAME = []
    for var_names in (VAR_PROTECTED_NAME_ORD, VAR_MATERIAL_NAME_ORD):
        if var_names:
            VAR_VI_X_NAME.extend([var_names] if isinstance(var_names, str) else var_names)

if VAR_VI_TO_DUMMY_NAME is None:
    VAR_VI_TO_DUMMY_NAME = []
    for var_names in (VAR_PROTECTED_NAME_UNORD, VAR_MATERIAL_NAME_UNORD):
        if var_names:
            VAR_VI_TO_DUMMY_NAME.extend([var_names] if isinstance(var_names, str) else var_names)

# Fairness methods
# ----------------
# 'scores', 'xvariables' or 'scores_xvariables'. None selects 'scores' for best_policy_score and
# 'xvariables' otherwise. best_policy_score rejects 'xvariables'.
FAIR_ADJUST_TARGET = None

# 'Quantiled' (default), 'Mean' or 'MeanVar'. Scores support all three methods. Discrete decision
# variables always use Quantiled; policy trees use it for all decision variables. Score
# adjustment, when requested, runs first with the selected method.
FAIR_TYPE = None

# Regression for Mean/MeanVar adjustments. Default 'RandomForest'; 'automatic' selects separately
# for each regression using min(5, n) CV folds. 'Mean' uses sample means. See the API for other
# estimator names.
FAIR_REGRESSION_METHOD = None

# Quantile-adjustment grouping: 'Kmeans' (default) clusters features jointly; 'EqualCell' bins
# each feature; 'NoDiscretization' keeps values continuous. If the sum of distinct-value counts is
# within the group limit, exact joint cells are used instead. For policy-tree decision-variable
# adjustment, NoDiscretization is replaced by Kmeans.
FAIR_PROTECTED_DISC_METHOD = None
FAIR_MATERIAL_DISC_METHOD = None

# Grouping limits (default 5): per-feature bins for EqualCell, requested joint clusters for
# Kmeans. Kmeans counts are capped by distinct profiles; EqualCell joint cells can exceed the per-
# feature limit.
FAIR_PROTECTED_MAX_GROUPS = None
FAIR_MATERIAL_MAX_GROUPS = None

# Compare differences of adjusted scores with adjusted differences of original scores. Only used
# when scores are adjusted; can be expensive. Default False.
FAIR_CONSISTENCY_TEST = None

# Distinct-value threshold for treating ordered decision variables as continuous (default 20).
# Unordered decision variables are always discrete. Also controls continuous/discrete handling of
# protected variables in reported diagnostics.
FAIR_CONT_MIN_VALUES = None

# Policy-tree depth
# -----------------
# Maximum split depth of the first policy tree. Default 3; this example uses 2. A depth d allows
# at most 2**d leaves; fitted trees can be smaller.
PT_DEPTH_TREE_1 = 2

# Maximum extra split depth within each first-stage leaf. Default 1; 0 disables the second stage.
# The two stages are optimized sequentially. Here 2 + 2 allows at most 4 split levels and 16
# leaves.
PT_DEPTH_TREE_2 = 2

# -------------- Other parameters (costs and restrictions) --------------------
OTHER_MAX_SHARES = (1, 1, 1)  # Maximum share allowed for each treatment.
#  Maximum allowed share for each treatment, in the order of VAR_POLSCORE_NAME.
#  Specify one finite value per treatment, with 0 <= share <= 1.
#  A value of 0 specifies zero capacity for that treatment.
#  Shares must sum to at least 1; feasibility is also checked using integer observation counts.
#  Default (or None) is (1, ..., 1), corresponding to no share restrictions.

# The following is an example on how the fairness adjustment of the
# OptimalPolicy object can be used.
# For convenience the parameters are collected and passed as a dictionary.
# Of course, they can also be passed as single parameters (or not at all, in
# which case default values are used).
params = {'fair_adjust_target': FAIR_ADJUST_TARGET,
          'fair_consistency_test': FAIR_CONSISTENCY_TEST,
          'fair_cont_min_values': FAIR_CONT_MIN_VALUES,
          'fair_regression_method': FAIR_REGRESSION_METHOD,
          'fair_protected_max_groups': FAIR_PROTECTED_MAX_GROUPS,
          'fair_material_max_groups': FAIR_MATERIAL_MAX_GROUPS,
          'fair_protected_disc_method': FAIR_PROTECTED_DISC_METHOD,
          'fair_material_disc_method': FAIR_MATERIAL_DISC_METHOD,
          'fair_type': FAIR_TYPE,
          'gen_method': GEN_METHOD,
          'gen_outfiletext': GEN_OUTFILETEXT,
          'gen_outpath': GEN_OUTPATH,
          'other_max_shares': OTHER_MAX_SHARES,
          'pt_depth_tree_1': PT_DEPTH_TREE_1,
          'pt_depth_tree_2': PT_DEPTH_TREE_2,
          'var_d_name': VAR_D_NAME,
          'var_id_name': VAR_ID_NAME,
          'var_polscore_desc_name': VAR_POLSCORE_DESC_NAME,
          'var_polscore_name': VAR_POLSCORE_NAME,
          'var_protected_name_ord': VAR_PROTECTED_NAME_ORD,
          'var_protected_name_unord': VAR_PROTECTED_NAME_UNORD,
          'var_material_name_ord': VAR_MATERIAL_NAME_ORD,
          'var_material_name_unord': VAR_MATERIAL_NAME_UNORD,
          'var_vi_x_name': VAR_VI_X_NAME,
          'var_vi_to_dummy_name': VAR_VI_TO_DUMMY_NAME,
          'var_x_name_ord': VAR_X_NAME_ORD,
          'var_x_name_unord': VAR_X_NAME_UNORD,
          }
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Part 1: Learn and evaluate the fairness-adjusted allocation

# Initialise the instance from the optimal policy class
myoptp_fair = OptimalPolicy(**params)

# Training data: Build decision rule
# Instead of the solve() method use the solvefair() method to get fairness
# adjustments selected by FAIR_ADJUST_TARGET. Automatic feature selection is unsupported.
alloc_train_fair_dict = myoptp_fair.solvefair(training_df.copy(), data_title='training')
# Evaluate the results using the training data
myoptp_fair.evaluate(alloc_train_fair_dict['allocation_df'], training_df.copy(),
                     data_title='training'
                     )

# allocate() automatically computes missing adjusted decision variables for the learned rule.
# evaluate() also needs the original protected columns for fairness diagnostics.
# Both inputs must have the same observation order; rows are matched by position.
alloc_pred_fair_dict = myoptp_fair.allocate(prediction_df.copy(), data_title='prediction')
results_pred_fair_dict = myoptp_fair.evaluate(alloc_pred_fair_dict['allocation_df'],
                                              prediction_df.copy(),
                                              data_title='prediction'
                                              )
outpath_fair = results_pred_fair_dict['outpath']

# Final reporting and creation of pdf
myoptp_fair.print_time_strings_all_steps()
my_report = McfOptPolReport(optpol=myoptp_fair, outputpath=outpath_fair,
                             outputfile='Report_OptP_Fairness')
my_report.report()
del my_report

# ------------------------------------------------------------------------------
# Part 2: Compare fairness adjusted allocation to unadjusted allocation
# 2.1 Compare winners and losers.
# 2.1.1 Learn an unadjusted allocation using the same permitted decision features.
myoptp = OptimalPolicy(**params)
myoptp.solve(training_df.copy(), data_title='training')
alloc_pred_dict = myoptp.allocate(prediction_df.copy(), data_title='prediction')
myoptp.evaluate(alloc_pred_dict['allocation_df'], prediction_df.copy(), data_title='prediction')
if myoptp_fair.gen_cfg.method != myoptp.gen_cfg.method:
    raise ValueError('Different allocation methods used in both allocations.')

# 2.1.2 Choose the same named allocation rule from both fitted models.

match myoptp.gen_cfg.method:
    case 'best_policy_score': ALLOC_NAME = 'bb'
    case 'bps_classifier':    ALLOC_NAME = 'bps_classif_bb'
    case 'policy_tree':       ALLOC_NAME = 'Policy Tree'
    case method: raise ValueError(f'Winner-loser comparison not available for {method}')


# 2.1.3 Choose training or prediction data for comparison
if TRAIN_DATA_WELFARE:  # Use training data for comparison
    alloc_train_dict = myoptp.allocate(training_df.copy(), data_title='training')
    alloc_train_fair_dict = myoptp_fair.allocate(training_df.copy(), data_title='training fair')
    alloc_np = alloc_train_dict['allocation_df'][ALLOC_NAME].to_numpy().reshape(-1)
    alloc_fair_np = alloc_train_fair_dict['allocation_df'][ALLOC_NAME].to_numpy().reshape(-1)
    data_eval_df = training_df
else:                  # Use data not used for training for comparison
    alloc_np = alloc_pred_dict['allocation_df'][ALLOC_NAME].to_numpy().reshape(-1)
    alloc_fair_np = alloc_pred_fair_dict['allocation_df'][ALLOC_NAME].to_numpy().reshape(-1)
    data_eval_df = prediction_df

# 2.1.4 Find winners and losers
# Comparison is based on (unadjusted) policy scores
valuations_np = data_eval_df[myoptp.var_cfg.polscore_name].to_numpy()
row_indices = np.arange(valuations_np.shape[0])
# Pick chosen treatment
welfare_np = valuations_np[row_indices, alloc_np]
welfare_fair_np = valuations_np[row_indices, alloc_fair_np]
welfare_df = pd.DataFrame(welfare_np.reshape(-1, 1), columns=('Welfare',))
welfare_fair_df = pd.DataFrame(welfare_fair_np.reshape(-1, 1), columns=('Welfare_fair_alloc',))
# Collect protected variables
protected_var = myoptp_fair.var_cfg.protected_ord_name.copy()
protected_var.extend(myoptp_fair.var_cfg.protected_unord_name)

# Compare dependence of adjusted and unadjusted allocations on protected variables.
dependence_nonfair = dependence_allocation_variables(alloc_np, data_eval_df[protected_var])
dependence_fair = dependence_allocation_variables(alloc_fair_np, data_eval_df[protected_var])
# Compare individual welfare under the adjusted and unadjusted allocations.
winner_loser_dict = myoptp.winners_losers(data_eval_df, welfare_fair_df,
                                          welfare_reference_df=welfare_df,
                                          outpath=WEIGHTPLOT_PATH,
                                          title='Winners and Losers of Fairness Adjustments',
                                          )
myoptp.print_time_strings_all_steps(title='Comparison to unadjusted allocation.')

# 2.2 Sensitivity to the information supplied about protected variables.
# At setting 0, reshuffle all protected-variable rows; at 1, keep the original data.
# Intermediate settings reshuffle a decreasing share. solvefair() runs at every setting,
# so 0 is not an exact no-adjustment benchmark and changes need not be monotone.
# Evaluate welfare with original scores and dependence with original protected data.

# Collect names of protected variables and respective data in np.ndarray
protected_name = []
if params['var_protected_name_ord']:
    if not isinstance(params['var_protected_name_ord'], (list, tuple)):
        params['var_protected_name_ord'] = [params['var_protected_name_ord']]
    protected_name.extend(params['var_protected_name_ord'])

if params['var_protected_name_unord']:
    if not isinstance(params['var_protected_name_unord'], (list, tuple)):
        params['var_protected_name_unord'] = [params['var_protected_name_unord']]
    protected_name.extend(params['var_protected_name_unord'])

prot_train_np = training_df[protected_name].to_numpy(copy=True)
prot_pred_np = prediction_df[protected_name].to_numpy(copy=True)

# Define the protected-information settings.
grid = np.linspace(0, 1, NO_GRID_VALUES_FAIRNESS_PLOTS+1)
print('Protected-information setting (0: shuffled, 1: original): ', end=' ')
# Define dataframe to store welfare and dependency measures
measures = ['welfare']
measures.extend([f'{i}{j}' for i, j in product(['Corr', 'CrV', 'BayF'], protected_name)])

welfare_dependence_df = pd.DataFrame(np.nan, index=np.round(grid, 3), columns=measures)
params_tradeoff = deepcopy(params)
params_tradeoff['_int_with_output'] = False
params_tradeoff['_int_output_no_new_dir'] = True
for grid_idx, fair_weight in enumerate(grid):
    print(round(fair_weight, 2), end=' ', flush=True)
    params_loop = deepcopy(params_tradeoff)
    training_df_loop = training_df.copy()
    prediction_df_loop = prediction_df.copy()

    # Shuffle protected rows jointly, preserving their within-row combinations.
    training_df_loop[protected_name] = reshuffle_share_rows(prot_train_np, 1-fair_weight, seed=1234)
    prediction_df_loop[protected_name] = reshuffle_share_rows(prot_pred_np, 1-fair_weight, seed=1234
                                                              )
    # Obtain new allocation with modified protected variable
    myoptp_loop = OptimalPolicy(**params_loop)
    myoptp_loop.solvefair(training_df_loop)

    # Use training or prediction data
    data_eval_df = training_df_loop if TRAIN_DATA_WELFARE else prediction_df_loop
    data_org_df = training_df.copy() if TRAIN_DATA_WELFARE else prediction_df.copy()
    alloc_dict = myoptp_loop.allocate(data_eval_df.copy())

    # Evaluate the same allocation as in the winner-loser comparison.
    alloc_loop_np = alloc_dict['allocation_df'][ALLOC_NAME].to_numpy()

    # Evaluations are based on (unmodified) policy scores.
    valuations_np = data_eval_df[myoptp.var_cfg.polscore_name].to_numpy()
    row_indices = np.arange(valuations_np.shape[0])
    values = valuations_np[row_indices, alloc_loop_np]
    welfare_dependence_df.iloc[grid_idx, 0] = np.round(np.mean(values), 3)

    dep1 = dependence_allocation_variables(alloc_loop_np,
                                           data_org_df[protected_name]
                                           )
    dep = [dep_m_df.iloc[0].tolist() for dep_m_df in dep1]
    welfare_dependence_df.iloc[grid_idx, 1:] = [round(i, 3) for dep2 in dep for i in dep2]

    del myoptp_loop

#    Preparations for plots
liste = ['Fair', myoptp_fair.fair_cfg.adjust_target, myoptp_fair.fair_cfg.adj_type,]
if myoptp_fair.fair_cfg.adj_type == 'Quantiled':
    liste.extend(['Mat', myoptp_fair.fair_cfg.material_disc_method,
                  'Prot', myoptp_fair.fair_cfg.protected_disc_method,
                  ])
titel = '_'.join(liste)
TITEL_PLOT = 'Welfare for different protected-information settings'
fig, ax = plt.subplots()
ax.set_title(TITEL_PLOT)
ax.set_xlabel('Protected-information setting (0: shuffled, 1: original)')
welfare = welfare_dependence_df['welfare'].tolist()
ax.set_ylabel('Welfare')
label = params['gen_method']
ax.plot(grid, welfare, label=label, color='b')
ax.legend()

WEIGHTPLOT_PATH.mkdir(parents=True, exist_ok=True)
fig.savefig(WEIGHTPLOT_PATH / (titel + '.jpeg'), format='jpeg')
fig.savefig(WEIGHTPLOT_PATH / (titel + '.pdf'), format='pdf')
plt.show()
plt.close()

txt = ('\nWelfare and protected-variable dependence across information settings.'
       f'\nWelfare is measured by {" ".join(myoptp.var_cfg.polscore_name)}.'
       '\nWelfare across protected-information settings: '
       f'{' '.join([str(round(w, 3)) for w in welfare])}'
       )
txt += '\n' * 2 + str(welfare_dependence_df)
txt += (f'\n\nComparative outputs (plot, tables) are stored in {WEIGHTPLOT_PATH}'
        f'\nDetails of (full) fairness adjustment are stored in {outpath_fair}'
        f'\nDetails of unadjusted assignment are stored in {winner_loser_dict["outpath"]}'
        )
PRINT_FILE = WEIGHTPLOT_PATH / ('Fairlevels_' + GEN_OUTFILETEXT + '.txt')
print_f(PRINT_FILE, txt)

print(txt)
del myoptp, myoptp_fair

print('End of example estimation.\n\nThanks for using OptimalPolicy with Fairness correction.'
      ' \n\nYours sincerely\nExperimental OptimalPolicy MCF module \U0001F600'
      )
