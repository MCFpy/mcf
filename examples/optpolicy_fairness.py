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

Version: 0.8.0

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

from mcf.example_data_functions import example_data
from mcf.mcf_print_stats_functions import print_f
from mcf.optpolicy_evaluation_functions import dependence_allocation_variables
from mcf.optpolicy_fair_add_functions import reshuffle_share_rows
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
    descr_stats=True,
    correlation_x='high')

if PREDDATA_IS_TRAINDATA:
    prediction_df = training_df

# ------------- Methods used in Optimal Policy Module --------------------------
GEN_METHOD = 'policy_tree'
# METHODS = ('best_policy_score', 'bps_classifier', 'policy_tree',)

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

VAR_POLSCORE_DESC_NAME = (('zero', 'ite1vs0', 'ite2vs0',),
                          ('x_cont0', 'iate1vs0', 'iate2vs0',)
                          )
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

VAR_VI_X_NAME = None  # ('x_cont0', 'x_cont1', 'x_cont2',)
#   Names of variables for which variable importance is computed.
#   Default is None.
VAR_VI_TO_DUMMY_NAME = []  # ('x_unord0',)
#   Names of categorical variables for which variable importance is computed.
#   These variables will be broken up into dummies. Default is None.

# ---------------------- Method specific parameters ---------------------------
TRAIN_DATA_WELFARE = False  # Use training / prediction data to evaluate welfare

# - - - - - - - - - - - - - - - Fairness - - - - - - - - - - - - - - - - - - - -
#   The 'fairscores' method adjusts the policy scores by removing the influence
#   of some features that are specificied in the VAR_PROTECTED_NAME variable.
#   If the 'fairscores' method is not explicitly called, the following variables
#   are not relevant and need not to be specifified.

NO_GRID_VALUES_FAIRNESS_PLOTS = 10
#  Combines the fairness adjusted and the original scores to create a plot
#  which shows welfare loss of when imposing fairness.
#  NO_GRID_VALUES_FAIRNESS_PLOTS is the number of grid points (between 0 and 1)
#  for the respective weighted means.

#   Protected variables (the effect of the policy scores on those variables will
#                        be removed, conditional on the 'materially important'
#                        variables).

VAR_PROTECTED_NAME_ORD = 'x_cont3'     # ('x_cont3', 'x_ord3',)
VAR_PROTECTED_NAME_UNORD = 'x_unord2'  # ('x_unord2', 'x_unord3',)
#   These variables should NOT be contained in decision variables, i.e.,
#   VAR_X_NAME_ORD and/or VAR_X_NAME_UNORD. If they are included, they will be
#   removed and VAR_X_NAME_ORD and/or VAR_X_NAME_UNORD will be adjusted
#   accordingly. Defaults are None (at least one of them must be specified, if
#   the fairness adjustments are to be used).

#  Materially relavant variables: An effect of the protected variables on the
#  scores is allowed, if captured by these variables (only).
#  These variables may (or may not) be included among the decision variables.
#  These variables must (!) not be included among the protected variables.
VAR_MATERIAL_NAME_ORD = 'x_cont4'      # ('x_cont4', 'x_ord1',)
VAR_MATERIAL_NAME_UNORD = 'x_unord0'   # ('x_unord0', 'x_unord4',)

#   Collect variable used to compute variable importance statistics for
#   allocations
if VAR_VI_X_NAME is None:
    VAR_VI_X_NAME = []
    if VAR_PROTECTED_NAME_ORD:
        list(VAR_VI_X_NAME).extend(VAR_PROTECTED_NAME_ORD)
    if VAR_MATERIAL_NAME_ORD:
        list(VAR_VI_X_NAME).extend(VAR_MATERIAL_NAME_ORD)

if VAR_VI_TO_DUMMY_NAME is None:
    VAR_VI_TO_DUMMY_NAME = []
    if VAR_PROTECTED_NAME_UNORD:
        list(VAR_VI_TO_DUMMY_NAME).extend(VAR_PROTECTED_NAME_UNORD)
    elif VAR_MATERIAL_NAME_UNORD:
        list(VAR_VI_TO_DUMMY_NAME).extend(VAR_MATERIAL_NAME_UNORD)

#  -----------------------------------------------------------------------------
#  Details of the fairness adjustment methods for policy scores
#  (see Bearth, Lechner, Muny, Mareckova, 2025, arXiv, for details)

#  Target for the fairness adjustment
FAIR_ADJUST_TARGET = None
#  Possible values: 'scores', 'xvariables', 'scores_xvariables' (or None)
#  'scores': Adjust policy scores
#  'xvariables': Adjust decision variables
#  'scores_xvariables': Adjust both decision variables and scores
#  Default (or None) is 'xvariables'.

#  Method to choose the type of correction for the policy scores
FAIR_TYPE = None
#  Possible values: 'Quantiled', 'Mean', 'MeanVar' (or None)
#  'Quantiled': Removing dependence via (an empricial version of) the approach
#               by Strack and Yang (2024).
#  'Mean':  Mean dependence of the policy score on protected var's is removed.
#  'MeanVar':  Mean dependence and heteroscedasticity is removed.
#  'Mean' and 'MeanVar' are only availble for adjusting the score
#  (not the decision variables)
#  Default (or None) is 'Quantiled'.

#  Method choice when predictions from machine learning are needed
FAIR_REGRESSION_METHOD = None
#  Possible values: 'automatic', 'Mean', None,
#  'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5'
#  'SupportVectorMachine', 'SupportVectorMachineC2', 'SupportVectorMachineC4',
#  'AdaBoost', 'AdaBoost100', 'AdaBoost200', 'GradBoost', 'GradBoostDepth6',
#  'GradBoostDepth12', 'LASSO', 'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger'
#  Regression method to adjust scores w.r.t. protected features.
#  Possible choices are based on scikit-learn's regression methods.
#  (Mean is included if other estimators are not useful.)
#  If 'automatic', an optimal methods will be chosen based on 5-fold
#  cross-validation in the training data. If 'automatic', a method is specified
#  it will be used for all scores and all adjustments. If None, every policy
#  score might be adjusted with a different method. 'Mean' is included for cases
#  in which regression methods have no explanatory power.
#  This method is relevant when "fair_type in ('Mean', 'MeanVar')"
#  Default (or None) is 'RandomForest'.

#  Parameters for discretization of features (necessary for 'Quantilized'):
FAIR_PROTECTED_DISC_METHOD = None
FAIR_MATERIAL_DISC_METHOD = None
#  Possible values: 'NoDiscretization', 'EqualCell', 'Kmeans' (or None)
#  Whether and how to perform the discretization for protected / materially
#  relevant features (for FAIR_TYPE = 'Quantiled'). Maximum number of groups
#  allowed is set by FAIR_MATERIAL_MAX_GROUPS.
#  'NoDiscretization': Variables are not changed. If one of the features has
#               more different values than FAIR_MATERIAL_MAX_GROUPS, all
#               materially relevant features will formally be treated as
#               continuous. In  this case quantile positions will be based on
#               nonparametrically estimated densities. The latter may become
#               unreliable if their dimension is not very small.
#  'EqualCell': Attempts to create equal cells for each feature. Maybe be
#               useful for a very small number of variables with few different
#               values.
#  'Kmeans':    Use Kmeans clustering algorithm to form homogeneous cells.
#  Default (or None) is 'Kmeans'.

#  Level of discretization of variables (if needed)
FAIR_PROTECTED_MAX_GROUPS = None
FAIR_MATERIAL_MAX_GROUPS = None
#  Number of different groups of values of features that are
#  materially relavant / protected (if discretization is used).
#  This is only necessary for 'Quantilized'.
#  Its meaning depends on fair_protected_disc_method, fair_material_disc_method:
#  If 'EqualCell': If more than 1 variable is included among the protected
#      features, this restriction is applied to each variable.
#  If 'Kmeans': This is the number of clusters used by Kmeans.
#  Default (or None) is 5.

#  Test for internally consistency
FAIR_CONSISTENCY_TEST = None
#  When FAIR_ADJUST_TARGET is 'scores' or 'scores_xvariables', then
#  the fairness corrections are applied independently to every policy score (
#  which usually is a potential outcome or an IATE(x) for each treatment
#  relative to some base treatment (i.e. comparing 1-0, 2-0, 3-0, etc.).
#  Thus the IATE for the 2-1 comparison can be computed as IATE(2-0)-IATE(1-0).
#  This tests compares two ways to compute a fair score for the 2-1 (and all
#  other comparisons) which should give simular results:
#  a) Difference of two fair (!) scores
#  b) Difference of corresponding scores, subsequently made fair.
#  Note: Depending on the number of treatments, this test may be computationally
#  more expensive than the orginal fairness corrections.
#  Therefore, the default is False (and it is not performed if
#  (fair_adjust_target == 'score_xvariables', although this requires a score
#  adjustment as well.

#  When should a decision variable be treated as continuous?
FAIR_CONT_MIN_VALUES = None
#  The methods used for fairness corrections depends on whether the variable
#  is consider as continuous or discrete. All unordered variables are considered
#  being discrete, and all ordered variables with more than FAIR_CONT_MIN_VALUES
#  are considered as being discrete as well. The default (or None) is 20.

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
params = {
    'fair_adjust_target': FAIR_ADJUST_TARGET,
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
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

# ------------------------------------------------------------------------------
# Part 1:Build and evaluate fairness adjusted allocation

# Initialise the instance from the optimal policy class
myoptp_fair = OptimalPolicy(**params)

# Training data: Build decision rule
# Instead of the solve() method use the solvefair() method to get fairness
# adjustments for scores and decision variables
alloc_train_fair_dict = myoptp_fair.solvefair(training_df.copy(),
                                              data_title='training'
                                              )
# Evaluate the results using the training data
myoptp_fair.evaluate(alloc_train_fair_dict['allocation_df'],
                     training_df.copy(),
                     data_title='training'
                     )

# Analyse the results for data not used for training (not implemented for
# 'best_policy_score')
if myoptp_fair.gen_dict['method'] != 'best_policy_score':
    # Compute the new allocation
    alloc_pred_fair_dict = myoptp_fair.allocate(
        prediction_df.copy(), data_title='prediction'
        )
    # Evaluate the new allocation
    results_pred_fair_dict = myoptp_fair.evaluate(
        alloc_pred_fair_dict['allocation_df'],
        prediction_df.copy(),
        data_title='prediction'
        )
    outpath_fair = results_pred_fair_dict['outpath']
else:
    results_pred_fair = outpath_fair = alloc_pred_fair_df = None

# Final reporting and creation of pdf
myoptp_fair.print_time_strings_all_steps()
my_report = McfOptPolReport(optpol=myoptp_fair,
                            outputfile='Report_OptP_')
my_report.report()
del my_report

# ------------------------------------------------------------------------------
# Part 2: Compare fairness adjusted allocation to unadjusted allocation
if myoptp_fair.gen_dict['method'] == 'best_policy_score':
    raise ValueError('Comparisons of allocations only for methods that learn '
                     'explicit variable-based rule in training data.'
                     )
# 2.1 Compare winners & losers (only for methods training an assignment rule)
# 2.1.1 Create allocation without fairness corrections
myoptp = OptimalPolicy(**params)
myoptp.solve(training_df.copy(), data_title='training')
alloc_pred_dict = myoptp.allocate(prediction_df.copy(),
                                  data_title='prediction'
                                  )
myoptp.evaluate(alloc_pred_dict['allocation_df'],
                prediction_df.copy(),
                data_title='prediction'
                )
if myoptp_fair.gen_dict['method'] != myoptp.gen_dict['method']:
    raise ValueError('Different allocation methods used in both allocations.')

# 2.1.2 Chose specific allocations to investigate (names depend on method used)
if myoptp.gen_dict['method'] == 'bps_classifier':
    ALLOC_NAME = 'bps_classif_bb'
elif myoptp.gen_dict['method'] == 'policy_tree':
    ALLOC_NAME = 'Policy Tree'
else:
    raise ValueError('Winner-looser comparison not available for '
                     f"{myoptp.gen_dict['method']}"
                     )
# 2.1.3 Choose training or predictiond data for comparison
if TRAIN_DATA_WELFARE:  # Use training data for comparison
    alloc_train_dict = myoptp.allocate(training_df.copy(),
                                       data_title='training'
                                       )
    alloc_train_fair_dict = myoptp_fair.allocate(training_df.copy(),
                                                 data_title='training fair'
                                                 )
    alloc_np = alloc_train_dict['allocation_df'][ALLOC_NAME].to_numpy(
        ).reshape(-1)
    alloc_fair_np = alloc_train_fair_dict['allocation_df'][ALLOC_NAME].to_numpy(
        ).reshape(-1)
    data_eval_df = training_df
else:                  # Use data not used for training for comparison
    alloc_np = alloc_pred_dict['allocation_df'][ALLOC_NAME].to_numpy(
        ).reshape(-1)
    alloc_fair_np = (alloc_pred_fair_dict['allocation_df'][ALLOC_NAME]
                     .to_numpy()
                     .reshape(-1)
                     )
    data_eval_df = prediction_df

# 2.1.4 Find winners and losers
# Comparison is based on (unadjusted) policy scores
valuations_np = data_eval_df[myoptp.var_dict['polscore_name']].to_numpy()
row_indices = np.arange(valuations_np.shape[0])
# Pick chosen treatment
welfare_np = valuations_np[row_indices, alloc_np]
welfare_fair_np = valuations_np[row_indices, alloc_fair_np]
welfare_df = pd.DataFrame(welfare_np.reshape(-1, 1), columns=('Welfare',))
welfare_fair_df = pd.DataFrame(welfare_fair_np.reshape(-1, 1),
                               columns=('Welfare_fair_alloc',)
                               )
# Collect protected variables
protected_var = myoptp_fair.var_dict['protected_ord_name']
protected_var.extend(myoptp_fair.var_dict['protected_unord_name'])

# Check the dependence of fair and 'un'fair allocation on protected variables
dependence_nonfair = dependence_allocation_variables(alloc_np,
                                                     data_eval_df[protected_var]
                                                     )
dependence_fair = dependence_allocation_variables(alloc_fair_np,
                                                  data_eval_df[protected_var]
                                                  )
# Analyse who wins and who loses when allocation becames fairer
winner_loser_dict = myoptp.winners_losers(
            data_eval_df, welfare_fair_df,
            welfare_reference_df=welfare_df,
            outpath=WEIGHTPLOT_PATH,
            title='Winners and Losers of Fairness Adjustments'
            )

myoptp.print_time_strings_all_steps(title='Comparison to unadjusted allocation.'
                                    )

# 2.2 Further investigate the efficiency-fairness tradeoff
#     Create plot with welfare comparisons for different levels of fairness
#     adjustments.

# 2.2.1 Create simulated protected variable and plot results for different
#       strengths of protection

#       Subsequently add more and more information from protected variable to
#       simulated protected variable

# Collect names of protected variables and respective data in np.ndarray
protected_name = []
if params['var_protected_name_ord']:
    if not isinstance(params['var_protected_name_ord'], (list, tuple)):
        params['var_protected_name_ord'] = [params['var_protected_name_ord']]
    protected_name.extend(params['var_protected_name_ord'])

if params['var_protected_name_unord']:
    if not isinstance(params['var_protected_name_unord'], (list, tuple)):
        params['var_protected_name_unord'] = [params['var_protected_name_unord']
                                              ]
        protected_name.extend(params['var_protected_name_unord'])

prot_train_np = training_df[protected_name].to_numpy()
prot_pred_np = prediction_df[protected_name].to_numpy()

# Define grid capturing strength of fairness
grid = np.linspace(0, 1, NO_GRID_VALUES_FAIRNESS_PLOTS+1)
print('Weight of fairness in artifically created materially relevant '
      'variable: ', end=' '
      )
# Define dataframe to store welfare and dependency measures
measures = ['welfare']
measures.extend([f"{i}{j}" for i, j
                 in product(['Corr', 'CrV', 'BayF'], protected_name)]
                )

welfare_dependence_df = pd.DataFrame(np.nan, index=np.round(grid, 3),
                                     columns=measures
                                     )
params_tradeoff = deepcopy(params)
params_tradeoff['_int_with_output'] = False
params_tradeoff['_int_output_no_new_dir'] = True
for grid_idx, fair_weight in enumerate(grid):
    print(round(fair_weight, 2), end=' ', flush=True)
    params_loop = deepcopy(params_tradeoff)
    training_df_loop = training_df.copy()
    prediction_df_loop = prediction_df.copy()

    # Reshuffle protected variable to make to make it less informative
    training_df_loop[protected_name] = reshuffle_share_rows(
        prot_train_np, 1-fair_weight, seed=1234
        )
    prediction_df_loop[protected_name] = reshuffle_share_rows(
        prot_pred_np, 1-fair_weight, seed=1234
        )
    # Obtain new allocation with modified protected variable
    myoptp_loop = OptimalPolicy(**params_loop)
    myoptp_loop.solvefair(training_df_loop)

    # Use training or prediction data
    data_eval_df = (training_df_loop
                    if TRAIN_DATA_WELFARE else prediction_df_loop)
    data_org_df = (training_df.copy()
                   if TRAIN_DATA_WELFARE else prediction_df.copy())
    alloc_dict = myoptp_loop.allocate(data_eval_df.copy())

    # Evaluations are based on (unmodified) policy scores
    valuations_np = data_eval_df[myoptp.var_dict['polscore_name']
                                 ].to_numpy()
    row_indices = np.arange(valuations_np.shape[0])
    values = valuations_np[row_indices, alloc_dict['allocation_df'].to_numpy(
        ).reshape(-1)]
    welfare_dependence_df.iloc[grid_idx, 0] = np.round(np.mean(values), 3)

    dep1 = dependence_allocation_variables(alloc_dict['allocation_df'].to_numpy(
        ).reshape(-1),
                                           data_org_df[protected_name])
    dep = [dep_m_df.iloc[0].tolist() for dep_m_df in dep1]
    welfare_dependence_df.iloc[grid_idx, 1:] = [
        round(i, 3) for dep2 in dep for i in dep2]

    del myoptp_loop

#    Preparations for plots
liste = ['Fair',
         myoptp_fair.fair_dict['adjust_target'],
         myoptp_fair.fair_dict['adj_type'],
         ]
if myoptp_fair.fair_dict['adj_type'] == 'Quantiled':
    liste.extend(['Mat', myoptp_fair.fair_dict['material_disc_method'],
                  'Prot', myoptp_fair.fair_dict['protected_disc_method']]
                 )
titel = '_'.join(liste)
TITEL_PLOT = 'Welfare for different degrees of fairness'
fig, ax = plt.subplots()
ax.set_title(TITEL_PLOT)
ax.set_xlabel('Weight of fairness adjustment (0: no adj., 1: full adj.)')
welfare = welfare_dependence_df['welfare'].tolist()
ax.set_ylabel('Welfare')
label = params['gen_method']
ax.plot(grid, welfare, label=label, color='b')
ax.legend()

fig.savefig(WEIGHTPLOT_PATH / (titel + '.jpeg'), format='jpeg')
fig.savefig(WEIGHTPLOT_PATH / (titel + '.pdf'), format='pdf')
plt.show()
plt.close()

txt = ('\nWelfare and correlation with protected attribute for different '
       'fairness levels.'
       '\nWelfare is measured by '
       f'{" ".join(myoptp.var_dict["polscore_name"])}.'
       '\nWelfare for different levels of fairness: '
       f'{' '.join([str(round(w, 3)) for w in welfare])}'
       )
txt += '\n' * 2 + str(welfare_dependence_df)
txt += ('\n\n'
        f'Comparative outputs (plot, tables) are stored in {WEIGHTPLOT_PATH}'
        f'\nDetails of (full) fairness adjustment are stored in {outpath_fair}'
        '\nDetails of unadjusted assignment are stored in '
        f'{winner_loser_dict["outpath"]}'
        )
PRINT_FILE = WEIGHTPLOT_PATH / ('Fairlevels_' + GEN_OUTFILETEXT + '.txt')
print_f(PRINT_FILE, txt)

print(txt)
del myoptp, myoptp_fair

print('End of example estimation.\n\nThanks for using OptimalPolicy with '
      'Fairness correction.'
      ' \n\nYours sincerely\nExperimental OptimalPolicy MCF module \U0001F600'
      )
