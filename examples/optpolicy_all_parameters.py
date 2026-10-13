"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Optimal Policy - Python implementation

Please note that the Optimal Policy (Decision) module is still beta.
It is less tested than the MCF main module.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

Comprehensive OptimalPolicy configuration example for the current API.
Comments give short usage notes; see the API for full validation rules.

"""
from pathlib import Path

from mcf.example_data import example_data
from mcf.optpolicy_main import OptimalPolicy
from mcf.reporting import McfOptPolReport

# ------------------------ Variables NOT passed to OptimalPolicy -----------------------------------
#  Define data to be used in this example

APPLIC_PATH = Path.cwd() / 'example'

# ------------------------------ Generate artificial data ------------------------------------------

# Parameters to generate artificial data (DataFrame) for this example
TRAIN_OBS = 1000
#  Number of observations of training data.
#  'best_policy_score': Training data must contain policy scores.
#  'policy_tree', 'bps_classifier': Training data must contain policy scores and features.
#  Default is 1000.

PRED_OBS = 1000
#  Number of observations of prediction data. Prediction data is used to allocate treatments.
#  'best_policy_score': Prediction data must contain policy scores.
#  'policy_tree', 'bps_classifier': Prediction data must contain the features that are used in
#  training. Policy scores are not required.
#  Default is 1000.

FAIRNESS_CORRECTION = False
#  If True, scores are adjusted for fairness
#  considerations a là Bearth, Lechner, Mareckova, and Muny (2024). Requires
#  the specification of some additional keywords below
#  (var_protected_xxx, fair_xxx, ...).

ESTRISK_CORRECTION = False
#  If True, scores are adjusted for estimation risk. Requires the specification of some
#  additional keywords below (estrisk_value)


PREDDATA_IS_TRAINDATA = False
#  If True, the same data will be used for training & prediction (in this case, PRED_OBS is ignored)

NO_FEATURES = 20
#  Number of features. Will generate different types of features (continuous, dummies, ordered,
#  unordered)

NO_TREATMENTS = 3
#  Number of treatments

training_all_df, prediction_df, name_dict = example_data(obs_y_d_x_iate=TRAIN_OBS,
                                                         obs_x_iate=PRED_OBS,
                                                         no_features=NO_FEATURES,
                                                         no_treatments=NO_TREATMENTS,
                                                         seed=12345,
                                                         type_of_heterogeneity='WagerAthey',
                                                         descr_stats=True,
                                                         )
if PREDDATA_IS_TRAINDATA:
    prediction_df = training_all_df.copy()

# --------------------- Policy Learners used in the Optimal Policy class ---------------------------
METHODS = ('best_policy_score', 'bps_classifier', 'policy_tree',)
#  Tuple used to set GEN_METHOD in this example
#  Currently valid methods are: 'best_policy_score', 'bps_classifier', 'policy_tree'


# Omitted keywords use defaults. Explicit None can differ; see PT_EVA_CAT_MULT.

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
VAR_POLSCORE_DESC_NAME = (('ite0vs0', 'ite1vs0', 'ite2vs0',),
                          ('iate0vs0', 'iate1vs0', 'iate2vs0',)
                         )

# Ordered decision variables, including binary features, for trees/classifiers and allocation
# descriptions. At least one ordered or unordered feature is required for these learning methods.
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2', 'x_ord0', 'x_ord1', 'x_ord2',)

# Unordered decision variables. Use integer-coded categories with at least three distinct values
# in retained training data; put binary features in VAR_X_NAME_ORD.
VAR_X_NAME_UNORD = ('x_unord0', 'x_unord1', 'x_unord2',)

# Features used to predict allocations for variable-importance diagnostics during evaluate().
# Default None.
VAR_VI_X_NAME = ('x_cont0', 'x_cont1', 'x_cont2',)

# Categorical predictors expanded into dummies for variable-importance diagnostics. Default None.
VAR_VI_TO_DUMMY_NAME = ('x_unord0',)

# Output and CPU use
# ------------------
# Output base directory. Explicit paths get a method subdirectory (BPS, PT or BPS_CLASSIF); None
# uses Path.cwd() / 'output' without that subdirectory. Nonempty directories normally get a
# numeric suffix.
GEN_OUTPATH = Path.cwd() / 'example/outputOPT'

# Base name of detailed .txt and _Summary.txt files. Existing files with these names are deleted
# on initialization. Default 'txtFileWithOutput'.
GEN_OUTFILETEXT = 'OptPolicy.0.11.0'

# Standard text output: 0 terminal, 1 files, 2 both (default).
GEN_OUTPUT_TYPE = None

# Predict each learned allocation with a random forest classifier to describe feature importance
# during evaluate(). Requires output and available VAR_VI_* predictors. Default True.
GEN_VARIABLE_IMPORTANCE = None

# CPU workers: None uses 80% of logical cores; numeric values <=1.5 use one worker, larger values
# are rounded. Policy-tree search with 1 is sequential.
GEN_MP_PARALLEL = None

# Allocation method and policy trees
# ----------------------------------
# 'best_policy_score' (default) allocates from scores; 'bps_classifier' learns classifiers for
# best-score allocations; 'policy_tree' fits policy trees. The legacy 'policy tree old' is also
# supported. The loop below sets this keyword.
GEN_METHOD = None

# Maximum split depth of the first policy tree. Default 3; this example uses 2. A depth d allows
# at most 2**d leaves; fitted trees can be smaller.
PT_DEPTH_TREE_1 = 2

# Maximum extra split depth within each first-stage leaf. Default 1; 0 disables the second stage.
# The two stages are optimized sequentially. Here 2 + 2 allows at most 4 split levels and 16
# leaves.
PT_DEPTH_TREE_2 = 2

# Candidate-value limit for ordered features (default 100). Smaller grids reduce work but can miss
# better splits. Candidate approximation and sequential stages mean the fitted tree is not a
# global optimum over all possible trees.
PT_NO_OF_EVALUPOINTS = None

# Multiplier for the categorical candidate-split limit in policy_tree. Omitting the keyword uses
# 1; explicit None, as below, uses 2.
PT_EVA_CAT_MULT = None

# When all categorical splits fit within the limit, consider all. Otherwise True samples
# categories; False uses score-difference orderings with bounded candidate counts. Default False;
# policy_tree only.
PT_SELECT_VALUES_CAT = None

# Minimum leaf size. None uses 0.1 * training rows / maximum leaf count, scaled by the smallest
# positive share limit when restricted. Rounded with a minimum of 1. First-stage leaves must also
# accommodate the second stage.
PT_MIN_LEAF_SIZE = None

# Additional feasibility checks and reward penalties during tree search; does not guarantee every
# capacity limit. Requires PT_DEPTH_TREE_2=0; otherwise disabled. Default False.
PT_ENFORCE_RESTRICTION = None

# Best-score allocations and evaluation benchmarks
# ------------------------------------------------
# Optional priority variable for an additional capacity-restricted best-score allocation: process
# larger values first. Also supplies training labels for an additional bps_classifier rule. Unused
# without restrictions.
VAR_BB_RESTRICT_NAME = 'x_unord0'

# Probabilities for the random evaluation benchmark: one finite, nonnegative value per treatment,
# summing to 1. None uses observed treatment shares when available, otherwise equal probabilities.
# Realized shares can differ.
RND_SHARES = None

# Treatment costs and capacity restrictions
# -----------------------------------------
# Maximum treatment shares, one per score, in [0, 1], summing to at least 1; 0 gives zero
# capacity. Integer capacities also need to be feasible. None is unrestricted. Best-score methods
# also return unrestricted bb allocations; classifier predictions and policy trees need not
# satisfy the requested limits.
OTHER_MAX_SHARES = None

# Base costs in policy-score units, one per treatment; subtracted from scores. None uses zero.
# Restricted policy trees calibrate additional costs using best-score allocations, without
# guaranteeing tree capacity limits.
OTHER_COSTS_OF_TREAT = None

# One finite positive multiplier per treatment for additional calibrated tree costs. Final cost =
# base cost + multiplier * additional cost. None uses 1 for each treatment; base costs are
# unchanged.
OTHER_COSTS_OF_TREAT_MULT = None

# Automatic data preparation
# --------------------------
# Policy learning always cleans required data. Trees/classifiers also screen covariates. This
# option removes redundancy when absolute Pearson correlation exceeds 0.999. Default True.
DC_CHECK_PERFECTCORR = None

# Screening threshold for the smaller category of a binary feature. Default 10; use 1 to disable
# this filter. Values below 1, including 0, use the default.
DC_MIN_DUMMY_OBS = None

# Feature selection
# -----------------
# Select decision variables for trees/classifiers using importance for each policy-score
# difference from treatment 0. Ignored for best_policy_score; unsupported with solvefair().
# Default False.
FS_YES = None

# Reserve observations for feature selection and exclude them from policy learning (default True).
# False reuses the same sample.
FS_OTHER_SAMPLE = None

# Reserved share for feature selection, in (0, 0.5]. Default 0.33.
FS_OTHER_SAMPLE_SHARE = None

# Remove a feature only when its importance is at or below the threshold for every score
# difference. Default 0; (0, 1] are fractions and (1, 100] are percentages. Stop when no feature
# qualifies or at most two original features remain.
FS_REL_VI_THRESHOLD = None

# Fairness variables and methods
# ------------------------------
# Protected features for fairness adjustments, conditional on materially relevant features. They
# are removed from the corresponding ordered/unordered decision lists. At least one protected
# variable is required for solvefair().
VAR_PROTECTED_NAME_ORD = ('x_cont0', 'x_ord0',)
VAR_PROTECTED_NAME_UNORD = ('x_unord0', 'x_unord1',)

# Materially relevant features through which protected-variable influence is allowed. They may
# also be decision variables, but must not overlap with protected features.
VAR_MATERIAL_NAME_ORD = ('x_cont1', 'x_ord1',)
VAR_MATERIAL_NAME_UNORD = ('x_unord2', 'x_unord3',)

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

# Estimation-risk adjustment
# --------------------------
# Finite k in policy_score - k * standard_error. Default 1. estrisk_adjust() selects the returned
# adjusted score columns for later learning; pass its data_estrisk_df to solve() or solvefair(). k
# is not a confidence level.
ESTRISK_VALUE = None

# Internal controls
# -----------------
# Plot legend size: indices 1-7 from xx-small to xx-large, or those size strings. Default 2
# (x-small).
_INT_FONTSIZE = None

# Requested saved-figure resolution (default 500). Large tree JPEGs may use less; tree PDFs are
# vector graphics.
_INT_DPI = None

# Enable text, figures, diagnostics and report information. Default True; False leaves output
# paths unset and disables variable-importance diagnostics.
_INT_WITH_OUTPUT = None

# True permits reusing a nonempty output directory; False (default) selects an empty directory or
# numeric suffix. Existing named text files are deleted during initialization.
_INT_OUTPUT_NO_NEW_DIR = None

# Display policy-tree and Qini figures (default True). False retains saved figures and tree
# figures in PDF reports. Requires output.
_INT_SHOW_PLOTS = None

# Use Numba for supported computations. Default True.
_INT_WITH_NUMBA = None

# Split ordered-feature first-split candidates into up to four tasks per feature during parallel
# policy_tree search. Default True.
_INT_XTR_PARALLEL = None

# Use the legacy Ray implementation. Default False.
_INT_MP_USE_OLD_RAY = None

# New parallel backend: 'ray', 'joblib' or 'sequential'. None selects joblib on Windows or when
# Ray is unavailable; otherwise Ray.
_INT_MP_BACKEND = None

# Joblib array-memory-mapping threshold, default 4 MiB. 0 removes the size threshold; object
# arrays are excluded.
_INT_MP_MEMMAP_MIN_BYTES = None

# Joblib temporary-file base directory. None uses Path.cwd() / 'joblibtemp'. Executors clean their
# own subdirectories; an existing base directory is retained.
_INT_MP_MEMMAP_DIR = None

# Requested maximum batches for parallel tree search. None or 'automatic' uses task-, worker- and
# backend-dependent sizing. Actual batch counts can be smaller than requested.
_INT_MP_BATCHES = None

# Collect constructor keywords; the loop chooses gen_method.
params = {'dc_check_perfectcorr': DC_CHECK_PERFECTCORR,
          'dc_min_dummy_obs': DC_MIN_DUMMY_OBS,

          'estrisk_value': ESTRISK_VALUE,

          'fair_type': FAIR_TYPE,
          'fair_adjust_target': FAIR_ADJUST_TARGET, 'fair_consistency_test': FAIR_CONSISTENCY_TEST,
          'fair_cont_min_values': FAIR_CONT_MIN_VALUES,
          'fair_regression_method': FAIR_REGRESSION_METHOD,
          'fair_protected_max_groups': FAIR_PROTECTED_MAX_GROUPS,
          'fair_material_max_groups': FAIR_MATERIAL_MAX_GROUPS,
          'fair_protected_disc_method': FAIR_PROTECTED_DISC_METHOD,
          'fair_material_disc_method': FAIR_MATERIAL_DISC_METHOD,

          'fs_rel_vi_threshold': FS_REL_VI_THRESHOLD, 'fs_other_sample': FS_OTHER_SAMPLE,
          'fs_other_sample_share': FS_OTHER_SAMPLE_SHARE, 'fs_yes': FS_YES,

          'gen_outfiletext': GEN_OUTFILETEXT, 'gen_outpath': GEN_OUTPATH,
          'gen_output_type': GEN_OUTPUT_TYPE, 'gen_mp_parallel': GEN_MP_PARALLEL,
          'gen_variable_importance': GEN_VARIABLE_IMPORTANCE,

          'other_costs_of_treat': OTHER_COSTS_OF_TREAT,
          'other_costs_of_treat_mult': OTHER_COSTS_OF_TREAT_MULT,
          'other_max_shares': OTHER_MAX_SHARES,

          'pt_depth_tree_1': PT_DEPTH_TREE_1, 'pt_depth_tree_2': PT_DEPTH_TREE_2,
          'pt_enforce_restriction': PT_ENFORCE_RESTRICTION, 'pt_eva_cat_mult': PT_EVA_CAT_MULT,
          'pt_no_of_evalupoints': PT_NO_OF_EVALUPOINTS, 'pt_min_leaf_size': PT_MIN_LEAF_SIZE,
          'pt_select_values_cat': PT_SELECT_VALUES_CAT,

          'rnd_shares': RND_SHARES,

          'var_bb_restrict_name': VAR_BB_RESTRICT_NAME, 'var_d_name': VAR_D_NAME,
          'var_id_name': VAR_ID_NAME, 'var_polscore_desc_name': VAR_POLSCORE_DESC_NAME,
          'var_polscore_name': VAR_POLSCORE_NAME, 'var_polscore_se_name': VAR_POLSCORE_SE_NAME,
          'var_protected_name_ord': VAR_PROTECTED_NAME_ORD,
          'var_protected_name_unord': VAR_PROTECTED_NAME_UNORD,
          'var_material_name_ord': VAR_MATERIAL_NAME_ORD,
          'var_material_name_unord': VAR_MATERIAL_NAME_UNORD, 'var_vi_x_name': VAR_VI_X_NAME,
          'var_vi_to_dummy_name': VAR_VI_TO_DUMMY_NAME, 'var_x_name_ord': VAR_X_NAME_ORD,
          'var_x_name_unord': VAR_X_NAME_UNORD,

          '_int_dpi': _INT_DPI, '_int_fontsize': _INT_FONTSIZE,
          '_int_output_no_new_dir': _INT_OUTPUT_NO_NEW_DIR,
          '_int_with_numba': _INT_WITH_NUMBA, '_int_with_output': _INT_WITH_OUTPUT,
          '_int_xtr_parallel': _INT_XTR_PARALLEL,
          '_int_mp_use_old_ray': _INT_MP_USE_OLD_RAY,
          '_int_mp_backend': _INT_MP_BACKEND,
          '_int_mp_memmap_min_bytes': _INT_MP_MEMMAP_MIN_BYTES,
          '_int_mp_memmap_dir': _INT_MP_MEMMAP_DIR,
          '_int_mp_batches': _INT_MP_BATCHES,
          '_int_show_plots': _INT_SHOW_PLOTS,
          }

APPLIC_PATH.mkdir(parents=True, exist_ok=True)

for method in METHODS:   # Loop over specified methods
    params['gen_method'] = method
    myoptp = OptimalPolicy(**params)

    if ESTRISK_CORRECTION:
        # Adjust scores for estimation uncertainty
        estrisk_dict = myoptp.estrisk_adjust(training_all_df, data_title='training')
        training_df = estrisk_dict['data_estrisk_df']
    else:
        training_df = training_all_df.copy()

    # solve() returns retained training rows aligned with its allocations.
    # solvefair() returns only a dictionary; this complete simulation keeps row order.
    if FAIRNESS_CORRECTION:
        solvefair_dict = myoptp.solvefair(training_df, data_title='training')
        alloc_train_df = solvefair_dict['allocation_df']
    else:
        solve_dict, training_df = myoptp.solve(training_df, data_title='training')
        alloc_train_df = solve_dict['allocation_df']

    # evaluate() pairs rows by position. After adjustment, missing current scores
    # fall back to the last complete available descriptive-score block.
    myoptp.evaluate(alloc_train_df, training_df, data_title='training')

    # All three methods can allocate these data: the simulation supplies policy scores.
    # Best-score allocation applies earlier risk/fairness adjustments when needed.
    results_alloc = myoptp.allocate(prediction_df, data_title='prediction')
    myoptp.evaluate(results_alloc['allocation_df'], prediction_df, data_title='prediction')

    myoptp.print_time_strings_all_steps()
    if myoptp.gen_cfg.with_output:
        my_report = McfOptPolReport(optpol=myoptp,
                                     outputpath=results_alloc['outpath'],
                                     outputfile='Report_OptP_' + method)
        my_report.report()
    del myoptp

print('End of example estimation.\n\nThanks for using OptimalPolicy. \n\nYours'
      ' sincerely\nOptimalPolicy MCF module (beta) \U0001F600'
      )
