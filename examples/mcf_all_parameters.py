"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economic Research
University of St. Gallen, Switzerland

Version: 0.10.0

This is an example to show how to use the mcf with full specification of all its keywords.
It may be seen as an add-on to the published mcf documentation.

"""
from copy import deepcopy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from mcf.example_data import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ----------------------- Generate artificial data (DataFrame) -------------------------------------

# Number of observations of training data.
TRAIN_OBS = 1_000
#  Training data is used to train the causal forest. Training data must contain outcome, treatment
#  and features. Default is 1000.

# Number of observations of prediction data.
PRED_OBS = 1_000
#  Prediction data is used to compute the effects. Prediction data must contain features. Treatment
#  effects on the treated additionally require treatment information. Default is 1000.

PREDDATA_IS_TRAINDATA = False  # If True, the same data will be used  for training and prediction
#  (in this case, PRED_OBS is ignored).

# Number of features.
NO_FEATURES = 20
#  Will generate different types of features (X: continuous, dummies, ordered, unordered)

# Number of treatments
NO_TREATMENTS = 3

# Get the data in DataFrames for training and predictions and dictionary with variable names
training_df, prediction_df, name_dict = example_data(obs_y_d_x_iate=TRAIN_OBS,
                                                     obs_x_iate=PRED_OBS,
                                                     no_features=NO_FEATURES,
                                                     no_treatments=NO_TREATMENTS,
                                                     seed=12345,
                                                     type_of_heterogeneity='WagerAthey',
                                                     descr_stats=True,
                                                     correlation_x='high'
                                                     )
if PREDDATA_IS_TRAINDATA:
    prediction_df = None

# ------------------ Parameters of the ModifiedCausalForest ----------------------------------------
#  The following parameters are all used to initialise the ModifiedCausalForest class.
#  Whenever None is specified, parameter will be set to default values.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Path were the output is written to (text, estimated effects, etc.)
GEN_OUTPATH = Path.cwd() / 'example/output'
#  If this is None, a */out directory below the current directory is used. If specified directory
#  does not exist, it will be created. OUTPATH is passed to ModifiedCausalForest.

# File for text output
GEN_OUTFILETEXT = 'mcf.py.0.10.0'
#   If gen_outfiletext is None, 'txtFileWithOutput' is used. *.txt file extension will be added.

# Variables used in training and prediction (=inference)
#  Convention: Internally all variable names are set to lowercase (including those in the
#  DataFrames provided by user). Therefore, avoid having variables in the data with names that
#  differ only with respect to capitalization.

#  The following variable names must at least be provided:
#      train method: d_name, y_name, (x_name_ord or x_name_unord)
#      predict and analyse methods: x_name_ord or x_name_unord
#  All other variables are optional.

# Name of treatment variable
VAR_D_NAME = name_dict['d_name']      # name_dict['d_cont'], name_dict['d_name']
# Type of treatment: 'discrete' (default) or 'continuous'
GEN_D_TYPE = 'discrete'               # 'continuous', 'discrete'

# Outcome variables (single variable (string) or several variables (list of string))
VAR_Y_NAME = name_dict['y_name']

# Variable to build trees.
VAR_Y_TREE_NAME = name_dict['y_name']
#  If None or [], the first variable in y_name is used to build trees.

# Features
VAR_X_NAME_ORD = name_dict['x_name_ord']      # Ordered (including dummy variables)
VAR_X_NAME_UNORD = name_dict['x_name_unord']   # Unordered

# Identifier
VAR_ID_NAME = name_dict['id_name']
#  If None (default) or []: Identifier will be added the data.

# Cluster and panel identifiers (default is no variable)
VAR_CLUSTER_NAME = name_dict['cluster_name']  # Name of variable defining clusters if used.
VAR_W_NAME = name_dict['weight_name']         # Name of weight, if weighting option is used.

# Names of variables always checked on when deciding next split
VAR_X_NAME_ALWAYS_IN_ORD = []
VAR_X_NAME_ALWAYS_IN_UNORD = []

# ------------------ Variables related to treatment heterogeneity ----------------------------------
#  Variables to define policy relevant heterogeneity in multiple treatment procedure. If not
#  already included, they will be added to the list of features. The default is no variable.

#  Ordered variables with many values
VAR_Z_NAME_CONT = name_dict['x_name_ord'][:2]
#  Note: Put discrete variables with only few values in categories below. They are discretized
#  as well as treated as continuous for GATE estimation.

# Ordered variables with few values
VAR_Z_NAME_ORD = name_dict['x_name_ord'][-2:]

#   Unordered variables
VAR_Z_NAME_UNORD = name_dict['x_name_unord'][:2]

# Variable to balance the GATEs on
VAR_X_NAME_BALANCE_BGATE = name_dict['x_name_ord'][3:5]
#  Only relevant if P_BGATE is True. The distribution of these variables is kept constant when a
#  BGATE is computed. The default (or None) is to use the other heterogeneity variables for
#  balancing, if thereare any.

# Variables to be used for bias adjustment (Note: Bias adjustment is still experimental)
VAR_X_NAME_BA = None # (name_dict['x_name_ord'][0], *name_dict['x_name_unord'][:1],)
#  They must be included among the ordered and / or unordered variables specified above.
#  Default is None.

# Variables that cannot be removed by feature selection
VAR_X_NAME_REMAIN_ORD = []       # Ordered
VAR_X_NAME_REMAIN_UNORD = []     # Unordered
# Default is no variable.

# Variables for balancing tests (cannot be removed by feature selection).
VAR_X_NAME_BALANCE_TEST_ORD = name_dict['x_name_ord'][:2]           # Ordered
VAR_X_NAME_BALANCE_TEST_UNORD = name_dict['x_name_unord'][:2]       # Unordered
#  Treatment specific descriptive statistics will only be printed for those variables.

# -------------------------- Output and use of CPUs ------------------------------------------------

# Where to direct the text output to
GEN_OUTPUT_TYPE = None
#  0: Output goes to terminal
#  1: output goes to file
#  2: Output goes to file and terminal (default)

# Multiprocessing (number of parallel processes  (>0))
GEN_MP_PARALLEL = None
#  Default is to use 80% of logical cores (reduce if memory problems!) If _int_obs_bigdata, it is
#  reduced to 75% of specified value. 0, 1: no parallel computations

# ------------------------------ Data cleaning -----------------------------------------------------
DC_SCREEN_COVARIATES = None  # If True (Default) screen covariates (sc)
DC_CHECK_PERFECTCORR = None  # if sc=True: if True (default), var's that are  perfectly correlated
#  with each others will be deleted.
DC_MIN_DUMMY_OBS = None      # if > 0 dummy variables with observations in one category smaller
#  than this value will be deleted. Default is 10.
DC_CLEAN_DATA = None         # if True (default), remove all rows with missing
#  observations & unnecessary variables from DataFrame.

# ------------------------- Training the causal forest ---------------------------------------------
CF_BOOT = None      # Number of Causal Trees. Default is 1000.

# Estimation methods
CF_MCE_VART = None  # Splitting rule
#  0: mse's of regression only considered
#  1: mse+mce criterion (default)
#  2: -var(effect): heterogy maximising splitting rule of Wager & Athey (2018)
#  3: randomly switching between outcome-mse+mce criterion & penalty functions

# - - - - - - - - - Penalty function (depends on value of mce_vart) - - - - - - - - - - - - - - - -
# Muliplier of penalty function (or else)
CF_P_DIFF_PENALTY = None
#  if mce_vart == 0: Penalty irrelevant
#  if mce_vart == 1: Multiplier of penalty (in terms of var(y))
#                    0: no penalty
#                    default (or None):  2*((n*subsam_share)**0.9)/(n*subsam_share)
#                                        sqrt(no_of_treatments*(no_of_treatments-1)/2).
#  if mce_vart == 2: Default (or None): Multiplier of penalty (in terms of value of MSE(y) value
#                    function without splits) for penalty.
#                    0: no penalty
#                    default: 100*4*(n*f_c.subsam_share)^0.8)/(n*f_c.subsam_share)
#  if mce_vart == 3: Probability of using p-score (0-1) Default is 0.5.
#  Increase if balancing tests indicate bad balance.

# Type of penalty function
CF_PENALTY_TYPE = None
#  'mse_d':  MSE of treatment prediction in daughter leaf (new in 0.7.0)
#  'diff_d': Penalty as squared leaf difference (as in Lechner, 2018)
#  Note that an important advantage of 'mse_d' that it can also be used for tuning (due to its
#  computation, this is not possible for 'diff_d'). Default (or None) is 'mse_d'.

# - - - - - - - - - Compare potential outcomes only to first potential outcome - - - - - - - - - -
CF_COMPARE_ONLY_TO_ZERO = None
#  If True, the computation of the MCE ignores all elements not related to the first treatment
#  (which usually is the control group). This speeds up computation and may be attractive when
#  interest is only in the comparisons of each treatment to the control group, and not among each
#  other. This may also be attractive for optimal policy analysis based on using estimated potential
#  outcomes normalized by the potential outcome of the control group (i.e., IATEs of treatments vs.
#  control group). Default is False.

# - - - - - - - - - - - - - - - - - - Subsampling - - - - - - - - - - - - - - - - - - - - - - - - -
# Size of subsampling sample to build tree
CF_SUBSAMPLE_FACTOR_FOREST = None
#  Default size of subsample is: max(min(0.67*n,(2*(n**0.85)/n)), n**0.5);
#  N: sample size;  n: 2x sample size of the smallest treatment group
#  if >0: reduces (<1) or increases (>1) the default subsample size, max is 0.67
#  Actual share of subsample = default size * SUBSAMPLE_FACTOR_FOREST
#  Default (or None): SUBSAMPLE_FACTOR_FOREST = 1

# Size of subsampling sample to evaluate tree
CF_SUBSAMPLE_FACTOR_EVAL = None
#  Tree-specific subsampling also in evaluation sample should increase speed
#  at which asymtotic bias disappears (at the expense of slower disappearance
#  of the variance; however, simulations show no relevant impact).
#  False: No subsampling in evaluation subsample
#  True or None: 2 * subsample size used for tree building (default)[to avoid too many empty leaves]
#  Float (>0): Multiplier of subsample size used for tree building
#  In particular for larger samples, using subsampling in evaluation may speed up computations and
#  reduces demand on memory.

# - - - - - - - - - -  Matching step to find neighbor for mce computation - - - - - - - - - - - - -
# Distance measure
CF_MATCH_NN_PROG_SCORE = None
# False: Use Mahalanobis matching. True:  Use prognostic scores (default).

# Use only main diagonal or full covariance matrix in Mahalanobis matching
CF_NN_MAIN_DIAG_ONLY = None
#  Only relevenat if match_nn_prog_score = False. True: use main diagonal only.
#                                                 False (default): inverse of covariance matrix.

# - - - - - - - - - - - - - - - - Other tuning parameters - - - - - - - - - - - - - - - - - - - - -
# Check all possible splits or use only randomly selected thresholds (for continuous features)
CF_RANDOM_THRESHOLDS = None
#  If > 0: Do not check all possible split values of ordered variables, but only RANDOM_THRESHOLDS
#  (new randomisation for each split). Fewer thresholds speeds up programme but may (!) lead to
#  less accurate results.
#  0: no random thresholds
#  > 0: number of random thresholds used for ordered var's
#  Default: 4+N_train**0.2

# Minimum leaf size (use of a grid is possible)
CF_N_MIN_MIN = None      # Smallest minimum leaf size
CF_N_MIN_MAX = None      # Largest minimum leaf size.
#  Default (or None): n_min_min = round(max((n_d_subsam**0.4) / 10, 1.5) * number of treatments
#                     n_min_max = round(max((n_d_subsam**0.5) / 10, 2) * number of treatments
#                                 n_d: Number of observations in the smallest treatment arm.

# Number of grid values
CF_N_MIN_GRID = None
#  If n_min_grid == 1: n_min=(N_MIN_MIN+N_MIN_MAX)/2
#  If grid is used, optimal value is determined by out-of-bag estimation of objective function.
#  Default (or None) is 1.

# Minimum number of observations per treatment in leaf
CF_N_MIN_TREAT = None
#  A higher value reduces the risk that a leaf cannot be filled with outcomes from all treatment
#  arms in the evaluation subsample. There is no grid based tuning for this parameter. This impacts
#  the minimum leaf size which will be at least n_min_treat * number of treatments.
#   Default (or None): (N_MIN_MIN + N_MIN_MAX) / 2 / # of treatments / 4. Minimum is 2.

# Alpha regularity (use of a grid is possible)
CF_ALPHA_REG_MIN = None      # smallest alpha, 0 <= alpha < 0.4 (def: 0.05)
CF_ALPHA_REG_MAX = None      # 0 <= alpha < 0.5 (def: 0.15)
CF_ALPHA_REG_GRID = None     # number of grid values (def: 1).
#  Results of Wager and Athey (2018) suggest alpha_reg < 0.2. A larger value may increase speed of
#  tree building.
#  If ALPHA_REG_GRID == 1, alpha = (ALPHA_REG_MIN+ALPHA_REG_AX)/2
#  If grid is used, optimal value is determined by out-of-bag estimation of objective function.

# Number of variables used at each new split of tree
CF_M_SHARE_MIN = None        # minimum share of variables (0-1); def = 0.1
CF_M_SHARE_MAX = None        # maximum share of variables (0-1); def = 0.6
#  Number of variables used for splitting = share * total # of variable. If variables randomly
#  selected for splitting do not show any variation in leaf considered for splitting, then all
#  variables will be used for that split.

# Number of grid values logarithmically spaced, including m_min m_max.
CF_M_GRID = None
#  Default is 1.
#  If M_GRID == 1: m_share = (M_SHARE_MIN+M_SHARE_MAX)/2
#   If grid is used, optimal value is determined by out-of-bag estimation of objective function.

# Tune all parameters
CF_TUNE_ALL = None
#  If True, all *_grid keywords will be set to 3. User specified values are respected if larger
#  than 3. Default is False.

# Randomly determine the number of variables to be used.
CF_M_RANDOM_POISSON = None
#  True: Number of randomly selected variables is stochastic for each split, drawn from a Poisson
#  distribution. Grid gives mean value of 1 + poisson distribution (m-1).
#  Default is True if M > 10, otherwise False (to avoid getting too few variables that not have
#  enough variation).
#  If grid is used, optimal value is determined by out-of-bag estimation of objective function.

# Maximum number of observations allowed per chunk.
CF_CHUNKS_MAXSIZE = None
#  For large samples, randomly split the training data into equally sized chunks, train a forest in
#  each chunk, and estimate effects for each forest. Final effect estimates are obtained by
#  averaging effects obtained for each forest. This procedures improves scalability by reducing
#  computation time and memory demand.
#  If CF_CHUNKS_MAXSIZE is smaller than sample size: Random splitting
#  If CF_CHUNKS_MAXSIZE is larger than sample size: No random splitting.
#  Default if _int_low_memory_predict is False:
#  If less than 100000 training observations: No splitting. Otherwise, the maximal size of
#  each chunksize is obtained as
#  100000 + (number of observations - 100000)**0.8 /(# of treatments-1).
#  Default if _int_low_memory_predict is True:
#  If 250000 or less training observations: No splitting. Otherwise, the maximal size of
#  each chunksize is obtained as 250000 + (number of observations - 250000)**0.8/(# of treatments-1)

# Variable importance for causal forest
CF_VI_OOB_YES = None
# True:  computes variable importance based on permuting every single x in oob prediction; time
# consuming. Default is False.

# ----------------------------- Feature selection --------------------------------------------------
FS_YES = None
#  True: feature selection active. False: no feature selection (default).
FS_OTHER_SAMPLE = None
#  False: Use sample used for causal forest estimation.
#  True (default): Random sample from training data used. These data will not be used for training
#  the causal forest.

# Share of sample to be used for feature selection (if FS_OTHER_SAMPLE is True)
FS_OTHER_SAMPLE_SHARE = None
#  Default is 0.33.

# Threshold in terms of relative loss of variable importance (0-1) for outcome variable
FS_REL_VI_THRESHOLD_Y = None
#  Default is 0.

# Threshold in terms of relative loss of variable importance (0-1) for outcome variable
FS_REL_VI_THRESHOLD_D = None
#  Default is 0.

# Define how the two thresholds are combined.
FS_REL_VI_KEEP_IF = None
#  Possible choice are: 'y_relevant', 'y_or_d_relevant', 'y_and_d_relevant'
#  Default is 'y_or_d_relevant'

# ------------- Joint parameters for Local Centering & Common Support adjustment -------------------
LC_CS_CV = None        # True: Use crossvalidation (default).
#  False (estimator choice in local centering): Use LC_CS_SHARE for testing.
#  False (common support and local centering): Use random subsample of the data that will then not
#  be used for training the CF.

# Share of data used for estimating E(y|x)
LC_CS_SHARE = None
#  0.1-0.9. Default is 0.25.

# Number of folds used in crossvalidation (if LC_CS_CV is True)
LC_CS_CV_K = None
#  Default depends on the size of the training sample (N): N < 100'000: 5
#                                               100'000 <= N < 250'000: 4
#                                               250'000 <= N < 500'000: 3
#                                               500'000 <= N: 2

# -------------------------- Local centering -------------------------------------------------------
# Local centering (or not)
LC_YES = None
#  Default is True.
LC_ESTIMATOR = None
#  The estimator used for local centering. Possible choices are scikit-learn's regression methods
#  'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5', 'SupportVectorMachine',
#  'SupportVectorMachineC2', 'SupportVectorMachineC4', 'AdaBoost', 'AdaBoost100', 'AdaBoost200',
#  'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12', 'LASSO', (if more than 1000 observations),
#  'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',  'Mean' ('Mean' is included for the case when
#  if other estimators are not useful). If set to 'automatic', the estimator with the lowest
#  out-of-sample mean squared error (MSE) is selected. Whether this selection is based on
#  cross-validation or a test sample is governed by LC_CS_CV. Default (or None) is 'RandomForest'.

# Compute uncentered potential outcomes
LC_UNCENTER_PO = None
#  Predicted potential outcomes for individual data  points (IATEs) re-adjusted for local centering
#  are added to data output.  Default is True.

# ----------------------------- Common support -----------------------------------------------------
# Type of common support adjustment
CS_TYPE = None
#  0: No common support adjustment
#  1,2: Support check based on estimated classification random forests.
#  1: Use min-max rules for probabilities in treatment subsamples .
#  2: Enforce minimum and maximum probabilities for all obs and all probs-1 observations off support
#  are removed.
#  Out-of-bag predictions are used to avoid overfitting (which would lead to a too large reduction
#  in the number of observations). Default (or None) is 1.

# Correction of computed common support cut-offs
CS_ADJUST_LIMITS = None
#  IF CS_TYPE > 0:  > 0 or None (default); 0: No adjustment
#  Adjustment: upper *= 1+SUPPORT_ADJUST_LIMITS, lower *= 1-SUPPORT_ADJUST_LIMITS
#  The restrictiveness of the common support criterion increases with the number of treatments. This
#  parameter reduces this restrictiveness.  Default (None): (number of treatments - 2) * 0.05

# Quantile used to compute upper and lower cut-off for common-support
CS_QUANTIL = None
#  If CS_TYPE == 1: 1: min-max rule; 0-1: if smaller than 1.  the respective quantil is taken for
#  the cut-offs; Default is to use the minumum and maximum (i.e., 0% and 100% quantile)

# Lower cut-off
CS_MIN_P = None
#  If cs_type == 2: Observations are deleted if p(d=m|x) <= 'cs_min_p' for at least one treatment.
#  Default (or None) is 0.01

# Maximum share of sample reduction allowed
CS_MAX_DEL_TRAIN = None
#  If share of observations in training data used that are OFF support is larger than
#  SUPPORT_MAX_DEL_TRAIN (0-1), programme will be terminated and user should change input data.
#  Default (or None) is 0.5.

# Check common support using individual variables.
CS_DETECT_CONST_VARS_STOP = None
#  Control variables that have no variation inside a treatment arm violate the common support
#  condition. If CS_DETECT_VARS_NO_VAR_STOP is True, data will checked for such variables and an
#  exception is raised if such a variable is detected. Then, the user has to decide to either adjust
#  the data (by deleting observations with the value of the variable that creates the problem) or
#  delete this variable. Default (or None) is True.

# ---- Option to switch role of forest and y-sample and average prediction to increase precision ---
# Caveat: Inference will be conservative.
# Second round effects are estimated based on switching training and estimation subsamples. As the
# final estimate is the average of the two point estimates, this increase their efficiency (and
# stability) but does not allow for weight-based inference on these more efficient estimates.
# Therefore, inference is based on the average variance and is therefore conservative. Note that
# turning efficient estimation on roughly doubles computation time.
GEN_ATE_EFF = None
#  Default (or None) is False.
GEN_GATE_EFF = None
#  This includes the BGATEs and CBGATES. Default (or None) is False.
GEN_IATE_EFF = None
#  Default (or None) is False.
GEN_QIATE_EFF = None
#  Default (or None) is False.
#  Note: If GEN_GATE_EFF is True or GEN_IATE_EFF is True or GEN_QIATE_EFF is True, then GEN_ATE_EFF
#  will be automatically set to True.

# ---------------------------- Clustering and panel data -------------------------------------------
# Treat data as panel data
GEN_PANEL_DATA = None
#  True if panel data; None or False: no panel data. True activates the clustered standard error.
#  Use cluster_name to define variable that contains identifier for panel unit.
#  Default (or None) is False.

# Uses the panel structure in training
GEN_PANEL_IN_RF = None
#  Uses the panel structure also when building the random samples within the forest procedure.
#  Default (or None) is True.

# One-way clustering robust standard errors
P_CLUSTER_STD = None
#  True: Clustered standard error. Default (or None) is False. It will be automatically set to True
#  if panel data option is activated. The clustering and panel options are experimental.

# ----------------------- Parameters for continuous treatment --------------------------------------
# The continuous treatment part is experimental (and not extensively tested).
# Number of grid point for discretization of continuous treatment (with 0 mass point).
# Grid is defined in terms of quantiles of continuous part of treatment.
CT_GRID_NN = None
#  Used to aproximate the neighbourhood matching. Default (or None) is 10.
CT_GRID_W = None
#  Used to aproximate the weights. Default (or None) is 10.
CT_GRID_DR = None
#  Used to aproximate the dose response function. Default (or None) is 10.

# ------------------------------ Balancing test (beta) ---------------------------------------------
# Balancing test (experimental)
P_BT_YES = None  # None
#  True: ATE based balancing test based on weights. Requires weight_based_inference. Relevance of
#  this test in its current implementation is not fully clear. Default (or None) is False.

# ---- Choice based sampling to speed up programme if treatment groups have very different sizes ---
# Choice-based sampling is experimental.
P_CHOICE_BASED_SAMPLING = None
#  True: Choice based sampling. Default (or None) is False.
P_CHOICE_BASED_PROBS = None
#  Sampling probabilities to be specified. These weights are used for (g,b)ates only. Treatment
#  information must therefore be available in the prediction data.

#  -------------------------- Sample weights -------------------------------------------------------
# Using sampling weights is experimental.
GEN_WEIGHTED = None
#  True: use sampling weights. Default is False. Sampling weights specified in var_w_name will be
#  used; slows down programme.

# ------------------------ Parameters for predicting effects ---------------------------------------
# Truncation of extreme weights
P_MAX_WEIGHT_SHARE = None
#  Maximum share of any weight, 0 <= 1. Default (or None) is 0.05.
#  Limit enforced by trimming excess weights and renormalisation for each effect separately.
#  Due to the renormalising, the final weights could be somewhat above this threshold.

# Confidence level for bounds shown in plots
P_CI_LEVEL = None
# 0 < 1: Confidence level for bounds used in plots. Default (or None) is 0.95.

# - - - - - - - - - - - - - - - - - Weight based inference - - - - - - - - - - - - - - - - - - - - -
# Variance estimation
P_COND_VAR = None
#  False: Variance estimation uses var(wy). True: conditional mean & variances are used.
#  Default (or None) is True.

# Computation of conditional variances
P_KNN = None
#  False: Nadaraya-Watson estimation. True: knn estimation (faster). Default is True.
#  Nadaray-Watson estimation gives a better approximaton of the variance, but k-NN is much faster,
#  in particular for larger datasets.

# Parameters of KNN and Nadarays-Watson estimators
P_KNN_MIN_K = None
#  k: Minimum number of neighbours k-nn estimation. Default (or None) is 10.
P_KNN_CONST = None
#  Multiplier of default number of observation used in moving average of analyses method.
#  Default is 1.
P_NW_BANDW = None
#  Bandwidth for nw estimation; multiplier of Silverman's optimal bandwidth. Default (or None) is 1.
P_NW_KERN = None
#  Kernel for nw estimation: 1: Epanechikov (def); 2: normal. Default (or None) is 1.

# - - - - - - - - - - - - Causal effects to be estimated - - - - - - - - - - - - - - - - - - - - - -
# Bootstrap of standard errors (P_SE_BOOT_ATE, P_SE_BOOT_GATE, P_SE_BOOT_IATE, P_SE_BOOT_IATE)
# Specify either a Boolean (if True, number of bootstrap replications will be set to 199) or an
# integer corresponding to the number of bootstrap replications; this implicity implies True). The
# default is False if CLUSTER_STD is False and True with 199 replications CLUSTER_STD is True.
# (w_ji * y_i) are bootstrapped SE_BOOT_xATE times.
P_SE_BOOT_ATE = None
P_SE_BOOT_GATE = None
P_SE_BOOT_IATE = None
P_SE_BOOT_QIATE = None
# False: No Bootstrap SE of effects. True: SE_BOOT_xATE = 199.

# GATE estimation of variables with many values
P_MAX_CATS_Z_VARS = None
#  Maximum number of categories for discretizing continuous z variables. Default (or None) is n**.3.

# Computation of standard error for ATE
P_ATE_NO_SE_ONLY = None
#  True: Computes only the ATE without standard errors. Default (or None) is False.

# Treatment-group specific effects
P_ATET = None
#  True: Average effects computed for subpopulations by treatments (if available). Default is False.
P_GATET = None
#  True: Gate's for subpopulations by treatments. Default is False. If there no variables specified
#  for gate estimation, p_bgate is set to False.

#  Balanced GATEs
P_BGATE = None
#  True: BGATEs will be computed. True requires to specify the variable names to
#  balance on in VAR_BGATE_NAME.
#  If no variables are specified for gate estimation, p_bgate is set to False.
#  Default is False.

P_CBGATE = None
#  True: CBGATEs will be computed. Default is False.
#  If there are no variables specified for gate estimation, p_cbgate is set to False.

#  More details for (CB)GATE estimation
#  Number of evluation points for continuous variables (GATE, BGATE, CBGATE).
P_GATES_NO_EVALU_POINTS = None
#  Default is 50.

# Alternative ways to estimate GATEs for continuous variables
P_GATES_SMOOTH = None
#  Instead of discretizing variable, its GATE is evaluated at p_gates_smooth_no_evalu_points. Since
#  there are likely to be no observations, a local neighbourhood around the evaluation points is
#  considered. Default is True.
P_GATES_SMOOTH_BANDWIDTH = None
#  Multiplier for SGATE aggregation. Def. is 1.
P_GATES_SMOOTH_NO_EVALU_POINTS = None
#  Number of evaluation points. Default is 50.

# More parameters for all (B)GATE(T)s
# GATEs will not only be compared to ATE but also to GATEs computed at previous evaluation point.
P_GATES_MINUS_PREVIOUS = None
#  Default (or None) is False. If True, GATE estimation is slower as it is not optimized for
#  multiprocessing and no plots are shown for this parameter.

# How much data to use for computing BGATEs
P_BGATE_SAMPLE_SHARE = None
#  (0<1) Implementation is very cpu intensive. Random samples are used to speed up programme if more
#  obs / # of evaluation points > 10. # Default is 1 if n_prediction < 1000; otherwise:
#  (1000 + (n_pred-1000) ** (3/4))) / # of evaluation points

# Estimation of IATEs and their standard errors
# Estimate IATEs
P_IATE = None
#  Default is True.

# Estimate standard errors of IATEs
P_IATE_SE = None  # None
# Default is False. Note: Estimating IATEs and their standard errors may be time consuming.

# Estimate IATE minus ATE
P_IATE_M_ATE = None
#  True: IATE(x) - ATE is estimated, including inference if p_iate_se == True. Increaes computation
#  time. Requires _int_low_memory_predict == False .
#  Default is False.

# Estimation of QIATEs and their standard errors (see Kutz, Lechner, 2025)
# Estimate QIATEs
P_QIATE = None
#  If True, p_iate will always be set to True. Requires _int_low_memory_predict == False .
#  Default is False.

# SE(QIATE) will be estimated.
P_QIATE_SE = None
#  Default is False.

# QIATE(x) - median(IATE(x)) is estimated
P_QIATE_M_MQIATE = None
#  Including inference if p_qiate_se == True. Increaes computation time. Default is False.

# QIATE(x, q) - QIATE(x, 1-q) is estimated
P_QIATE_M_OPP = None
#  Including inference if p_qiate_se == True. Increaes computation time. Default is False.

# Number of quantiles for which QIATEs are computed
P_QIATE_NO_OF_QUANTILES = None
#  Default is 99.

# Smooth estimated QIATEs using kernel smoothing
P_QIATE_SMOOTH = None
# Default is True.

# Multiplier applied to default bandwidth used for kernel smoothing of QIATE
P_QIATE_SMOOTH_BANDWIDTH = None
#  Default is 1.

# Bias correction procedure for QIATEs based on simulations.
P_QIATE_BIAS_ADJUST = None
#  Default is False. If P_QIATE_BIAS_ADJUST is True, P_IATE_SE is set to True as well.

# --------------- Bias adjustments using internal mcf weights (experimental!) ----------------------
P_BA = None
#  If True, bias adjustment is used. Default is False.

# Use propensity score as a regressor
P_BA_USE_PROP_SCORE = None
#  Default is True.

# Use prognostic scores are used as regressors
P_BA_USE_PROG_SCORE = None
#  Default is True.

# Use variables specified in VAR_X_BA_NAME as regressors
P_BA_USE_X = None
#  Default is False.

# Use weighted ridge regression or weighted OLS
P_BA_RIDGE = None
# If True use weighted ridge, otherwise  use weighted OLS. Default is True.

# Type of adjustment used.
P_BA_ADJ_METHOD = None
#  Possible methods are 'zeros', 'train_obs', 'weighted_train_obs', 'pred_obs'. Default is
# 'weighted_train_obs'. Defines how to evaluate estimated regressions in the adjustment procedures:
# 'zeros': The values of the (centered) covariates are set to zero.
# 'train_obs': They are set to their empirical distribution for the training data (unconditional on
#              treatment).
# 'weighted_train_obs': As observables, but observations are weighted given the weights from the
#              forests (across treatments).  This imposes some localness on the X-distribution and
#              still removes the impact of treatment control differences of X-values in the leaves.
# 'pred_obs': Unweighted prediction features are used to evaluate the respective regressions. This
#              option is not yet implemented.

# Force all adjusted weights to be positive.
P_BA_POS_WEIGHTS_ONLY = None
#  Default is False.

# Parameters deterimining how to compute the prognostic score
# (only relevantif P_BA_USE_PROP_SCORE == True)
P_BA_ESTIMATOR = None
#  Estimator used for computing the prognostic score. Possible choices are scikit-learn's
#  regression methods 'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5'
#  'SupportVectorMachine', 'SupportVectorMachineC2', 'SupportVectorMachineC4', 'AdaBoost',
#  'AdaBoost100', 'AdaBoost200', 'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12',
#  'LASSO', (if more than 1000 observations),  'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
#  'Mean'. (Mean is included if other estimators are not useful.)
#  If set to 'automatic', the estimator with the lowest out-of-sample mean squared error (MSE)
#  is selected. Whether this selection is based on cross-validation or a test sample is governed by
#  LC_CS_CV. The default (or None) is 'RandomForest'.

P_BA_CV_K = None
# if LC_CS_CV: # of folds used in crossvalidation. Default
#  depends on the size of the training sample (N): N < 100'000: 5.
#                                       100'000 <= N < 250'000: 4
#                                       250'000 <= N < 500'000: 3
#                                       500'000 <= N: 2

# ----------------------------- Analysis of effects ------------------------------------------------
# Analyses the predictions by binary correlations or some regression type methods
POST_EST_STATS = None
#  Default is True. False if p_iate == False.

# Use only effects relative to treatment with lowest treatment value.
POST_RELATIVE_TO_FIRST_GROUP_ONLY = None
#  Default is True.

# Checking the binary correlations of predictions with features
POST_BIN_CORR_YES = None
#  Default is True.

# Minimum threshhold of absolute correlation to be displayed
POST_BIN_CORR_THRESHOLD = None
#  Default is 0.1.

# Plots of estimated treatment effects.
POST_PLOTS = None
#  Default is True.

# Using k-means clustering to analyse patterns in the estimated effects
POST_KMEANS_YES = None
#  Default is True.
POST_KMEANS_SINGLE = None
# If True, clustering is also with respect to all single effects. Default is False.
POST_KMEANS_NO_OF_GROUPS = None  # post_kmeans_yes is True: # of clusters to be built:
#   Integer, list or tuple (or None --> default).
#   Default: List of 5 values: [a, b, c, d, e]; c = 5 to 10; depending on n;
#   c<7: a=c-2, b=c-1, d=c+1, e=c+2 else a=c-4, b=c-2, d=c+2, e=c+4.
POST_KMEANS_REPLICATIONS = None
#  if post_kmeans_yes is True: number of replications with random start centers to avoid local
#  extrema. Default is 10.
POST_KMEANS_MAX_TRIES = None
#  id post_kmeans_yes is True: maximum number of iterations in each replication to achive
#  convergence. Default is 1000.
POST_KMEANS_MIN_SIZE_SHARE = None
#  Smallest share of cluster size allowed in % (0-33). Default (None) is 1.
POST_RANDOM_FOREST_VI = None
#  Variable importance measure of predictive random forest used to learn factors influencing IATEs.
#  Default is True.
POST_TREE = None
#  Regression trees (honest and standard) of Depth 2 to 5 are estimated to describe IATES(x).
#  Default (or None) is True.

# ------------------------- Sensitivity method (experimental) -------------------------------------
SENS_CBGATE = None
#  Boolean (or None), optional. Compute CBGATEs for sensitivity analysis. Default is False.
SENS_BGATE = None
#  Boolean (or None), optional.  Compute BGATEs for sensitivity analysis. Default is False.
SENS_GATE = None
#  Boolean (or None), optional. Compute GATEs for sensitivity analysis. Default is False.
SENS_IATE = None
#  Boolean (or None), optional. Compute IATEs for sensitivity analysis. If the results dictionary is
#  passed, and it contains IATEs, then the default value is True, and False otherwise.
SENS_IATE_SE = None
#  Boolean (or None), optional. Compute Standard errors of IATEs for sensitivity analysis.
#  Default is False.
SENS_SCENARIOS = None
#  List or tuple of strings, optional. Different scenarios considered.
#  'basic' : Use estimated treatment probabilities for simulations. No confounding.
#  Default is ('basic',).
SENS_CV_K = None
#  Integer (or None), optional. Data to be used for any cross-validation:
#  Number of folds in cross-validation. Default (or None) is 5.
SENS_REPLICATIONS = None
#  Integer (or None), optional. Number of replications for simulating placebo treatments.
#  Default is 2.
SENS_REFERENCE_POPULATION = None
#  Integer or float (or None). Defines the treatment status of the reference population used by the
#  sensitivity analysis. Default is to use the treatment with most observed observations.

# ----- Internal variables: Change these variables only if you know what you do --------------------
_INT_MP_USE_OLD_RAY = None
#  Use 'old' (and stable) implementation of Ray for multiprocessing.
#  Default is False.

# The following _int_mp_* keywords are only relevant if int_mp_use_old_ray is False
_INT_MP_BACKEND = None     # Backend to be used for parallelisation.
#  Possible values 'ray' or 'joblib' or 'sequential'.
#  Note that joblib cannot handle more than 62 processes in parallel.
#  The default for Windows is 'joblib', else 'ray'. If there are more than 250'000 training
#  observations (that can be used to build the forest), then the default for Windows is 'ray' as
#  well.

_INT_MP_MEMMAP_MIN_BYTES = None     # Minimum size of objects to use memory maps in joblib.
#  Default is 4 * 1024 * 1024

_INT_MP_MEMMAP_DIR = None           # Tempory path to store memory maps. Will be removed when
#  computations are finished. Default is Path.cwd() / 'joblibtemp'.

_INT_MP_BATCHES = None        # Number of batches used when running multiprocessing (all backends).
#  This partly determines memory consumption. Possible values are 'automatic' or positive integers.
#  'automatic' means that the number of batches are set to minimize memory consumptions (at the cost
#  of sometimes some speed reduction), by setting them to ceil(no_of_tasks/number_of_workers)
#  Default is 'automatic'.

#  Low memory version used for effect computation (prediction)
_INT_LOW_MEMORY_PREDICT = None
#  If True, the memory footprint of the prediction step
#  will be drastically reduced (and computational speed significantly increased) by not keeping
#  the full weight matrix. This will allow for deeper forests and more prediction data points that
#  can reasonably be used when training data is very large. Therefore, the defaults for
#  _int_max_obs_prediction and cf_chunks_maxsize
#  As of now, this option is incompatible with instrumental variable estimation and the estimation
#  of QIATEs. p_iate_m_ate must also be set to False if _int_low_memory_predict is True.
#  Default is True.

_INT_LOW_MEMORY_MAX_CHUNKSIZE = None
#  Maximum number of prediction observations that are jointly computed by a single process.
#  Only relevant if _int_low_memory_predict is True.
#  Default (or None) is 1'000 - (N_training - 10'000)**0.5. Minimum is 10. Maximum is 1000.

# Use CUDA based GPU if available on hardware.
_INT_CUDA = None
#  Experimental feature. Default is False.

# Restrict multiprocessing
_INT_REPLICATION = None
#  True does not allow multiprocessing in local centering, feature selection, and common support.
#  Default is False.

# True: Provide information for McfOptPolReports to construct informative reports.
_INT_REPORT = None
#  Default is True.

# Print output on txt file and/or console.
_INT_WITH_OUTPUT = None
#  Default is True.

# Do not create a new directory when the path already exists.
_INT_OUTPUT_NO_NEW_DIR = None
#  Default is False.

 # Additional output about running of programme
_INT_VERBOSE = None
#  Default is True.

# Information about memory usage in some memory intensive steps of algorithm
_INT_MEMORY_PRINT = None
#  Default is False

# Print descriptive stats of input+output files controls for all figures
_INT_DESCRIPTIVE_STATS = None

# Parameters for figures
_INT_SHOW_PLOTS = None
#  Execute plt.show() command. Default is True.
_INT_FONTSIZE = None
#  Legend, 1(very small) to 7(very large); def:2
_INT_DPI = None
#  > 0: default (or None): 500.

# Only for (B, AM) GATEs: What type of plot to use for continuous variables.
_INT_NO_FILLED_PLOT = None      # Use filled plot if more than x points(def:20)

# Weights matrix stored as sparse matrix.
_INT_WEIGHT_AS_SPARSE = None
#  Default (or None) is True.

# Number of IATEs that are estimated in a ray worker
_INT_IATE_CHUNK_SIZE = None
#  If _INT_LOW_MEMORY_PREDICT is True, then the this is set to _INT_LOW_MEMORY_MAX_CHUNKSIZE.
#  If _INT_LOW_MEMORY_MAX is False, the default is number of prediction observations / workers.
#  If programme crashes when computing IATEs, reduce this number.

# Sparse weight matrix computed in several chunks
_INT_WEIGHT_AS_SPARSE_SPLITS = None
#  If _INT_LOW_MEMORY_PREDICT is False, then the default is
#  Rows of prediction data * rows of Fill_y data / number of forests / (25'000 * 25'000))
#  If _INT_LOW_MEMORY_PREDICT is True, then the default is 1.

# Share of sample used to build forest
_INT_SHARE_FOREST_SAMPLE = None
#  0-1. Default is 0.5.

# Discretising of continuous variables
_INT_MAX_CATS_CONT_VARS = None
# Maximum number of categories for continuous variables n values < n speed up programme.
# Default is False.

# Save values of x only if < 50 (cont. vars).
_INT_MAX_SAVE_VALUES = None
#  Default is 50.

# Seeding is redone when building forest
_INT_SEED_SAMPLE_SPLIT = None
#  Default is 67567885.

# Variable importance: type of mp
_INT_MP_VIM_TYPE = None
#  1: variable based (fast, lots of memory)
#  2: bootstrap based (slower, less memory)
#  Default: 1 if n < 20000, 2 otherwise

# Weights computation: type of mp (relevant if )
_INT_MP_WEIGHTS_TYPE = None
#  1: groups-of-obs based (fast, lots of memory)
#  2: tree based (takes forever, less memory)
#  Default is 1. Variable is overwritten (set to 1 if multiprocessing is used).

# Weight computation: Split forests for variable importance computations
_INT_MP_WEIGHTS_TREE_BATCH = None
#  Few batches: More speed, more memory. Default: Automatically determined. Soon to be depreciated.
#  Only relevant if _INT_LOW_MEMORY_PREDICT is False.

# Ray multiprocessing
_INT_MP_RAY_DEL = None
# Tuple with any of the following:
#   'refs': Delete references to object store (default)
#   'rest': Delete all other objects of Ray task
#   'none': Nothing is deleted.
#   These 3 options can be combined.
#  Default (or None) is 'refs'

_INT_MP_RAY_SHUTDOWN = None
#  Shutdown ray task by task (default is False if N < 100'000 and True otherwise)
#  If programme collapses because of excess memory reduce workers or set _INT_MP_RAY_SHUTDOWN is
#  True.  When using this programme repeatedly like in Monte Carlo studies always using
#  _INT_MP_RAY_SHUTDOWN is True may be a good idea.

_INT_MP_RAY_OBJSTORE_MULTIPLIER = None
#  Increase internal default values for Ray object store above 1 if programme crashes because
#  object store is full. Default is 1. ONLY RELEVANT if _INT_MP_RAY_SHUTDOWN is True.

# Return all data with predictions despite with_output = False
_INT_RETURN_IATE_SP = None
#  Useful for cross-validation and simulations. Default is False.

# Delete forests from instance
_INT_DEL_FOREST = None
#  If True, less memory is needed, but the trained instance of the class cannot be reused when
#  calling predict with the same instance again, i.e. the forest has to be retrained.
#  Default is False.

# Keep all zeros weights when computing standard errors (slows down computation)
_INT_KEEP_W0 = None
#  Default is False.

# The following keywords define upper limits for sample size. If actual number is larger then the
# prespecified number, then the respective data will randomly reduced to the specified upper limit.
_INT_OBS_BIGDATA = None
#  If number of training observations is larger than _INT_OBS_BIGDATA, the following happens
#  during training and prediction:
#    (i) Number of workers is halved in local centering.
#    (ii) The number of workers used is reduced to 75% of default.
#    (iii) The data type for many numpy arrays is reduced from float64 to
#         float32.
#    Default (or None) is 1'000'000.

# Maximum number of observations used in training
_INT_MAX_OBS_TRAINING = None
#  Note: Reducing observations for training (increases MSE and thus should be avoided).
#  Default (or None) is infinity.

# Maximum number of observations used in prediction
_INT_MAX_OBS_PREDICTION = None
#  Reducing observations for prediction does not much affect MSE. It may reduce detectable
#  heterogeneity, but may also dramatically reduce computation time.
#  Default (or None) is 250'000 if _int_low_memory_predict is False and 1'000'000 otherwise.

# Reduce observations when computing kmeans in analye method
_INT_MAX_OBS_KMEANS = None
#  Reducing observations may reduce detectable heterogeneity, but also reduces computation time.
#  Default (or None) is 200'000.

# Figures showing relation of IATEs and features
_INT_MAX_OBS_POST_REL_GRAPHS = None
#  Note: In-built non-parametric regression is computationally intensive.  Default is 50'000.

# --------------------------------------------------------------------------------------------------
# For convenience the mcf parameters are collected and passed as a dictionary. Of course, they can
# also be passed as single parameters (or not at all, in which case default values are used).

params = {'cf_alpha_reg_grid': CF_ALPHA_REG_GRID,
          'cf_alpha_reg_max': CF_ALPHA_REG_MAX, 'cf_alpha_reg_min': CF_ALPHA_REG_MIN,
          'cf_boot': CF_BOOT, 'cf_chunks_maxsize': CF_CHUNKS_MAXSIZE,
          'cf_compare_only_to_zero': CF_COMPARE_ONLY_TO_ZERO, 'cf_n_min_grid': CF_N_MIN_GRID,
          'cf_n_min_max': CF_N_MIN_MAX, 'cf_n_min_min': CF_N_MIN_MIN,
          'cf_n_min_treat': CF_N_MIN_TREAT, 'cf_nn_main_diag_only': CF_NN_MAIN_DIAG_ONLY,
          'cf_m_grid': CF_M_GRID, 'cf_m_share_max': CF_M_SHARE_MAX,
          'cf_m_random_poisson': CF_M_RANDOM_POISSON, 'cf_m_share_min': CF_M_SHARE_MIN,
          'cf_match_nn_prog_score': CF_MATCH_NN_PROG_SCORE, 'cf_mce_vart': CF_MCE_VART,
          'cf_random_thresholds': CF_RANDOM_THRESHOLDS, 'cf_p_diff_penalty': CF_P_DIFF_PENALTY,
          'cf_penalty_type': CF_PENALTY_TYPE, 'cf_subsample_factor_eval': CF_SUBSAMPLE_FACTOR_EVAL,
          'cf_subsample_factor_forest': CF_SUBSAMPLE_FACTOR_FOREST, 'cf_tune_all': CF_TUNE_ALL,
          'cf_vi_oob_yes': CF_VI_OOB_YES,

          'cs_adjust_limits': CS_ADJUST_LIMITS, 'cs_max_del_train': CS_MAX_DEL_TRAIN,
          'cs_min_p': CS_MIN_P, 'cs_quantil': CS_QUANTIL, 'cs_type': CS_TYPE,

          'ct_grid_dr': CT_GRID_DR, 'ct_grid_nn': CT_GRID_NN, 'ct_grid_w': CT_GRID_W,

          'dc_check_perfectcorr': DC_CHECK_PERFECTCORR, 'dc_clean_data': DC_CLEAN_DATA,
          'dc_min_dummy_obs': DC_MIN_DUMMY_OBS, 'dc_screen_covariates': DC_SCREEN_COVARIATES,

          'fs_rel_vi_keep_if': FS_REL_VI_KEEP_IF,
          'fs_other_sample': FS_OTHER_SAMPLE, 'fs_other_sample_share': FS_OTHER_SAMPLE_SHARE,
          'fs_rel_vi_threshold_y': FS_REL_VI_THRESHOLD_Y,
          'fs_rel_vi_threshold_d': FS_REL_VI_THRESHOLD_D, 'fs_yes': FS_YES,

          'gen_ate_eff': GEN_ATE_EFF, 'gen_d_type': GEN_D_TYPE, 'gen_gate_eff': GEN_GATE_EFF,
          'gen_iate_eff': GEN_IATE_EFF, 'gen_panel_data': GEN_PANEL_DATA,
          'gen_panel_in_rf': GEN_PANEL_IN_RF, 'gen_qiate_eff': GEN_QIATE_EFF,
          'gen_weighted': GEN_WEIGHTED, 'gen_mp_parallel': GEN_MP_PARALLEL,
          'gen_outfiletext': GEN_OUTFILETEXT, 'gen_outpath': GEN_OUTPATH,
          'gen_output_type': GEN_OUTPUT_TYPE,

          'lc_cs_cv': LC_CS_CV, 'lc_cs_cv_k': LC_CS_CV_K, 'lc_cs_share': LC_CS_SHARE,
          'lc_estimator': LC_ESTIMATOR, 'lc_uncenter_po': LC_UNCENTER_PO, 'lc_yes': LC_YES,

          'post_bin_corr_threshold': POST_BIN_CORR_THRESHOLD,
          'post_bin_corr_yes': POST_BIN_CORR_YES, 'post_est_stats': POST_EST_STATS,
          'post_kmeans_no_of_groups': POST_KMEANS_NO_OF_GROUPS,
          'post_kmeans_max_tries': POST_KMEANS_MAX_TRIES,
          'post_kmeans_replications': POST_KMEANS_REPLICATIONS, 'post_kmeans_yes': POST_KMEANS_YES,
          'post_kmeans_single': POST_KMEANS_SINGLE,
          'post_kmeans_min_size_share': POST_KMEANS_MIN_SIZE_SHARE,
          'post_random_forest_vi': POST_RANDOM_FOREST_VI,
          'post_relative_to_first_group_only': POST_RELATIVE_TO_FIRST_GROUP_ONLY,
          'post_plots': POST_PLOTS, 'post_tree': POST_TREE,

          'p_ate_no_se_only': P_ATE_NO_SE_ONLY, 'p_cbgate': P_CBGATE, 'p_atet': P_ATET,
          'p_bgate': P_BGATE, 'p_bt_yes': P_BT_YES,
          'p_choice_based_sampling': P_CHOICE_BASED_SAMPLING,
          'p_choice_based_probs': P_CHOICE_BASED_PROBS, 'p_ci_level': P_CI_LEVEL,
          'p_cluster_std': P_CLUSTER_STD, 'p_cond_var': P_COND_VAR,
          'p_gates_minus_previous': P_GATES_MINUS_PREVIOUS, 'p_gates_smooth': P_GATES_SMOOTH,
          'p_gates_smooth_bandwidth': P_GATES_SMOOTH_BANDWIDTH,
          'p_gates_smooth_no_evalu_points': P_GATES_SMOOTH_NO_EVALU_POINTS,
          'p_gatet': P_GATET, 'p_gates_no_evalu_points': P_GATES_NO_EVALU_POINTS,
          'p_bgate_sample_share': P_BGATE_SAMPLE_SHARE,
          'p_iate': P_IATE, 'p_iate_se': P_IATE_SE, 'p_iate_m_ate': P_IATE_M_ATE, 'p_knn': P_KNN,
          'p_knn_const': P_KNN_CONST, 'p_knn_min_k': P_KNN_MIN_K, 'p_nw_bandw': P_NW_BANDW,
          'p_nw_kern': P_NW_KERN, 'p_max_cats_z_vars': P_MAX_CATS_Z_VARS,
          'p_max_weight_share': P_MAX_WEIGHT_SHARE, 'p_qiate': P_QIATE, 'p_qiate_se': P_QIATE_SE,
          'p_qiate_m_mqiate': P_QIATE_M_MQIATE, 'p_qiate_m_opp': P_QIATE_M_OPP,
          'p_qiate_no_of_quantiles': P_QIATE_NO_OF_QUANTILES, 'p_qiate_smooth': P_QIATE_SMOOTH,
          'p_qiate_smooth_bandwidth': P_QIATE_SMOOTH_BANDWIDTH,
          'p_qiate_bias_adjust': P_QIATE_BIAS_ADJUST, 'p_se_boot_ate': P_SE_BOOT_ATE,
          'p_se_boot_gate': P_SE_BOOT_GATE, 'p_se_boot_iate': P_SE_BOOT_IATE,
          'p_se_boot_qiate': P_SE_BOOT_QIATE, 'p_ba': P_BA,

          'p_ba_adj_method': P_BA_ADJ_METHOD, 'p_ba_pos_weights_only': P_BA_POS_WEIGHTS_ONLY,
          'p_ba_use_prop_score': P_BA_USE_PROP_SCORE, 'p_ba_use_prog_score': P_BA_USE_PROG_SCORE,
          'p_ba_use_x': P_BA_USE_X, 'p_ba_estimator': P_BA_ESTIMATOR, 'p_ba_cv_k': P_BA_CV_K,
          'p_ba_ridge': P_BA_RIDGE,
          'var_cluster_name': VAR_CLUSTER_NAME, 'var_d_name': VAR_D_NAME,
          'var_id_name': VAR_ID_NAME, 'var_w_name': VAR_W_NAME,
          'var_x_name_balance_test_ord': VAR_X_NAME_BALANCE_TEST_ORD,
          'var_x_name_balance_test_unord': VAR_X_NAME_BALANCE_TEST_UNORD,
          'var_x_name_balance_bgate': VAR_X_NAME_BALANCE_BGATE, 'var_x_name_ba': VAR_X_NAME_BA,
          'var_x_name_always_in_ord': VAR_X_NAME_ALWAYS_IN_ORD,
          'var_x_name_always_in_unord': VAR_X_NAME_ALWAYS_IN_UNORD,
          'var_x_name_remain_ord': VAR_X_NAME_REMAIN_ORD,
          'var_x_name_remain_unord': VAR_X_NAME_REMAIN_UNORD,
          'var_x_name_ord': VAR_X_NAME_ORD, 'var_x_name_unord': VAR_X_NAME_UNORD,
          'var_y_name': VAR_Y_NAME, 'var_y_tree_name': VAR_Y_TREE_NAME,
          'var_z_name_cont': VAR_Z_NAME_CONT, 'var_z_name_ord': VAR_Z_NAME_ORD,
          'var_z_name_unord': VAR_Z_NAME_UNORD,

          '_int_cuda': _INT_CUDA, '_int_del_forest': _INT_DEL_FOREST,
          '_int_descriptive_stats': _INT_DESCRIPTIVE_STATS, '_int_dpi': _INT_DPI,
          '_int_fontsize': _INT_FONTSIZE, '_int_iate_chunk_size': _INT_IATE_CHUNK_SIZE,
          '_int_keep_w0': _INT_KEEP_W0,
          '_int_low_memory_predict': _INT_LOW_MEMORY_PREDICT,
          '_int_low_memory_max_chunksize': _INT_LOW_MEMORY_MAX_CHUNKSIZE,
          '_int_no_filled_plot': _INT_NO_FILLED_PLOT,
          '_int_max_cats_cont_vars': _INT_MAX_CATS_CONT_VARS,
          '_int_max_save_values': _INT_MAX_SAVE_VALUES,
          '_int_max_obs_training': _INT_MAX_OBS_TRAINING,
          '_int_max_obs_prediction': _INT_MAX_OBS_PREDICTION,
          '_int_max_obs_kmeans': _INT_MAX_OBS_KMEANS,
          '_int_max_obs_post_rel_graphs': _INT_MAX_OBS_POST_REL_GRAPHS,
          '_int_mp_use_old_ray': _INT_MP_USE_OLD_RAY,
          '_int_mp_backend': _INT_MP_BACKEND,
          '_int_mp_memmap_min_bytes': _INT_MP_MEMMAP_MIN_BYTES,
          '_int_mp_memmap_dir': _INT_MP_MEMMAP_DIR,
          '_int_mp_batches': _INT_MP_BATCHES,
          '_int_mp_ray_del': _INT_MP_RAY_DEL,
          '_int_mp_ray_objstore_multiplier': _INT_MP_RAY_OBJSTORE_MULTIPLIER,
          '_int_mp_ray_shutdown': _INT_MP_RAY_SHUTDOWN, '_int_mp_vim_type': _INT_MP_VIM_TYPE,
          '_int_mp_weights_tree_batch': _INT_MP_WEIGHTS_TREE_BATCH,
          '_int_mp_weights_type': _INT_MP_WEIGHTS_TYPE,
          '_int_obs_bigdata': _INT_OBS_BIGDATA, '_int_report': _INT_REPORT,
          '_int_output_no_new_dir': _INT_OUTPUT_NO_NEW_DIR, '_int_replication': _INT_REPLICATION,
          '_int_return_iate_sp': _INT_RETURN_IATE_SP,
          '_int_memory_print': _INT_MEMORY_PRINT,
          '_int_seed_sample_split': _INT_SEED_SAMPLE_SPLIT,
          '_int_share_forest_sample': _INT_SHARE_FOREST_SAMPLE, '_int_show_plots': _INT_SHOW_PLOTS,
          '_int_verbose': _INT_VERBOSE, '_int_weight_as_sparse': _INT_WEIGHT_AS_SPARSE,
          '_int_weight_as_sparse_splits': _INT_WEIGHT_AS_SPARSE_SPLITS,
          '_int_with_output': _INT_WITH_OUTPUT
          }
# These are additional parameters used in the sensitivity analysis
params_sensitivity = {'sens_cbgate': SENS_CBGATE, 'sens_bgate': SENS_BGATE, 'sens_gate': SENS_GATE,
                      'sens_iate': SENS_IATE, 'sens_iate_se': SENS_IATE_SE,
                      'sens_scenarios': SENS_SCENARIOS, 'sens_cv_k': SENS_CV_K,
                      'sens_replications': SENS_REPLICATIONS,
                      'sens_reference_population': SENS_REFERENCE_POPULATION
                      }
# These parameters can be changed in the prediction and analysis module
# (it enough to change them only in the predict). Here, as a demonstration,
# parameters are set to arbitrary values, including their defaults.
# Note that 'None' is not a valid value for any for these parameters.

params_predict_analysis_2nd_round = {
    # If feature selection is active: Be careful not to select variables that have been removed by
    #     feature selection. If so, the programme will end with an exception.
    'var_x_name_balance_test_ord': name_dict['x_name_ord'][:1],
    'var_x_name_balance_test_unord': name_dict['x_name_unord'][:1],
    'var_x_name_balance_bgate': name_dict['x_name_ord'][3:6],
    'var_x_name_ba': (name_dict['x_name_ord'][0], *name_dict['x_name_unord'][:2],),
    'var_z_name_ord': name_dict['x_name_ord'][-3:],
    'var_z_name_unord': name_dict['x_name_unord'][:1],

    'cs_type': 0,  # Can only be changed to 0  (use only if prediction data are on common support)

    'p_ba': False, 'p_ba_adj_method': 'zeros', 'p_ba_pos_weights_only': False, 'p_ba_use_x': False,
    'p_ba_use_prop_score': False, 'p_ba_use_prog_score': False,
    'p_ate_no_se_only': False, 'p_atet': True, 'p_gatet': False,

    'p_bgate': True, 'p_cbgate': True, 'p_iate': True, 'p_iate_se': True, 'p_iate_m_ate': False,
    'p_bgate_sample_share': 0.5, 'p_gates_minus_previous': False, 'p_gates_smooth_bandwidth': 1,
    'p_gates_smooth': True, 'p_gates_smooth_no_evalu_points': 50, 'p_gates_no_evalu_points': 50,
    'p_qiate': False, 'p_qiate_se': False, 'p_qiate_m_mqiate': False, 'p_qiate_m_opp': False,
    'p_qiate_no_of_quantiles': 50, 'p_qiate_smooth': True, 'p_qiate_smooth_bandwidth': 1,
    'p_qiate_bias_adjust': False, 'p_bt_yes': False,  'p_choice_based_sampling': False,
    'p_cond_var': True, 'p_knn': True, 'p_knn_const': 1,
    'p_knn_min_k': 5, 'p_nw_bandw': 1, 'p_nw_kern': 1, 'p_ci_level': 0.90,
    'p_se_boot_ate': False, 'p_se_boot_gate': False, 'p_se_boot_iate': False,
    'p_se_boot_qiate': False, # 'p_iv_aggregation_method':  IV only

    'gen_output_type': 2,

    'post_bin_corr_threshold': 0.1, 'post_bin_corr_yes': True, 'post_est_stats': True,
    'post_kmeans_yes': False, 'post_kmeans_no_of_groups': 5, 'post_kmeans_max_tries': 10,
    'post_kmeans_min_size_share': 0.1, 'post_kmeans_replications': 100, 'post_kmeans_single': False,
    'post_random_forest_vi': False, 'post_relative_to_first_group_only': True, 'post_plots': False,
    'post_tree': False, 
    }
if NO_TREATMENTS == 2:  # Valid specification depends on number of treatments
    params_predict_analysis_2nd_round['p_choice_based_probs'] = (0.3, 0.7)
elif NO_TREATMENTS == 3:
    params_predict_analysis_2nd_round['p_choice_based_probs'] = (0.1, 0.2, 0.7)

# ------------------------------------------------------------------------------
# Modules may send many irrelevant warnings: Globally ignore them
warnings.filterwarnings('ignore')

# Some data to be used by the method
zeros_df = pd.DataFrame(np.zeros(len(prediction_df)), columns=('zeros',))
ones_df = pd.DataFrame(np.ones(len(prediction_df)), columns=('ones',))

if NO_TREATMENTS < 3:
    test_alloc_df = pd.concat((prediction_df[VAR_D_NAME], zeros_df, ones_df, ), axis=1)
else:
    twos_df = pd.DataFrame(np.ones(len(prediction_df))*2, columns=('twos',))
    test_alloc_df = pd.concat((prediction_df[VAR_D_NAME], zeros_df, ones_df, twos_df,), axis=1)

test_alloc_df.rename(columns={'treat': 'observed'}, inplace=True)
# ------------------------------------------------------------------------------

mymcf = ModifiedCausalForest(**params)

mymcf.train(training_df)
# Copy mymcf for later use with changed keyword values (later) as mymcf will be modified by the
# predict method (see the BGATE example for an alternative way to do this)
mymcf_2 = deepcopy(mymcf)

# Continue using the keyword values as specified when initializing the instance
results = mymcf.predict(prediction_df)
results_with_cluster_id_df = mymcf.analyse(results)

if GEN_WEIGHTED is not True and GEN_D_TYPE != 'continuous':
    mymcf.predict_different_allocations(prediction_df, test_alloc_df)

if GEN_D_TYPE != 'continuous' and not FS_YES:
    # Sensitivity (experimental)
    params['gen_outpath'] = results['path_output'] / 'sensitivity'
    mymcf_sens = ModifiedCausalForest(**params)
    params_sensitivity['results'] = results
    mymcf_sens.sensitivity(training_df, prediction_df, **params_sensitivity) # pylint: disable=E1125

    # Write report in pdf file
    my_report = McfOptPolReport(mcf=mymcf, mcf_sense=mymcf_sens)
else:
    my_report = McfOptPolReport(mcf=mymcf)

my_report.report()

if GEN_D_TYPE != 'continuous' and not FS_YES:
    # Use new values of some keywords as specified in dictionary with copied instance
    results_2 = mymcf_2.predict(prediction_df, new_keywords=params_predict_analysis_2nd_round,)
    results_with_cluster_id_df_2 = mymcf_2.analyse(results_2)

    my_report2 = McfOptPolReport(mcf=mymcf_2)
    my_report2.report()

print('End of computations.\n\nThanks for using the ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
