"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economic Research
University of St. Gallen, Switzerland

Version: 0.11.0

Comprehensive configuration example for ModifiedCausalForest.
Comments summarise the current API; see the API for full validation rules.
Instrumental-variable and treatment-version options have separate examples.

"""
from copy import deepcopy
from pathlib import Path

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

PREDDATA_IS_TRAINDATA = False  # If True, the same data will be used for training and prediction
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
    prediction_df = training_df.copy()

# ------------------ Parameters of ModifiedCausalForest ---------------------------------------
# Values below are passed explicitly to the constructor. None usually selects an automatic or
# default setting, but is not always equivalent to omitting a keyword (see FS_MSE_FOR_CLASSIFIER).
# Variable names are case-insensitive; do not use columns differing only in capitalization.

# Output paths
# ------------
# Output directory. None uses Path.cwd() / 'output'. Missing directories are created; nonempty
# directories normally get a numeric suffix.
GEN_OUTPATH = Path.cwd() / 'example/output'

# Base name of the detailed .txt and _Summary.txt files. None selects 'txtFileWithOutput'.
# Existing files with these names are deleted on initialization.
GEN_OUTFILETEXT = 'mcf.py.0.11.0'

# Variables for training and prediction
# -------------------------------------
# Treatment variable. Discrete labels are recoded internally. Prediction data must use the
# original training labels; unseen labels are rejected.
VAR_D_NAME = name_dict['d_name']

# Treatment type: 'discrete' (default) or 'continuous' (experimental).
GEN_D_TYPE = 'discrete'

# Outcome name or list of names. Training requires an outcome and treatment; prediction requires
# the trained features and any variables needed by the enabled options.
VAR_Y_NAME = name_dict['y_name']

# Single outcome used to build trees. None uses the first VAR_Y_NAME.
VAR_Y_TREE_NAME = name_dict['y_name']

# Ordered features, including binary variables.
VAR_X_NAME_ORD = name_dict['x_name_ord']

# Unordered features with at least three training categories. Prediction rows containing unseen
# categories are removed.
VAR_X_NAME_UNORD = name_dict['x_name_unord']

# Observation identifier. None uses 'id'; data cleaning keeps an existing identifier or creates
# one. With cleaning disabled, supply the identifier column.
VAR_ID_NAME = name_dict['id_name']

# Cluster identifier, required for panel data or clustered inference.
VAR_CLUSTER_NAME = name_dict['cluster_name']

# Sampling-weight variable, used when GEN_WEIGHTED is True.
VAR_W_NAME = name_dict['weight_name']

# Ordered/unordered variables prioritised in split-candidate draws. They are added to the features
# and protected from screening and feature selection. If the candidate count is smaller than this
# set, only a random subset of this set is drawn.
VAR_X_NAME_ALWAYS_IN_ORD = []
VAR_X_NAME_ALWAYS_IN_UNORD = []

# Variables for heterogeneity, balance and bias adjustment
# --------------------------------------------------------
# Continuous heterogeneity variables: discretized for GATEs, with additional smoothed GATEs when
# P_GATES_SMOOTH is True. Use the next lists for few categories.
VAR_Z_NAME_CONT = name_dict['x_name_ord'][:2]

# Ordered heterogeneity variables with few values.
VAR_Z_NAME_ORD = name_dict['x_name_ord'][-2:]

# Unordered heterogeneity variables. Heterogeneity variables are added to the features unless only
# ATEs without standard errors are requested.
VAR_Z_NAME_UNORD = name_dict['x_name_unord'][:2]

# Variables balanced across BGATE evaluation values. None uses the other heterogeneity variables,
# if available. Requires P_BGATE=True.
VAR_X_NAME_BALANCE_BGATE = name_dict['x_name_ord'][3:5]

# Training features used directly in bias adjustment when P_BA and P_BA_USE_X are True. The active
# set must be specified before training.
VAR_X_NAME_BA = None

# Ordered/unordered features protected from screening and feature selection. These lists do not
# add new features.
VAR_X_NAME_REMAIN_ORD = []
VAR_X_NAME_REMAIN_UNORD = []

# Variables for balance tests and treatment-specific descriptive statistics. Training descriptives
# can use these even when P_BT_YES=False. Active prediction tests need the corresponding
# prediction columns.
VAR_X_NAME_BALANCE_TEST_ORD = name_dict['x_name_ord'][:2]
VAR_X_NAME_BALANCE_TEST_UNORD = name_dict['x_name_unord'][:2]

# Output and CPU use
# ------------------
# Standard text output: 0 terminal, 1 files, 2 both (default or None).
GEN_OUTPUT_TYPE = None

# CPU workers: None selects 80% of logical cores; values <=1.5 select one worker; larger numbers
# are rounded. Large training samples reduce the count to 75%. See _INT_MP_BACKEND for backend
# selection and Windows limits.
GEN_MP_PARALLEL = None

# Data cleaning and screening
# ---------------------------
# Screen covariates for low variation and redundancy. Default True.
DC_SCREEN_COVARIATES = None

# With screening enabled, remove redundant covariates with absolute Pearson correlation above
# 0.999. Default True.
DC_CHECK_PERFECTCORR = None

# Minimum binary-category count during screening. Default 10; use 1 to disable this filter. Values
# below 1, including 0, select the default.
DC_MIN_DUMMY_OBS = None

# Keep needed columns and remove incomplete rows. Default True. Earlier input checks reject
# missing or nonnumeric entries in supplied columns, including unused columns: prepare numeric,
# complete input before calling train() or predict().
DC_CLEAN_DATA = None

# Causal forest and splitting rules
# ---------------------------------
# Number of trees in each forest. Default 1000.
CF_BOOT = None

# Splitting rule: 0 outcome MSE; 1 MSE + mean correlated error (default); 2 maximise treatment-
# effect heterogeneity; 3 randomly switch between MSE + MCE and the treatment criterion.
CF_MCE_VART = None

# MCE definition for rules 1 and 3: 'cross_error' (default) uses observed treatment-specific leaf
# means; 'covariance' uses centred matched outcomes.
CF_MCE_TYPE = None

# Treatment-penalty multiplier for rules 1 and 2; 0 disables it and None selects a sample-
# dependent value. Rule 2 uses 100 times the rule-1 default. For rule 3, this is the probability
# of selecting the treatment criterion (default 0.5).
CF_P_DIFF_PENALTY = None

# Treatment penalty: 'mse_d' (default) uses treatment-prediction MSE; 'diff_d' uses squared
# differences between daughter leaves. Only 'mse_d' supports tuning with the penalty. Constant
# outcomes stop splitting unless a treatment penalty is active.
CF_PENALTY_TYPE = None

# Restrict treatment pairs in MCE/heterogeneity terms to comparisons with the first treatment. All
# outcome-MSE terms remain. Default False.
CF_COMPARE_ONLY_TO_ZERO = None

# Tree subsampling and matching
# -----------------------------
# Multiplier for the automatic construction-subsample share (default 1). The rule uses the
# effective training size per chunk, with a 0.67 upper bound and a sample-dependent lower bound.
CF_SUBSAMPLE_FACTOR_FOREST = None

# Filling-sample multiplier: None or True selects 2; False or a numeric value below 0.01 uses all
# filling rows. At the default power, factor 1 halves the uncapped count. Smaller samples reduce
# work but add subsampling variability.
CF_SUBSAMPLE_FACTOR_EVAL = None

# Exponent for filling-sample size, from 0.85 to 1 (default 0.90). At 0.85, use the construction-
# share rule from version 0.10.0; larger powers use the filling-pool size independently of
# CF_SUBSAMPLE_FACTOR_FOREST.
CF_SUBSAMPLE_POWER_EVAL = None

# Use prognostic scores for Mahalanobis matching when enough matching features are available;
# otherwise use covariates. Default True. False always uses covariates.
CF_MATCH_NN_PROG_SCORE = None

# Use only diagonal covariance terms in Mahalanobis matching. Applies to both score and covariate
# matching. Default False uses the full covariance.
CF_NN_MAIN_DIAG_ONLY = None

# Forest tuning
# -------------
# Initial threshold count per ordered variable; 0 uses all. None selects round(4 + n**0.2), using
# effective training size per chunk. If no admissible split is found, retry all variables and
# supported thresholds with the same restrictions.
CF_RANDOM_THRESHOLDS = None

# Lower/upper bounds for the minimum-leaf-size grid. None selects automatic bounds based on
# treatment count and the smallest construction-arm subsample. These bounds also enter the
# automatic treatment-specific minimum.
CF_N_MIN_MIN = None
CF_N_MIN_MAX = None

# Number of leaf-size grid values (default 1, using the rounded midpoint). Multiple values are
# selected by the out-of-bag objective; rounding can remove duplicates.
CF_N_MIN_GRID = None

# Minimum observations per treatment in a construction leaf, for discrete treatments. None selects
# round(max((lower + upper) / (20 * treatments), 1)). Higher values also raise the total minimum
# leaf size.
CF_N_MIN_TREAT = None

# Lower/upper alpha-regularity bounds (defaults 0.05/0.15). Larger alpha requires more balanced
# daughter samples.
CF_ALPHA_REG_MIN = None
CF_ALPHA_REG_MAX = None

# Number of alpha grid values (default 1, the midpoint). Multiple values use out-of-bag selection.
CF_ALPHA_REG_GRID = None

# None selects 'treatment' for discrete treatments and 'pooled' for continuous treatments.
# 'treatment' applies within-arm unweighted counts and ceilings; 'pooled' applies to the total
# sample. Minimum leaf sizes still apply.
CF_ALPHA_REG_TYPE = None

# Lower/upper shares of candidate split variables (defaults 0.1/0.6), converted to integer
# candidate counts.
CF_M_SHARE_MIN = None
CF_M_SHARE_MAX = None

# Number of logarithmically spaced candidate-count values (default 1, using the midpoint of the
# count bounds). Multiple values use out-of-bag selection.
CF_M_GRID = None

# Raise the three tuning-grid sizes to at least 3; larger requested grids are retained. Rounding
# can reduce the number of distinct values. Default False.
CF_TUNE_ALL = None

# Default True draws 1 + Poisson(M - 1) for nominal candidate count M, capped at the available
# features. False uses M. Always-in priority applies to either choice.
CF_M_RANDOM_POISSON = None

# Target maximum training rows per forest chunk, including construction and filling samples.
# Chunk-specific effects are averaged. None uses automatic sizing, with splitting starting above
# 100,000 rows, or 250,000 in low-memory prediction mode.
CF_CHUNKS_MAXSIZE = None

# Permutation importance for the first regular forest chunk, using out-of-bag predictions.
# Requires output and can be expensive. Default False.
CF_VI_OOB_YES = None

# Feature selection
# -----------------
# Select features using auxiliary outcome and treatment forests. Default False.
FS_YES = None

# Default True reserves a separate sample for feature selection. False also uses these
# observations for causal-forest training.
FS_OTHER_SAMPLE = None

# Reserved share when FS_OTHER_SAMPLE=True. Default 0.33; choose a positive share no larger than
# 0.5.
FS_OTHER_SAMPLE_SHARE = None

# Relative variable-importance threshold for the tree outcome. Default 0; values up to 1 are
# fractions, larger values up to 100 are percentages.
FS_REL_VI_THRESHOLD_Y = None

# Corresponding threshold for treatment prediction. Default 0; uses the same fraction/percentage
# convention.
FS_REL_VI_THRESHOLD_D = None

# Keep features relevant for 'y_relevant', 'y_or_d_relevant' (default), or 'y_and_d_relevant'.
# Protected features remain.
FS_REL_VI_KEEP_IF = None

# True uses negative MSE of class probabilities for classifier importance; False or explicit None
# uses accuracy. Omitting this keyword defaults to True, so the explicit None below selects
# accuracy.
FS_MSE_FOR_CLASSIFIER = None

# Shared local-centering and common-support settings
# --------------------------------------------------
# Default True uses cross-fitting. False reserves independent data separately for each enabled
# stage, excluding those observations from forest training. For reversed rounds, local centering
# uses each round's construction/filling roles.
LC_CS_CV = None

# Holdout share for each enabled stage when LC_CS_CV=False. Default 0.25.
LC_CS_SHARE = None

# Cross-validation folds. Automatic defaults for training sizes below 100,000 / 250,000 / 500,000
# / otherwise are 5 / 4 / 3 / 2.
LC_CS_CV_K = None

# Local centering
# ---------------
# Subtract predicted conditional outcome means before forest estimation. Default True.
LC_YES = None

# Default 'RandomForest'. 'automatic' selects the lowest validation-MSE estimator; LC_CS_CV
# controls cross-validation versus holdout selection. See the API for the available estimator
# names.
LC_ESTIMATOR = None

# Add uncentred individual potential-outcome columns for VAR_Y_TREE_NAME; the centred columns and
# treatment effects remain. Reversed-round estimates use averaged centering predictions. Default
# True.
LC_UNCENTER_PO = None

# Common support
# --------------
# 0 disables trimming; 1 intersects treatment-specific probability ranges (default); 2 bounds
# every treatment probability by CS_MIN_P and 1 - CS_MIN_P. Probabilities use classification
# forests and the LC_CS_CV sampling scheme. Continuous treatments force 0.
CS_TYPE = None

# Relax type-1 limits: multiply lower bounds by 1 - value and upper bounds by 1 + value, clipped
# to [0, 1]. Default (treatments - 2) * 0.05; 0 disables this relaxation.
CS_ADJUST_LIMITS = None

# Type-1 range quantile. Default 1 uses minima/maxima; a value between 0.5 and 1 uses the
# corresponding lower and upper quantiles.
CS_QUANTIL = None

# Type-2 lower probability bound (default 0.01); the upper bound is 1 - CS_MIN_P. Probabilities at
# either bound are retained.
CS_MIN_P = None

# Maximum off-support share, checked separately for construction and filling samples. Default 0.5;
# exceeding it raises an error. No corresponding prediction-share limit is imposed.
CS_MAX_DEL_TRAIN = None

# With support checks enabled, stop if a feature is approximately constant within a treatment arm
# before trimming. Default True. False skips this check but retains probability-based trimming.
CS_DETECT_CONST_VARS_STOP = None

# Efficient estimation with reversed sample roles
# -----------------------------------------------
# Average estimates from two rounds with construction/filling roles exchanged. ATE variances are
# averaged conservatively. Default False; enabled automatically if any of the following three
# options is True.
GEN_ATE_EFF = None

# Average GATE/BGATE/CBGATE estimates and variances over both rounds. Default False.
GEN_GATE_EFF = None

# Average IATEs over both rounds. Requested IATE variances use half the regular-round variance,
# without a second variance calculation. Default False.
GEN_IATE_EFF = None

# Average QIATE estimates and variances from separately ranked rounds. GEN_IATE_EFF separately
# controls the reported IATE averages. Default False.
GEN_QIATE_EFF = None

# Clustering and panel data
# -------------------------
# Enable panel handling and clustered inference. Requires VAR_CLUSTER_NAME. Default False.
GEN_PANEL_DATA = None

# Default True for panel data; otherwise forced False. Subsample clusters within each construction
# chunk. The initial sample split and chunking still operate on rows.
GEN_PANEL_IN_RF = None

# One-way clustered standard errors. Default False, forced True by GEN_PANEL_DATA=True. Requires
# the cluster identifier.
P_CLUSTER_STD = None

# Continuous treatment (experimental)
# -----------------------------------
# Target grid size for matching the positive continuous-treatment part. The zero-treatment mass is
# handled separately. Default 10.
CT_GRID_NN = None

# Target grid size for weight computation. Default 10.
CT_GRID_W = None

# Requested positive-treatment quantile points for the dose-response grid. Default 100, raised to
# at least CT_GRID_W. A zero point is added; interpolation can change the final grid size.
CT_GRID_DR = None

# Balance tests and prediction-population weights
# -----------------------------------------------
# ATE-weight balance tests for the specified balance variables. Default True, including None;
# disabled without balance variables or in ATE-only mode.
P_BT_YES = None

# Reweight prediction treatment groups for aggregate effects; this does not draw a new sample.
# Requires discrete treatments and prediction treatment information. Default False.
P_CHOICE_BASED_SAMPLING = None

# Positive relative weights, one per treatment in internal order. They are normalized to mean 1,
# not inverted as sampling probabilities.
P_CHOICE_BASED_PROBS = None

# Use sampling weights from VAR_W_NAME. Default False (experimental).
GEN_WEIGHTED = None

# Prediction and analytic inference
# ---------------------------------
# Cap extreme normalized weights before renormalizing (default 0.05); final weights can exceed the
# cap after renormalization. Values >=1 disable capping; 0 selects the default. Negative weights
# are capped by absolute size.
P_MAX_WEIGHT_SHARE = None

# Confidence level for reported intervals and plots. Default 0.95.
P_CI_LEVEL = None

# Default True uses conditional outcome moments given estimation weights for analytic variances;
# False uses weighted outcomes. Bootstrap/residual-based paths bypass this choice.
P_COND_VAR = None

# Conditional-moment estimator: True uses moving averages over sorted weights (default); False
# uses Nadaraya-Watson kernel regression.
P_KNN = None

# Minimum moving-average neighbour count. Default 10.
P_KNN_MIN_K = None

# Window-size multiplier for conditional moments and sorted-IATE plot smoothing. Default 1.
P_KNN_CONST = None

# Nadaraya-Watson bandwidth multiplier. Default 1.
P_NW_BANDW = None

# Nadaraya-Watson kernel: 1 Epanechnikov (default), 2 Gaussian.
P_NW_KERN = None

# Bootstrap replications for the requested standard errors. At construction, None selects 199 with
# clustered inference and otherwise disables bootstrapping. False disables it; True selects 199;
# integers >=49 specify the count. In new_keywords, True selects 99, so use an explicit integer
# for the same count across both interfaces.
P_SE_BOOT_ATE = None
P_SE_BOOT_GATE = None
P_SE_BOOT_IATE = None
P_SE_BOOT_QIATE = None

# ATEs and group effects
# ----------------------
# Target bins for continuous heterogeneity variables. Default round(n_training_effective**0.3).
# Prediction reuses training quantile bins and means.
P_MAX_CATS_Z_VARS = None

# Compute only ATEs without standard errors, disabling other effect estimates and balance tests.
# Default False.
P_ATE_NO_SE_ONLY = None

# Effects by observed treatment group (default False). Requires discrete treatment information and
# every main treatment arm in prediction data before trimming.
P_ATET = None

# GATEs by observed treatment group; also enables ATET at initialization. Requires heterogeneity
# variables and prediction treatments. Default False.
P_GATET = None

# Balance variables in VAR_X_NAME_BALANCE_BGATE across GATE evaluation values; None for that list
# uses other heterogeneity variables. Default False; disabled without heterogeneity variables.
P_BGATE = None

# Balance all other features across GATE evaluation values. Default False; disabled without
# heterogeneity variables.
P_CBGATE = None

# Fallback quantile-grid size for BGATE/CBGATE variables without stored evaluation values. Default
# 50; separate from the smoothed-GATE grid.
P_GATES_NO_EVALU_POINTS = None

# Add smoothed GATEs for continuous heterogeneity variables alongside their discretized estimates.
# Uses an Epanechnikov kernel. Default True.
P_GATES_SMOOTH = None

# Smoothed-GATE bandwidth multiplier. Default 1.
P_GATES_SMOOTH_BANDWIDTH = None

# Target smoothed-GATE evaluation points; distinct values or quantiles are used. Default 50.
P_GATES_SMOOTH_NO_EVALU_POINTS = None

# Report differences from the preceding evaluation value instead of the overall ATE. Differences
# use weights before separate caps, so they can differ from subtracting capped estimates. Default
# False.
P_GATES_MINUS_PREVIOUS = None

# Reference-sample size for BGATE/CBGATE: min(N, max(50, floor(N * share / m))).
# N is the number of available prediction rows; m is the number of evaluation points.
# None selects an automatic share. The same base sample is reused across evaluation points.
P_BGATE_SAMPLE_SHARE = None

# Individual and quantile effects
# -------------------------------
# Estimate individual effects. Default True.
P_IATE = None

# Estimate individual-effect standard errors, which can be expensive. Default False.
P_IATE_SE = None

# Estimate IATE-minus-ATE deviations, with inference when P_IATE_SE=True. Requires
# _INT_LOW_MEMORY_PREDICT=False. Weight differences precede capping. Default False.
P_IATE_M_ATE = None

# Estimate quantile individual effects and enable IATE computation. Requires discrete treatments,
# _INT_LOW_MEMORY_PREDICT=False and P_BA=False; unavailable with treatment versions or IV
# estimation. Default False.
P_QIATE = None

# QIATE standard errors; also enables IATE standard errors, but does not enable P_QIATE itself.
# Default False.
P_QIATE_SE = None

# QIATE differences from the median effect, with inference if P_QIATE_SE=True. Default False.
P_QIATE_M_MQIATE = None

# QIATE differences between quantiles q and 1 - q, with inference if P_QIATE_SE=True. Default
# False.
P_QIATE_M_OPP = None

# Number Q of midpoint quantiles (j + 0.5) / Q, j=0,...,Q-1. Default 99. The median is computed
# separately when needed.
P_QIATE_NO_OF_QUANTILES = None

# Smooth forest weights over fractional IATE ranks (default True). False uses the nearest observed
# rank.
P_QIATE_SMOOTH = None

# Bandwidth multiplier for smoothing over IATE ranks. Default 1.
P_QIATE_SMOOTH_BANDWIDTH = None

# Adjust ranks using normal quantiles and IATE standard errors; also enables P_IATE_SE. Default
# False.
P_QIATE_BIAS_ADJUST = None

# Bias adjustment with forest weights (experimental)
# --------------------------------------------------
# Enable bias adjustment (default False). Requires at least one score/covariate source.
# Incompatible with continuous treatments, treatment versions, sampling weights, clustered/panel
# inference and QIATEs.
P_BA = None

# Use random-forest propensity scores as adjustment regressors. Default True.
P_BA_USE_PROP_SCORE = None

# Use treatment-specific prognostic scores for VAR_Y_TREE_NAME as adjustment regressors. The same
# scores serve all outcomes. Default True.
P_BA_USE_PROG_SCORE = None

# Use the training features in VAR_X_NAME_BA directly as adjustment regressors. Default False; the
# active set must be fixed before training.
P_BA_USE_X = None

# Use weighted ridge regression (default True); False uses weighted OLS.
P_BA_RIDGE = None

# Evaluation target: 'train_obs' (default, including None) uses the regular round's full filling-
# sample distribution; 'zeros' uses centred-regressor zeros; 'weighted_train_obs' uses forest-
# weighted filling rows in the current chunk and round.
P_BA_ADJ_METHOD = None

# Restrict adjusted weights to nonnegative values, including zero. Default False.
P_BA_POS_WEIGHTS_ONLY = None

# Prognostic-score estimator (default 'RandomForest'). For regression outcomes, 'automatic'
# selects by treatment-specific cross-validated MSE. Outcomes with fewer than 10 distinct values
# use a classifier. See the API for estimator choices.
P_BA_ESTIMATOR = None

# Folds for bias-adjustment cross-validation. Automatic defaults by training size match
# LC_CS_CV_K: 5 / 4 / 3 / 2.
P_BA_CV_K = None

# Analysis of estimated individual effects
# ----------------------------------------
# Enable analyse() diagnostics (default True, disabled without IATEs). Analysis requires output
# and a returned IATE DataFrame.
POST_EST_STATS = None

# Analyse effects relative to the first treatment (default True). False includes all discrete-
# treatment comparisons.
POST_RELATIVE_TO_FIRST_GROUP_ONLY = None

# Report Pearson correlations involving estimated effects. Default True.
POST_BIN_CORR_YES = None

# Minimum absolute feature-effect correlation to display. Does not filter effect-effect
# correlations. Default 0.1.
POST_BIN_CORR_THRESHOLD = None

# Save IATE figures (default True); display depends on _INT_SHOW_PLOTS. False still allows
# density/sorted-effect CSV output and separately enabled tree analysis.
POST_PLOTS = None

# Cluster IATE profiles in their original units, without automatic scaling. Default True.
POST_KMEANS_YES = None

# Also cluster each effect separately. Default False.
POST_KMEANS_SINGLE = None

# Integer for one cluster count; list/tuple for selection by the highest silhouette score; None
# for an automatic five-value grid. Feasible counts depend on observations and distinct profiles;
# small clusters can be merged.
POST_KMEANS_NO_OF_GROUPS = None

# Random starts for k-means. Default 10.
POST_KMEANS_REPLICATIONS = None

# Maximum iterations per k-means start. Default 1000.
POST_KMEANS_MAX_TRIES = None

# Minimum cluster size in percent, not as a fraction. Default 1 (one percent).
POST_KMEANS_MIN_SIZE_SHARE = None

# Random-forest importance for predicting IATEs from features. Describes associations with
# estimated effects. Default True.
POST_RANDOM_FOREST_VI = None

# Describe IATEs using standard and honest regression trees of depths 2 through 5. Default True.
POST_TREE = None

# Sensitivity analysis (experimental)
# -----------------------------------
# Compute placebo CBGATEs; also requires P_CBGATE=True. Default False.
SENS_CBGATE = None

# Compute placebo BGATEs; also requires P_BGATE=True. Default False.
SENS_BGATE = None

# Compute placebo GATEs. Default False; also enabled by placebo BGATEs/CBGATEs.
SENS_GATE = None

# Compute placebo IATEs. Passing results with an IATE DataFrame enables this even if False and
# replaces the supplied prediction sample with those data. Without previous IATEs, the default is
# False.
SENS_IATE = None

# Compute placebo IATE standard errors. Default False.
SENS_IATE_SE = None

# Default ('basic',), currently the supported scenario. Simulate treatments from estimated
# probabilities within one observed treatment population, keeping outcomes unchanged, so simulated
# treatment has no effect.
SENS_SCENARIOS = None

# Propensity-score cross-validation folds. Default 5; use at least 2.
SENS_CV_K = None

# Placebo-treatment simulation replications. Default 2; at least 1.
SENS_REPLICATIONS = None

# Reference treatment in internal recoded order (0, 1, ...), not the original labels. None uses
# the largest retained treatment group.
SENS_REFERENCE_POPULATION = None

# Internal controls: change only when needed
# ------------------------------------------
# Use the legacy Ray implementation. Default False.
_INT_MP_USE_OLD_RAY = None

# Backend with legacy Ray disabled: 'ray', 'joblib' or 'sequential'. None selects joblib if Ray is
# unavailable, or on Windows with at most 250,000 adjusted training rows; otherwise Ray. On
# Windows, joblib rejects more than 63 workers; with the new backend, counts 61-63 are also
# rejected.
_INT_MP_BACKEND = None

# Minimum eligible NumPy-array size for joblib memory mapping. Default 4 * 1024 * 1024 bytes; 0
# removes the size threshold.
_INT_MP_MEMMAP_MIN_BYTES = None

# Base directory for joblib temporary files. None uses Path.cwd() / 'joblibtemp'. Executors clean
# up their own subdirectories; an existing base directory is retained.
_INT_MP_MEMMAP_DIR = None

# Requested maximum task batches, or 'automatic'/None (default). Actual batching depends on task
# count, workers and computation; streaming weight tasks bypass this setting. Used with the new
# backends.
_INT_MP_BATCHES = None

# Predict without retaining the full weight matrix. Default True. Incompatible with IV estimation,
# QIATEs and P_IATE_M_ATE=True.
_INT_LOW_MEMORY_PREDICT = None

# Target prediction rows per low-memory chunk. None uses automatic sizing, from 1000 down to 10 as
# training size grows; explicit values >=1 are accepted.
_INT_LOW_MEMORY_MAX_CHUNKSIZE = None

# Request experimental GPU support. Default False; unavailable PyTorch/CUDA disables it with a
# warning. IATE GPU computation also needs full weights, no bias adjustment and fewer than 16
# selected IATE workers.
_INT_CUDA = None

# Use one scikit-learn worker in feature selection, support checks, local centering and
# sensitivity propensity forests. Does not disable MCF parallel computations or all auxiliary
# forests. Default False.
_INT_REPLICATION = None

# Enable standard output, diagnostics, figures and report information. Default True; False leaves
# output paths unset. Enable _INT_RETURN_IATE_SP explicitly to retain the IATE DataFrame when
# output is disabled.
_INT_WITH_OUTPUT = None

# True permits reusing a nonempty output directory; default False uses an empty directory or a
# numeric suffix. A new_keywords dictionary, even {}, triggers that directory search regardless of
# this flag.
_INT_OUTPUT_NO_NEW_DIR = None

# Additional progress output when output is enabled. Default True.
_INT_VERBOSE = None

# Print RAM statistics at selected stages; requires output and verbose output. Default False.
_INT_MEMORY_PRINT = None

# Print descriptive statistics when output is enabled. Default True.
_INT_DESCRIPTIVE_STATS = None

# Display generated plots when output is enabled. Default True.
_INT_SHOW_PLOTS = None

# Legend/title font-size index, 1 through 7 from xx-small to xx-large. Default 2 (x-small).
_INT_FONTSIZE = None

# Saved-figure resolution. Default 500 dots per inch.
_INT_DPI = None

# Ordered-GATE plotting threshold for connected estimates and confidence limits; smoothed plots
# also use filled bands above it. Default 20.
_INT_NO_FILLED_PLOT = None

# Store forest weights as sparse matrices. Default True; required for IV estimation.
_INT_WEIGHT_AS_SPARSE = None

# Target prediction rows per parallel IATE-effect task. None uses ceil(prediction rows / workers),
# independently of low-memory chunk size. Use values >=1; ignored in direct single-worker
# computation.
_INT_IATE_CHUNK_SIZE = None

# Positive integer number of chunks for sparse-weight construction; 1 disables extra splitting.
# None becomes 1 with low-memory prediction and otherwise uses automatic sizing when weights are
# computed.
_INT_WEIGHT_AS_SPARSE_SPLITS = None

# Training share for filling trees; the remainder builds them. Default 0.5; valid values lie
# strictly between 0.01 and 0.99. Efficient estimation exchanges these roles in the additional
# round.
_INT_SHARE_FOREST_SAMPLE = None

# Optional quantile-bin target for ordered features. None disables it. Discretize only features
# with more than target + 2 distinct values; prediction reuses training bins and means.
_INT_MAX_CATS_CONT_VARS = None

# Store distinct ordered-feature values only below this count (default 50). Unordered categories
# are always retained.
_INT_MAX_SAVE_VALUES = None

# Parallel forest-weight strategy: 1 distributes groups of prediction rows (default); 2
# distributes trees for each row. Ignored with one worker, including within low-memory prediction
# chunks.
_INT_MP_WEIGHTS_TYPE = None

# Forest-batch count for the tree-batching branch of out-of-bag variable importance. None selects
# automatically; 1/False/True disables batching. Explicit counts must not exceed the number of
# trees.
_INT_MP_WEIGHTS_TREE_BATCH = None

# Legacy Ray cleanup: 'refs' releases supplied object references (default); 'rest' releases
# result/task-reference lists; ('refs', 'rest') does both. Use ('none',) for neither. Other
# references can keep objects alive.
_INT_MP_RAY_DEL = None

# Shut down Ray at legacy cleanup points with multiple workers. None resolves during training to
# False below 150,000 effective rows per chunk, True otherwise. Does not control the new executor
# or error-recovery shutdowns.
_INT_MP_RAY_SHUTDOWN = None

# Scale explicit Ray object-store memory requests before memory caps. Default 1; independent of
# _INT_MP_RAY_SHUTDOWN. Does not resize a running Ray instance.
_INT_MP_RAY_OBJSTORE_MULTIPLIER = None

# Return processed prediction data and individual estimates in results['iate_data_df']. Requires
# IATEs. Output enabled forces True; without output, only explicit True enables it. Separate
# effect arrays are unaffected.
_INT_RETURN_IATE_SP = None

# Delete forests after prediction to save memory. The instance then needs retraining before
# another predict() call. Default False.
_INT_DEL_FOREST = None

# Retain zero weights in standard-error calculations. Default False.
_INT_KEEP_W0 = None

# Threshold for memory reductions (default 1,000,000). Large training samples reduce worker
# counts, local-centering selection can halve them again, and selected arrays use float32. This is
# not a row-sampling limit.
_INT_OBS_BIGDATA = None

# Optional training-row limit, sampled before preprocessing with a fixed seed. Use an integer
# >100; None imposes no limit.
_INT_MAX_OBS_TRAINING = None

# Prediction-row limit, sampled before preprocessing with a fixed seed. Aggregates use the
# retained sample. Default 1,000,000 with low-memory prediction, otherwise 250,000.
_INT_MAX_OBS_PREDICTION = None

# Maximum rows for k-means in analyse() (default 200,000). If sampled, the returned DataFrame with
# cluster labels contains only those rows.
_INT_MAX_OBS_KMEANS = None

# Maximum rows per IATE-feature graph, including its CSV data and LOWESS fit. Default 50,000.
_INT_MAX_OBS_POST_REL_GRAPHS = None

# --------------------------------------------------------------------------------------------------
# For convenience the mcf parameters are collected and passed as a dictionary. Of course, they can
# also be passed as single parameters (or not at all, in which case default values are used).

params = {'cf_alpha_reg_grid': CF_ALPHA_REG_GRID,
          'cf_alpha_reg_max': CF_ALPHA_REG_MAX, 'cf_alpha_reg_min': CF_ALPHA_REG_MIN,
          'cf_alpha_reg_type': CF_ALPHA_REG_TYPE,
          'cf_boot': CF_BOOT, 'cf_chunks_maxsize': CF_CHUNKS_MAXSIZE,
          'cf_compare_only_to_zero': CF_COMPARE_ONLY_TO_ZERO, 'cf_n_min_grid': CF_N_MIN_GRID,
          'cf_n_min_max': CF_N_MIN_MAX, 'cf_n_min_min': CF_N_MIN_MIN,
          'cf_n_min_treat': CF_N_MIN_TREAT, 'cf_nn_main_diag_only': CF_NN_MAIN_DIAG_ONLY,
          'cf_m_grid': CF_M_GRID, 'cf_m_share_max': CF_M_SHARE_MAX,
          'cf_m_random_poisson': CF_M_RANDOM_POISSON, 'cf_m_share_min': CF_M_SHARE_MIN,
          'cf_match_nn_prog_score': CF_MATCH_NN_PROG_SCORE,
          'cf_mce_type': CF_MCE_TYPE, 'cf_mce_vart': CF_MCE_VART,
          'cf_random_thresholds': CF_RANDOM_THRESHOLDS, 'cf_p_diff_penalty': CF_P_DIFF_PENALTY,
          'cf_penalty_type': CF_PENALTY_TYPE, 'cf_subsample_factor_eval': CF_SUBSAMPLE_FACTOR_EVAL,
          'cf_subsample_factor_forest': CF_SUBSAMPLE_FACTOR_FOREST,
          'cf_subsample_power_eval': CF_SUBSAMPLE_POWER_EVAL, 'cf_tune_all': CF_TUNE_ALL,
          'cf_vi_oob_yes': CF_VI_OOB_YES,

          'cs_adjust_limits': CS_ADJUST_LIMITS, 'cs_max_del_train': CS_MAX_DEL_TRAIN,
          'cs_detect_const_vars_stop': CS_DETECT_CONST_VARS_STOP,
          'cs_min_p': CS_MIN_P, 'cs_quantil': CS_QUANTIL, 'cs_type': CS_TYPE,

          'ct_grid_dr': CT_GRID_DR, 'ct_grid_nn': CT_GRID_NN, 'ct_grid_w': CT_GRID_W,

          'dc_check_perfectcorr': DC_CHECK_PERFECTCORR, 'dc_clean_data': DC_CLEAN_DATA,
          'dc_min_dummy_obs': DC_MIN_DUMMY_OBS, 'dc_screen_covariates': DC_SCREEN_COVARIATES,

          'fs_rel_vi_keep_if': FS_REL_VI_KEEP_IF,
          'fs_other_sample': FS_OTHER_SAMPLE, 'fs_other_sample_share': FS_OTHER_SAMPLE_SHARE,
          'fs_rel_vi_threshold_y': FS_REL_VI_THRESHOLD_Y,
          'fs_rel_vi_threshold_d': FS_REL_VI_THRESHOLD_D, 'fs_yes': FS_YES,
          'fs_mse_for_classifier': FS_MSE_FOR_CLASSIFIER,

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
          '_int_mp_memmap_dir': _INT_MP_MEMMAP_DIR, '_int_mp_batches': _INT_MP_BATCHES,
          '_int_mp_ray_del': _INT_MP_RAY_DEL,
          '_int_mp_ray_objstore_multiplier': _INT_MP_RAY_OBJSTORE_MULTIPLIER,
          '_int_mp_ray_shutdown': _INT_MP_RAY_SHUTDOWN,
          '_int_mp_weights_tree_batch': _INT_MP_WEIGHTS_TREE_BATCH,
          '_int_mp_weights_type': _INT_MP_WEIGHTS_TYPE,
          '_int_obs_bigdata': _INT_OBS_BIGDATA,
          '_int_output_no_new_dir': _INT_OUTPUT_NO_NEW_DIR, '_int_replication': _INT_REPLICATION,
          '_int_return_iate_sp': _INT_RETURN_IATE_SP,
          '_int_memory_print': _INT_MEMORY_PRINT,
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
# Demonstrate supported keyword updates on a separate, trained copy.
# Pass new_keywords to its first predict() call; even {} is rejected after that first call.
# Updates persist on the instance and therefore also apply to its subsequent analyse() call.
# Use explicit values. None is accepted only for ignored, inactive variable-name lists.
# The flags p_ba, p_ba_use_prop_score, p_ba_use_prog_score, p_qiate, p_qiate_se and p_bt_yes
# can only be disabled through new_keywords; enable them before training when needed.

params_predict_analysis_2nd_round = {
    # Use trained features in list/tuple form, with their original ordered/unordered roles.
    # Screening and feature selection may remove features; those cannot be selected here.
    # var_x_name_ba below is ignored because bias adjustment/direct covariates are disabled.
    'var_x_name_balance_test_ord': name_dict['x_name_ord'][:1],
    'var_x_name_balance_test_unord': name_dict['x_name_unord'][:1],
    'var_x_name_balance_bgate': name_dict['x_name_ord'][3:6],
    'var_x_name_ba': (name_dict['x_name_ord'][0], *name_dict['x_name_unord'][:2],),
    'var_z_name_ord': name_dict['x_name_ord'][-3:],
    'var_z_name_unord': name_dict['x_name_unord'][:1],

    # 'cs_type': 0,  # Use only after restricting prediction data to common support.

    'p_ba': False, 'p_ba_adj_method': 'zeros', 'p_ba_pos_weights_only': False, 'p_ba_use_x': False,
    'p_ba_use_prop_score': False, 'p_ba_use_prog_score': False,
    'p_ate_no_se_only': False, 'p_atet': True, 'p_gatet': False,

    'p_bgate': True, 'p_cbgate': True, 'p_iate': True, 'p_iate_se': True, 'p_iate_m_ate': False,
    'p_bgate_sample_share': 0.5, 'p_gates_minus_previous': False, 'p_gates_smooth_bandwidth': 1,
    'p_gates_smooth': True, 'p_gates_smooth_no_evalu_points': 50, 'p_gates_no_evalu_points': 50,
    'p_qiate': False, 'p_qiate_se': False, 'p_qiate_m_mqiate': False, 'p_qiate_m_opp': False,
    'p_qiate_no_of_quantiles': 50, 'p_qiate_smooth': True, 'p_qiate_smooth_bandwidth': 1,
    'p_qiate_bias_adjust': False, 'p_bt_yes': False, 'p_choice_based_sampling': False,
    'p_cond_var': True, 'p_knn': True, 'p_knn_const': 1,
    'p_knn_min_k': 5, 'p_nw_bandw': 1, 'p_nw_kern': 1, 'p_ci_level': 0.90,
    'p_se_boot_ate': False, 'p_se_boot_gate': False, 'p_se_boot_iate': False,
    'p_se_boot_qiate': False,

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

# -------------------------- Train, predict and report -----------------------------------------
# Compare observed treatment with assigning everyone to treatment 0, 1 or 2.
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
# Copy before the first prediction: new_keywords requires a trained, not-yet-predicted instance.
mymcf_2 = deepcopy(mymcf)

# Continue using the keyword values as specified when initializing the instance
results = mymcf.predict(prediction_df)
# analyse() returns a result dictionary with cluster labels, or None if k-means is disabled.
results_with_clusters = None
if (mymcf.gen_cfg.with_output and mymcf.post_cfg.est_stats
        and results['iate_data_df'] is not None):
    results_with_clusters = mymcf.analyse(results)

# Allocation evaluation requires retained forests and supports these sampling settings.
if (mymcf.gen_cfg.d_type == 'discrete' and not mymcf.gen_cfg.weighted
        and not mymcf.p_cfg.choice_based_sampling and not mymcf.int_cfg.del_forest):
    mymcf.predict_different_allocations(prediction_df, test_alloc_df)

if GEN_D_TYPE != 'continuous' and not FS_YES:
    # Sensitivity is unavailable with continuous treatments or feature selection.
    # Previous IATE results, when present, determine the sensitivity prediction sample.
    sensitivity_params = params.copy()
    if results['path_output'] is not None:
        sensitivity_params['gen_outpath'] = results['path_output'] / 'sensitivity'
    mymcf_sens = ModifiedCausalForest(**sensitivity_params)
    mymcf_sens.sensitivity(training_df, prediction_df, results=results, **params_sensitivity)

    if mymcf.gen_cfg.with_output:
        my_report = McfOptPolReport(mcf=mymcf, mcf_sense=mymcf_sens)
        my_report.report()
elif mymcf.gen_cfg.with_output:
    my_report = McfOptPolReport(mcf=mymcf)
    my_report.report()

if GEN_D_TYPE != 'continuous' and not FS_YES:
    # Use new values of some keywords as specified in dictionary with copied instance
    results_2 = mymcf_2.predict(prediction_df, new_keywords=params_predict_analysis_2nd_round,)
    results_with_clusters_2 = None
    if (mymcf_2.gen_cfg.with_output and mymcf_2.post_cfg.est_stats
            and results_2['iate_data_df'] is not None):
        results_with_clusters_2 = mymcf_2.analyse(results_2)

    if mymcf_2.gen_cfg.with_output:
        my_report2 = McfOptPolReport(mcf=mymcf_2)
        my_report2.report()

print('End of computations.\n\nThanks for using the ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
