"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economic Research
University of St. Gallen, Switzerland

Version: 0.8.0

This is an example to show how to use the mcf with full specification of all
its keywords. It may be seen as an add-on to the published mcf documentation.

"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from mcf.example_data_functions import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport

# ------------------ NOT parameters of the ModifiedCausalForest ----------------

# ---------------------- Generate artificial data ------------------------------

# Parameters to generate artificial data (DataFrame) for this example
TRAIN_OBS = 1000        # Number of observations of training data.
#                          Training data is used to train the causal forest.
#                          Training data must contain outcome, treatment
#                          and features. Default is 1000.

PRED_OBS = 1000          # Number of observations of prediction data.
#                          Prediction data is used to compute the effects.
#                          Prediction data must contain features. Treatment
#                          effects on the treated additionally require
#                          treatment information. Default is 1000.

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
    correlation_x='high'
    )

if PREDDATA_IS_TRAINDATA:
    prediction_df = None


# ------------------ Parameters of the ModifiedCausalForest --------------------
#   Whenever None is specified, parameter will be set to default values.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
GEN_OUTPATH = Path.cwd() / 'example/output'    # Path object, String, or None
#   Path were the output is written too (text, estimated effects, etc.)
#   If this is None, a */out directory below the current directory is used.
#   If specified directory does not exist, it will be created.
#   OUTPATH is passed to ModifiedCausalForest.

GEN_OUTFILETEXT = 'mcf.py.0.8.0'
#   File for text output. If gen_outfiletext is None, 'txtFileWithOutput' is
#   used. *.txt file extension will be added by the programme.

#   ----------------------------------------------------------------------------
#   The following parameters are all used to initialise ModifiedCausalForest.
#   ----------------------------------------------------------------------------
#   Convention: Internally all variable names are set to lowercase (including
#   those in the DataFrames provided by user). Therefore, avoid having
#   variables in the data with names that differ only with respect to
#   capitalization.

#   The following variable names must at least be provided:
#      train method: d_name, y_name, (x_name_ord or x_name_unord)
#      predict and analyse methods: x_name_ord or x_name_unord
#   All other variables are optional.

VAR_D_NAME = name_dict['d_name']   # Name of treatment variable
GEN_D_TYPE = 'discrete'  # Type of treatment: 'discrete' (default)
#                                              or 'continuous'

# Outcome variables
#    Single variable (string) or several variables (list of string)
VAR_Y_NAME = name_dict['y_name']

VAR_Y_TREE_NAME = name_dict['y_name']     # Variable to build trees.
#   If None or [], the first variable in y_name is used to build trees.

#   Features
VAR_X_NAME_ORD = name_dict['x_name_ord']   # Ordered (including dummy variables)
VAR_X_NAME_UNORD = name_dict['x_name_unord']   # Unordered

#   Identifier
VAR_ID_NAME = name_dict['id_name']
#   If None (default) or []: Identifier will be added the data.

#   Cluster and panel identifiers (default is no variable)
VAR_CLUSTER_NAME = name_dict['cluster_name']  # Name of variable defining
#                                               clusters if used.
VAR_W_NAME = name_dict['weight_name']         # Name of weight, if weighting
#                                               option is used.

# Names of variables always checked on when deciding next split
VAR_X_NAME_ALWAYS_IN_ORD = []
VAR_X_NAME_ALWAYS_IN_UNORD = []

#   ------------- Variables related to treatment heterogeneity -----------------
#   Variables to define policy relevant heterogeneity in multiple treatment
#   procedure. If not already included, they will be added to the list of
#    features. The default is no variable.

#   Ordered variables with many values (put discrete variables with only few
#   values in categories below). They are discretized as well as treated as
#   continuous for GATE estimation.
VAR_Z_NAME_CONT = name_dict['x_name_ord'][:2]

# Ordered variables with few values
VAR_Z_NAME_ORD = name_dict['x_name_ord'][-2:]

#   Unordered variables
VAR_Z_NAME_UNORD = name_dict['x_name_unord'][:2]

# Variable to balance the GATEs on. Only relevant if P_BGATE is True. The
# distribution of these variables is kept constant when a BGATE is computed.
# The default (or None) is to use the other heterogeneity variables if there
# are any for balancing.
VAR_X_NAME_BALANCE_BGATE = name_dict['x_name_ord'][3:5]

#   Variables that cannot be removed by feature selection. Default is no
#   variable.
VAR_X_NAME_REMAIN_ORD = []
VAR_X_NAME_REMAIN_UNORD = []

#   Variables for balancing tests (cannot be removed by feature selection).
#   Treatment specific descriptive statistics will only be printed for those
#   variables.
VAR_X_NAME_BALANCE_TEST_ORD = name_dict['x_name_ord'][:2]
VAR_X_NAME_BALANCE_TEST_UNORD = name_dict['x_name_unord'][:2]

# --------------------- End of variables section -------------------------------

#   ------------- Where to direct the text output to ---------------------------
GEN_OUTPUT_TYPE = None      # 0: Output goes to terminal
#                             1: output goes to file
#                             2: Output goes to file and terminal (default)

#   ------------- Multiprocessing ----------------------------------------------
GEN_MP_PARALLEL = None      # Number of parallel processes  (>0)
#   Default is to use 80% of logical cores (reduce if memory problems!)
#   If _int_obs_bigdata, it is reduced to 75% of specified value.
#   0, 1: no parallel computations

#   ------------- Data cleaning ------------------------------------------------
DC_SCREEN_COVARIATES = None  # If True (Default) screen covariates (sc)
DC_CHECK_PERFECTCORR = None  # if sc=True: if True (default), var's that are
#   perfectly correlated with each others will be deleted.
DC_MIN_DUMMY_OBS = None      # if > 0 dummy variables with observations in one
#    category smaller than this value will be deleted. Default is 10.
DC_CLEAN_DATA = None         # if True (default), remove all rows with missing
#      observations & unnecessary variables from DataFrame.

#   ------------- Training the causal forest -----------------------------------
CF_BOOT = None    # Number of Causal Trees. Default is 1000.

#   Estimation methods
CF_MCE_VART = None  # Splitting rule
#   0: mse's of regression only considered
#   1: mse+mce criterion (default)
#   2: -var(effect): heterogy maximising splitting rule of Wager & Athey (2018)
#   3: randomly switching between outcome-mse+mce criterion & penalty functions
#   Penalty function (depends on value of mce_vart)
CF_P_DIFF_PENALTY = None   # Muliplier of penalty function (or else)
#   mce_vart == 0: Penalty irrelevant
#   mce_vart == 1: Multiplier of penalty (in terms of var(y))
#                  0: no penalty
#                  default (or None):
#                      2*((n*subsam_share)**0.9)/(n*subsam_share)*
#                      sqrt(no_of_treatments*(no_of_treatments-1)/2).
#                      Increase if balancing tests indicate bad balance.
#   mce_vart == 2: Default (or None): Multiplier of penalty (in terms of value
#                  of MSE(y) value function without splits) for penalty.
#                  0: no penalty
#                  def: 100*4*(n*f_c.subsam_share)^0.8)/(n*f_c.subsam_share).
#                       Increase if balancing tests indicate bad balance.
#   mce_vart == 3: Probability of using p-score (0-1)
#                  Default is 0.5. Increase if balancing tests indicate bad
#                  balance.
CF_PENALTY_TYPE = None     # Type of penalty function.
#   'mse_d':  MSE of treatment prediction in daughter leaf (new in 0.7.0)
#   'diff_d': Penalty as squared leaf difference (as in Lechner, 2018)
#    Note that an important advantage of 'mse_d' that it can also be used
#    for tuning (due to its computation, this is not possible for 'diff_d').
#    Default (or None) is 'mse_d'.

CF_COMPARE_ONLY_TO_ZERO = None      # If True, the computation of the MCE
#   ignores all elements not related to the first treatment (which usually is
#   the control group). This speeds up computation and may be attractive when
#   interest is only in the comparisons of each treatment to the control group,
#   and not among each other. This may also be attractive for optimal policy
#   analysis based on using estimated potential outcomes normalized by the
#   potential outcome of the control group (i.e., IATEs of treatments vs.
#   control group). Default is False.

#   Subsampling
CF_SUBSAMPLE_FACTOR_FOREST = None   # Size of subsampling sample to build tree
#   Default size of subsample is: max(min(0.67*n,(2*(n**0.85)/n)), n**0.5);
#   N: sample size;  n: 2x sample size of the smallest treatment group
#   >0: reduces (<1) or increases (>1) the default subsample size, max is 0.67
#   Actual share of subsample = default size * SUBSAMPLE_FACTOR_FOREST
#   Default: SUBSAMPLE_FACTOR_FOREST = 1

#   Tree-specific subsampling also in evaluation sample should increase speed
#   at which asymtotic bias disappears (at the expense of slower disappearance
#   of the variance; however, simulations show no relevant impact).
CF_SUBSAMPLE_FACTOR_EVAL = None   # Size of subsampling sample to evaluate tree
#   False: No subsampling in evaluation subsample
#   True or None: 2 * subsample size used for tree building (default)
#                 [to avoid too many empty leaves]
#   Float (>0): Multiplier of subsample size used for tree building
#   In particular for larger samples, using subsampling in evaluation may speed
#   up computations and reduces demand on memory.

#   Matching step to find neighbor for mce computation
CF_MATCH_NN_PROG_SCORE = None   # False: Use Mahalanobis matching.
#                                 True:  Use prognostic scores (default).

CF_NN_MAIN_DIAG_ONLY = None    # Only if match_nn_prog_score = False
#                                True: use main diagonal only.
#                                False (default): inverse of covariance matrix.

CF_RANDOM_THRESHOLDS = None     # If > 0: Do not check all possible split
#   values of ordered variables, but only RANDOM_THRESHOLDS (new randomisation
#   for each split)
#   0: no random thresholds
#   > 0: number of random thresholds used for ordered var's
#   Default: 4+N_train**0.2 (fewer thresholds speeds up programme but may (!)
#   lead to less accurate results.)

#   Minimum leaf size (use of a grid is possible)
CF_N_MIN_MIN = None      # Smallest minimum leaf size
CF_N_MIN_MAX = None      # Largest minimum leaf size.
#   Defaults are: n_min_min = round(max((n_d_subsam**0.4) / 10, 1.5)
#                                   * number of treatments
#   Defaults are: n_min_max = round(max((n_d_subsam**0.5) / 10, 2)
#                                   * number of treatments
#   (n_d denotes the number of observations in the smallest treatment arm)
CF_N_MIN_GRID = None     # Number of grid values (def: 1)
#   If n_min_grid == 1: n_min=(N_MIN_MIN+N_MIN_MAX)/2
#   If grid is used, optimal value is determined by out-of-bag estimation of
#   objective function.
CF_N_MIN_TREAT = None    # Minimum number of observations per treatment in leaf
#   A higher value reduces the risk that a leaf cannot be filled with
#   outcomes from all treatment arms in the evaluation subsample.
#   There is no grid based tuning for this parameter. This impacts the minimum
#   leaf size which will be at least n_min_treat * number of treatments.
#   Default is (N_MIN_MIN+N_MIN_MAX)/2 / # of treatments / 4. Minimum is 2.

#   Alpha regularity (use of a grid is possible)
#   Results of Wager and Athey (2018) suggest alpha_reg < 0.2. A larger value
#   may increase speed of tree building.
CF_ALPHA_REG_MIN = None      # smallest alpha, 0 <= alpha < 0.4 (def: 0.05)
CF_ALPHA_REG_MAX = None      # 0 <= alpha < 0.5 (def: 0.15)
CF_ALPHA_REG_GRID = None     # number of grid values (def: 1).
#   If ALPHA_REG_GRID == 1, alpha = (ALPHA_REG_MIN+ALPHA_REG_AX)/2
#   If grid is used, optimal value is determined by out-of-bag estimation of
#   objective function.

#    Variables used at each new split of tree
CF_M_SHARE_MIN = None        # minimum share of variables (0-1); def = 0.1
CF_M_SHARE_MAX = None        # maximum share of variables (0-1); def = 0.6
#   Number of variables used for splitting = share * total # of variable
#   If variables randomly selected for splitting do not show any variation in
#   leaf considered for splitting, then all variables will be used for that
#   split.
CF_M_GRID = None
#   Number of grid values logarithmically spaced, including m_min m_max.
#   Default is 1.
#   If M_GRID == 1: m_share = (M_SHARE_MIN+M_SHARE_MAX)/2
#   If grid is used, optimal value is determined by out-of-bag estimation of
#   objective function.

CF_TUNE_ALL = None           # Tune all parameters.
#   If True, all *_grid keywords will be set to 3. User specified values are
#   respected if larger than 3. Default is False.

CF_M_RANDOM_POISSON = None
#   True: Number of randomly selected variables is stochastic for each split,
#   drawn from a Poisson distribution.
#   Grid gives mean value of 1 + poisson distribution (m-1).
#   Default is True if M > 10, otherwise False (to avoid getting too few
#   variables that not have enough variation)
#   If grid is used, optimal value is determined by out-of-bag estimation of
#   objective function.

#   For large samples, randomly split the training data into equally sized
#   chunks, train a forest in each chunk, and estimate effects for each forest.
#   Final effect estimates are obtained by averaging effects obtained for each
#   forest. This procedures improves scalability by reducing computation time
#   and memory demand.
CF_CHUNKS_MAXSIZE = None  # Maximum number of observations allowed per chunk.
#   If CF_CHUNKS_MAXSIZE is smaller than sample size: Random splitting
#   If CF_CHUNKS_MAXSIZE is larger than sample size: No random splitting.
#   Default: If less than 90000 training observations: No splitting. Otherwise,
#   the maximal size of each chunksize is obtained as
#   100000 + (number of observations - 100000)**0.8 / (# of treatments-1)

#   Variable importance for causal forest
CF_VI_OOB_YES = None
#   True:  computes variable importance based on permuting every single x in
#          oob prediction; time consuming. Default is False.

#   ------------- Feature selection --------------------------------------------
FS_YES = None           # True: feature selection active.
#                         False: no feature selection (default).
FS_OTHER_SAMPLE = None  # False: Use sample used for causal forest estimation.
#                         True (default): Random sample from training data
#                         used. These data will not be used for causal forest.
FS_OTHER_SAMPLE_SHARE = None  # If FS_OTHER_SAMPLE: share of sample (def: 0.33)
FS_RF_THRESHOLD = None   # Threshold in terms of relative loss of variable
#                          importanance in % (default is 1).

#   ---------How to compute local centering (estimator choice and prediction) --
# & common support adjustment
LC_CS_CV = None        # True: Use crossvalidation (default).
#  False (estimator choice in local centering): Use LC_CS_SHARE for testing.
#  False (common support and local centering): Use
#  random subsample of the data that will then not be used for training the CF.
LC_CS_SHARE = None  # Share of data used for estimating E(y|x).
#                     0.1-0.9 (def = 0.25)
LC_CS_CV_K = None   # if LC_CS_CV: # of folds used in crossvalidation. Default
#   depends on the size of the training sample (N): N < 100'000: 5.
#                                                   100'000 <= N < 250'000: 4
#                                                   250'000 <= N < 500'000: 3
#                                                   500'000 <= N: 2

#   ------------- Local centering ----------------------------------------------
LC_YES = None              # Local centering. Default is True.
LC_ESTIMATOR = None         # The estimator used for local centering.

#   Possible choices are scikit-learn's regression methods
#  'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5'
#  'SupportVectorMachine', 'SupportVectorMachineC2', 'SupportVectorMachineC4',
#  'AdaBoost', 'AdaBoost100', 'AdaBoost200',
#  'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12',
#  'LASSO', 'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
#  'Mean'. (Mean is included if other estimators are not useful.)
#  If set to 'automatic', the estimator with the lowest out-of-sample
#  mean squared error (MSE) is selected. Whether this selection is based on
#  cross-validation or a test sample is governed by LC_CS_CV.
#  The default (or None) is 'RandomForest'.

LC_UNCENTER_PO = None       # Predicted potential outcomes for individual data
#   points (IATEs) re-adjusted for local centering are added to data output.
#   Default is True.

#   ------------- Common support -----------------------------------------------
CS_TYPE = None
#   0: No common support adjustment
#   1,2: Support check based on estimated classification random forests.
#   1: Use min-max rules for probabilities in treatment subsamples .
#   2: Enforce minimum and maximum probabilities for all obs and all probs-1
#   observations off support are removed.
#   Out-of-bag predictions are used to avoid overfitting (which would lead to a
#   too large reduction in the number of observations). Default (None) is 1.
CS_ADJUST_LIMITS = None
#   IF CS_TYPE > 0:  > 0 or None (default); 0: No adjustment
#   max *= 1+SUPPORT_ADJUST_LIMITS, min *= 1-SUPPORT_ADJUST_LIMITS
#   The restrictiveness of the common support criterion increases with the
#   number of treatments. This parameter reduces this restrictiveness.
#   Default (None): (number of treatments - 2) * 0.05
CS_QUANTIL = None
#   If CS_TYPE == 1: 1: min-max rule; 0-1: if smaller than 1,
#   the respective quantil is taken for the cut-offs; default: min-max
CS_MIN_P = None
#   If cs_type == 2: observations are deleted if p(d=m|x) is less or equal
#                   than 'cs_min_p' for at least one treatment.  def: 0.01
CS_MAX_DEL_TRAIN = None
#   If share of observations in training data used that are
#   OFF support is larger than SUPPORT_MAX_DEL_TRAIN (0-1), programme will be
#   terminated and user should change input data. Default is 0.5.
CS_DETECT_CONST_VARS_STOP = None
#   Control variables that have no variation inside a treatment arm violate the
#   common support condition. If CS_DETECT_VARS_NO_VAR_STOP is True, data will
#   checked for such variables and an exception is raised if such a variable is
#   detected. Then, the user has to decide to either adjust the data (by
#   deleting observations with the value of the variable that creates the
#   problem) or delete this variable. Default is True.

#   --- Switch role of forest and y-sample and average prediction
#                    -- inference is not possible --
GEN_IATE_EFF = None      # True: Second round of IATEs are estimated
#   based on switching training and estimation subsamples. This increase their
#   efficiency, but does not allow inference on these more efficient estimates.
#   If False, execution time is faster. Default is False.

#   ------------- Clustering and panel data ------------------------------------
GEN_PANEL_DATA = None    # True if panel data; None or False: no panel data.
#   True activates the clustered standard error. Default is False.
#   Use cluster_name to define variable that contains identifier for panel unit
GEN_PANEL_IN_RF = None   # Uses the panel structure also when building the
#   random samples within the forest procedure. Default is True.
P_CLUSTER_STD = None    # True: Clustered standard error. Default is False.
#   Will be automatically set to True if panel data option is activated.

#   ------------- Parameters for continuous treatment --------------------------
#   Number of grid point for discretization of continuous treatment (with 0
#   mass point).
#   Grid is defined in terms of quantiles of continuous part of treatment.
CT_GRID_NN = None  # Used to aproximate the neighbourhood matching (def. is 10)
CT_GRID_W = None   # Used to aproximate the weights (def. is 10)
CT_GRID_DR = None  # Used to aproximate the dose response function (def.: 100)

#   ------------- Balancing test (beta) ----------------------------------------
P_BT_YES = None  # True: ATE based balancing test based on weights.
#   Requires weight_based_inference. Relevance of this test in its current
#   implementation is not fully clear. Default is False.

#   ---- Choice based sampling to speed up programme if treatment groups have
#   ---- very different sizes
P_CHOICE_BASED_SAMPLING = None  # True: Choice based sampling.Default is False.
P_CHOICE_BASED_PROBS = None     # Sampling probabilities to be
#   specified. These weights are used for (g,b)ates only. Treatment information
#   must therefore be available in the prediction data.

#   ------------- Sample weights -----------------------------------------------
GEN_WEIGHTED = None          # True: use sampling weights. Default is False.
#   Sampling weights specified in var_w_name will be used;
#   slows down programme. Experimental.

#   ------------- Predicting effects -------------------------------------------
#   Truncation of extreme weights
P_MAX_WEIGHT_SHARE = None  # Maximum share of any weight, 0 <= 1, default=0.05
#   Enforced by trimming excess weights and renormalisation for each
#   (b,g,i) ate separately; because of renormalising, the final weights could
#    be somewhat above this threshold.

P_CI_LEVEL = None     # 0 < 1: Confidence level for bounds used in plots.
#                              Default is 0.95.

#   Weight based inference
P_COND_VAR = None   # False: Variance estimation uses var(wy).
#   True: conditional mean & variances are used. Default is True.
P_KNN = None        # False: Nadaraya-Watson estimation.
#                     True: knn estimation (faster). Default is True.
#   Nadaray-Watson estimation gives a better approximaton of the variance, but
#   k-NN is much faster, in particular for larger datasets.
P_KNN_MIN_K = None  # k: Minimum number of neighbours k-nn estimation(def:10).
P_KNN_CONST = None  # Multiplier of default number of observation used in
#                      moving average of analyses method. Default is 1.
P_NW_BANDW = None   # Bandwidth for nw estimation; multiplier of Silverman's
#                     optimal bandwidth. Default is 1.
P_NW_KERN = None    # Kernel for nw estimation: 1: Epanechikov (def); 2: normal
#                     Default is 1.

# Bootstrap of standard errors (P_SE_BOOT_ATE, P_SE_BOOT_GATE, P_SE_BOOT_IATE,
# P_SE_BOOT_IATE)
# Specify either a Boolean (if True, number of bootstrap replications will be
# set to 199) or an integer corresponding to the number of bootstrap
# replications; this implicity implies True). The default is False if
# CLUSTER_STD is False and True with 199 replications CLUSTER_STD is True.
P_SE_BOOT_ATE = None     # (w_ji * y_i) are bootstrapped SE_BOOT_xATE times
P_SE_BOOT_GATE = None    # False: No Bootstrap SE of effects
P_SE_BOOT_IATE = None    # True: SE_BOOT_xATE = 199
P_SE_BOOT_QIATE = None

#   GATE estimation of variables with many values
P_MAX_CATS_Z_VARS = None  # Maximum number of categories for discretizing
#                           continuous z variables. Default is n**.3.

P_ATE_NO_SE_ONLY = None  # True: Computes only the ATE without standard errors.
#   Default is False.

#   Treatment-group specific effects
P_ATET = None      # True: Average effects computed for subpopulations by
#   treatments (if available). Default is False.
P_GATET = None     # True: Gate's for subpopulations by treatments. Default
#   is False. If there no variables specified for gate estimation, p_bgate is
#   set to False.

#  Balanced GATEs
P_BGATE = None     # True: BGATEs will be computed. Default is False.
#   True requires to specify the variable names to balance on in VAR_BGATE_NAME
#   If no variables are specified for gate estimation, p_bgate is set to
#   False.
P_CBGATE = None    # True: CBGATEs will be computed. Default is False.
#   If there are no variables specified for gate estimation, p_cbgate is set to
#   False.

#   More details for (CB)GATE estimation
P_GATES_NO_EVALU_POINTS = None  # Number of evluation points for
#   continuous variables (GATE, BGATE, CBGATE). Default is 50.
P_GATES_SMOOTH = None  # Alternative way to estimate GATEs for continuous
#   variables. Instead of discretizing variable, its GATE is evaluated at
#   p_gates_smooth_no_evalu_points. Since there are likely to be no
#   observations, a local neighbourhood around the evaluation points is
#   considered. Default is True.
P_GATES_SMOOTH_BANDWIDTH = None  # Multiplier for SGATE aggregation. Def. is 1.
P_GATES_SMOOTH_NO_EVALU_POINTS = None  # Default is 50.
P_GATES_MINUS_PREVIOUS = None   # GATES will not only be compared to ATE but
#   also to GATES computed at the previous evaluation point. Default is False.
#   If True, GATE estimation is slower as it is not optimized for
#   multiprocessing and no plots are shown for this parameter.
P_BGATE_SAMPLE_SHARE = None   # (0<1) Implementation is very cpu intensive.
#   Random samples are used to speed up programme if more obs / # of evaluation
#   points > 10. # Default is 1 if n_prediction < 1000; otherwise:
#   (1000 + (n_pred-1000) ** (3/4))) / # of evaluation points

#   Estimation of IATEs and their standard errors
P_IATE = None         # True: IATEs will be estimated. Default is True.
P_IATE_SE = None      # True: SE(IATE) will be estimated. Default is False.
#   Estimating IATEs and their standard errors may be time consuming
P_IATE_M_ATE = None      # True: IATE(x) - ATE is estimated,
#   including inference if p_iate_se == True. Increaes computation time.
#   Default is False.

#   Estimation of QIATEs and their standard errors
P_QIATE = None         # True: QIATEs will be estimated. If True, p_iate will
#   always be set to True. Default is False.
P_QIATE_SE = None      # True: SE(QIATE) will be estimated. Default is False.
#   Estimating IATEs and their standard errors may be time consuming
P_QIATE_M_MQIATE = None      # True: QIATE(x) - median(IATE(x)) is estimated,
#   including inference if p_qiate_se == True. Increaes computation time.
#   Default is False.
P_QIATE_M_OPP = True      # True: QIATE(x, q) - QIATE(x, 1-q) is estimated,
#   including inference if p_qiate_se == True. Increaes computation time.
#   Default is False.
P_QIATE_NO_OF_QUANTILES = None   # Number of quantiles for which QIATEs are
#   computed. Default is 99.
P_QIATE_SMOOTH = None            # Smooth estimated QIATEs using kernel
#   smoothing. Default is True.
P_QIATE_SMOOTH_BANDWIDTH = None  # Multiplier applied to default bandwidth
#   used for kernel smoothing of QIATE. Default is 1.
P_QIATE_BIAS_ADJUST = None       # Bias correction procedure for QIATEs based on
#   simulations. Default is True.
#   If P_QIATE_BIAS_ADJUST is True, P_IATE_SE is set to True as well.

#   ------------- Analysis of effects ------------------------------------------
POST_EST_STATS = None   # Analyses the predictions by binary correlations
#   or some regression type methods. Default is True. False if p_iate == False.
POST_RELATIVE_TO_FIRST_GROUP_ONLY = None  # Use only effects relative to
#   treatment with lowest treatment value. Default is True.
POST_BIN_CORR_YES = None    # Checking the binary correlations of predictions
#   with features. Default is True.
POST_BIN_CORR_THRESHOLD = None  # Minimum threshhold of absolute correlation
#                                 to be displayed. Default is 0.1.
POST_PLOTS = None      # Plots of estimated treatment effects. Default is True.
POST_KMEANS_YES = None  # Using k-means clustering to analyse
#                         patterns in the estimated effects. Default is True.
POST_KMEANS_SINGLE = None   # If True, clustering is also with respect to
#                              all single effects. Default is False.
POST_KMEANS_NO_OF_GROUPS = None  # post_kmeans_yes is True: # of clusters
#   to be built: Integer, list or tuple (or None --> default).
#   Default: List of 5 values: [a, b, c, d, e]; c = 5 to 10; depending on n;
#   c<7: a=c-2, b=c-1, d=c+1, e=c+2 else a=c-4, b=c-2, d=c+2, e=c+4
POST_KMEANS_REPLICATIONS = None  # post_kmeans_yes is True: # of replications
#   with random start centers to avoid local extrema. Default is 10.
POST_KMEANS_MAX_TRIES = None     # post_kmeans_yes is True: maximum number
#   of iterations in each replication to achive convergence. Default is 1000.
POST_KMEANS_MIN_SIZE_SHARE = None  # Smallest share of cluster size allowed
#                                    in % (0-33). Default (None) is 1.
POST_RANDOM_FOREST_VI = None     # Variable importance measure of predictive
#   random forest used to learn factors influencing IATEs. Default is True.

POST_TREE = None     # Regression trees (honest and standard) of Depth 2 to 5
#   are estimated to describe IATES(x). Default (or None) is True.

#   ----------------- Sensitivity method (experimental) ------------------------
SENS_CBGATE = None
#   Boolean (or None), optional. Compute CBGATEs for sensitivity analysis.
#   Default is False.
SENS_BGATE = None
#   Boolean (or None), optional.  Compute BGATEs for sensitivity analysis.
#   Default is False.
SENS_GATE = None
#   Boolean (or None), optional. Compute GATEs for sensitivity analysis.
#   Default is False.
SENS_IATE = None
#   Boolean (or None), optional. Compute IATEs for sensitivity analysis.
#   If the results dictionary is passed, and it contains IATEs, then the
#   default value is True, and False otherwise.
SENS_IATE_SE = None
#   Boolean (or None), optional. Compute Standard errors of IATEs for
#   sensitivity analysis. Default is False.
SENS_SCENARIOS = None
#  List or tuple of strings, optional. Different scenarios considered.
#  Default is ('basic',).
#  'basic' : Use estimated treatment probabilities for simulations.
#            No confounding.
SENS_CV_K = None
#  Integer (or None), optional. Data to be used for any cross-validation:
#  Number of folds in cross-validation. Default (or None) is 5.
SENS_REPLICATIONS = None
#  Integer (or None), optional. Number of replications for simulating placebo
#  treatments. Default is 2.
SENS_REFERENCE_POPULATION = None
#   Integer or float (or None). Defines the treatment status of the reference
#   population used by the sensitivity analysis. Default is to use the
#   treatment with most observed observations.

# ----- Internal variables: Change these variables only if you know what you do

_INT_CUDA = None                 # Use CUDA based GPU if available on hardware.
#                                  Experimental feature. Default is False.
_INT_REPLICATION = None          # True does not allow multiprocessing in
#   local centering, feature selection, and common support. Default is False.
_INT_REPORT = None              # True: Provide information for McfOptPolReports
#                                 to construct informative reports.
#                                 Default is True.
_INT_WITH_OUTPUT = None         # Print output on txt file and/or console.
#                                 Default is True.
_INT_OUTPUT_NO_NEW_DIR = None   # Do not create a new directory when the path
#                                 already exists. Default is False.
_INT_VERBOSE = None             # True (def): Output about running of programme
_INT_DESCRIPTIVE_STATS = None   # Print descriptive stats of input+output files
#                                 controls for all figures
_INT_SHOW_PLOTS = None          # Execute plt.show() command. Default is True.
#                                 turn off if programme runs on console.
_INT_FONTSIZE = None            # Legend, 1(very small) to 7(very large); def:2
_INT_DPI = None                 # > 0: default (or None): 500
#   Only for (B, AM) GATEs: What type of plot to use for continuous variables.
_INT_NO_FILLED_PLOT = None      # Use filled plot if more than x points(def:20)
_INT_WEIGHT_AS_SPARSE = None    # Weights matrix stored as sparse matrix.
#                                 Default (or None) is True.

_INT_IATE_CHUNK_SIZE = None     # Number of IATEs that are estimated in a
#   ray worker. Default is number of prediction observations / workers.
#   If programme If programme crashes in IATE in the second part of IATE
#   estimation (2/2) because of excess memory consumption, reduce this number.

_INT_WEIGHT_AS_SPARSE_SPLITS = None  # Sparse weight matrix computed in several
#   chunks. Default is
#   Rows of prediction data * rows of Fill_y data / number of forests /
#   / (25'000 * 25'000))
_INT_SHARE_FOREST_SAMPLE = None
#   0-1: Share of sample used build forest. Default is 0.5.
_INT_MAX_CATS_CONT_VARS = None  # Discretising of continuous variables: maximum
#   number of categories for continuous variables n values < n speed up
#   programme, default: not used.

_INT_MAX_SAVE_VALUES = None     # Save values of x only if < 50 (cont. vars).
#                                 Default is 50.
_INT_SEED_SAMPLE_SPLIT = None   # Seeding is redone when building forest
#                                 Default is 67567885.
_INT_MP_VIM_TYPE = None         # Variable importance: type of mp
#                                 1: variable based (fast, lots of memory)
#                                 2: bootstrap based (slower, less memory)
#                                 Def: 1 if n < 20000, 2 otherwise
_INT_MP_WEIGHTS_TYPE = None     # Weights computation: type of mp
#                                 1: groups-of-obs based (fast, lots of memory)
#                                 2: tree based (takes forever, less memory)
#                                 Default is 1. Variable is overwritten (set to
#                                 1 if multiprocessing is used).
_INT_MP_WEIGHTS_TREE_BATCH = None  # Weight computation: Split forests for
#   variable importance computations.
#   Few batches: More speed, more memory.  Default: Automatically determined.
#   Soon to be depreciated.

_INT_MP_RAY_DEL = None           # Tuple with any of the following:
#   'refs': Delete references to object store (default)
#   'rest': Delete all other objects of Ray task
#   'none': Nothing is deleted.
#   These 3 options can be combined.
# Default (or None) is 'refs'
_INT_MP_RAY_SHUTDOWN = None  # Shutdown ray task by task (default is False if
#   N < 100'000 and True otherwise)
#   If programme collapses because of excess memory reduce workers or set
#   _INT_MP_RAY_SHUTDOWN is True.
#   When using this programme repeatedly like in Monte Carlo studies always
#   using _INT_MP_RAY_SHUTDOWN is True may be a good idea.
_INT_MP_RAY_OBJSTORE_MULTIPLIER = None  # Increase internal default values for
#   Ray object store above 1 if programme crashes because object store is full
#   (default is 1); ONLY RELEVANT if _INT_MP_RAY_SHUTDOWN is True.

_INT_RETURN_IATE_SP = None   # Return all data with predictions
#   despite with_output = False (useful for cross-validation and simulations.
#   Default is False.
_INT_DEL_FOREST = None     # Delete forests.
#   Delete forests from instance. If True, less memory is needed, but the
#   trained instance of the class cannot be reused when calling predict with
#   the same instance again, i.e. the forest has to be retrained.
#   Default is False.
_INT_KEEP_W0 = None            # Keep all zeros weights when computing
#   standard errors (slows down computation). Default is False.

# The following keywords define upper limits for sample size. If actual number
# is larger then the prespecified number, then the respective data will
# randomly reduced to the specified upper limit.
_INT_OBS_BIGDATA = None  # If number of training observations is larger
# than _INT_OBS_BIGDATA, the following happens during training and prediction:
#    (i) Number of workers is halved in local centering.
#    (ii) The number of workers used is reduced to 75% of default.
#    (iii) The data type for many numpy arrays is reduced from float64 to
#         float32.
#    Default is 1'000'000.
_INT_MAX_OBS_TRAINING = None     # Training method: Reducing observations
#    for training increases MSE and thus should be avoided.
#    Default is infinity.
_INT_MAX_OBS_PREDICTION = None   # Prediction method: Reducing observations
#    for prediction does not much affect MSE. It may reduce detectable
#    heterogeneity, but may also dramatically reduce computation time.
#    Default is 250'000.
_INT_MAX_OBS_KMEANS = None      # kmeans in analye method: Reducing observations
#    may reduce detectable heterogeneity, but also reduces computation time.
#    Default is 200'000.
_INT_MAX_OBS_POST_REL_GRAPHS = None  # Figures showing relation of IATEs and
#   features (in-built non-parametric regression is computationally intensive).
#   Default is 50'000.

# ---------------------------------------------------------------------------
# For convenience the mcf parameters are collected and passed as a dictionary.
# Of course, they can also be passed as single parameters (or not at all, in
# which case default values are used).

params = {
    'cf_alpha_reg_grid': CF_ALPHA_REG_GRID,
    'cf_alpha_reg_max': CF_ALPHA_REG_MAX, 'cf_alpha_reg_min': CF_ALPHA_REG_MIN,
    'cf_boot': CF_BOOT, 'cf_chunks_maxsize': CF_CHUNKS_MAXSIZE,
    'cf_compare_only_to_zero': CF_COMPARE_ONLY_TO_ZERO,
    'cf_n_min_grid': CF_N_MIN_GRID,
    'cf_n_min_max': CF_N_MIN_MAX, 'cf_n_min_min': CF_N_MIN_MIN,
    'cf_n_min_treat': CF_N_MIN_TREAT,
    'cf_nn_main_diag_only': CF_NN_MAIN_DIAG_ONLY,
    'cf_m_grid': CF_M_GRID, 'cf_m_share_max': CF_M_SHARE_MAX,
    'cf_m_random_poisson': CF_M_RANDOM_POISSON,
    'cf_m_share_min': CF_M_SHARE_MIN,
    'cf_match_nn_prog_score': CF_MATCH_NN_PROG_SCORE,
    'cf_mce_vart': CF_MCE_VART,
    'cf_random_thresholds': CF_RANDOM_THRESHOLDS,
    'cf_p_diff_penalty': CF_P_DIFF_PENALTY, 'cf_penalty_type': CF_PENALTY_TYPE,
    'cf_subsample_factor_eval': CF_SUBSAMPLE_FACTOR_EVAL,
    'cf_subsample_factor_forest': CF_SUBSAMPLE_FACTOR_FOREST,
    'cf_tune_all': CF_TUNE_ALL,
    'cf_vi_oob_yes': CF_VI_OOB_YES,
    'cs_adjust_limits': CS_ADJUST_LIMITS, 'cs_max_del_train': CS_MAX_DEL_TRAIN,
    'cs_min_p': CS_MIN_P, 'cs_quantil': CS_QUANTIL, 'cs_type': CS_TYPE,
    'ct_grid_dr': CT_GRID_DR, 'ct_grid_nn': CT_GRID_NN, 'ct_grid_w': CT_GRID_W,
    'dc_check_perfectcorr': DC_CHECK_PERFECTCORR,
    'dc_clean_data': DC_CLEAN_DATA, 'dc_min_dummy_obs': DC_MIN_DUMMY_OBS,
    'dc_screen_covariates': DC_SCREEN_COVARIATES,
    'fs_rf_threshold': FS_RF_THRESHOLD, 'fs_other_sample': FS_OTHER_SAMPLE,
    'fs_other_sample_share': FS_OTHER_SAMPLE_SHARE, 'fs_yes': FS_YES,
    'gen_d_type': GEN_D_TYPE,  'gen_iate_eff': GEN_IATE_EFF,
    'gen_panel_data': GEN_PANEL_DATA, 'gen_panel_in_rf': GEN_PANEL_IN_RF,
    'gen_weighted': GEN_WEIGHTED, 'gen_mp_parallel': GEN_MP_PARALLEL,
    'gen_outfiletext': GEN_OUTFILETEXT, 'gen_outpath': GEN_OUTPATH,
    'gen_output_type': GEN_OUTPUT_TYPE,
    'lc_cs_cv': LC_CS_CV, 'lc_cs_cv_k': LC_CS_CV_K, 'lc_cs_share': LC_CS_SHARE,
    'lc_estimator': LC_ESTIMATOR,
    'lc_uncenter_po': LC_UNCENTER_PO, 'lc_yes': LC_YES,
    'post_bin_corr_threshold': POST_BIN_CORR_THRESHOLD,
    'post_bin_corr_yes': POST_BIN_CORR_YES, 'post_est_stats': POST_EST_STATS,
    'post_kmeans_no_of_groups': POST_KMEANS_NO_OF_GROUPS,
    'post_kmeans_max_tries': POST_KMEANS_MAX_TRIES,
    'post_kmeans_replications': POST_KMEANS_REPLICATIONS,
    'post_kmeans_yes': POST_KMEANS_YES,
    'post_kmeans_single': POST_KMEANS_SINGLE,
    'post_kmeans_min_size_share': POST_KMEANS_MIN_SIZE_SHARE,
    'post_random_forest_vi': POST_RANDOM_FOREST_VI,
    'post_relative_to_first_group_only': POST_RELATIVE_TO_FIRST_GROUP_ONLY,
    'post_plots': POST_PLOTS, 'post_tree': POST_TREE,
    'p_ate_no_se_only': P_ATE_NO_SE_ONLY, 'p_cbgate': P_CBGATE,
    'p_atet': P_ATET, 'p_bgate': P_BGATE, 'p_bt_yes': P_BT_YES,
    'p_choice_based_sampling': P_CHOICE_BASED_SAMPLING,
    'p_choice_based_probs': P_CHOICE_BASED_PROBS, 'p_ci_level': P_CI_LEVEL,
    'p_cluster_std': P_CLUSTER_STD, 'p_cond_var': P_COND_VAR,
    'p_gates_minus_previous': P_GATES_MINUS_PREVIOUS,
    'p_gates_smooth': P_GATES_SMOOTH,
    'p_gates_smooth_bandwidth': P_GATES_SMOOTH_BANDWIDTH,
    'p_gates_smooth_no_evalu_points': P_GATES_SMOOTH_NO_EVALU_POINTS,
    'p_gatet': P_GATET, 'p_gates_no_evalu_points': P_GATES_NO_EVALU_POINTS,
    'p_bgate_sample_share': P_BGATE_SAMPLE_SHARE,
    'p_iate': P_IATE, 'p_iate_se': P_IATE_SE, 'p_iate_m_ate': P_IATE_M_ATE,
    'p_knn': P_KNN, 'p_knn_const': P_KNN_CONST, 'p_knn_min_k': P_KNN_MIN_K,
    'p_nw_bandw': P_NW_BANDW, 'p_nw_kern': P_NW_KERN,
    'p_max_cats_z_vars': P_MAX_CATS_Z_VARS,
    'p_max_weight_share': P_MAX_WEIGHT_SHARE,
    'p_qiate': P_QIATE, 'p_qiate_se': P_QIATE_SE,
    'p_qiate_m_mqiate': P_QIATE_M_MQIATE,
    'p_qiate_m_opp': P_QIATE_M_OPP,
    'p_qiate_no_of_quantiles': P_QIATE_NO_OF_QUANTILES,
    'p_qiate_smooth': P_QIATE_SMOOTH,
    'p_qiate_smooth_bandwidth': P_QIATE_SMOOTH_BANDWIDTH,
    'p_qiate_bias_adjust': P_QIATE_BIAS_ADJUST,
    'p_se_boot_ate': P_SE_BOOT_ATE, 'p_se_boot_gate': P_SE_BOOT_GATE,
    'p_se_boot_iate': P_SE_BOOT_IATE,
    'var_x_name_balance_bgate': VAR_X_NAME_BALANCE_BGATE,
    'p_se_boot_qiate': P_SE_BOOT_QIATE,
    'var_cluster_name': VAR_CLUSTER_NAME, 'var_d_name': VAR_D_NAME,
    'var_id_name': VAR_ID_NAME,
    'var_w_name': VAR_W_NAME,
    'var_x_name_balance_test_ord': VAR_X_NAME_BALANCE_TEST_ORD,
    'var_x_name_balance_test_unord': VAR_X_NAME_BALANCE_TEST_UNORD,
    'var_x_name_always_in_ord': VAR_X_NAME_ALWAYS_IN_ORD,
    'var_x_name_always_in_unord': VAR_X_NAME_ALWAYS_IN_UNORD,
    'var_x_name_remain_ord': VAR_X_NAME_REMAIN_ORD,
    'var_x_name_remain_unord': VAR_X_NAME_REMAIN_UNORD,
    'var_x_name_ord': VAR_X_NAME_ORD, 'var_x_name_unord': VAR_X_NAME_UNORD,
    'var_y_name': VAR_Y_NAME, 'var_y_tree_name': VAR_Y_TREE_NAME,
    'var_z_name_cont': VAR_Z_NAME_CONT, 'var_z_name_ord': VAR_Z_NAME_ORD,
    'var_z_name_unord': VAR_Z_NAME_UNORD,
    '_int_cuda': _INT_CUDA,
    '_int_del_forest': _INT_DEL_FOREST,
    '_int_descriptive_stats': _INT_DESCRIPTIVE_STATS, '_int_dpi': _INT_DPI,
    '_int_fontsize': _INT_FONTSIZE,
    '_int_iate_chunk_size': _INT_IATE_CHUNK_SIZE,
    '_int_keep_w0': _INT_KEEP_W0,
    '_int_no_filled_plot': _INT_NO_FILLED_PLOT,
    '_int_max_cats_cont_vars': _INT_MAX_CATS_CONT_VARS,
    '_int_max_save_values': _INT_MAX_SAVE_VALUES,
    '_int_max_obs_training': _INT_MAX_OBS_TRAINING,
    '_int_max_obs_prediction': _INT_MAX_OBS_PREDICTION,
    '_int_max_obs_kmeans': _INT_MAX_OBS_KMEANS,
    '_int_max_obs_post_rel_graphs': _INT_MAX_OBS_POST_REL_GRAPHS,
    '_int_obs_bigdata': _INT_OBS_BIGDATA,
    '_int_report': _INT_REPORT,
    '_int_mp_ray_del': _INT_MP_RAY_DEL,
    '_int_mp_ray_objstore_multiplier': _INT_MP_RAY_OBJSTORE_MULTIPLIER,
    '_int_mp_ray_shutdown': _INT_MP_RAY_SHUTDOWN,
    '_int_mp_vim_type': _INT_MP_VIM_TYPE,
    '_int_mp_weights_tree_batch': _INT_MP_WEIGHTS_TREE_BATCH,
    '_int_mp_weights_type': _INT_MP_WEIGHTS_TYPE,
    '_int_output_no_new_dir': _INT_OUTPUT_NO_NEW_DIR,
    '_int_replication': _INT_REPLICATION,
    '_int_return_iate_sp': _INT_RETURN_IATE_SP,
    '_int_seed_sample_split': _INT_SEED_SAMPLE_SPLIT,
    '_int_share_forest_sample': _INT_SHARE_FOREST_SAMPLE,
    '_int_show_plots': _INT_SHOW_PLOTS, '_int_verbose': _INT_VERBOSE,
    '_int_weight_as_sparse': _INT_WEIGHT_AS_SPARSE,
    '_int_weight_as_sparse_splits': _INT_WEIGHT_AS_SPARSE_SPLITS,
    '_int_with_output': _INT_WITH_OUTPUT
    }

params_sensitivity = {
    'sens_cbgate': SENS_CBGATE, 'sens_bgate': SENS_BGATE,
    'sens_gate': SENS_GATE, 'sens_iate': SENS_IATE,
    'sens_iate_se': SENS_IATE_SE, 'sens_scenarios': SENS_SCENARIOS,
    'sens_cv_k': SENS_CV_K, 'sens_replications': SENS_REPLICATIONS,
    'sens_reference_population': SENS_REFERENCE_POPULATION
     }

# ------------------------------------------------------------------------------
# Modules may sent many irrelevant warnings: Globally ignore them
warnings.filterwarnings('ignore')

# Some data to be used by the method
zeros_df = pd.DataFrame(np.zeros(len(prediction_df)), columns=('zeros',))
ones_df = pd.DataFrame(np.ones(len(prediction_df)), columns=('ones',))
if NO_TREATMENTS < 3:
    test_alloc_df = pd.concat((prediction_df[VAR_D_NAME], zeros_df, ones_df, ),
                              axis=1)
else:
    twos_df = pd.DataFrame(np.ones(len(prediction_df))*2, columns=('twos',))
    test_alloc_df = pd.concat((prediction_df[VAR_D_NAME], zeros_df, ones_df,
                               twos_df,),
                              axis=1)
test_alloc_df.rename(columns={'treat': 'observed'}, inplace=True)
# ------------------------------------------------------------------------------

mymcf = ModifiedCausalForest(**params)

mymcf.train(training_df)
results = mymcf.predict(prediction_df)

results_with_cluster_id_df = mymcf.analyse(results)

mymcf.predict_different_allocations(prediction_df, test_alloc_df)

params['gen_outpath'] = GEN_OUTPATH / 'sensitivity'
mymcf_sens = ModifiedCausalForest(**params)
params_sensitivity['results'] = results
mymcf_sens.sensitivity(training_df, **params_sensitivity)

my_report = McfOptPolReport(mcf=mymcf, mcf_sense=mymcf_sens)
my_report.report()

print('End of computations.\n\nThanks for using the ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600')
