"""Created on Wed Apr  1 15:58:30 2020.

Modified Causal Forest - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

Version: 0.0.2

-*- coding: utf-8 -*- .

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Change-log:
New in version 0.0.1
- Optional: Bootstrap standard errors
  SE_BOOT_ATE       # False: No Bootstrap standard errors for ATE, ...
  SE_BOOT_GATE      # > 99: (w_ji y_i) are bootstrapped SE_BOOT_*** times
  SE_BOOT_IATE      # (weights renormalized). Default is False
- Bug fix in objective function (incorrect weighting and incorrect
                                 aggregation for tuning and VI)
- Changes in dealing with common support
  SUPPORT_MAX_DEL_TRAIN = 0.5
    If share of observations in training data used for forest data that are
    OFF support is larger than SUPPORT_MAX_DEL_TRAIN (0-1), programme will be
   terminated and user should change input data.
  Common support checks and corrections are now done before any estimation and
  relate to all data files.
- Use Ray 1.4.1 or higher for Python 3.9 support [pip install ray[default]]
- If specified path for output does already exist, a new path will be created 
  and used (solves some issue with writing rights that occured occassionaly)
  
New in version 0.0.2
- Tool for optimal policy allocation now also available (uses predicted
                                                         effects)
- Kmeans uses a ranges of number of clusters and determines optimal cluster
  by silhoutte analysis (POST_KMEANS_NO_OF_GROUPS can now be a list or tuple
                         with possible cluster sizes)
- (Experimental) Optimal Policy Tool included bases on estimated IATEs 
  (allowing implicitly for constraints and programme costs)
- Bug fixes for problems computing treatment effects for treatment populations
- Detection of numerical variables added. Raises Exception.
- All variables used in initial treatment specific statistics (important
  information to detect common support issues)
- Documentation only: Treatment specific statistics will only be printed for
  those variables used to check the balancing of the sample
- Fix some bugs for use of panel data and clustering
- Improved stat's for common support analysis
- Bug fix for dealing with missings in data cleaning
"""
from mcf import mcf_functions as mcf

TRAIN_MCF = True       # Train the forest; def: True
PREDICT_MCF = True     # Estimate effects; def: True
SAVE_FOREST = True    # Save forest for predict. w/o reestimation def:False
FOREST_FILES = None   # File name for all information needed for prediction
#   If a name is specified, then files with an *.csv, *.pickle, *ps.npy, *d.npy
#   extensions will be automatically specified.
#   If None, file names will be same as indat with extensions
#   *_savepred.pickle, *_savepred.csv,  *_savepredps.npy, *_savepredd.npy
#                                      These 4 files will be saved in outpfad.

APPLIC_PATH = 'D:/mlechner/mcftest'  # NOT passed to MCF
OUTPATH = APPLIC_PATH + '/out'
#   If this None a */out directory below the current directory is used
#   If specified directory does not exist, it will be created.
DATPATH = APPLIC_PATH + '/testdata'
#   If a path is None, path of this file is used.

INDATA = 'dgp_mcfN1000S5'           # csv for estimation (without extension)
PREDDATA = 'dgp_mcfN1000S5_x_only'  # csv for effects
#   csv extension is added automatically to both file names
#   If preddata is not specified, indata will be used as file name.

OUTFILETEXT = INDATA + "mcf.py.0.0.2"  # File for text output
#   if outfiletext is None, name of indata with extension .out is used

#   Variables for estimation
VAR_FLAG1 = False   # No variable of mcf, used just for this test

#   The following variables must be provided:
#                 d_name, y_name, x_name_ord or x_name_unord
#   All other variables are optional. Lists or None must be used.

D_NAME = ['d']          # Treatment: Must be discrete (not needed for pred)
Y_NAME = ['y']          # List of outcome variables (not needed for pred)
Y_TREE_NAME = ['y']     # Variable to build trees (not needed for pred)
#   if None or [], the first variable in y_name is used to build trees.
#   it will be added to the list of outcome variablkes

#   Features, predictors, independent variables, confounders: ordered
X_NAME_ORD = []
if VAR_FLAG1:
    for i in range(120):
        X_NAME_ORD.append('cont' + str(i))
    for i in range(10):
        X_NAME_ORD.append('dum' + str(i))
    for i in range(10):
        X_NAME_ORD.append('ord' + str(i))
else:
    for i in range(5):
        X_NAME_ORD.append('cont' + str(i))
    for i in range(3):
        X_NAME_ORD.append('dum' + str(i))
    for i in range(3):
        X_NAME_ORD.append('ord' + str(i))

#   Features, predictors, independent variables: unordered, categorial

X_NAME_UNORD = []
if VAR_FLAG1:
    for i in range(10):
        X_NAME_UNORD.append('cat' + str(i))
else:
    for i in range(2):
        X_NAME_UNORD.append('cat' + str(i))

#   Identifier
ID_NAME = ['ID']
#   If no identifier -> it will be added the data that is saved for later use
#   Cluster and panel identifiers
CLUSTER_NAME = ['cluster']  # Variable defining the clusters if used
W_NAME = ['weight']         # Name of weight, if weighting option is used

# Variables always included when deciding next split
X_NAME_ALWAYS_IN_ORD = []      # (not needed for pred)
X_NAME_ALWAYS_IN_UNORD = []    # (not needed for pred)

#   Variables to define policy relevant heterogeneity in multiple treatment
#   procedure they will be added to the conditioning set. If not included,
#   they will be automatically added to the list of confounders.

#   Ordered variables with many values (put discrete variables with few values
#   in category below). These variables are recoded to define the split, they
#   will be added to the list of confounders. Since they are broken up in
#   categories for all their observed values, it does not matter whether they
#   are coded as ordered or unordered.
if VAR_FLAG1:
    Z_NAME_LIST = ['cont0', 'cont2']
else:
    Z_NAME_LIST = ['cont0']

#   Variables that are discrete and define a unique sample split for each value
#   Ordered variables
if VAR_FLAG1:
    Z_NAME_SPLIT_ORD = ['dum0', 'dum2', 'ord0', 'ord2']
else:
    Z_NAME_SPLIT_ORD = ['ord0', 'dum0']
#   Unordered variables
if VAR_FLAG1:
    Z_NAME_SPLIT_UNORD = ['cat0', 'cat2']
else:
    Z_NAME_SPLIT_UNORD = ['cat0']

#   Names of variables for which marginal GATE (at median) will be computed
#   Variable must be in included in x_name_ord or x_name_unord; otherwise
#   variables will be deleted from list
if VAR_FLAG1:
    Z_NAME_MGATE = ['cont0', 'cont2', 'cat0', 'cat2', 'ord0', 'ord2']
else:
    Z_NAME_MGATE = ['cont0', 'cat0', 'ord0']

#   Names of variables for which average marginal GATE will be computed
#   Variable must be in included x_name_ord or x_name_unord; otherwise
#   variables will be deleted from list.
if VAR_FLAG1:
    Z_NAME_AMGATE = ['cont0', 'cont2', 'cat0', 'cat2', 'ord0', 'ord2']
else:
    Z_NAME_AMGATE = ['cont0',  'cat0', 'ord0']

#   Variable to be excluded from preliminary feature selection
X_NAME_REMAIN_ORD = []
X_NAME_REMAIN_UNORD = []

#   Variables for balancing tests (also excluded from feature selection)
#   Treatment specific statistics will only be printed for those variables
if VAR_FLAG1:
    X_BALANCE_NAME_ORD = ['cont0', 'cont1', 'cont2', 'cont3', 'cont4',
                          'cont5', 'cont6', 'cont7', 'cont8', 'cont9',
                          'dum0', 'dum1', 'ord0', 'ord1']
    X_BALANCE_NAME_UNORD = ['cat0', 'cat1']
else:
    X_BALANCE_NAME_ORD = ['cont0']
    X_BALANCE_NAME_UNORD = ['cat0']

#   Variables that control output. If not specified or set to None, defaults
#   will be used.

OUTPUT_TYPE = None          # 0: Output goes to terminal
#                             1: output goes to file
#                             2: Output goes to file and terminal (def)
VERBOSE = True             # True (def): Output about running of programme
DESCRIPTIVE_STATS = True  # print descriptive stats of input + output files
SHOW_PLOTS = True          # execute plt.show() command (def: True)

# controls for all figures
FONTSIZE = None       # legend, 1 (very small) to 7 (very large); def: 2
DPI = None            # > 0: def: 500
CI_LEVEL = None       # 0 < 1: confidence level for bounds: def: 0.90
# Only for (A, AM) GATEs: What type of plot to use for continuous variables
NO_FILLED_PLOT = None  # use filled plot if more than xx points (None: 20)

# Multiprocessing
MP_PARALLEL = None         # number of parallel processes  (>0)
#                                def: of cores*0.8 (reduce if memory problems!)
#                                0, 1: no parallel computations
MP_WITH_RAY = True  # True: Ray, False: Concurrent futures for Multiproces.
#                           False may be faster with small samples
#                           True (def): Should be superior with larger samples
MP_RAY_OBJSTORE_MULTIPLIER = None  # Increase internal default values for
#                                      Ray object store above 1 if programme
#                                      crashes because object store is full
#                                      (def: 1)
MP_VIM_TYPE = None        # Variable importance: type of mp
#                               1: variable based (fast, lots of memory)
#                               2: bootstrap based (slower, less memory)
#                               Def: 1 if n < 20000, 2 otherwise
MP_WEIGHTS_TYPE = None    # Weights computation: type of mp
#                               1: groups-of-obs based (fast, lots of memory)
#                               2: bootstrap based (takes forever, less memory)
#                               Def: 1
MP_WEIGHTS_TREE_BATCH = None  # Weight computation:Split Forests in batches
#                               Few batches: More speed, more memory.
#                               Def: Automatically determined

WEIGHT_AS_SPARSE = True   # Weights matrix as sparse matrix (def: True)
BOOT = None                 # of bootstraps / subsamplings (def: 1000)

# data cleaning
SCREEN_COVARIATES = True  # True (Default): screen covariates (sc)
CHECK_PERFECTCORR = True  # if sc=True: if True (default), var's that are
#                             perfectly correlated with others will be deleted
MIN_DUMMY_OBS = None      # if sc=1: dummy variable with obs in one
#                    category smaller than this value will be deleted (def: 10)
CLEAN_DATA_FLAG = True    # if True (def), remove all missing & unnecessary
#                             variables from data set
# Estimation methods
MCE_VART = None  # splitting rule
#               0: mse's of regression only considered
#               1: mse+mce criterion; (def)
#               2: -var(effect): heterogy maximising splitting rule of
#                       wager & athey (2018)
#               3: randomly switching between outcome-mse+mce criterion
#                   and penalty functions
# Penalty function
P_DIFF_PENALTY = None  # depends on mce_vart
#               if mce_vart == 0: irrelevant
#               if mce_vart == 1: multiplier of penalty (in terms of var(y))
#                   0: no penalty
#                  def: 4*((n*subsam_share)^0.8)/(n*subsam_share)*
#                      no_of_treatments*(no_of_treatments-1)/2; increase if
#                      balancing tests indicate bad balance
#               if mce_vart == 2: multiplier of penalty (in terms of value of
#                      MSE(y) value function without splits) for penalty;
#                   0: no penalty
#                  def: 100*4*(n*f_c.subsam_share)^0.8)/(n*f_c.subsam_share)
#                      increase if balancing tests indicate bad balance
#               if mce_vart == 3: probability of using p-score (0-1)
#                   def:0.5; increase if balancing tests indicate badt balance
# a priori feature pre-selection by random forest
FS_YES = False           # True: feature selection active
#                             False: not active (def)
FS_OTHER_SAMPLE = True  # False: same sample as for rf estimation used
#                         True (def): random sample taken from overall sample
FS_OTHER_SAMPLE_SHARE = None  # share of sample to be used for feature
#                                   selection  (def: 0.2);

FS_RF_THRESHOLD = None   # rf: threshold in % of loss of variable
#                                  importanance (def: 0)

# Local centering
L_CENTERING = False  # False: No local centering (def)
#   Note: Local centering is unlikely to be useful for other outcomes than the
#         one used for building the forest.
L_CENTERING_NEW_SAMPLE = False  # (def: False)
#   True: A random sample is used for computing predictions. This prediction
#     is subtracted from the outcome variables. The data used to compute it, is
#     not used anywhere else (and thus the sample size is reduced)
#   False: Cross-validation is used to compute the centering within the major
#     samples (tree buildung, filling with y, feature selection). These 2 / 3
#     adjustments are computed independent of each other.
#     This version is computationally more intensive (but stat. more efficient)
L_CENTERING_SHARE = None   # Share of data used for estimating E(y|x).
#    This data is not available for other estimations. 0.1-0.9; (def = 0.25)
#    Only valid if l_centering_new_sample is True
L_CENTERING_CV_K = None    # Number of folds used in crossvalidation
#    The larger the better the estimation quality and the longer computation.
#    > 1 (def: 5)
#    Only valid if l_centering_new_sample is False

# Common support
SUPPORT_CHECK = None
#   0: no common support
#   1,2 (def: 1): support check based on the estimated predictive rfs for each
#   treatment probability but one.
#   1: use min max rules for probabilities in treatment subsamples .
#   2: enforce minumum and maximum probabilities for all obs and all probs-1
#   observations off support are removed.
#   Out-of-bag predictions are used to avoid overfitting (which leads to a
#   too large reduction in observations).
SUPPORT_QUANTIL = None
#   if support check == 1: 1: min-max rule; 0-1: if smaller than 1,
#           the respective quantil is taken for the cut-offs; def: min-max
SUPPORT_MIN_P = None
#   if support check == 2: observation is deleted if p(d=m|x) is less or equal
#           than 'support_min_p' for at least one treatment  def: 0.01
SUPPORT_MAX_DEL_TRAIN = 0.5
#   If share of observations in training data used for forest data that are
#   OFF support is larger than SUPPORT_MAX_DEL_TRAIN (0-1), programme will be
#   terminated and user should change input data. Default is 0.5.

VARIABLE_IMPORTANCE_OOB = None
#   True:  computes variable importance based on permuting every single x in
#          oob prediction; time consuming (def: False)
BALANCING_TEST = True  # True: ATE based balancing test based on weights
#                             requires weight_based_inference (def: True)

# Truncation of extreme weights
MAX_WEIGHT_SHARE = None  # maximum share of any weight, 0 <= 1, def: 0.05
#   enforced by trimming excess weights and renormalisation for each ate, gate
#   & iate separately; because of renormalising, the final weights be somewhat
#   above this threshold

# Subsampling
SUBSAMPLE_FACTOR = None   # size of subsampling sample
#   0-1: reduces the default subsample size by 1-subsample_factor
#   Default: min(0.67,(2*(n^0.8)/n))
#   n is computed as twice the sample size in the smallest treatment group

# matching step to find neighbor for mce computation
MATCH_NN_PROG_SCORE = True   # False: use Mahalanobis matching, else
#                                  use prognostic scores (default: True)
NN_MAIN_DIAG_ONLY = False    # Only if match_nn_prog_score = False
#   True: use main diagonal only; False (def): inverse of covariance matrix
STOP_EMPTY = None     # x: stops splitting the tree if the next x5
#   randomly chosen variable did not led to a new leaf
#   0: new variables will be drawn & splitting continues n times
#   (faster if smaller, but nonzero); (def:25)

SHARE_FOREST_SAMPLE = None
#   0-1: share of sample used for predicting y given forests (def: 0.5)
#        other sample used for building forest

RANDOM_THRESHOLDS = None  # 0: no random thresholds
#               > 0: number of random thresholds used
#               Def: 20 (using only few thresholds may speed up the programme)

# Minimum leaf size
N_MIN_MIN = None      # smallest minimum leaf size (def: -1)
#   scalar or vector: minimum number of observations in leave
#     (if grid is used, optimal value will be determined by oob)
#   -2: n**0.4/20, at least 3; -1: n**0.4/10; at least 5 (def)
#    relevant n is twice the number of observations in smallest treatment group
N_MIN_MAX = None      # largest minimum leaf size (def=-1)
#   -2: sqrt(n)/10, at least 3; -1: minc([sqrt(n)/5; at least 5 (def)
#    relevant n is twice the number of observations in smallest treatment group
N_MIN_GRID = None     # numer of grid values (def: 1)
#                           If n_min_grid == 1: n_min_min is used for leaf size

# Alpha regularity
#   Results of Wager and Athey (2018) suggest alpha_reg < 0.2. A larger value
#   may increase speed of tree building. if grid is used, optimal value by OOB
ALPHA_REG_MIN = None      # smallest alpha, 0 < alpha < 0.4 (def: 0.1)
ALPHA_REG_MAX = None      # 0 < alpha < 0.5 (def: 0.2)
ALPHA_REG_GRID = None     # number of grid values (def: 2)

# grid for number of variables drawn at each new split of tree
M_MIN_SHARE = None
#   minimum share of variables used for next split (0-1); def = -1
M_MAX_SHARE = None
# maximum share of variables used for next split (0-1); def = -1
#   note that if m_max_share = 1, the algorithm corresponds to bagging trees
#   -1: m_min_share = 0.1 * anzahl_variables; m_max_share=anzahl_variables*0.66
#   -2: m_min_share = 0.2 * anzahl_variables; m_max_share=anzahl_variables*0.8
#      default values reduced to 70% of these values if feature learning
M_GRID = None  # m_try
#   number of grid values logarithmically spaced including m_min m_max (def: 2)
M_RANDOM_POISSON = True
#   if True / def:# of randomly selected variables is stochastic for each split
#   grid gives mean value of 1+poisson distribution(m-1)

# option for weight based inference (the only inference useful for aggregates)
COND_VAR_FLAG = True  # False: variance estimation uses var(wy)
#                           True: conditional mean & variances are used (True)
KNN_FLAG = False        # False: Nadaraya-Watson estimation (def)
#                           True: knn estimation (faster)
KNN_MIN_K = None        # k: minimum number of neighbours in
#                           k-nn estimation(def: 10)
KNN_CONST = None        # constant in number of neighbour
#                           asymptotic expansion formula of knn (def: 1)
NW_BANDW = None         # bandwidth for nw estimation; multiplier
#                           of silverman's optimal bandwidth (None: 1)
NW_KERN_FLAG = None     # kernel for nw estimation:
#                           1: epanechikov (def); 2: normal
SE_BOOT_ATE = False     # False: No Bootstrap standard errors for ATE, ...
SE_BOOT_GATE = False    # > 99: (w_ji y_i) are bootstrapped SE_BOOT_*** times
SE_BOOT_IATE = False    # Default is False

MAX_CATS_Z_VARS = None  # maximum number of categories for
#                           continuous z variables (def: n**.3)

PANEL_DATA = True       # True if panel data; None or False: no panel data
#                              this activates the clustered standard error,
#                              does perform only weight based inference
# use cluster_name to define variable that contains identifier for panel unit
PANEL_IN_RF = False      # uses the panel structure also when building the
#                       random samples within the forest procedure (def: True)
#       if panel == 1
CLUSTER_STD = False    # True:clustered standard error; cluster variable in
#                           variable file; (None: False) will be automatically
#                           set to one if panel data option is activated

CHOICE_BASED_SAMPLING = False  # True: choice based sampling (def: False)
CHOICE_BASED_WEIGHTS = [1, 1, 1]   # sampling probabilities to be specified
#   these weights relate to 'pred_eff_data' and used for (g)ates only.

# Sample weights
WEIGHTED = False               # True: use sampling weights,  def: False
#   if 1: sampling weights specified in w_name will be used; slows down progr.

# Estimate treatment-group specific effects if possible
ATET_FLAG = False      # True (def: False): average effects computed for
#                          subpopulations by treatments (if available)
#                          works only if at least one z variable is specified
GATET_FLAG = False     # True: gate(t)s for subpopulations by treatments
#                           (def: False)

# Marginal GATEs (MGATE) and Average Marginal GATEs (AMGATE)
GMATE_NO_EVALUATION_POINTS = None  # Number of evluation points for
#                              continuous variables (MGATE & AMGATE) (def: 50)
GMATE_SAMPLE_SHARE = None  # (0<1) Implementation of is very cpu intensive.
#   Random samples are used to speed up programme if more obs / # of evaluation
#   points > 10. # def: 1 if n_prediction < 1000; otherwise:
# (1000 + (n_pred-1000) ** (3/4))) / # of evaluation points
SMOOTH_GATES = True  # Alternative way to estimate GATEs for continuous
#   variables. Instead of discretizing variable, it's GATE is evaluated at
#   smooth_gates_no_evaluation_points. Since there are likely to be no
#   observations, a local neighbourhood around the evaluation points is
#   considered. (def: True)
SMOOTH_GATES_BANDWIDTH = None  # Multiplier for SGATE aggregation (def:1)
SMOOTH_GATES_NO_EVALUATION_POINTS = None  # (def: 50)

# analysis of predicted values
POST_EST_STATS = True   # analyses the predictions by binary correlations
#                       or some regression type methods (def: True)
RELATIVE_TO_FIRST_GROUP_ONLY = True  # use only effects relative to lowest
#                                          treatment value (def: True)
BIN_CORR_YES = True     # checking the binary predictions (def: True)
BIN_CORR_THRESHOLD = None  # minimum threshhold of absolute correlation
#                                to be displayed    (def: 0.1)
POST_PLOTS = True       # plots of estimated treatment effects
#                             in pred_eff_data         (def: True)
POST_KMEANS_YES = True  # using k-means clustering to analyse
#                       patterns in the estimated effects (def: True)
POST_KMEANS_NO_OF_GROUPS = None  # post_kmeans_yes is True: # of clusters
# to be build: Integer, list or tuple (or None --> default).
# Def: List of 5 values: [a, b, c, d, e]; c = 5 to 10; depending on n;
# c<7: a=c-2, b=c-1, d=c+1, e=c+2 else a=c-4, b=c-2, d=c+2, e=c+4
POST_KMEANS_REPLICATIONS = None  # post_kmeans_yes is True:# of replications
#                    with random start centers to avoid local extrema (def:10)
POST_KMEANS_MAX_TRIES = None     # post_kmeans_yes is True: maximum number
#           of iterations in each replication to achive convergence (def:1000)
POST_RANDOM_FOREST_VI = True     # use variable importance measure of
#  predictive random forest to learn major factors influencing IATEs (def:True)

# ---------------------------------------------------------------------------
_SMALLER_SAMPLE = 0      # 0<test_only<1: test prog.with smaller sample
_MAX_CATS_CONT_VARS = None  # discretising of continuous variables: maximum
#                          number of categories for continuous variables n
#                          values < n speed up programme, def: not used.
_WITH_OUTPUT = True       # use print statements
_MAX_SAVE_VALUES = 50  # save value of x only if less than 50 (cont. vars)
_SEED_SAMPLE_SPLIT = 67567885   # seeding is redone when building forest
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    mcf.ModifiedCausalForest(
        outpfad=OUTPATH, datpfad=DATPATH, indata=INDATA, preddata=PREDDATA,
        outfiletext=OUTFILETEXT, output_type=OUTPUT_TYPE, verbose=VERBOSE,
        save_forest=SAVE_FOREST, forest_files=FOREST_FILES,
        fontsize=FONTSIZE, dpi=DPI, ci_level=CI_LEVEL,
        no_filled_plot=NO_FILLED_PLOT, show_plots=SHOW_PLOTS,
        descriptive_stats=DESCRIPTIVE_STATS, clean_data_flag=CLEAN_DATA_FLAG,
        screen_covariates=SCREEN_COVARIATES, min_dummy_obs=MIN_DUMMY_OBS,
        check_perfectcorr=CHECK_PERFECTCORR,
        panel_data=PANEL_DATA, panel_in_rf=PANEL_IN_RF, weighted=WEIGHTED,
        cluster_std=CLUSTER_STD, choice_based_sampling=CHOICE_BASED_SAMPLING,
        choice_based_weights=CHOICE_BASED_WEIGHTS,
        match_nn_prog_score=MATCH_NN_PROG_SCORE,
        nn_main_diag_only=NN_MAIN_DIAG_ONLY,
        n_min_grid=N_MIN_GRID, n_min_min=N_MIN_MIN, n_min_max=N_MIN_MAX,
        m_min_share=M_MIN_SHARE, m_max_share=M_MAX_SHARE, m_grid=M_GRID,
        m_random_poisson=M_RANDOM_POISSON, alpha_reg_min=ALPHA_REG_MIN,
        alpha_reg_max=ALPHA_REG_MAX, alpha_reg_grid=ALPHA_REG_GRID,
        mce_vart=MCE_VART, p_diff_penalty=P_DIFF_PENALTY, boot=BOOT,
        knn_flag=KNN_FLAG, knn_const=KNN_CONST, nw_kern_flag=NW_KERN_FLAG,
        knn_min_k=KNN_MIN_K, cond_var_flag=COND_VAR_FLAG, nw_bandw=NW_BANDW,
        subsample_factor=SUBSAMPLE_FACTOR, stop_empty=STOP_EMPTY,
        random_thresholds=RANDOM_THRESHOLDS,
        share_forest_sample=SHARE_FOREST_SAMPLE,
        atet_flag=ATET_FLAG, gatet_flag=GATET_FLAG,
        max_cats_z_vars=MAX_CATS_Z_VARS,
        gmate_no_evaluation_points=GMATE_NO_EVALUATION_POINTS,
        gmate_sample_share=GMATE_SAMPLE_SHARE, smooth_gates=SMOOTH_GATES,
        smooth_gates_bandwidth=SMOOTH_GATES_BANDWIDTH,
        smooth_gates_no_evaluation_points=SMOOTH_GATES_NO_EVALUATION_POINTS,
        l_centering=L_CENTERING, l_centering_share=L_CENTERING_SHARE,
        l_centering_new_sample=L_CENTERING_NEW_SAMPLE,
        l_centering_cv_k=L_CENTERING_CV_K, fs_yes=FS_YES,
        fs_other_sample=FS_OTHER_SAMPLE, fs_rf_threshold=FS_RF_THRESHOLD,
        fs_other_sample_share=FS_OTHER_SAMPLE_SHARE,
        support_min_p=SUPPORT_MIN_P, support_check=SUPPORT_CHECK,
        support_max_del_train=SUPPORT_MAX_DEL_TRAIN,
        support_quantil=SUPPORT_QUANTIL, max_weight_share=MAX_WEIGHT_SHARE,
        variable_importance_oob=VARIABLE_IMPORTANCE_OOB,
        balancing_test=BALANCING_TEST,
        post_kmeans_max_tries=POST_KMEANS_MAX_TRIES,
        post_random_forest_vi=POST_RANDOM_FOREST_VI,
        bin_corr_yes=BIN_CORR_YES, post_plots=POST_PLOTS,
        post_est_stats=POST_EST_STATS, post_kmeans_yes=POST_KMEANS_YES,
        relative_to_first_group_only=RELATIVE_TO_FIRST_GROUP_ONLY,
        bin_corr_threshold=BIN_CORR_THRESHOLD,
        post_kmeans_no_of_groups=POST_KMEANS_NO_OF_GROUPS,
        post_kmeans_replications=POST_KMEANS_REPLICATIONS,
        id_name=ID_NAME, cluster_name=CLUSTER_NAME, w_name=W_NAME,
        d_name=D_NAME, y_tree_name=Y_TREE_NAME, y_name=Y_NAME,
        x_name_ord=X_NAME_ORD, x_name_unord=X_NAME_UNORD,
        x_name_always_in_ord=X_NAME_ALWAYS_IN_ORD,
        x_name_always_in_unord=X_NAME_ALWAYS_IN_UNORD,
        x_name_remain_ord=X_NAME_REMAIN_ORD,
        x_name_remain_unord=X_NAME_REMAIN_UNORD,
        x_balance_name_ord=X_BALANCE_NAME_ORD,
        x_balance_name_unord=X_BALANCE_NAME_UNORD,
        z_name_list=Z_NAME_LIST, z_name_split_ord=Z_NAME_SPLIT_ORD,
        z_name_split_unord=Z_NAME_SPLIT_UNORD, z_name_mgate=Z_NAME_MGATE,
        z_name_amgate=Z_NAME_AMGATE,
        mp_parallel=MP_PARALLEL, mp_vim_type=MP_VIM_TYPE,
        weight_as_sparse=WEIGHT_AS_SPARSE, mp_weights_type=MP_WEIGHTS_TYPE,
        mp_weights_tree_batch=MP_WEIGHTS_TREE_BATCH, mp_with_ray=MP_WITH_RAY,
        mp_ray_objstore_multiplier=MP_RAY_OBJSTORE_MULTIPLIER,
        _with_output=_WITH_OUTPUT, _max_save_values=_MAX_SAVE_VALUES,
        _smaller_sample=_SMALLER_SAMPLE, _seed_sample_split=_SEED_SAMPLE_SPLIT,
        _max_cats_cont_vars=_MAX_CATS_CONT_VARS,
        predict_mcf=PREDICT_MCF, train_mcf=TRAIN_MCF,
        se_boot_ate=SE_BOOT_ATE, se_boot_gate=SE_BOOT_GATE,
        se_boot_iate=SE_BOOT_IATE)
