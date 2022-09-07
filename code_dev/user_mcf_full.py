"""Created on Wed Apr  1 15:58:30 2020.

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner, SEW, University of St. Gallen, Switzerland

Version: 0.3.0 dev

-*- coding: utf-8 -*- .

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

New in version 0.1.0
- Kmeans uses a ranges of number of clusters and determines optimal cluster
  by silhoutte analysis (POST_KMEANS_NO_OF_GROUPS can now be a list or tuple
                         with possible cluster numbers)
- (Experimental) Optimal Policy Tool included bases on estimated IATEs
  (allowing implicitly for constraints and programme costs) building
  policy trees
- Bug fixes for problems computing treatment effects for treatment populations
- Detection of numerical variables added. Raises Exception.
- All variables used are shown in initial treatment specific statistics
  (important information to detect common support issues)
- Documentation only: Treatment specific statistics will only be printed for
  those variables used to check the balancing of the sample
- Fix some bugs for use of panel data and clustering
- Improved stat's for common support analysis
- Bug fix for dealing with missings in data cleaning

New in version 0.1.1
- Optional tree-specific subsampling for evaluation sample (subsample variables
                                                            got new names)
- Evaluation points of GATE figures are included in csv-file
- Minor bug fixes (_with_output, _smaller_sample, (A,AM)GATE/IATE-ATE plots,
                   sampling weights)
- k means cluster indicator saved in file with IATE predictions
- Exception raised if choice based sampling is activated and there is no
  treatment information in predictions file
- New defaults for 'random_thresholds'
- Further stabilizing ray by deleting references to object store and tasks
- MCF returns now ATE, Standarderror(ATE), GATE, SE(GATE), IATE, SE(IATE), and
                  the name of the file with the predictions

New in version 0.1.2
- Minor bug fixes (common support with very few observations deselected;
                   _with_output=False led to error in optimal policy modul)
- New default values for penalty function, random_thresholds, alpha regularity,
  number of bootstrap replications for SE estimation, Local centering,
  STOP_EMPTY
- Consistent use of new random number generator
- Documentation on the web is now much clearer which version is the default
  version used (MSE based with matching on outcome score & penalty function)
- As a new default Ray is only initialized ones and uses workers=logical_cores
  - 2;  further ways to finetune Ray are added (_mp_ray_del,_mp_ray_shutdown,
  _mp_ray_shutdown, mp_ray_objstore_multiplier becomes
  _mp_ray_objstore_multiplier).
- For obtaining cluster standard errors, using block-bootstrap on the w_i*y_i
  terms are now the default. This is slower but likely to me more accurate
  (less conservative) than the aggregation within-clusters used before.
- There are few options included now that help with large data sets
  a) reduce_split_sample: Split sample in parts used for estimation the effects
     and the predicting the effects for given x (outcome information is not
     needed for that part). This may also be useful/required for some optimal
     policy analysis. Note that having a large prediction sample may
     substantially increase computation time.
  While this sample split is done in the beginning of the programme,
  the following sample reductions are performed after determining common
  support:
  b) reduce_training: takes random sample from training data
     reduce_prediction: takes random sample from prediction data
  c) reduce_largest_group_train: Reduces the largest group in training. This
     should be less costly in terms of precision lost than taking random
     samples.
- Computing IATEs and their standard erros is optional
  (default is to compute IATE and their standard errors). If IATEs and/or their
  standard errors are not needed this may significantly speed up the programme
  (IATE_FLAG, IATE_SE_FLAG)
- Additional returns from main function ModifiedCausalForest()
- Change of default for variance computation. While Nadaray-Watson estimation
  (previous default) gives a better approximaton of the variance,
  k-NN is much faster, in particular for larger datasets. Therefore, k-NN is
  the new default.
- Fix of possible bug in MSE computation for multiple treatments (+ speed up
  for more than 4 treatments)
- Optimal policy module may use Ray for multiprocessing

New in version 0.1.3
- Minor bug fixes that led to crashing the programme
- New default for stop_empty
- Optimal policy module has the new option to compute a sequential policy tree
- Default for SUBSAMPLE_FACTOR_EVAL is now False (note that values and
                                                  description changed somewhat)

New in version 0.1.4
- _MP_RAY_SHUTDOWN has new default.
- Bug fix for predicting from previously trained and saved forests.
- Bug fix in mcf_init_function when there are missing values

New in version 0.2.0
- Improved plots.
- New overlap plots for common support analysis.
- Bug fix for GATE estimation (only relevant when weighting is used)
- Some keyword arguments changed names:
    verbose -> _verbose, descriptive_stats -> _descriptive_stats,
    show_plots -> _show_plots, fontsize -> _fontsize, dpi -> _dpi,
    no_filled_plot -> _no_filled_plot, mp_with_ray -> _mp_with_ray,
    mp_vim_type -> _mp_vim_type, mp_weights_type -> _mp_weights_type,
    mp_weights_tree_batch -> _mp_weights_tree_batch,
    weight_as_sparse -> _weight_as_sparse
- New keyword added for common support: support_adjust_limits (see description
                                                               below)
- stop_empty removed as parameter
- Main function changed name: ModifiedCausalForest -> modified_causal_forest
- Results replicate for discrete treatments (complete seeding of random number
                                             generators added)
- Experimental version of continuous treatment module added
 (not yet fully tested, method description will be added in the future)
  - new keyword arguments relating to continuous treatments:
    d_type, ct_grid_nn, ct_grid_w, ct_grid_dr
- The optimal policy modul contains new rules based on 'black box' approaches,
  i.e., using the potential outcomes directly to obtain optimal allocations
- The optimal policy modul allows for describing allocations with respect to
  other policy variables than the ones used for determining the allocation.
  If an observed allocation exists, results will also be computed (i) relative
  to an such an allocation , (ii) for those who's allocated treatment is
  different to the observed treatment
- Bug fix for policy trees under restrictions
- Black Box Optimal Policy allocations comes now with some bootstrap results
  (given the policy scores) as a crude measure for the uncertainty

New in version 0.2.1
- Bug fix in mgate

New in version 0.2.2
- Bug fix in plotting Gates.
- ATEs are now saved in csv file (same as data for figures and other effects).

New in version 0.2.3
- Nothing, just a bug removal for pip install

New in version 0.2.4
- Bug fix for cases when y had no variation when splitting
- File with IATEs also contains indicator of specific cluster in k-means
  clustering (post_est_stats == True and post_kmeans_yes == True)
- There is a problem some in sklearn.ensemble.RandomForestRegressor that leads
  to (slightly) different results when rerunning the programming if there is
  local centering: A new keyword arguments is added that slows down local
  centering a but but removes that problem (l_centering_replication,
                                            default is False)

New in version 0.2.5
- Some small increases in code efficiency
- Bug fix in bootstrap of optimal policy module
- Better organisation of output
    - Name change of file with predicted IATE. It ends as X_IATE.csv
    - Key results in short summary file
    - More intuitive directory structure
    - default value of l_centering_replication changed from False to True
- More efficient estimation of IATE -> EffIATE
    - EffIATE is computed by reversing the role of training and estimation
      sample and taking mean of both runs (this is the new default)
    - No inference is available for the EffIATE, but it will be more efficient
      than the IATE of the first round (as single sample splitting is mainly
                                        needed for inference)
    - Efficient versions of GATEs and ATEs are not provided as inference
      appears to be crucial for these parameters (which is lost in the
                                                  averaging step)
    - EFFIATE_FLAG is a new control variable for this feature
    - eff_iate will be saved together with iate and returned by the
      modified_causal_forest()

New in version 0.3.0
- Bug fixes:
    - Training and prediction are done in separate runs of programme
    - Issue in opt policy for case when treatment is unobserved is resolved
- New Features:
    - The mcf can be run either with a functional interface or an OOP interface
      - Functional:
          (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
                  pred_outfile, eff_pot_y, eff_iate
                  ) = mcf.modified_causal_forest
        It can seperately train the forest or predict the forest.  This is
        governed by the train_mcf and predict_mcf parameters. If training and
        prediction is separated, all information for prediction is contained in
        files in the directory used for the output.
        When predicting in a separate step, everything is reloaded from this
        directory.
        Although, training and prediction can be separated it is usually most
        efficient to do it together.
      - Object-orientated:
          The is a mcf object with three methods. Proceed as follows:
          (i) Create the mcf object:  mymcf = mcf.ModifiedCausalForest()
          (ii) Use the train method if training only:  mymcf.train()
               * The information needed for prediction is saved on file.
          (iii) Use the predict method if prediction only:
              (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
               pred_outfile, eff_pot_y, eff_iate) =  mymcf.predict()
          (ii+iii) Use the train_predict method if training and prediction
                   us performed in one step (usually most efficient):
                  (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
                   pred_outfile, eff_pot_y, eff_iate) = mymcf.train_predict()
         NOTE: Efficient IATE estimation is not yet implemented for OO version.
    - Replicability of results when there is local centering is added as an
      option (new variable: l_centering_replication). Default  is True.
    - Plots are now more consistent with each other.  
    - Predicted potential outcomes for individual data points (IATEs) contain
      also additional predictions for which local centering is reversed
      (new variable: l_centering_undo_iate). Default is True.
    - Nicely formated tables for Gates, AMGates, and MGates are created and
      saved in the respective folders (by Hugo Bodory).

    - OptPolicy.py has a new option that allows stochastic treatment
      allocations
      * details are explained in the OptPolicy.py file.

"""
from mcf import mcf_functions as mcf

TRAIN_MCF = True       # Train the forest; def: True
PREDICT_MCF = True     # Estimate effects; def: True
SAVE_FOREST = False    # Save forest for predict. w/o reestimation def:False
FOREST_FILES = None    # File name for all information needed for prediction
#   If a name is specified, then files with an *.csv, *.pickle, *ps.npy, *d.npy
#   extensions will be automatically specified.
#   If None, file names will be same as indat with extensions
#   *_savepred.pickle, *_savepred.csv,  *_savepredps.npy, *_savepredd.npy
#                                      These 4 files will be saved in outpfad.

APPLIC_PATH = 'c:/mlechner/mcftest'  # NOT passed to MCF
OUTPATH = APPLIC_PATH + '/out'
#   If this None a */out directory below the current directory is used
#   If specified directory does not exist, it will be created.
DATPATH = APPLIC_PATH + '/testdata'
#   If a path is None, path of this file is used.

INDATA = 'dgp_mcfN1000S5'    # 'dgp_mcfN1000S5'
# csv for estimation (without extension)
PREDDATA = 'dgp_mcfN1000S5_x_only'  # csv for effects (no outcomes needed)

#   csv extension is added automatically to both file names
#   If preddata is not specified, indata will be used as file name.

OUTFILETEXT = INDATA + 'mcf.py.0.3.0'  # File for text output
#   if outfiletext is None, name of indata with extension .out is used

#   Variables for estimation
VAR_FLAG1 = False   # No variable of mcf, used just for this test

#   The following variables must be provided:
#                 d_name, y_name, x_name_ord or x_name_unord
#   All other variables are optional. Lists or None must be used.

# D_NAME = 'D_cont'    # Treatment (not needed for pred)
# D_TYPE = 'continuous'     # 'discrete' (default) or 'continuous'
D_NAME = 'D0_1_12'  # , 'D' 'D01_12', D0_1_12
D_TYPE = 'discrete'     # 'discrete' (default) or 'continuous'

# List of outcome variables (not needed for pred)
Y_NAME = ['y', 'Cont118']  # ['y', 'Cont118', 'Cont119']

Y_TREE_NAME = None     # Variable to build trees (not needed for pred)
#   if None or [], the first variable in y_name is used to build trees.
#   it will be added to the list of outcome variablkes

#   Features, predictors, independent variables, confounders: ordered
if VAR_FLAG1:
    X_NAME_ORD = ['cont' + str(i) for i in range(120)]
    for i in range(10):
        X_NAME_ORD.append('dum' + str(i))
    for i in range(10):
        X_NAME_ORD.append('ord' + str(i))
else:
    X_NAME_ORD = ['cont' + str(i) for i in range(5)]
    for i in range(3):
        X_NAME_ORD.append('dum' + str(i))
    for i in range(3):
        X_NAME_ORD.append('ord' + str(i))

#   Features, predictors, independent variables: unordered, categorial

X_NAME_UNORD = []
if VAR_FLAG1:
    X_NAME_UNORD = ['cat' + str(i) for i in range(10)]
else:
    X_NAME_UNORD = ['cat' + str(i) for i in range(2)]

#   Identifier
ID_NAME = ['ID']
#   If no identifier -> it will be added the data that is saved for later use
#   Cluster and panel identifiers
CLUSTER_NAME = ['cluster']  # Variable defining the clusters if used
W_NAME = ['weight']         # Name of weight, if weighting option is used

# Variables always included when deciding next split
X_NAME_ALWAYS_IN_ORD = []      # (not needed for pred)
X_NAME_ALWAYS_IN_UNORD = []    # (not needed for pred)

# ******* Variables related to treatment heterogeneity ****
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

# Multiprocessing
MP_PARALLEL = None        # number of parallel processes  (>0)
# default: Logical cores -2 (reduce if memory problems!)
# 0, 1: no parallel computations

BOOT =40              # of bootstraps / subsamplings (def: 1000)

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
#               1: mse+mce criterion; (def, None)
#               2: -var(effect): heterogy maximising splitting rule of
#                       wager & athey (2018)
#               3: randomly switching between outcome-mse+mce criterion
#                   and penalty functions
# Penalty function
P_DIFF_PENALTY = None  # depends on mce_vart
#               if mce_vart == 0: irrelevant
#               if mce_vart == 1: multiplier of penalty (in terms of var(y))
#                   0: no penalty
#                  def: 2*((n*subsam_share)**0.9)/(n*subsam_share)*
#                      sqrt(no_of_treatments*(no_of_treatments-1)/2); increase
#                      if balancing tests indicate bad balance
#               if mce_vart == 2: multiplier of penalty (in terms of value of
#                      MSE(y) value function without splits) for penalty;
#                   0: no penalty
#                  def: 100*4*(n*f_c.subsam_share)^0.8)/(n*f_c.subsam_share)
#                      increase if balancing tests indicate bad balance
#               if mce_vart == 3: probability of using p-score (0-1)
#                   def:0.5; increase if balancing tests indicate badt balance
# a priori feature pre-selection by random forest
FS_YES = False          # True: feature selection active
#                             False: not active (def)
FS_OTHER_SAMPLE = True  # False: same sample as for rf estimation used
#                         True (def): random sample taken from overall sample
FS_OTHER_SAMPLE_SHARE = None  # share of sample to be used for feature
#                                   selection  (def: 0.2);

FS_RF_THRESHOLD = None   # rf: threshold in % of loss of variable
#                                  importanance (def: 0)

# Local centering
L_CENTERING = True  # False: No local centering (def: True)

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
L_CENTERING_REPLICATION = True
#    True does not allow multiprocessing in
#    the local centering step. Slower, but leads to replicability of the
#    results. Default is True.
L_CENTERING_UNDO_IATE = True
#   Predicted potential outcomes for individual data points (IATEs) re-adjusted
#   for local centering are added to data output. Default is True.


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
SUPPORT_ADJUST_LIMITS = None
#   IF SUPPORT_CHECK == 1:  > 0 or None (=default); 0: No adjustment
#   max *= 1+SUPPORT_ADJUST_LIMITS, min *= 1-SUPPORT_ADJUST_LIMITS
#   The restrictiveness of the common support criterion increases with the
#   number of treatments. This parameter reduces this restrictiveness.
#   Default (None) = (number of treatments - 2) * 0.05
SUPPORT_QUANTIL = None
#   if support check == 1: 1: min-max rule; 0-1: if smaller than 1,
#           the respective quantil is taken for the cut-offs; def: min-max
SUPPORT_MIN_P = None
#   if support check == 2: observation is deleted if p(d=m|x) is less or equal
#           than 'support_min_p' for at least one treatment  def: 0.01
SUPPORT_MAX_DEL_TRAIN = None
#   If share of observations in training data used for forest data that are
#   OFF support is larger than SUPPORT_MAX_DEL_TRAIN (0-1), programme will be
#   terminated and user should change input data. Default is 0.5.

VARIABLE_IMPORTANCE_OOB = False
#   True:  computes variable importance based on permuting every single x in
#          oob prediction; time consuming (def: False)
BALANCING_TEST = True  # True: ATE based balancing test based on weights
# Requires weight_based_inference (def: True); relevance of this test is not
# yet fully clear.

# Truncation of extreme weights
MAX_WEIGHT_SHARE = None  # maximum share of any weight, 0 <= 1, def: 0.05
#   enforced by trimming excess weights and renormalisation for each ate, gate
#   & iate separately; because of renormalising, the final weights could be
#   somewhat above this threshold

# Subsampling
SUBSAMPLE_FACTOR_FOREST = None   # size of subsampling sample to build tree
#   Default subsample share is: min(0.67,(2*(n^0.8)/n));
#   N: sample size;  n: 2x sample size of the smallest treatment group
#   >0: reduces (<1) or increases (>1) the default subsample size, max is 0.8
#   Actual share of subsample = def_share * SUBSAMPLE_FACTOR_FOREST
#   Default: SUBSAMPLE_FACTOR_FOREST = 1

# Tree-specific subsampling also in evaluation sample should increase speed
# at which asymtotic bias disappears (at the expense of a slower disappearance
# of the variance). However, simulations show no relevant impact.
SUBSAMPLE_FACTOR_EVAL = True   # size of subsampling sample to evaluate tree
# False or None: No subsampling in evaluation subsample (default)
# True: Same values as for SUBSAMPLE_FACTOR_FOREST
# Float (>0): Multiplier of subsample size used for tree building
# In particular for larger samples, using subsampling in evaluation may speed
# up computations and reduces demand on memory.

# matching step to find neighbor for mce computation
MATCH_NN_PROG_SCORE = True   # False: use Mahalanobis matching
#                              True:  use prognostic scores (default: True)
NN_MAIN_DIAG_ONLY = False    # Only if match_nn_prog_score = False
#   True: use main diagonal only; False (def): inverse of covariance matrix

RANDOM_THRESHOLDS = None  # 0: no random thresholds
#               > 0: number of random thresholds used for ordered var's
#               Def: 4+N_train**0.2 (few thresholds speeds up programme)

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
ALPHA_REG_MIN = None      # smallest alpha, 0 <= alpha < 0.4 (def: 0)
ALPHA_REG_MAX = None      # 0 <= alpha < 0.5 (def: 0.1)
ALPHA_REG_GRID = None     # number of grid values (def: 1)

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

# GATE and other plots
CI_LEVEL = None       # 0 < 1: confidence level for bounds on plot: def: 0.90

# option for weight based inference (the only inference useful for aggregates)
COND_VAR_FLAG = True  # False: variance estimation uses var(wy)
#                       True: conditional mean & variances are used (def: True)
KNN_FLAG = None        # False: Nadaraya-Watson estimation
#                        True: knn estimation (faster) (def: True)
# Nadaray-Watson estimation gives a better approximaton of the variance, but
# k-NN is much faster, in particular for larger datasets
KNN_MIN_K = None        # k: minimum number of neighbours in
#                           k-nn estimation(def: 10)
KNN_CONST = None        # constant in number of neighbour
#                           asymptotic expansion formula of knn (def: 1)
NW_BANDW = None         # bandwidth for nw estimation; multiplier
#                           of silverman's optimal bandwidth (None: 1)
NW_KERN_FLAG = None     # kernel for nw estimation:
#                           1: Epanechikov (def); 2: normal

SE_BOOT_ATE = None     # (w_ji y_i) are bootstrapped SE_BOOT_xATE times
SE_BOOT_GATE = None    # False: No Bootstrap SE of effects
SE_BOOT_IATE = None    # True: 199 bootstraps
# if CLUSTER_STD == False: Default is False
# if CLUSTER_STD == True default is 199; block-bootstrap is used

MAX_CATS_Z_VARS = None  # maximum number of categories for discretizing
#                           continuous z variables (def: n**.3) (GATE)

PANEL_DATA = False      # True if panel data; None or False: no panel data
#                              this activates the clustered standard error,
#                              does perform only weight based inference
# use cluster_name to define variable that contains identifier for panel unit
PANEL_IN_RF = False      # uses the panel structure also when building the
#                       random samples within the forest procedure (def: True)
#                       if panel == 1
CLUSTER_STD = False   # True:clustered standard error; cluster variable in
#                           variable file; (None: False) will be automatically
#                           set to one if panel data option is activated

CHOICE_BASED_SAMPLING = False  # True: choice based sampling (def: False)
CHOICE_BASED_WEIGHTS = [0.9, 0.8, 0.9, 0.95]   # sampling probabilities to be
# specified. These weights relate to 'pred_eff_data' and used for (g)ates only.
# Treatment information must therefore be available in the prediction file.

# Sample weights
WEIGHTED = None               # True: use sampling weights,  def: False
#   if 1: sampling weights specified in w_name will be used; slows down progr.

# Estimate treatment-group specific effects if possible
ATET_FLAG = False      # True (def: False): average effects computed for
#                          subpopulations by treatments (if available)
#                          works only if at least one z variable is specified
GATET_FLAG = False     # True: gate(t)s for subpopulations by treatments
#                           (def: False)

# Estimate IATEs and their standard errors
IATE_FLAG = True         # True (default): IATEs will be estimated; False: No
IATE_SE_FLAG = True      # True (default): SE(IATE) will be estimated;False: No
# Estimating IATEs and their standard errors may be time consuming
EFFIATE_FLAG = True      # True (default): Second round of IATEs are estimated
#   based on switching training and estimation subsamples.
#   If False, execution time is considerable faster.

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

# The following parameters are only relevant if treatment is continuous

# Number of grid point for discretization of cont. treatment
#   Grid is defined in terms of quantiles of continuous part of treatment.
CT_GRID_NN = None  # Used to aproximate the neighbourhood matching (def. is 10)
CT_GRID_W = None  # Used to aproximate the weights (def. is 10)
CT_GRID_DR = None  # Used to aproximate the dose response function (def.: 100)

# analysis of predicted values
POST_EST_STATS = True   # analyses the predictions by binary correlations
#  or some regression type methods (def: True); False if iate_flag == False
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

# Sample splitting to reduce computational costs
REDUCE_SPLIT_SAMPLE = False            # Default (None) is False
REDUCE_SPLIT_SAMPLE_PRED_SHARE = None  # 0<..<1: Share used for prediction
#   Split sample randomly in parts used for estimation the effects and the
#   predicting of the effects for given x (outcome information is not needed
#   for that part). This may also be useful/required for some optimal policy
#   analysis.
#   This works only if indata and preddata are identical or preddata is not
#   specified.

#   While this sample split is done in the beginning of the programme,
#   the following sample reductions are performed after determining common
#   support (only observations on common support are used).
REDUCE_TRAINING = False         # Random sample of indata. Default is False.
REDUCE_TRAINING_SHARE = None    # 0<...<1: Share to keep. Default (None) is 0.5
REDUCE_PREDICTION = False      # Random sample of preddata. Default is False.
REDUCE_PREDICTION_SHARE = None   # 0<...<1: Share to keep.Default (None) is 0.5
REDUCE_LARGEST_GROUP_TRAIN = False  # Reduces the largest treatment group in
#   indata. Should be less costly in terms of precision lost than taking random
#   samples.
REDUCE_LARGEST_GROUP_TRAIN_SHARE = None  # 0<...<1: Default (None) is 0.5
#   Note: The largest group will never become smaller than the 2nd largest.

# ---------------------------------------------------------------------------
_VERBOSE = True             # True (def): Output about running of programme
_DESCRIPTIVE_STATS = True  # print descriptive stats of input + output files
_SHOW_PLOTS = True          # execute plt.show() command (def: True)
# controls for all figures
_FONTSIZE = None       # legend, 1 (very small) to 7 (very large); def: 2
_DPI = None            # > 0: def (None): 500
# Only for (A, AM) GATEs: What type of plot to use for continuous variables
_NO_FILLED_PLOT = None  # use filled plot if more than xx points (def, None:20)

_WEIGHT_AS_SPARSE = True   # Weights matrix as sparse matrix (def: True)

_SHARE_FOREST_SAMPLE = None
#   0-1: share of sample used for predicting y given forests (def: 0.5)
#        other sample used for building forest
_SMALLER_SAMPLE = None      # 0<test_only<1: test prog.with smaller sample
_MAX_CATS_CONT_VARS = None  # discretising of continuous variables: maximum
# number of categories for continuous variables n values < n speed up
# programme, def: not used.
_WITH_OUTPUT = True       # use print statements
_MAX_SAVE_VALUES = 50  # save value of x only if less than 50 (cont. vars)
_SEED_SAMPLE_SPLIT = 67567885   # seeding is redone when building forest
_MP_WITH_RAY = True  # True: Ray, False: Concurrent futures for Multiproces.
#                           False may be faster with small samples
#                           True (def): Should be superior with larger samples
_MP_VIM_TYPE = None        # Variable importance: type of mp
#                               1: variable based (fast, lots of memory)
#                               2: bootstrap based (slower, less memory)
#                               Def: 1 if n < 20000, 2 otherwise
_MP_WEIGHTS_TYPE = None    # Weights computation: type of mp
#                               1: groups-of-obs based (fast, lots of memory)
#                               2: tree based (takes forever, less memory)
#                               Def: 1
_MP_WEIGHTS_TREE_BATCH = None  # Weight computation:Split Forests in batches
#                               Few batches: More speed, more memory.
#                               Def: Automatically determined
_MP_RAY_DEL = None     # None (leads to default) or Tuple with any of the
# the following: 'refs': Delete references to object store (default)
#                'rest': Delete all other objects of Ray task
#                These 2 can be combined. If deleting nothing is intended, use
#                ('none',)
_MP_RAY_SHUTDOWN = None  # Do shutdown ray task by task (Default is False if N
#  N < 100'000 and True otherwise)
#  If programme collapse because of excess memory reduce workers or set
#  _MP_RAY_SHUTDOWN = True
#  When using this programme repeatedly like in Monte Carlo studies always use
#  _MP_RAY_SHUTDOWN = True
_MP_RAY_OBJSTORE_MULTIPLIER = None  # Increase internal default values for
# Ray object store above 1 if programme crashes because object store is full
# (def: 1); ONLY RELEVANT if _MP_RAY_SHUTDOWN = True
# ---------------------------------------------------------------------------
_FUNCTIONAL = True      # This is not an MCF parameter. Just to distinghish
#  the functional and object-oriented call to mcf for demonstration purposes.
# ---------------------------------------------------------------------------
#if __name__ == '__main__':
if _FUNCTIONAL:
    (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var, pred_outfile,
     eff_pot_y, eff_iate
    ) = mcf.modified_causal_forest(
        outpfad=OUTPATH, datpfad=DATPATH, indata=INDATA, preddata=PREDDATA,
        outfiletext=OUTFILETEXT, output_type=OUTPUT_TYPE,
        save_forest=SAVE_FOREST, forest_files=FOREST_FILES,
        ci_level=CI_LEVEL,
        clean_data_flag=CLEAN_DATA_FLAG,
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
        subsample_factor_forest=SUBSAMPLE_FACTOR_FOREST,
        subsample_factor_eval=SUBSAMPLE_FACTOR_EVAL,
        atet_flag=ATET_FLAG, gatet_flag=GATET_FLAG,
        iate_flag=IATE_FLAG, iate_se_flag=IATE_SE_FLAG,
        effiate_flag=EFFIATE_FLAG,
        max_cats_z_vars=MAX_CATS_Z_VARS,
        gmate_no_evaluation_points=GMATE_NO_EVALUATION_POINTS,
        gmate_sample_share=GMATE_SAMPLE_SHARE, smooth_gates=SMOOTH_GATES,
        smooth_gates_bandwidth=SMOOTH_GATES_BANDWIDTH,
        smooth_gates_no_evaluation_points=SMOOTH_GATES_NO_EVALUATION_POINTS,
        l_centering=L_CENTERING, l_centering_share=L_CENTERING_SHARE,
        l_centering_new_sample=L_CENTERING_NEW_SAMPLE,
        l_centering_cv_k=L_CENTERING_CV_K,
        l_centering_replication=L_CENTERING_REPLICATION,
        l_centering_undo_iate=L_CENTERING_UNDO_IATE, fs_yes=FS_YES,
        fs_other_sample=FS_OTHER_SAMPLE, fs_rf_threshold=FS_RF_THRESHOLD,
        fs_other_sample_share=FS_OTHER_SAMPLE_SHARE,
        support_min_p=SUPPORT_MIN_P, support_check=SUPPORT_CHECK,
        support_max_del_train=SUPPORT_MAX_DEL_TRAIN,
        support_quantil=SUPPORT_QUANTIL,
        support_adjust_limits=SUPPORT_ADJUST_LIMITS,
        max_weight_share=MAX_WEIGHT_SHARE,
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
        d_name=D_NAME, d_type=D_TYPE, y_tree_name=Y_TREE_NAME, y_name=Y_NAME,
        x_name_ord=X_NAME_ORD, x_name_unord=X_NAME_UNORD,
        x_name_always_in_ord=X_NAME_ALWAYS_IN_ORD,
        x_name_always_in_unord=X_NAME_ALWAYS_IN_UNORD,
        x_name_remain_ord=X_NAME_REMAIN_ORD,
        x_name_remain_unord=X_NAME_REMAIN_UNORD,
        x_balance_name_ord=X_BALANCE_NAME_ORD,
        x_balance_name_unord=X_BALANCE_NAME_UNORD,
        z_name_list=Z_NAME_LIST, z_name_split_ord=Z_NAME_SPLIT_ORD,
        z_name_split_unord=Z_NAME_SPLIT_UNORD, z_name_mgate=Z_NAME_MGATE,
        z_name_amgate=Z_NAME_AMGATE, random_thresholds=RANDOM_THRESHOLDS,
        mp_parallel=MP_PARALLEL,
        predict_mcf=PREDICT_MCF, train_mcf=TRAIN_MCF,
        se_boot_ate=SE_BOOT_ATE, se_boot_gate=SE_BOOT_GATE,
        se_boot_iate=SE_BOOT_IATE,
        reduce_split_sample=REDUCE_SPLIT_SAMPLE,
        reduce_split_sample_pred_share=REDUCE_SPLIT_SAMPLE_PRED_SHARE,
        reduce_training=REDUCE_TRAINING,
        reduce_training_share=REDUCE_TRAINING_SHARE,
        reduce_prediction=REDUCE_PREDICTION,
        reduce_prediction_share=REDUCE_PREDICTION_SHARE,
        reduce_largest_group_train=REDUCE_LARGEST_GROUP_TRAIN,
        reduce_largest_group_train_share=REDUCE_LARGEST_GROUP_TRAIN_SHARE,
        ct_grid_nn=CT_GRID_NN, ct_grid_w=CT_GRID_W, ct_grid_dr=CT_GRID_DR,
        _mp_vim_type=_MP_VIM_TYPE, _mp_with_ray=_MP_WITH_RAY,
        _weight_as_sparse=_WEIGHT_AS_SPARSE, _mp_weights_type=_MP_WEIGHTS_TYPE,
        _mp_weights_tree_batch=_MP_WEIGHTS_TREE_BATCH,
        _no_filled_plot=_NO_FILLED_PLOT, _show_plots=_SHOW_PLOTS,
        _descriptive_stats=_DESCRIPTIVE_STATS,
        _share_forest_sample=_SHARE_FOREST_SAMPLE,
        _verbose=_VERBOSE, _fontsize=_FONTSIZE, _dpi=_DPI,
        _mp_ray_objstore_multiplier=_MP_RAY_OBJSTORE_MULTIPLIER,
        _with_output=_WITH_OUTPUT, _max_save_values=_MAX_SAVE_VALUES,
        _smaller_sample=_SMALLER_SAMPLE, _seed_sample_split=_SEED_SAMPLE_SPLIT,
        _max_cats_cont_vars=_MAX_CATS_CONT_VARS, _mp_ray_del=_MP_RAY_DEL,
        _mp_ray_shutdown=_MP_RAY_SHUTDOWN
        )
else:
    if not (TRAIN_MCF and PREDICT_MCF):
        EFFIATE_FLAG = False
    mymcf = mcf.ModifiedCausalForest(
        outpfad=OUTPATH, datpfad=DATPATH, indata=INDATA, preddata=PREDDATA,
        outfiletext=OUTFILETEXT, output_type=OUTPUT_TYPE,
        save_forest=SAVE_FOREST, forest_files=FOREST_FILES,
        ci_level=CI_LEVEL,
        clean_data_flag=CLEAN_DATA_FLAG,
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
        subsample_factor_forest=SUBSAMPLE_FACTOR_FOREST,
        subsample_factor_eval=SUBSAMPLE_FACTOR_EVAL,
        atet_flag=ATET_FLAG, gatet_flag=GATET_FLAG,
        iate_flag=IATE_FLAG, iate_se_flag=IATE_SE_FLAG,
        effiate_flag=EFFIATE_FLAG,
        max_cats_z_vars=MAX_CATS_Z_VARS,
        gmate_no_evaluation_points=GMATE_NO_EVALUATION_POINTS,
        gmate_sample_share=GMATE_SAMPLE_SHARE, smooth_gates=SMOOTH_GATES,
        smooth_gates_bandwidth=SMOOTH_GATES_BANDWIDTH,
        smooth_gates_no_evaluation_points=SMOOTH_GATES_NO_EVALUATION_POINTS,
        l_centering=L_CENTERING, l_centering_share=L_CENTERING_SHARE,
        l_centering_new_sample=L_CENTERING_NEW_SAMPLE,
        l_centering_cv_k=L_CENTERING_CV_K,
        l_centering_replication=L_CENTERING_REPLICATION, fs_yes=FS_YES,
        fs_other_sample=FS_OTHER_SAMPLE, fs_rf_threshold=FS_RF_THRESHOLD,
        fs_other_sample_share=FS_OTHER_SAMPLE_SHARE,
        support_min_p=SUPPORT_MIN_P, support_check=SUPPORT_CHECK,
        support_max_del_train=SUPPORT_MAX_DEL_TRAIN,
        support_quantil=SUPPORT_QUANTIL,
        support_adjust_limits=SUPPORT_ADJUST_LIMITS,
        max_weight_share=MAX_WEIGHT_SHARE,
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
        d_name=D_NAME, d_type=D_TYPE, y_tree_name=Y_TREE_NAME, y_name=Y_NAME,
        x_name_ord=X_NAME_ORD, x_name_unord=X_NAME_UNORD,
        x_name_always_in_ord=X_NAME_ALWAYS_IN_ORD,
        x_name_always_in_unord=X_NAME_ALWAYS_IN_UNORD,
        x_name_remain_ord=X_NAME_REMAIN_ORD,
        x_name_remain_unord=X_NAME_REMAIN_UNORD,
        x_balance_name_ord=X_BALANCE_NAME_ORD,
        x_balance_name_unord=X_BALANCE_NAME_UNORD,
        z_name_list=Z_NAME_LIST, z_name_split_ord=Z_NAME_SPLIT_ORD,
        z_name_split_unord=Z_NAME_SPLIT_UNORD, z_name_mgate=Z_NAME_MGATE,
        z_name_amgate=Z_NAME_AMGATE, random_thresholds=RANDOM_THRESHOLDS,
        mp_parallel=MP_PARALLEL,
        predict_mcf=PREDICT_MCF, train_mcf=TRAIN_MCF,
        se_boot_ate=SE_BOOT_ATE, se_boot_gate=SE_BOOT_GATE,
        se_boot_iate=SE_BOOT_IATE,
        reduce_split_sample=REDUCE_SPLIT_SAMPLE,
        reduce_split_sample_pred_share=REDUCE_SPLIT_SAMPLE_PRED_SHARE,
        reduce_training=REDUCE_TRAINING,
        reduce_training_share=REDUCE_TRAINING_SHARE,
        reduce_prediction=REDUCE_PREDICTION,
        reduce_prediction_share=REDUCE_PREDICTION_SHARE,
        reduce_largest_group_train=REDUCE_LARGEST_GROUP_TRAIN,
        reduce_largest_group_train_share=REDUCE_LARGEST_GROUP_TRAIN_SHARE,
        ct_grid_nn=CT_GRID_NN, ct_grid_w=CT_GRID_W, ct_grid_dr=CT_GRID_DR,
        _mp_vim_type=_MP_VIM_TYPE, _mp_with_ray=_MP_WITH_RAY,
        _weight_as_sparse=_WEIGHT_AS_SPARSE, _mp_weights_type=_MP_WEIGHTS_TYPE,
        _mp_weights_tree_batch=_MP_WEIGHTS_TREE_BATCH,
        _no_filled_plot=_NO_FILLED_PLOT, _show_plots=_SHOW_PLOTS,
        _descriptive_stats=_DESCRIPTIVE_STATS,
        _share_forest_sample=_SHARE_FOREST_SAMPLE,
        _verbose=_VERBOSE, _fontsize=_FONTSIZE, _dpi=_DPI,
        _mp_ray_objstore_multiplier=_MP_RAY_OBJSTORE_MULTIPLIER,
        _with_output=_WITH_OUTPUT, _max_save_values=_MAX_SAVE_VALUES,
        _smaller_sample=_SMALLER_SAMPLE, _seed_sample_split=_SEED_SAMPLE_SPLIT,
        _max_cats_cont_vars=_MAX_CATS_CONT_VARS, _mp_ray_del=_MP_RAY_DEL,
        _mp_ray_shutdown=_MP_RAY_SHUTDOWN)

    if TRAIN_MCF and not PREDICT_MCF:
        mymcf.train()
    elif not TRAIN_MCF and PREDICT_MCF:
        mymcf.predict()
    elif TRAIN_MCF and PREDICT_MCF:
        (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
         pred_outfile, eff_pot_y, eff_iate) = mymcf.train_predict()
