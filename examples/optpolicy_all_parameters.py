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

Version: 0.7.2

This is an example to show how to use the OptimalPolicy class of the mcf
module with full specification of all its keywords. It may be seen as an add on
to the published mcf documentation.

"""
from pathlib import Path

from mcf.example_data_functions import example_data
from mcf.optpolicy_functions import OptimalPolicy
from mcf.reporting import McfOptPolReport

# ------------- NOT passed to OptimalPolicy -----------------------------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example'

# ---------------------- Generate artificial data ------------------------------

# Parameters to generate artificial data (DataFrame) for this example
TRAIN_OBS = 1000        # Number of observations of training data.
#      'best_policy_score': Training data must contain policy scores.
#      'policy tree', 'bps_classifier': Training data must contain policy scores
#                         and features. Default is 1000.

PRED_OBS = 1000         # Number of observations of prediction data.
#                         Prediction data is used to allocate.
#      'best_policy_score': Prediction data must contain policy scores.
#      'policy tree', 'bps_classifier': Prediction data must contain the
#                         features that are used in training. Policy scores are
#                         not required. Default is 1000.

FAIRNESS_CORRECTION = False
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

training_all_df, prediction_df, name_dict = example_data(
    obs_y_d_x_iate=TRAIN_OBS,
    obs_x_iate=PRED_OBS,
    no_features=NO_FEATURES,
    no_treatments=NO_TREATMENTS,
    seed=12345,
    type_of_heterogeneity='WagerAthey',
    descr_stats=True)

if PREDDATA_IS_TRAINDATA:
    prediction_df = training_all_df

# ------------- Methods used in Optimal Policy Module --------------------------
METHODS = ('best_policy_score', 'bps_classifier', 'policy tree',)

#  Tuple used to set GEN_METHOD in this example
#  Currently valid methods are: 'best_policy_score', 'bps_classifier',
#  'policy tree'

# -------- All what follows are parameters of the OptimalPolicy --------
#   Whenever None is specified, parameter will be set to default values.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
GEN_OUTPATH = Path.cwd() / 'example/outputOPT'  # Directory for output.
#   If it does not exist, it will be created. Default is an *.out directory
#   just below to the directory where the programme is run.

GEN_OUTFILETEXT = "OptPolicy.0.7.2"  # File for text output
#   Default is 'txtFileWithOutput'.
#   *.txt file extension will be added by the programme

#   ------------- Where to direct the text output to --------------------------
GEN_OUTPUT_TYPE = None      # 0: Output goes to terminal
#                             1: output goes to file
#                             2: Output goes to file and terminal (default)
GEN_METHOD = None
#   Available methods: 'best_policy_score', 'policy tree', 'bps_classifier'
#   Default is 'best_policy_score'.
#   Note for this example only: This variable will be overwritten inside the
#   loop at the end of the programme.
GEN_VARIABLE_IMPORTANCE = None    # Compute variable importance statistics
#   based on random forest  classifiers. Default is True.

#   ------------- Multiprocessing ----------------------------------------------
GEN_MP_PARALLEL = None       # Number of parallel processes  (>0)
#   Default is to use 80% of logical cores (reduce if memory problems!)
#   0, 1: no parallel computations

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
                          (('x_cont0', 'iate1vs0', 'iate2vs0',)))
#   Tuple of tuples. Each tuple of dimension equal to the number of treatments
#   contains treatment specific variables that are used to evaluate the effect
#   of the allocation with respect to those variables. This could be for
#   example policy score not used in training, but which are relevant
#   nevertheless. Default is no variables.

VAR_EFFECT_VS_0 = ('iate1vs0', 'iate2vs0')
#   Effects of treatment relative to treatment zero. Dimension is equal to the
#   different treatments minus 1. Default is no variables.

VAR_EFFECT_VS_0_SE = None
#   Standard errors of effects relative to treatment zero. Dimension is equal
#    to the different treatments minus 1. Default is no variables.

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

# ---------------------- Method specific parameters ---------------------------
# - - - - - - - - - - - - - - - Fairness - - - - - - - - - - - - - - - - - - - -
#   The 'fairscores' method adjusts the policy scores by removing the influence
#   of some features that are specificied in the VAR_PROTECTED_NAME variable.
#   If the 'fairscores' method is not explicitly called, the following variables
#   are not relevant and need not to be specifified.

#   Protected variables (the effect of the policy scores on those variables will
#                        be removed, conditional on the 'materially important'
#                        variables).
VAR_PROTECTED_NAME_ORD = ('x_cont0', 'x_ord4',)
VAR_PROTECTED_NAME_UNORD = ('x_unord0', 'x_unord3',)
#   These variables should NOT be contained in decision variables, i.e.,
#   VAR_X_NAME_ORD and/or VAR_X_NAME_UNORD. If they are included, they will be
#   removed and VAR_X_NAME_ORD and/or VAR_X_NAME_UNORD will be adjusted
#   accordingly. Defaults are None (at least one of them must be specified, if
#   the fairness adjustments are to be used).

#  Materially relavant variables: An effect of the protected variables on the
#  scores is allowed, if captured by these variables (only).
VAR_MATERIAL_NAME_ORD = ('x_cont1', 'x_ord2',)
VAR_MATERIAL_NAME_UNORD = ('x_unord1', 'x_unord2',)

#  These variables may (or may not) be included among the decision variables.
#  These variables must (!) not be included among the protected variables.

#  - - - - - - - Fairness adjustment methods for policy scores - - - - - - - -
#     (see Bearth, Lechner, Muny, Mareckova, 2024, for details)
FAIR_TYPE = None
#  Method to choose the type of correction for the policy scores.
#  'Mean':  Mean dependence of the policy score on protected var's is removed.
#  'MeanVar':  Mean dependence and heteroscedasticity is removed.
#  'Quantiled': Removing dependence via (an empricial version of) the approach
#               by Strack and Yang (2024).
#  Default (or None) is 'Quantiled'.

#  Method choice when predictions from machine learning are needed
FAIR_REGRESSION_METHOD = None
#  Regression method to adjust scores w.r.t. protected features.
#  Possible choices are scikit-learn's regression methods
#  'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5'
#  'SupportVectorMachine', 'SupportVectorMachineC2', 'SupportVectorMachineC4',
#  'AdaBoost', 'AdaBoost100', 'AdaBoost200',
#  'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12',
#  'LASSO', 'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
#  'Mean'. (Mean is included if other estimators are not useful.)
#  If 'automatic', an optimal methods will be chosen based on 5-fold
#  cross-validation in the training data. If 'automatic', a method is specified
#  it will be used for all scores and all adjustments. If None, every policy
#  score might be adjusted with a different method. 'Mean' is included for cases
#  in which regression methods have no explanatory power.
#  This method is relevant when "fair_type in ('Mean', 'MeanVar')"
#  Default (or None) is 'RandomForest'.

#  Parameters for discretization of features (necessary for 'Quantilized')

FAIR_PROTECTED_DISC_METHOD = None
#  Whether and how to perform the discretization for
#  protected features (for FAIR_TYPE = 'Quantiled'). Maximum number of groups
#  allowed is set by FAIR_PROTECTED_MAX_GROUPS.
#  'NoDiscretization': Variables are not changed. If one of the features has
#               more different values than FAIR_PROTECTED_MAX_GROUPS, all
#               protected features will formally be treated as continuous. In
#               this case quantile positions will be based on nonparametrically
#               estimated densities. The latter may become unreliable if their
#               dimension is not very small.
#  'EqualCell': Attempts to create equal cells for each feature. Maybe be
#               useful for a very small number of variables with few different
#               values.
#  'Kmeans':    Use Kmeans clustering algorithm to form homogeneous cells.
#  Default (or None) is 'Kmeans'.

FAIR_MATERIAL_DISC_METHOD = None
#  Whether and how to perform the discretization for materially relevant
#  features (for FAIR_TYPE = 'Quantiled'). Maximum number of groups
#  allowed is set by FAIR_MATERIAL_MAX_GROUPS.
#  'NoDiscretization': Variables are not changed. If one of the features has
#               more different values than FAIR_MATERIAL_MAX_GROUPS, all
#               materially relevant features will formally be treated as
#               continuous. In  this case quantile positions will be based on
#               nonparametrically estimated densities. The latter may become
#                unreliable if their dimension is not very small.
#  'EqualCell': Attempts to create equal cells for each feature. Maybe be
#               useful for a very small number of variables with few different
#               values.
#  'Kmeans':    Use Kmeans clustering algorithm to form homogeneous cells.
#  Default (or None) is 'Kmeans'.

#  Level of discretization of variables (only if needed)
FAIR_PROTECTED_MAX_GROUPS = None
FAIR_MATERIAL_MAX_GROUPS = None
#  Number of different groups of values of features that are
#  materially relavant /protected (if discretization is used).
#  This is currently only necessary for 'Quantilized'.
#  Its meaning depends on fair_protected_disc_method, fair_material_disc_method:
#  If 'EqualCell': If more than 1 variable is included among the protected
#      features, this restriction is applied to each variable.
#  If 'Kmeans': This is the number of clusters used by Kmeans.
#  Default (or None) is 5.

#  Test for internally consistency.
FAIR_CONSISTENCY_TEST = None
#  The fairness corrections are applied independently to every policy score (
#  which usually is a potential outcome or an IATE(x) for each treatment
#  relative to some base treatment (i.e. comparing 1-0, 2-0, 3-0, etc.).
#  Thus the IATE for the 2-1 comparison can be computed as IATE(2-0)-IATE(1-0).
#  This tests compares two ways to compute a fair score for the 2-1 (and all
#  other comparisons) which should give simular results:
#  a) Difference of two fair (!) scores
#  b) Difference of corresponding scores, subsequently made fair.
#  Note: Depending on the number of treatments, this test may be computationally
#  more expensive than the orginal fairness corrections.
#  Therefore, the default is False.

# - - - - - - - - - - - -  Black-Box allocations - - - - - - - - - - - - - - -
#   The method 'best_policy_score' conducts Black-Box allocations.
#   'Black-Box allocations' are those that are obtained by using the scores
#   directly (potentially subject to restrictions).
#   When the Black-Box allocations are used for allocation of new data, the
#   respective scores must be available in the prediction data.
VAR_BB_RESTRICT_NAME = 'x_unord0'   # Variable name related to a restriction.
#   If there is a capacity constraint, preference will be given to obs. with
#   highest values of this variable. Default is that no such variable is used.

# - - - - - - - - -  Black-Box allocations based Classifier - - - - - - - - - -
#   The bps_classifier classifier algorithm runs a classifier for each of the
#   allocations obtained by the 'best_policy_score' algorithm. One
#   advantage compared of this approach compared to the
#   'best_policy_score' algorithm is that the prediction of the allocation
#   for new observations is fast as it does not require to recompute the
#   policy score (as it is case with the 'best_policy_score' algorithm).
#   The classifier is selected among four different classifiers offered by
#   sci-kit learn, namely a simple neural network, two classification
#   random forests with minimum leaf size of 2 and 5, and ADDABoost. The
#   selection is a made according to the out-of-sample performance on
#   scikit-learns Accuracy Score.

#   There no parameters specific to this method.

# - - - - - - - - - - - Optimal shallow decision trees - - - - - - - - - - - -
#   These decision trees are optimal trees, in the sense that all possible
#   trees are checked if they lead to a better performance. If restrictions
#   are specified, then this is incorporated into treatment specific cost
#   parameters (see below). They are calibrated by finding values that lead to
#   to desired capacity constraints with (unrestricted) Black-Box allocations
#   (as the latter are much, much faster to compute than decision trees).
#   Many ideas of the implementation follow Zhou, Athey, Wager (2022). If the
#   provided policy score fulfils their conditions (i.e., they use a doubly
#   robust double machine learning like score), then they also provide
#   attractive theoretical properties.

PT_DEPTH_TREE_1 = 2   # Depth of 1st optimal tree. Default is 3.
#   In this example, this parameter is set to 2 to speed up computation.

PT_DEPTH_TREE_2 = 2   # Depth of 2nd optimal tree. This set is built within
#   the strata obtained from the leaves of the first tree. If set to 0, a second
#   tree is not built. Default is 1 (together with the default for
#   pt_depth_tree_1 this leads to a (not optimal) total tree of level of 4.
#   In this example, this parameter is set to 2.

#   Note that tree depth is defined such that a depth of 1 implies 2 leaves,
#   a depth of 2 implies 4 leaves, a depth of = 3 implies 8 leaves, etc.

PT_NO_OF_EVALUPOINTS = None  # No of evaluation points for continuous variables.
#   The lower this value, the faster the algorithm, but it may also deviate
#   more from the optimal splitting rule. This parameter is closely related to
#   approximation parameter of Zhou, Athey, Wager (2022)(A) with
#   A =  # of observation / PT_NO_OF_EVALUPOINTS. Default is 100.

PT_EVA_CAT_MULT = None      # Changes the number of the evaluation points
#   (pt_no_of_evalupoints) for the unordered (categorical) variables to:
#   pt_eva_cat_mult * pt_no_of_evalupoints (available only for the method
#   'policy tree').  Default is 2.

PT_SELECT_VALUES_CAT = None   # Approximation method for larger categorical
#   variables.
#   Since we search among optimal trees, for categorical variables
#   variables we need to check for all possible combinations of the different
#   values that lead to binary splits. This number could be huge. There-
#   fore, we compare only PT_NO_OF_EVALUPOINTS * PT_EVA_CAT_MULT different
#   combinations.
#   Method 1 (PT_SELECT_VALUES_CAT == True) does this by randomly drawing
#   values from the particular categorical variable and forming groups only
#   using those values. Method 2 (PT_SELECT_VALUES_CAT==False) sorts the
#   values of the categorical variables according to the values of the policy
#   score as one would do for a standard random forest. If this set is still
#   too large, a random sample of the entailed combinations is drawn.
#   Method 1 is only available for the method 'policy tree'. The default is
#   False.

PT_MIN_LEAF_SIZE = None    # Minimum leaf size. Leaves that are smaller than
#   PT_MIN_LEAF_SIZE in the training data will not be considered. A larger
#   number reduces computation time and avoids some overfitting. Default is
#   0.1 x # of training observations / # of leaves (if treatment shares are
#   restricted this is multiplied by the smallest share allowed).

PT_ENFORCE_RESTRICTION = None
#   Enforces the imposed restriction (to some extent) during the
#   computation of the policy tree. This increases the quality of trees
#   concerning obeying restrictions, but can be very time consuming.
#   It will be automatically set to False if more than 1 policy tree is
#   estimated. Default is False.

# --------------------------  Data cleaning  ----------------------------------
DC_SCREEN_COVARIATES = None  # Check covariates. Default is True.

#   The following parameters are only relevant if
#   DATA_SCREEN_COVARIATES is True.

DC_CHECK_PERFECTCORR = None  # Check if some variables are perfectly
#   correlated with each other. If yes, these variables are deleted.
#   Default is True.

DC_MIN_DUMMY_OBS = None      # Dummy variable with obs in one category
#   smaller than this value will be deleted. Default is 10.

DC_CLEAN_DATA = None         # Remove all missing & unnecessary variables from
#   Data. Default is True.

# ------------------------ Stochastic assignment ------------------------------
RND_SHARES = None       # Create a stochastic assignment of the data passed.
#   Tuple of size of number treatments. Sum of all elements must add to 1.
#   This used only used as a comparison in the evaluation of other allocations.
#   Default is shares of treatments in the allocation under investigation.

# -------------- Other parameters (costs and restrictions) --------------------
OTHER_MAX_SHARES = (1, 1, 0.3)  # Maximum share allowed for each treatment.
#   This is a tuple with as many elements as treatments.
#   0 <  OTHER_MAX_SHARES <= 1.  (1,...,1) implies unconstrained optimization
#   (default).

OTHER_COSTS_OF_TREAT = None   # Treatment specific costs. These costs
#   are directly subtracted from the policy scores. Therefore, they should be
#   measured in the same units as the scores. Default is 0.
#   If None, and when there are constraints, costs will be automatically
#   determined to  enforce constraints in the training data by finding
#   costs that lead to an allocation (not a tree, but chosing individually
#   best treatments by unconstrained Black-Box methods) that fulfils
#   restrictions in OTHER_MAX_SHARES.

OTHER_COSTS_OF_TREAT_MULT = None   # Multiplier of automatically determined
#   costs. None or Tuple of positive numbers with dimension equal to the number
#   of treatments. Default is (1, ..., 1).
#   Use only when automatic costs do not lead to a satisfaction of the
#   constraints given by OTHER_MAX_SHARES. This allows to increase (>1) or
#   decrease (<1) the share of treated in particular treatment.

# -----------Internal parameters. Change only if good reason to do so. --------
_INT_FONTSIZE = None            # Legend, 1(very small) to 7(very large); def: 2
_INT_DPI = None                 # > 0: Default (None): 500
_INT_WITH_OUTPUT = None   # Print output on file and screen. Default is True.
_INT_OUTPUT_NO_NEW_DIR = None   # Do not create a new directory when the path
#                                 already exists. Default is False.
_INT_REPORT = None              # True: Provide information for McfOptPolReports
#                                 to construct informative reports.
#                                 Default is True.
_INT_WITH_NUMBA = None    # Use Numba to speed up computations: Default is True.
_INT_XTR_PARALLEL = None  # Parallelize to a larger degree to make sure all
#   CPUs are busy for most of the time. Default is True. Only used for
#   'policy tree' and only used if _INT_PARALLEL_PROCESSING > 1 (or None)
# ------------------------------------------------------------------------

# The following is an example on how the methods of the OptimalPolicy object
# can be used.
# For convenience the mcf parameters are collected and passed as a dictionary.
# Of course, they can also be passed as single parameters (or not at all, in
# which case default values are used).

params = {
    'dc_check_perfectcorr': DC_CHECK_PERFECTCORR,
    'dc_clean_data': DC_CLEAN_DATA, 'dc_min_dummy_obs': DC_MIN_DUMMY_OBS,
    'dc_screen_covariates': DC_SCREEN_COVARIATES,
    'fair_type': FAIR_TYPE,
    'fair_consistency_test': FAIR_CONSISTENCY_TEST,
    'fair_regression_method': FAIR_REGRESSION_METHOD,
    'fair_protected_max_groups': FAIR_PROTECTED_MAX_GROUPS,
    'fair_material_max_groups': FAIR_MATERIAL_MAX_GROUPS,
    'fair_protected_disc_method': FAIR_PROTECTED_DISC_METHOD,
    'fair_material_disc_method': FAIR_MATERIAL_DISC_METHOD,
    'gen_outfiletext': GEN_OUTFILETEXT,
    'gen_outpath': GEN_OUTPATH, 'gen_output_type': GEN_OUTPUT_TYPE,
    'gen_mp_parallel': GEN_MP_PARALLEL,
    'gen_variable_importance': GEN_VARIABLE_IMPORTANCE,
    'other_costs_of_treat': OTHER_COSTS_OF_TREAT,
    'other_costs_of_treat_mult': OTHER_COSTS_OF_TREAT_MULT,
    'other_max_shares': OTHER_MAX_SHARES,
    'pt_depth_tree_1': PT_DEPTH_TREE_1, 'pt_depth_tree_2': PT_DEPTH_TREE_2,
    'pt_enforce_restriction': PT_ENFORCE_RESTRICTION,
    'pt_eva_cat_mult': PT_EVA_CAT_MULT,
    'pt_no_of_evalupoints': PT_NO_OF_EVALUPOINTS,
    'pt_min_leaf_size': PT_MIN_LEAF_SIZE,
    'pt_select_values_cat': PT_SELECT_VALUES_CAT,
    'rnd_shares': RND_SHARES,
    'var_bb_restrict_name': VAR_BB_RESTRICT_NAME,
    'var_d_name': VAR_D_NAME, 'var_effect_vs_0': VAR_EFFECT_VS_0,
    'var_effect_vs_0_se': VAR_EFFECT_VS_0_SE, 'var_id_name': VAR_ID_NAME,
    'var_polscore_desc_name': VAR_POLSCORE_DESC_NAME,
    'var_polscore_name': VAR_POLSCORE_NAME,
    'var_protected_name_ord': VAR_PROTECTED_NAME_ORD,
    'var_protected_name_unord': VAR_PROTECTED_NAME_UNORD,
    'var_material_name_ord': VAR_MATERIAL_NAME_ORD,
    'var_material_name_unord': VAR_MATERIAL_NAME_UNORD,
    'var_vi_x_name': VAR_VI_X_NAME,
    'var_vi_to_dummy_name': VAR_VI_TO_DUMMY_NAME,
    'var_x_name_ord': VAR_X_NAME_ORD, 'var_x_name_unord': VAR_X_NAME_UNORD,
    '_int_dpi': _INT_DPI, '_int_fontsize': _INT_FONTSIZE,
    '_int_output_no_new_dir': _INT_OUTPUT_NO_NEW_DIR,
    '_int_report': _INT_REPORT,
    '_int_with_numba': _INT_WITH_NUMBA,
    '_int_with_output': _INT_WITH_OUTPUT,
    '_int_xtr_parallel': _INT_XTR_PARALLEL,
    }
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

for method in METHODS:
    params['gen_method'] = method
    myoptp = OptimalPolicy(**params)
    # --- Fairness correction ------
    if FAIRNESS_CORRECTION:
        training_df, _, _, _ = myoptp.fairscores(training_all_df,
                                                 data_title='training')
    else:
        training_df = training_all_df

    # ----- Training data ----------
    alloc_train_df, _, _ = myoptp.solve(training_df,
                                        data_title='training fair')

    results_eva_train, _ = myoptp.evaluate(alloc_train_df, training_df,
                                           data_title='training fair')
    if method != 'best_policy_score':
        alloc_pred_df, _ = myoptp.allocate(prediction_df,
                                           data_title='prediction')
        # ----- Evaluate using prediction data ------
        results_eva_pred, _ = myoptp.evaluate(alloc_pred_df, prediction_df,
                                              data_title='prediction')
    myoptp.print_time_strings_all_steps()
    my_report = McfOptPolReport(optpol=myoptp,
                                outputfile='Report_OptP_' + method)
    my_report.report()
    del myoptp, my_report

print('End of example estimation.\n\nThanks for using OptimalPolicy. \n\nYours'
      ' sincerely\nOptimalPolicy MCF modul (beta) \U0001F600')
