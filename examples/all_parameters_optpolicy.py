"""Created on Wed Apr  1 15:58:30 2020.

Optimal Policy - Python implementation

Please note that this Optimal Policy (Decision) module is experimental.
It is much less tested than the MCF main module.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.4.1

-*- coding: utf-8 -*- .

This is an example to show how to use the OptimalPolicy class of the mcf
module with full specification of all its keywords. It may be seen as an add on
to the published mcf documentation.

"""
import os
import pickle

import pandas as pd

from mcf import OptimalPolicy

# ------------- NOT passed to OptimalPolicy -----------------------------------
#  Define data to be used in this example
APPLIC_PATH = 'c:/mlechner/py_neu/example'
DATPATH = APPLIC_PATH + '/data'
TRAINDATA = 'data_x_ps_1_1000.csv'
#  'best_policy_score': Training data must contain policy scores.
#  'policy tree': Training data must contain policy scores and features.
PREDDATA = 'data_x_ps_2_1000.csv'
#  'best_policy_score': Prediction data must contain policy scores.
#  'policy tree': Prediction data must contain features that are used in
#                 training. Policy scores are not required.

#  Define which methods of the OptimalPolicy class will be run
SOLVE = True               # Train allocation algorithm with train method
ALLOCATE = True            # Use allocate method to allocate data
EVALUATE = True            # Evaluate allocation

METHODS = ('best_policy_score', 'policy tree',)  # Tuple used to set GEN_METHOD
#  Currently valid methods are: 'best_policy_score', 'policy tree'

# -------- All what follows are parameters of the ModifiedCausalForest --------
#   Whenever, None is specified, parameter will set to default values.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
GEN_OUTPATH = APPLIC_PATH + '/outputOPT'     # Directory for the output.
#   If it does not exist, it will be created. Default is an *.out directory
#   just below to the directory where the programme is run.

GEN_OUTFILETEXT = TRAINDATA + "OptPolicy.0.4.1"  # File for text output
#   Default is 'txtFileWithOutput'.
#   *.txt file extension will be added by the programme

#   ------------- Where to direct the text output to --------------------------
GEN_OUTPUT_TYPE = None      # 0: Output goes to terminal
#                             1: output goes to file
#                             2: Output goes to file and terminal (default)
GEN_METHOD = None     # Available methods: 'best_policy_score', 'policy tree'
#   Default is 'best_policy_score'.
#   Note fo this example only: This variable will be overwritten inside the
#   loop at the end of the programme.

# ---------------- Names of variables used ------------------------------------
VAR_ID_NAME = 'ID'   # Name of identifier in data. Default is None.

VAR_D_NAME = 'D'  # 'DUM1' #  Name of discrete treatment.
#   Needed in training data only if 'changers' (different treatment than
#   observed treatment) are analysed and if allocation is compared to observed
#   allocation.

VAR_POLSCORE_NAME = ('YLC0_pot', 'YLC1_pot', 'YLC2_Pot', 'YLC3_Pot')
#   Treatment specific variables to measure the value of individual treatments.
#   This is ususally the estimated potential outcome or any other score related

VAR_POLSCORE_DESC_NAME = (('CONT0', 'CONT1', 'CONT2', 'CONT3'),
                          ('CONT5', 'CONT6', 'CONT7', 'CONT8'))
#   Tuple of tuples. Each tuple of dimension equal to the different treatments
#   contains treatment specific variables that are used to evaluate the effect
#   of the allocation with respect to those variables. This could be for
#   example policy score not used in training,but which are relevant
#   nevertheless. Default is no variables.

VAR_EFFECT_VS_0 = ('YLC1vs0_iate', 'YLC2vs0_iate', 'YLC3vs0_iate')
#   Effects of treatment relative to treatment zero. Dimension is equal to the
#   different treatments minus 1. Default is no variables.

VAR_EFFECT_VS_0_SE = ('YLC1vs0_iate_se', 'YLC2vs0_iate_se', 'YLC3vs0_iate_se')
#   Standard errors of effects relative to treatment zero. Dimension is equal
#    to the different treatments minus 1. Default is no variables.

VAR_X_ORD_NAME = ['CONT0', 'CONT1', 'DUM0', 'DUM1', 'ORD0', 'ORD1']
#   Ordered variables used to build policy tree.They are also used to
#   characterise the allocation.

VAR_X_UNORD_NAME = ['CAT0PR', 'CAT1PR']
#   Unordered variables used to build policy tree.They are also used to
#   characterise the allocation.

# ---------------------- Method specific parameters ---------------------------
# - - - - - - - - - - - -  Black-Box allocations - - - - - - - - - - - - - - -
#   The method 'best_policy_score' conducts Black-Box allocations.
#   'Black-Box allocations' are those that are obtained by using the scores
#   directly (potentially subject to restrictions).
#   When the Black-Box allocations are used for allocation of new data, the
#   respective scores must be available in the prediction data.

VAR_BB_RESTRICT_NAME = 'Cont0'   # Variable name related to a restriction.
#   If there is a capacity constraint, preference will be given to obs. with
#   highest values of this variable. Default is that no such variable is used.

# --------------------- Optimal shallow decision trees ------------------------
#   These decision trees are optimal trees, in the sense that all possible
#   trees are checked if they lead to a better performance. If restrictions
#   are specified, then this is incorparated into treatment specific cost
#   parameters (see below). They are calibrated by finding values that lead to
#   to desired capacity constraints with (unrestricted) Black-Box allocations
#   (as the latter are much, much faster to compute than decision trees).
#   Many ideas of the implementation follow Zhou, Athey, Wager (2022). If the
#   provided policy score fulfils their conditions (i.e., they use a doubly
#   robust double machine learning like score), then they also provide
#   attractive theoretical properties.

PT_DEPTH = 2         # Depth of tree. Defined such that PT_DEPTH == 1 implies 2
#   splits, PT_DEPTH = 2 implies 4 leafs, PT_DEPTH = 3 implies 8 leafs, etc.
#   Default is 3.
#   In this example, this parameter is set to 2 to speed up computation.

PT_NO_OF_EVALUPOINTS = None  # No of evaluation points for continous variables.
#   The lower this value, the faster the algorithm, but it may also deviate
#   more from the optimal splitting rule. This parameter is closely related to
#   approximation parameter of Zhou, Athey, Wager (2022)(A) with
#   A =  # of observation / PT_NO_OF_EVALUPOINTS. Default is 100.

PT_MIN_LEAF_SIZE = None    # Minimum leaf size. Leaves that are smaller than
#   PT_MIN_LEAF_SIZE in the training data will not be considered. A larger
#   number reduces computation time and avoids some overfitting. Default is
#   0.1 x # of training observations / # of leaves.

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
OTHER_MAX_SHARES = (1, 1, 1, 0.5)  # Maximum share allowed for each treatment.
#   This is a tuple with as many elements as treatments.
#   0 <  OTHER_MAX_SHARES <= 1.  (1,...,1) implies unconstrained optimization
#   (default).

OTHER_COSTS_OF_TREAT = None   # Treatment specific costs. These costs
#   are directly substracted from the policy scores. Therefore, they should be
#   measured in the same units as the scores. Default is 0.
#   If None, when are constraints, costs will be automatically determined to
#   enforce constraints in thetraining data by finding costs that lead to an
#   allocation (not a tree, but chosing individually best treatments by
#   unconstrained Black-Box methods) that fulfils restrictions in
#   OTHER_MAX_SHARES.

OTHER_COSTS_OF_TREAT_MULT = None   # Multiplier of automatically determined
#   costs. None or Tuple of positive numbers with dimension equal to the number
#   of treatments. Default is (1, ..., 1)
#   Use only when automatic costs do not lead to a satisfaction of the
#   constraints given by OTHER_MAX_SHARES. This allows to increase (>1) or
#    decrease (<1) the share of treated in particular treatment.

# -----------Internal parameters. Change only if good reason to do so. --------
INT_WITH_OUTPUT = None   # Print output on file and screen. Default is True.

INT_PARALLEL_PROCESSING = None  # False: No parallel computations
#                             True: Multiprocessing (def)
INT_HOW_MANY_PARALLEL = None  # Number of parallel process. Default is 80% of
#   cores, if this can be effectively implemented.

INT_WITH_NUMBA = None    # Use Numba to speed up computations: Default is True.
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
    'gen_method': GEN_METHOD, 'gen_outfiletext': GEN_OUTFILETEXT,
    'gen_outpath': GEN_OUTPATH, 'gen_output_type': GEN_OUTPUT_TYPE,
    'int_how_many_parallel': INT_HOW_MANY_PARALLEL,
    'int_parallel_processing': INT_PARALLEL_PROCESSING,
    'int_with_numba': INT_WITH_NUMBA, 'int_with_output': INT_WITH_OUTPUT,
    'other_costs_of_treat': OTHER_COSTS_OF_TREAT,
    'other_costs_of_treat_mult': OTHER_COSTS_OF_TREAT_MULT,
    'other_max_shares': OTHER_MAX_SHARES,
    'pt_depth': PT_DEPTH, 'pt_no_of_evalupoints': PT_NO_OF_EVALUPOINTS,
    'pt_min_leaf_size': PT_MIN_LEAF_SIZE,
    'rnd_shares': RND_SHARES,
    'var_bb_restrict_name': VAR_BB_RESTRICT_NAME,
    'var_d_name': VAR_D_NAME, 'var_effect_vs_0': VAR_EFFECT_VS_0,
    'var_effect_vs_0_se': VAR_EFFECT_VS_0_SE, 'var_id_name': VAR_ID_NAME,
    'var_polscore_desc_name': VAR_POLSCORE_DESC_NAME,
    'var_polscore_name': VAR_POLSCORE_NAME,
    'var_x_ord_name': VAR_X_ORD_NAME, 'var_x_unord_name': VAR_X_UNORD_NAME
    }

PICKLE_FILE_BB = DATPATH + '/myoptptrain_BB.pickle'  # Storing results on disc
PICKLE_FILE_PT = DATPATH + '/myoptppredict_PT.pickle'
CSV_FILE_BB = DATPATH + '/myoptptrain_BB.csv'  # Storing results on disc
CSV_FILE_PT = DATPATH + '/myoptppredict_PT.csv'

train_df = pd.read_csv(DATPATH + '/' + TRAINDATA)
pred_df = pd.read_csv(DATPATH + '/' + PREDDATA)


def save_load(file_name, object_to_save=None, save=True, output=True):
    """Save and load objects via pickle."""
    if save:
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, "wb+") as file:
            pickle.dump(object_to_save, file)
        object_to_load = None
        text = '\nObject saved to '
    else:
        with open(file_name, "rb") as file:
            object_to_load = pickle.load(file)
        text = '\nObject loaded from '
    if output:
        print(text + file_name)
    return object_to_load


for method in METHODS:
    if method == 'best_policy_score':
        pickle_file, alloc_file = PICKLE_FILE_BB, CSV_FILE_BB
    elif method == 'policy tree':
        pickle_file, alloc_file = PICKLE_FILE_PT, CSV_FILE_PT
    else:
        raise ValueError('Invalid method specified')

    params['gen_method'] = method
    if SOLVE:
        myoptp = OptimalPolicy(**params)
        # ----- Training data ----------
        alloc_train_df = myoptp.solve(train_df, data_title=TRAINDATA)
        save_load(pickle_file, object_to_save=myoptp, save=True, output=True)
    else:
        myoptp = save_load(pickle_file, object_to_save=None, save=False,
                           output=True)
        alloc_train_df = pd.read_csv(alloc_file)
    if EVALUATE:    # Evaluate using training data
        results_eva_train = myoptp.evaluate(alloc_train_df, train_df,
                                            data_title=TRAINDATA)
    if ALLOCATE:
        alloc_pred_df = myoptp.allocate(pred_df, data_title=PREDDATA)
    if EVALUATE:    # Evaluate using prediction data
        results_eva_pred = myoptp.evaluate(alloc_pred_df, pred_df,
                                           data_title=PREDDATA)
    myoptp.print_time_strings_all_steps()

print('End of example estimation.\n\nThanks for using OptimalPolicy. \n\nYours'
      ' sincerely\nExperimental OptimalPolicy MCF modul \U0001F600')
