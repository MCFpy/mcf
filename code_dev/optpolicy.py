"""Created on Wed Apr  1 15:58:30 2020.

Optimal Policy Trees - Python implementation

Please note that this Optimal Policy (Decision) module is experimental.
It is much less tested than the MCF main module.

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

Version: 0.3.0
-*- coding: utf-8 -*- .

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

The change log (for the different versions of the programme) is contained in
the file user_mcf_full.py .

New black-box simulation algorithm added in version 0.3.0
- Main idea: Account for uncertainty of the policy score in the allocation 
             decision beyond doing non-zero allocations only for effects
             which are statistically different from zero (which is already
                                                          implemented)
- Implementation: Draw value from the normal distribution of each IATE[i]
  assuming N(EFFECT_VS_0[i], EFFECT_VS_0_SE [i]), i=1:M.
  If largest value is positive: Assign treatment corresponding to this IATE.
  If largest value is negative: Assign zero treatment.
- This approach is much less sophisticated but in the spirit of
  Kitagawa,Lee, Qiu (2022, arXiv) using the sampling distribution instead of
  the posterior distribution.
- A note on the assumed correlation structure of the IATEs in the case of 
  multiple treatments: Implicitly this simulation strategy assumes
  (incorrectly) that the IATEs are independent (which of course is no
  restriction for the binary treatment)
- Necessary condition: EFFECT_VS_0 and EFFECT_VS_0_SE must be available
- Control variable: BB_STOCHASTIC (default is False implying deterministic
                                   assignment)
- Advantages of this approach:
    * Takes sampling uncertainty into account.
    * Average ret of optimal allocation should have much less sampling variance
      (i.e. it will be robust to small changes of the policy-scores)
- Disadvantages of this approach:
    * Black-box nature
    * Some additional simulation uncertainty is added.
    * Estimator of sampling uncertainty of estimated IATE is required.
    * It has some ad-hoc notion
    * Not yet tested in a simulation or theoretical analysis
    * Incorrect correlation assumptions in case of multiple treatments
- Stochastic simulations are currently only used for black-box approaches
"""
from mcf import optp_functions as optp
from psutil import cpu_count

APPLIC_PATH = 'c:/mlechner/mcftest'  # NOT passed to MCF
OUTPATH = APPLIC_PATH + '/out'
#   If this None a */out directory below the current directory is used
#   If specified directory does not exist, it will be created.
DATPATH = APPLIC_PATH + '/testdata'
#   If a path is None, path of this file is used.

INDATA = 'dgp_mcfN1000S5PredpredXXX'  # csv for estimation (without extension)
PREDDATA = 'dgp_mcfN1000S5Predpredxxxxx'
# PREDDATA = 'dgp_mcfN1000S5PredpredXXX'

OUTFILETEXT = INDATA + "OptPolicy.0.3.0"  # File for text output
#   if outfiletext is None, name of indata with extension .out is used

SAVE_PRED_TO_FILE = None  # If True predictions will be save with PREDDATA
#                           Default is True.

ID_NAME = 'ID'
#  If no identifier -> it will be added to the data that is saved for later use

D_NAME = None #'DUM1'
# Treatment name. Only necessary for changers analysis in training sample
# Default is None.

POLSCORE_NAME = ['YLC0_pot', 'YLC1_pot', 'YLC2_Pot', 'YLC3_Pot']
#   This is ususally the potential outcome
#   (or the policy scores in the language of Zhou, Athey, Wager (2019)

POLSCORE_DESC_NAME = ['CONT0', 'CONT1', 'CONT2', 'CONT3',
                      'CONT5', 'CONT6', 'CONT7', 'CONT8']
# These should relate to other potential outcomes. They will be used to
# describe the obtained allocation in terms of their effects. Should be a list
# or None (not used)

EFFECT_VS_0 = ['YLC1vs0_iate', 'YLC2vs0_iate', 'YLC3vs0_iate']
#   Effects relative to treatment zero: Default is None.
EFFECT_VS_0_SE = ['YLC1vs0_iate_se', 'YLC2vs0_iate_se', 'YLC3vs0_iate_se']
#   Standard errors of effects relative to treatment zero. Default is None.

X_ORD_NAME = ['CONT0', 'CONT1',
              'DUM0','DUM1',
              'ORD0', 'ORD1']
#   Ordered variables used to build policy tree

X_UNORD_NAME = ['CAT0PR',
                'CAT1PR']
#   Unordered variables used to build policy tree

#   Variables that control output. If not specified or set to None, defaults
#   will be used.
OUTPUT_TYPE = None          # 0: Output goes to terminal
#                             1: output goes to file
#                             2: Output goes to file and terminal (def)

PARALLEL_PROCESSING = True  # False: No parallel computations
#                             True: Multiprocessing (def)
HOW_MANY_PARALLEL = cpu_count(logical=True)-1

MP_WITH_RAY = True  # True: Ray, False: Concurrent futures for Multiproces.
#                           False may be faster with small samples/trees
#                           True (def): Should be superior with larger samples

# Number of parallel processes(def:# of cores)
WITH_NUMBA = True           # Use Numba to speed up programme (def: True)

# Data cleaning
SCREEN_COVARIATES = True  # True (Default): screen covariates (sc)
CHECK_PERFECTCORR = True  # if sc=True: if True (default), var's that are
#                             perfectly correlated with others will be deleted
MIN_DUMMY_OBS = None      # if sc=1: dummy variable with obs in one
#                    category smaller than this value will be deleted (def: 10)
CLEAN_DATA_FLAG = True    # if True (def), remove all missing & unnecessary
#                             variables from data set

# --------------------------- Black Box allocations -------------------------
BB_YES = True             # Allocate according to potential outcomes (def:True)
# This 'black box approach will be conducted on INDATA and PREDDATA
# if PREDDATA contains the potential outcomes as well. If not, it will only
# be conducted on indata.
BB_REST_VARIABLE = 'Cont0'  # If BB_YES = True
# If there is a capacity constraint, preference will be given to obs. with
# highest values of this variable; must inside a list

BB_BOOTSTRAPS = None    # Bootstrap replications to measure uncertainty of
#  allocation (with given policy score). Default value (None) is 499

BB_STOCHASTIC = False   # Use stochastic assignment. Default is False.
# --------------------------- Policy Trees -----------------------------------
# Parameters for building full decision tree
FT_YES = True             # Build full decision tree (def: True)
FT_NO_OF_EVALUPOINTS = None   # This parameter is closely related to
#                           approx. parameter of Zhou, Athey, Wager (A)
#                           A =  N / NO_OF_EVALUPOINTS (def: 100)
FT_DEPTH = 2      # > 0: (def: 3)

FT_MIN_LEAF_SIZE = None   # Minimum leaf size: def: 10% of sample size

# Parameters for building sequential decision tree (faster, less efficient)
ST_YES = FT_YES              # Build sequential decision tree (def: True)
ST_DEPTH = FT_DEPTH      # > 0: (def: 4)

ST_MIN_LEAF_SIZE = None   # Minimum leaf size: def: 10% of sample size
#                                                divided by  number of leaves
# ------------------------- Other parameters ---------------------------------
MAX_SHARES = (1,1,1,0.5)  # Maximum shares allowed for each policy
#   This is a tuple with as many elements as potential outcomes 0-1
#   (1,...,1) implies unconstrained optimization
COSTS_OF_TREAT = None   # Cost per treatment, will be subtracted
#   from policy scores. (same dimension as maxshares): Treatment specific costs
#   0: No costs
#   def (None): Cost will be automatically determined: Find costs that lead to
#   allocation (not a tree, but chosing individually best treatments) that
#   fullfils restrictions in max_shares with small(est) costs
#   This is used to enforce restrictions in policy trees. It is not used
#   for the black-box approaches.
COSTS_OF_TREAT_MULT = None   # Multiplier of automatic costs (0-1);
#   None or Tuple of positive numbers with dimension equal to the number of
#   treatments. Default is (1, ..., 1)
#   Used only when automatic costs do not lead to a satisfaction of the
#   constraints given by MAX_SHARES. This allows to increase (>1) or decrease
#   (<1) the share of treated in particular treatment.

#   Base assignment only on policy scores that are significantly better than
#   first score in polscore_name
ONLY_IF_SIG_BETTER_VS_0 = False       # Default: False
SIG_LEVEL_VS_0 = None                 # Default: 0.05
# ----------------------------------------------------------------------------
_WITH_OUTPUT = True       # Show output
_SMALLER_SAMPLE = 0       # 0 < test_only < 1: test prog.with smaller sample
_FT_BOOTSTRAPS = 0      # Not yet implemented
_ST_BOOTSTRAPS = 0      # Not yet implemented
# ------------------------------------------------------------------------
if __name__ == '__main__':
    optp.optpoltree(
        indata=INDATA, preddata=PREDDATA, datpath=DATPATH, outpath=OUTPATH,
        save_pred_to_file=SAVE_PRED_TO_FILE,
        id_name=ID_NAME, polscore_name=POLSCORE_NAME,
        polscore_desc_name=POLSCORE_DESC_NAME, d_name=D_NAME,
        x_ord_name=X_ORD_NAME, x_unord_name=X_UNORD_NAME,
        effect_vs_0=EFFECT_VS_0, effect_vs_0_se=EFFECT_VS_0_SE,
        output_type=OUTPUT_TYPE, outfiletext=OUTFILETEXT,
        mp_with_ray=MP_WITH_RAY, parallel_processing=PARALLEL_PROCESSING,
        how_many_parallel=HOW_MANY_PARALLEL, with_numba=WITH_NUMBA,
        screen_covariates=SCREEN_COVARIATES, min_dummy_obs=MIN_DUMMY_OBS,
        check_perfectcorr=CHECK_PERFECTCORR, clean_data_flag=CLEAN_DATA_FLAG,
        ft_yes=FT_YES, ft_no_of_evalupoints=FT_NO_OF_EVALUPOINTS,
        ft_depth=FT_DEPTH, ft_min_leaf_size=FT_MIN_LEAF_SIZE,
        st_yes=ST_YES, st_depth=ST_DEPTH,  st_min_leaf_size=ST_MIN_LEAF_SIZE,
        max_shares=MAX_SHARES,  costs_of_treat=COSTS_OF_TREAT,
        costs_of_treat_mult=COSTS_OF_TREAT_MULT,
        only_if_sig_better_vs_0=ONLY_IF_SIG_BETTER_VS_0,
        sig_level_vs_0=SIG_LEVEL_VS_0, bb_yes=BB_YES,
        bb_rest_variable=BB_REST_VARIABLE, ft_bootstraps=_FT_BOOTSTRAPS,
        st_bootstraps=_ST_BOOTSTRAPS, bb_bootstraps=BB_BOOTSTRAPS,
        bb_stochastic=BB_STOCHASTIC,
        _smaller_sample=_SMALLER_SAMPLE, _with_output=_WITH_OUTPUT)
