"""
Combining mcf estimation of IATEs with optimal policy implementation.

Goal is to end up with good allocations, not with estimates of causal effects.

# -*- coding: utf-8 -*-

Created on Thu Sep 22 08:31:20 2022

@author: MLechner

The change-log is contained user_mcf_full.py.

A.	Evaluation of Black-Box procedures: There is no E-T sample split needed.
    We compute the IATEs via cross-validation with a standard mcf
    including standard errors.

B.	Evaluation of policy tree
a.	E-T split (80-20)
b.	Estimate IATEs-8020 on E and predict for T. This will also automatically
    take account of common support. The programme will output a common-support
    adjusted T that will be used further.
c.	Using E only: For all observations in E (inside common support), estimate
    IATEs by CV. No standard errors, cross-fitting in side each mcf estimation.
d.	Using E only: For all observations in E (inside common support), build
    policy tree based on IATEs estimated in step c).
e.	Using T only: Predict allocations for all observations in T
    (inside common support).
f.	Using T only: Compare predicted outcome to observed and random allocations
    (as we did in the paper) for the policy tree. For this evaluation use the
    IATEs computed in step b)


Version 0.3.1.dev

"""
from mcf import optp_mcf_functions as opt_mcf

VAR_FLAG1 = False   # No variable of mcf, used just for this test

APPLIC_PATH = 'c:/mlechner/mcftest'  # NOT passed to MCF
OUTPATH = APPLIC_PATH + '/out'
#   If this None a */out directory below the current directory is used
#   If specified directory does not exist, it will be created.
DATAPATH = APPLIC_PATH + '/testdata'
#   If a path is None, path of this file is used.

INDATA = 'dgp_mcfN1000S5'    # 'dgp_mcfN1000S5'
# csv for estimation (without extension)

CV_IATE = 3   # Number of folds in cross-validated estimation of IATEs

OUTFILETEXT = INDATA + 'OptPolmcf'  # File for text output

EVALUATE_WHAT = ('poltree',)
# ('blackb', 'poltree') Tuple of which methods to evaluate

# Variables for mcf estimation
D_NAME = 'D0_1_12'
Y_NAME = ['y']
ID_NAME = ['ID']
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

if VAR_FLAG1:
    X_NAME_UNORD = ['cat' + str(i) for i in range(10)]
else:
    X_NAME_UNORD = ['cat' + str(i) for i in range(2)]
X_NAME_ALWAYS_IN_ORD = X_NAME_ALWAYS_IN_UNORD = X_NAME_REMAIN_ORD = None

# mcf paramamters (used to compute IATE(x))
BOOT = None
L_CENTERING = True
L_CENTERING_UNDO_IATE = False    # Could be problematic in CV
SUPPORT_CHECK = None
SUPPORT_QUANTIL = None

# Opt Policy parameters
POLSCORE_DESC_NAME = ['CONT0', 'CONT1', 'CONT2', 'CONT3', 'CONT4', 'CONT5']
OPT_X_ORD_NAME = ['CONT0', 'CONT1', 'DUM0','DUM1', 'ORD0', 'ORD1']
OPT_X_UNORD_NAME = ['CAT0PR', 'CAT1PR']
BB_REST_VARIABLE = 'Cont0'
BB_BOOTSTRAPS = None    # Bootstrap replications to measure uncertainty of
BB_STOCHASTIC = False   # Use stochastic assignment. Default is False.
FT_NO_OF_EVALUPOINTS = None   # This parameter is closely related to
FT_DEPTH = 2      # > 0: (def: 3)
FT_MIN_LEAF_SIZE = None   # Minimum leaf size: def: 10% of sample size
FT_MAX_TRAIN = 300
# Maximum number of observations used to train the policy tree; if training
# data is larger, an corresponding random sample will be drawn

# General parameters
MAX_SHARES = (1,1,0.5)  # Maximum shares allowed for each policy
COSTS_OF_TREAT = None   # Cost per treatment, will be subtracted
COSTS_OF_TREAT_MULT = None
ONLY_IF_SIG_BETTER_VS_0 = False       # Default: False
SIG_LEVEL_VS_0 = None                 # Default: 0.05

# --- Ususally no need to change these parameters -----------------------------
_TRAIN_SHARE = 0.8
_WITH_OUTPUT = _PRINT_TO_FILE = _PRINT_TO_TERMINAL = True
_SEED_SAMPLE_SPLIT = 122435467
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

opt_mcf.optpolicy_with_mcf(
    boot=BOOT, bb_bootstraps=BB_BOOTSTRAPS, bb_stochastic=BB_STOCHASTIC,
    bb_rest_variable=BB_REST_VARIABLE,
    cv_iate=CV_IATE, costs_of_treat=COSTS_OF_TREAT,
    costs_of_treat_mult=COSTS_OF_TREAT_MULT,
    datapath=DATAPATH, d_name=D_NAME,
    evaluate_what=EVALUATE_WHAT,
    ft_depth=FT_DEPTH, ft_min_leaf_size=FT_MIN_LEAF_SIZE,
    ft_no_of_evalupoints=FT_NO_OF_EVALUPOINTS, ft_max_train=FT_MAX_TRAIN,
    id_name=ID_NAME, indata=INDATA,
    l_centering=L_CENTERING, l_centering_undo_iate=L_CENTERING_UNDO_IATE,
    max_shares=MAX_SHARES,
    only_if_sig_better_vs_0=ONLY_IF_SIG_BETTER_VS_0,
    opt_x_ord_name=OPT_X_ORD_NAME, opt_x_unord_name=OPT_X_UNORD_NAME,
    outpath=OUTPATH, outfiletext=OUTFILETEXT,
    print_to_file=_PRINT_TO_FILE, print_to_terminal=_PRINT_TO_TERMINAL,
    polscore_desc_name=POLSCORE_DESC_NAME,
    sig_level_vs_0=SIG_LEVEL_VS_0,
    support_check=SUPPORT_CHECK, support_quantil=SUPPORT_QUANTIL,
    train_share=_TRAIN_SHARE,
    with_output=_WITH_OUTPUT,
    x_name_always_in_ord=X_NAME_ALWAYS_IN_ORD,
    x_name_always_in_unord=X_NAME_ALWAYS_IN_UNORD, x_name_ord=X_NAME_ORD,
    x_name_unord=X_NAME_UNORD, x_name_remain_ord=X_NAME_REMAIN_ORD,
    y_name=Y_NAME,
    _seed_sample_split = _SEED_SAMPLE_SPLIT,
    )
