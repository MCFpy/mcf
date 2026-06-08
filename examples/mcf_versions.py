"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.10.0

This is an example to show how the version of treatments can be dealt with in
the mcf (using defaults for other parameters). For more details see Lechner &
Kutz (2026): XXXX (in prepration).

"""
from copy import deepcopy
from pathlib import Path
import warnings

from mcf.example_data import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example/output'

# ---------------------- Generate artificial data ------------------------------
OBS_TRAIN = 1000
OBS_PRED = 1000
NO_MAIN_TREATMENTS = 3
D_NO_VERSIONS = 3

training_df, prediction_df, name_dict = example_data(obs_y_d_x_iate=OBS_TRAIN,
                                                     obs_x_iate=OBS_PRED,
                                                     no_treatments=NO_MAIN_TREATMENTS,
                                                     d_no_versions=D_NO_VERSIONS,
                                                     )
# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = name_dict['d_name']   # Name of treatment variable
#  If d_no_versions > 2, this is a list with the treatment version as second
#  variable. Note that the value of versions are conditional on the main
#  treatment. In other words, version 2 of treatment 1 and version 2 of
#  treatment 2 lead to different potential outcomes.
#  IMPORTANT: Main treatment must always be the first element in the treatments
#             list. This variable is also used for the programme to determine
#             if there are treatment versions at all.
#             Only one element in list/tuple or string: No treatment versions.
#             Two elements in list/tuple: 1st element is main treatment,
#                                         2nd element is subtreatment.

VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1',)
VAR_X_NAME_UNORD = ('x_unord0', 'x_unord1', )
VAR_Z_NAME_ORD = name_dict['x_name_ord'][-2:]

VAR_X_NAME_TV = ('x_cont1', 'x_unord0',)
#  List or tuple of names of features used as regressors. These variables must be included
#  var_x_name_ord or var_x_name_unord. Default is None.

GEN_TV_ESTIMATOR = None
#  Estimation used to predict effect of specific version given (the weights of) the main treatment.
#  Possible options are 'ols', 'ridge'. Default (or None) is 'ridge'.

GEN_TV_SPECIFICATION = None   # 'interacted', 'separable'
#  This defines how the covariates enter the version-regressions inside the main treatments
#  'interacted': Covariates are interacted with the version dummies (V*X * b).
#  'separable': Covariates and version dummies enter in a linearly separable way ((V*b1 + X*b2).
#  Default (or None) is 'interacted'.

GEN_TV_MIN_SUBTREAT = None
#  Minimum number of subtreated per treatment. If actually number of subtreated with positive
#  weight in effct estimation is below gen_tv_min_subtreat, the average effect is used instead of
#  this subtreatment. Default is 10.

GEN_TV_CV_K = None
#  Number of folds in cross-validation for treatment version estimation (to find optimal penalty
#  for ridge regression). Only relevant if GEN_TV_ESTIMATOR == 'ridge' is used.
#  Default value (or None) depends on the size of the training sample (N):
#  N < 100'000: 5;  100'000 <= N < 250'000: 4; 250'000 <= N < 500'000: 3; 500'000 <= N: 2.

GEN_TV_PENALIZE_VERSION = None
#  Determines whether the coefficients of version dummies are penalized in a particular main
#  treatment. Only relevant if GEN_TV_ESTIMATOR == 'ridge' is used.
#  This is either a Boolean or a list or tuple of Booleans. The number of elements of the
#  list/tuple MUST equal the number of main treatments. If a single Boolean is provided it will be
#  internally expanded to such a list for which all elements are equal to this single Boolean.
#  True: Coefficients of the version dummies in the version ridge regression are (also) penalized
#        (including treatment covariate interactions).
#        Maybe useful, if there are very many treatment versions.
#  False: Only coefficients of covariates are penalized.
#  Default is False.

# -----------------------------------------------------------------------------
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

# Modules may send many irrelevant warnings: Globally ignore them
warnings.filterwarnings('ignore')
# -----------------------------------------------------------------------------
params = {'gen_outpath': APPLIC_PATH,
          'var_d_name': VAR_D_NAME,
          'var_y_name': VAR_Y_NAME,
          'var_x_name_ord': VAR_X_NAME_ORD,
          'var_x_name_unord': VAR_X_NAME_UNORD,
          'var_x_name_tv': VAR_X_NAME_TV,
          'var_z_name_ord': VAR_Z_NAME_ORD,
          'gen_tv_cv_k': GEN_TV_CV_K,
          'gen_tv_estimator': GEN_TV_ESTIMATOR,
          'gen_tv_min_subtreat': GEN_TV_MIN_SUBTREAT,
          'gen_tv_specification': GEN_TV_SPECIFICATION,
          'gen_tv_penalize_version': GEN_TV_PENALIZE_VERSION,
          }
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(**params)
mymcf.train(training_df)
mymcf2 = deepcopy(mymcf)   # Use below for a 2nd estimation without versions
results = mymcf.predict(prediction_df)
results_with_cluster_id_df = mymcf.analyse(results)
my_report = McfOptPolReport(mcf=mymcf)
my_report.report()

# Restimate without versions using the already trained forest (which does not
# depend on the versions)
results2 = mymcf2.predict(prediction_df, new_keywords={'no_treatment_versions': True},)
results_with_cluster_id_df = mymcf2.analyse(results2)
my_report2 = McfOptPolReport(mcf=mymcf2)
my_report2.report()

print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
