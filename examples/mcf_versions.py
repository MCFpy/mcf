"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

This example uses the mcf to estimate effects of the main treatments and
mcf-weighted regressions (OLS or ridge) to estimate effects of treatment versions
within each main treatment. It also shows how to reuse the trained forest for
estimation without treatment versions.

"""
from copy import deepcopy
from pathlib import Path

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

VAR_D_NAME = name_dict['d_name']
# Supply main treatment first and version second. Version labels are interpreted
# within each main treatment. A single treatment name disables version estimation.

VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1',)
VAR_X_NAME_UNORD = ('x_unord0', 'x_unord1', )
VAR_Z_NAME_ORD = name_dict['x_name_ord'][-2:]

VAR_X_NAME_TV = ('x_cont1', 'x_unord0',)
# Direct version-regression covariates; these must also be forest features. None uses none.

GEN_TV_ESTIMATOR = None
# 'ridge' (default): forest-weighted ridge with cross-validated penalty; 'ols': weighted OLS.
# Versions are incompatible with sampling weights, clustered/panel inference, QIATEs and P_BA.

GEN_TV_SPECIFICATION = None
# 'interacted' (default): version dummies and their covariate interactions.
# 'separable': version-specific intercepts with common covariate slopes.

GEN_TV_MIN_SUBTREAT = None
# Minimum observations with positive forest weight per version for the current effect.
# Below the threshold, use the main-treatment weights for that version. Default 10.

GEN_TV_CV_K = None
# Ridge-penalty CV folds; explicit values must be integers >=2. Automatic folds by training size:
# below 100,000 / 250,000 / 500,000 / otherwise: 5 / 4 / 3 / 2.

GEN_TV_PENALIZE_VERSION = None
# Penalize version-dummy coefficients as well as covariates and interactions. Default False.
# A Boolean or one-element list/tuple applies to all main treatments; otherwise supply one Boolean
# per main treatment. Covariate interactions remain penalized when this option is False.

GEN_TV_ADDSCORES = None
# Add the selected propensity/prognostic scores to version regressions. Default False.

GEN_TV_SCORE_ESTIMATOR = None
# Prognostic regression: 'RandomForest' (default), 'automatic' for CV selection, or another API
# estimator. Propensity scores and prognostic scores for outcomes with fewer than 10 distinct
# training values use random-forest classification; very small groups use their mean outcome.

GEN_TV_SCORE_CV_K = None
# Folds for automatic prognostic regression selection and local score cross-fitting.
# Automatic folds follow GEN_TV_CV_K; group sizes can reduce the effective count.
# Global propensity/classifier-based scores do not use this fold setting.

GEN_TV_SCORE_TYPE = None
# Select 'prop_version', 'prog_version', 'prop_main' and/or 'prog_main'.
# Default ('prop_version', 'prop_main'). Main scores use all forest features.
# Global version scores use VAR_X_NAME_TV; local version scores also use selected main scores.

GEN_TV_SCORE_VERSION_LOCAL = None
# False (default): fit global version-score models on tree-building data and reuse predictions
# for aggregate effects and IATEs. True fits version scores for each aggregate effect; IATE
# regressions then omit version-specific scores. Main-treatment scores are unaffected.
# Relevant only with GEN_TV_ADDSCORES and a selected version score.

# -----------------------------------------------------------------------------
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

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
          'gen_tv_addscores': GEN_TV_ADDSCORES,
          'gen_tv_score_estimator': GEN_TV_SCORE_ESTIMATOR,
          'gen_tv_score_cv_k': GEN_TV_SCORE_CV_K,
          'gen_tv_score_type': GEN_TV_SCORE_TYPE,
          'gen_tv_score_version_local': GEN_TV_SCORE_VERSION_LOCAL,
          }
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(**params)
mymcf.train(training_df)
# Copy before prediction: new_keywords requires a trained, not-yet-predicted instance.
mymcf2 = deepcopy(mymcf)
results = mymcf.predict(prediction_df)
results_with_clusters = mymcf.analyse(results)
my_report = McfOptPolReport(mcf=mymcf, outputpath=results['path_output'])
my_report.report()

# Re-estimate main-treatment effects using the same trained forest.
# Include no_treatment_versions only to disable versions; omit it to retain them.
# Any non-None value, including False, disables versions in the current API.
results2 = mymcf2.predict(prediction_df, new_keywords={'no_treatment_versions': True},)
results2_with_clusters = mymcf2.analyse(results2)
my_report2 = McfOptPolReport(mcf=mymcf2, outputpath=results2['path_output'])
my_report2.report()

print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
