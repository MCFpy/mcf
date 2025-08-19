"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.8.0

This is an example to show how IV estimation can be implemented relying on
defaults. Note that usually in applications it is very likely to be appropriate
to deviate from some of the default specifications.

"""
from pathlib import Path
import warnings

from mcf.example_data_functions import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport

# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
OBS_TRAINING = 2000
OBS_PREDICTION = 2000
# ---------------------- Generate artificial data ------------------------------
training_df, prediction_df, name_dict = example_data(
    no_treatments=2,
    obs_y_d_x_iate=OBS_TRAINING,
    obs_x_iate=OBS_PREDICTION,
    strength_iv=10,
    no_effect=True
    )

# ------------------ Parameters of the ModifiedCausalForest -------------------
APPLIC_PATH = Path.cwd() / 'example/output'

CF_BOOT = None

VAR_D_NAME = name_dict['d_name']          # Name of treatment variable
#   Note: Currently, treatment variables are restricted to be binary (IV only)

VAR_Y_NAME = name_dict['y_name']          # Name of outcome variable

VAR_X_NAME_ORD = name_dict['x_name_ord']

VAR_X_NAME_UNORD = name_dict['x_name_unord']

VAR_IV_NAME = name_dict['inst_bin_name']  # Name of instrumental variable
#   Note: Currently, instrumental variables are restricted to be binary

VAR_Z_NAME_CONT = name_dict['x_name_ord'][:2]

# Ordered variables with few values
VAR_Z_NAME_ORD = name_dict['x_name_ord'][-2:]

#   Unordered variables
VAR_Z_NAME_UNORD = name_dict['x_name_unord'][:2]


P_IV_AGGREGATION_METHOD = None
#    Defines method used to obtain aggregated effects.
#    Possible values are local', 'global', ('local', 'global',)
#    'local':        LIATEs will be computed and aggregated to obtain
#                    LGATEs, LBGATEs, LATEs, etc..
#                    This estimator is "internally consistent".
#    'global':       LATEs (only) will be directly computed as
#                    the ratio of reduced form and first stage predictions.
#                    This estimator is not necessarily "internally consistent".
#     For the differences in assumptions and properties of the two approaches
#     see Lechner and Mareckova (2025).
#     Default is ('local', 'global',).

# -----------------------------------------------------------------------------
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

# Modules may sent many irrelevant warnings: Globally ignore them
warnings.filterwarnings('ignore')
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(gen_outpath=APPLIC_PATH,
                             var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             var_x_name_unord=VAR_X_NAME_UNORD,
                             var_z_name_cont=VAR_Z_NAME_CONT,
                             var_z_name_ord=VAR_Z_NAME_ORD,
                             var_z_name_unord=VAR_Z_NAME_UNORD,
                             var_iv_name=VAR_IV_NAME,
                             cf_boot=CF_BOOT,
                             p_iv_aggregation_method=P_IV_AGGREGATION_METHOD,
                             )

mymcf.train_iv(training_df)
results_global, results_local = mymcf.predict_iv(prediction_df)

if results_global:
    mymcf.analyse(results_global)

if results_local:
    mymcf.analyse(results_local)

my_report = McfOptPolReport(mcf=mymcf)

my_report.report()

print('End of computations.\n\nThanks for using ModifiedCausalForest (IV).'
      ' \n\nYours sincerely\nMCF \U0001F600')
