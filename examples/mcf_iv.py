"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

Estimate IV effects with a binary treatment and a binary instrument.
Compare local aggregation of individual effects with global LATE estimation.
Other settings use API defaults, except full-weight prediction required for IV.

"""
from pathlib import Path

from mcf.example_data import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport

# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
OBS_TRAINING = 2000
OBS_PREDICTION = 2000
# ---------------------- Generate artificial data ------------------------------
training_df, prediction_df, name_dict = example_data(no_treatments=2,
                                                     obs_y_d_x_iate=OBS_TRAINING,
                                                     obs_x_iate=OBS_PREDICTION,
                                                     strength_iv=10,
                                                     no_effect=True,
                                                     )
# ------------------ Parameters of the ModifiedCausalForest -------------------
APPLIC_PATH = Path.cwd() / 'example/output'

CF_BOOT = None

VAR_D_NAME = name_dict['d_name']          # Name of treatment variable
#   IV requires two treatment values; training recodes the main treatment internally.

VAR_Y_NAME = name_dict['y_name']          # Name of outcome variable

VAR_X_NAME_ORD = name_dict['x_name_ord']

VAR_X_NAME_UNORD = name_dict['x_name_unord']

VAR_IV_NAME = name_dict['inst_bin_name']  # Name of instrumental variable
#   Instrument values must be exactly 0 or 1; other encodings are rejected.

VAR_Z_NAME_CONT = name_dict['x_name_ord'][:2]

# Ordered variables with few values
VAR_Z_NAME_ORD = name_dict['x_name_ord'][-2:]

#   Unordered variables
VAR_Z_NAME_UNORD = name_dict['x_name_unord'][:2]


P_IV_AGGREGATION_METHOD = None
# 'local': aggregate LIATEs into LATEs and requested group effects.
# 'global': compute LATE as the ratio of aggregate reduced-form/first-stage effects.
# Requested LIATEs use local first-stage scaling with either selection.
# None selects both. For one method, use 'local' or 'global' as a string;
# single-element lists/tuples are not accepted. Group effects require 'local'.

# -----------------------------------------------------------------------------
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

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
                             _int_low_memory_predict=False,
                             )
# IV requires sparse forest weights (the default) and low-memory prediction disabled.
mymcf.train_iv(training_df)
# Unselected aggregation methods return an empty dictionary.
results_global, results_local = mymcf.predict_iv(prediction_df)

# Pass one result dictionary at a time to analyse().
if results_global:
    mymcf.analyse(results_global)

if results_local:
    mymcf.analyse(results_local)

my_report = McfOptPolReport(mcf=mymcf,
                             outputpath=(results_global or results_local)['path_output'])

my_report.report()

print('End of computations.\n\nThanks for using ModifiedCausalForest (IV).'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
