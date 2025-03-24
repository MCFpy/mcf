"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.7.2

This is an example to show how IV estimation can be implemented relying on
defaults. Note that usually in applications it is very likely to be appropriate
to deviate from some of the default specifications.

"""
from pathlib import Path

from mcf.example_data_functions import example_data
from mcf.mcf_functions import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example/output'

# ---------------------- Generate artificial data ------------------------------

training_df, prediction_df, name_dict = example_data(no_treatments=2,
                                                     obs_y_d_x_iate=1000,
                                                     obs_x_iate=1000,
                                                     strength_iv=10)

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = name_dict['d_name']          # Name of treatment variable
#   Note: Currently, treatment variables are restricted to be binary (IV only)

VAR_Y_NAME = name_dict['y_name']          # Name of outcome variable

VAR_X_NAME_ORD = name_dict['x_name_ord']

VAR_X_NAME_UNORD = name_dict['x_name_unord']

VAR_IV_NAME = name_dict['inst_bin_name']  # Name of instrumental variable
#   Note: Currently, instrumental variables are restricted to be binary
# -----------------------------------------------------------------------------
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(gen_outpath=APPLIC_PATH,
                             var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             var_x_name_unord=VAR_X_NAME_UNORD,
                             var_iv_name=VAR_IV_NAME,
                             cf_boot=1000,
                             )

mymcf.train_iv(training_df)  # Returns not used here
results, _ = mymcf.predict_iv(prediction_df)

results_with_cluster_id_df, _ = mymcf.analyse(results)

my_report = McfOptPolReport(mcf=mymcf)

my_report.report()

print('End of computations.\n\nThanks for using ModifiedCausalForest (IV).'
      ' \n\nYours sincerely\nMCF \U0001F600')
