"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.7.1

This is an example to show how the mcf can be implemented relying completely on
defaults. Note that usually in application it is very likely to be appropriate
to deviate from some of the default specifications.

"""
import os

from mcf.example_data_functions import example_data
from mcf.mcf_functions import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = os.getcwd() + '/example'

# ---------------------- Generate artificial data ------------------------------

training_df, prediction_df, name_dict = example_data()

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'treat'   # Name of treatment variable
VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0',)  # Using VAR_X_NAME_UNORD instead is fine as well
# -----------------------------------------------------------------------------
if not os.path.exists(APPLIC_PATH):
    os.makedirs(APPLIC_PATH)
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD)

mymcf.train(training_df)  # Returns not used here
results, _ = mymcf.predict(prediction_df)
results_with_cluster_id_df, _ = mymcf.analyse(results)
my_report = McfOptPolReport(mcf=mymcf)
my_report.report()
print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600')
