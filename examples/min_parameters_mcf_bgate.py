"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.7.0

This is an example to show how the mcf with BGATE estimation can be implemented
relying completely on defaults. Note that usually in application it is very
likely to be appropriate to deviate from some of the default specifications.

"""
import os

from mcf.example_data_functions import example_data
from mcf.mcf_functions import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = os.getcwd() + '/example'
DATPATH = APPLIC_PATH + '/data'
TRAINDATA = 'data_y_d_x_1000.csv'
#  Training data must contain outcome, treatment and features.

PREDDATA = 'data_x_1000.csv'
#  Training data must contain features. Treated effects on the treated
#  additionally require treatment information.

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'treat'            # Name of treatment variable
VAR_Y_NAME = 'outcome'          # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2')
VAR_Z_NAME = 'x_cont1'
VAR_BGATE = 'x_cont2'
# -----------------------------------------------------------------------------
if not os.path.exists(APPLIC_PATH):
    os.makedirs(APPLIC_PATH)
# ---------------------- Generate artificial data ------------------------------

training_df, prediction_df, name_dict = example_data()
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             var_z_name_list=VAR_Z_NAME,
                             p_cbgate=True, p_bgate=True,
                             var_bgate_name=VAR_BGATE,
                             # gen_mp_parallel=1
                             )

mymcf.train(training_df)  # Returns not used here
results, _ = mymcf.predict(prediction_df)
results_with_cluster_id_df, _ = mymcf.analyse(results)
my_report = McfOptPolReport(mcf=mymcf)
my_report.report()
print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600')
