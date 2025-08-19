"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.8.0

This is an example to show how the mcf with BGATE estimation can be implemented
relying on defaults for parameters not related to the BGATE. Note that usually
in application it is very likely to be appropriate to deviate from some of the
default specifications.

Related paper: Nora Bearth & Michael Lechner (2025): Causal Machine Learning
               for Moderation Effects, Journal of Business & Economic Statistics
               (JBES).

"""
from pathlib import Path
import warnings

from mcf.example_data_functions import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example'
DATPATH = APPLIC_PATH / 'data'
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
VAR_X_NAME_BALANCE_BGATE = 'x_cont2'
# -----------------------------------------------------------------------------
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)
# ---------------------- Generate artificial data ------------------------------

training_df, prediction_df, name_dict = example_data()

# Modules may sent many irrelevant warnings: Globally ignore them
warnings.filterwarnings('ignore')
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(gen_outpath=APPLIC_PATH,
                             var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             var_z_name_cont=VAR_Z_NAME,
                             p_cbgate=True, p_bgate=True,
                             var_x_name_balance_bgate=VAR_X_NAME_BALANCE_BGATE,
                             )

mymcf.train(training_df)  # Returns not used here
results = mymcf.predict(prediction_df)
results_with_cluster_id_df = mymcf.analyse(results)
my_report = McfOptPolReport(mcf=mymcf)
my_report.report()
print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600')
