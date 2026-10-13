"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

Train an MCF and predict for its retained outcome-filling sample.
These observations come from the training data; this is not an independent
test sample or prediction for every original training row.

"""
from pathlib import Path

from mcf.example_data import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example/output'

# ---------------------- Generate artificial data ------------------------------

training_df, _, name_dict = example_data()
# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'treat'   # Name of treatment variable
VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0',)
VAR_X_NAME_UNORD = ('x_unord0',)
# At least one feature list is required. Binary features belong in VAR_X_NAME_ORD;
# unordered features need at least three distinct training categories.
# -----------------------------------------------------------------------------
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(gen_outpath=APPLIC_PATH,
                             var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             var_x_name_unord=VAR_X_NAME_UNORD,
                             )
results_train = mymcf.train(training_df)

# train() returns the processed construction and filling samples separately.
# Use the filling sample, after preparation and common-support trimming.
results = mymcf.predict(results_train['fill_y_df'])

# analyse() returns a dictionary with cluster labels under the default settings.
results_with_clusters = mymcf.analyse(results)

my_report = McfOptPolReport(mcf=mymcf, outputpath=results['path_output'])
my_report.report()

print('End of computations.'
      '\n\nThanks for using ModifiedCausalForest. '
      '\n\nYours sincerely '
      '\nMCF \U0001F600'
      )
