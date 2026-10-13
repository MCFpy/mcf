"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

Minimal MCF example: specify treatment, outcome and at least one feature.
Other estimation settings use their defaults; adapt them to the application.

"""
from pathlib import Path

from mcf.example_data import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example/output'

# ---------------------- Generate artificial data ------------------------------

training_df, prediction_df, name_dict = example_data()

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'treat'   # Name of treatment variable
VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0',)  # Ordered features include binary variables.
# Alternatively, use var_x_name_unord for categories with at least three values.
# -----------------------------------------------------------------------------
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(gen_outpath=APPLIC_PATH,
                             var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD
                             )
mymcf.train(training_df)
results = mymcf.predict(prediction_df)
# The returned dictionary adds k-means labels to results['iate_data_df'].
results_with_clusters = mymcf.analyse(results)
my_report = McfOptPolReport(mcf=mymcf, outputpath=results['path_output'])
my_report.report()
print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
