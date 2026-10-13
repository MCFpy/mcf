"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

Compare GATEs, BGATEs and CBGATEs along x_cont1. The first BGATE balances x_cont2;
the second also balances x_cont0, reusing the same trained forest.
Other estimation settings use their defaults; adapt them to the application.

Related paper: Nora Bearth & Michael Lechner (2025): Causal Machine Learning
               for Moderation Effects, Journal of Business & Economic Statistics
               (JBES).

"""
from pathlib import Path

from mcf.example_data import example_data
from mcf.mcf_general_sys import save_load
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
# Base directory for estimation output and the saved trained forest.
# Each PDF is saved in its prediction run's output directory.
APPLIC_PATH = Path.cwd() / 'example/output'

# ------------------ Parameters of the ModifiedCausalForest -------------------
VAR_D_NAME = 'treat'            # Treatment variable
VAR_Y_NAME = 'outcome'          # Outcome variable
VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2')

# Continuous heterogeneity variable: discretized effects plus smoothed estimates by default.
VAR_Z_NAME = 'x_cont1'

# BGATEs keep the distribution of these features constant across values of VAR_Z_NAME.
VAR_X_NAME_BALANCE_BGATE_1 = 'x_cont2'
VAR_X_NAME_BALANCE_BGATE_2 = ('x_cont2', 'x_cont0')
# CBGATEs balance all other forest features, here x_cont0 and x_cont2, in both runs.

APPLIC_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------- Generate artificial data ------------------------------
training_df, prediction_df, name_dict = example_data()

# ---------------------- Train and save the forest -----------------------------
mymcf = ModifiedCausalForest(gen_outpath=APPLIC_PATH,
                             var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             var_z_name_cont=VAR_Z_NAME,
                             p_cbgate=True, p_bgate=True,
                             var_x_name_balance_bgate=VAR_X_NAME_BALANCE_BGATE_1,
                             )
mymcf.train(training_df)  # The returned training dictionary is not needed here.

# new_keywords is accepted only before the first predict() call on a trained instance.
# Save that state now and reload it for the alternative specification below.
# The mcf_all_parameters example uses deepcopy() instead of saving to disk.
save_load(APPLIC_PATH / 'mymcf.pickle', object_to_save=mymcf, save=True, output=True)

# ---------------------- First balancing specification -------------------------
results = mymcf.predict(prediction_df)
# With default k-means enabled, analyse() returns a results dictionary with cluster labels
# in 'iate_data_df'. With k-means disabled, it returns None while other analyses still run.
results_with_clusters = mymcf.analyse(results)
my_report = McfOptPolReport(mcf=mymcf, outputpath=results['path_output'])
my_report.report()

# ---------------------- Second balancing specification ------------------------
# Reload the trained state before its first prediction; no retraining is needed.
# Supply updated names as a list/tuple, even for one variable. They must be among the
# retained training features; screening or feature selection may remove unprotected features.
# This update changes BGATE balancing only; CBGATEs still balance all other features.
mymcf2 = save_load(APPLIC_PATH / 'mymcf.pickle', save=False, output=True)
results2 = mymcf2.predict(prediction_df,
                          new_keywords={'var_x_name_balance_bgate': VAR_X_NAME_BALANCE_BGATE_2}
                          )
results2_with_clusters = mymcf2.analyse(results2)
my_report2 = McfOptPolReport(mcf=mymcf2, outputpath=results2['path_output'])
my_report2.report()


print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
