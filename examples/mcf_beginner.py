import pandas as pd

from mcf import ModifiedCausalForest

# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = 'c:/mlechner/py_neu/example'
DATPATH = APPLIC_PATH + '/data'
TRAINDATA = 'data_y_d_x_1000.csv'
#  Training data must contain outcome, treatment and features.

PREDDATA = 'data_x_1000.csv'
#  Training data must contain features. Treated effects on the treated
#  additionally require treatment information.

VAR_FLAG1 = False             # Related to specification used in this example.
#                               False: Few features. True: Many features.

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'D0_1_12'   # Name of treatment variable
VAR_Y_NAME = 'y'         # Name of outcome variable
VAR_X_NAME_ORD = ('cont0',)  # Using VAR_X_NAME_UNORD is fine as well

# -----------------------------------------------------------------------------
train_df = pd.read_csv(DATPATH + '/' + TRAINDATA)
pred_df = pd.read_csv(DATPATH + '/' + PREDDATA)
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD)

tree_df, fill_y_df = mymcf.train(train_df)  # Returns not used here

results = mymcf.predict(pred_df)

results_with_cluster_id_df = mymcf.analyse(results)

print('End of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600')
