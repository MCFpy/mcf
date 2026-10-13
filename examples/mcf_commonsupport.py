"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

Restrict one prediction sample to common support under several training samples.
Each iteration retains only rows accepted by all support checks so far.
The early-exit options skip local centering, forest training and effect estimation.

"""
from pathlib import Path

from mcf.example_data import example_data
from mcf.mcf_main import ModifiedCausalForest


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example/output'

NO_OF_TRAINING_DATA = 4   # Number of different training data sets considered
OBS_TRAIN = 800          # Observations used for training
OBS_PREDICT = 2000       # Initial observations to check for common support

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = 'treat'   # Name of treatment variable
VAR_Y_NAME = 'outcome'         # Name of outcome variable
VAR_X_NAME_ORD = ('x_cont0',)  # Using VAR_X_NAME_UNORD instead is fine as well
# -----------------------------------------------------------------------------
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# ---------------- Generate artificial prediction data -------------------------
_, prediction_df, name_dict = example_data(obs_x_iate=OBS_PREDICT)
print(f'Observations initially in prediction data: {len(prediction_df)}')

for i in range(NO_OF_TRAINING_DATA):
    # ---------------- Generate artificial training data -----------------------
    new_seed = 42 * (i + 1)
    training_df, _, _ = example_data(obs_y_d_x_iate=OBS_TRAIN,
                                     seed=new_seed,
                                     no_printing=True,
                                     )
    # Default cs_type=1 uses treatment-probability ranges estimated from training data.
    # train() stores the support rule even though it exits before forest training.
    mymcf = ModifiedCausalForest(gen_outpath=APPLIC_PATH,
                                 var_d_name=VAR_D_NAME,
                                 var_y_name=VAR_Y_NAME,
                                 var_x_name_ord=VAR_X_NAME_ORD,
                                 _int_with_output=True,
                                 )
    mymcf.train(training_df, exit_after_commonsupport=True)
    results = mymcf.predict(prediction_df, exit_after_commonsupport=True)

    # 'inputdata_on_support' contains the processed rows retained by this check.
    prediction_df = results['inputdata_on_support'].copy()

    print(f'Training data set {i+1}: Observations left in prediction data after cs-check: '
          f'{len(prediction_df)}')
    del mymcf
    if prediction_df.empty:
        print('No prediction observations remain on the joint common support.')
        break

print('\nEnd of computations.\n\nThanks for using ModifiedCausalForest.'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
