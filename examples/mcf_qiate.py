"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.11.0

Estimate QIATEs, their standard errors and differences across quantiles.
QIATEs require discrete treatments, full-weight prediction and no bias adjustment
or treatment versions. Other estimation settings use API defaults.

"""
from pathlib import Path

from mcf.example_data import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example/output'

# ---------------------- Generate artificial data ------------------------------

training_df, prediction_df, name_dict = example_data(no_treatments=2,
                                                     obs_y_d_x_iate=2000,
                                                     obs_x_iate=2000,
                                                     no_effect=False,
                                                     )
# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = name_dict['d_name']          # Name of treatment variable

VAR_Y_NAME = name_dict['y_name']          # Name of outcome variable

VAR_X_NAME_ORD = name_dict['x_name_ord']

VAR_X_NAME_UNORD = name_dict['x_name_unord']

# QIATE estimation also enables IATE computation. Default False.
P_QIATE = True
# QIATE standard errors also enable IATE standard errors; this alone does not enable QIATEs.
P_QIATE_SE = True
# QIATE(q) - QIATE(0.5), with standard errors when P_QIATE_SE=True. Default False.
# The median reference is computed separately when absent from the quantile grid.
P_QIATE_M_MQIATE = True
# QIATE(q) - QIATE(1 - q), on both sides of the median. Default False.
P_QIATE_M_OPP = True
# Number Q of midpoint quantiles (j + 0.5) / Q, j=0,...,Q-1. Default 99; minimum 10.
P_QIATE_NO_OF_QUANTILES = None
# Smooth forest weights over fractional IATE ranks with an Epanechnikov kernel.
# Default True; False selects the observation nearest the requested rank.
P_QIATE_SMOOTH = None
# Multiplier of the rank-smoothing bandwidth. Default 1.
P_QIATE_SMOOTH_BANDWIDTH = None
# Adjust IATE ranks using normal quantiles and IATE standard errors. Default False.
# True also enables IATE standard errors; it does not change the reported IATE estimates.
P_QIATE_BIAS_ADJUST = None

# -----------------------------------------------------------------------------
APPLIC_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(gen_outpath=APPLIC_PATH,
                             var_d_name=VAR_D_NAME,
                             var_y_name=VAR_Y_NAME,
                             var_x_name_ord=VAR_X_NAME_ORD,
                             var_x_name_unord=VAR_X_NAME_UNORD,
                             p_qiate=P_QIATE,
                             p_qiate_se=P_QIATE_SE,
                             p_qiate_m_mqiate=P_QIATE_M_MQIATE,
                             p_qiate_m_opp=P_QIATE_M_OPP,
                             p_qiate_no_of_quantiles=P_QIATE_NO_OF_QUANTILES,
                             p_qiate_smooth=P_QIATE_SMOOTH,
                             p_qiate_smooth_bandwidth=P_QIATE_SMOOTH_BANDWIDTH,
                             p_qiate_bias_adjust=P_QIATE_BIAS_ADJUST,
                             cf_boot=None,
                             gen_mp_parallel=None,
                             _int_low_memory_predict=False,
                             )

mymcf.train(training_df)  # The returned training dictionary is not needed here.
results = mymcf.predict(prediction_df)

# analyse() describes IATEs and returns a dictionary with k-means labels by default.
results_with_clusters = mymcf.analyse(results)

my_report = McfOptPolReport(mcf=mymcf, outputpath=results['path_output'])

my_report.report()

print('End of computations.\n\nThanks for using ModifiedCausalForest (QIATE).'
      ' \n\nYours sincerely\nMCF \U0001F600'
      )
