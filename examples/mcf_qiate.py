"""Created on Wed Apr  1 15:58:30 2020.  -*- coding: utf-8 -*- .

Modified Causal Forest - Python implementation

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.8.0

This is an example to show how the qiates of mcf can be computed relying
on defaults for parameters not related to QIATE estimation. Note that usually in
applications it is very likely to be appropriate to deviate from some of the
default specifications.

"""
from pathlib import Path
import warnings

from mcf.example_data_functions import example_data
from mcf.mcf_main import ModifiedCausalForest
from mcf.reporting import McfOptPolReport


# ------------------ NOT parameters of the ModifiedCausalForest ---------------
#  Define data to be used in this example
APPLIC_PATH = Path.cwd() / 'example/output'

# ---------------------- Generate artificial data ------------------------------

training_df, prediction_df, name_dict = example_data(no_treatments=2,
                                                     obs_y_d_x_iate=2000,
                                                     obs_x_iate=2000,
                                                     no_effect=False)

# ------------------ Parameters of the ModifiedCausalForest -------------------

VAR_D_NAME = name_dict['d_name']          # Name of treatment variable
#   Note: Currently, treatment variables are restricted to be binary (IV only)

VAR_Y_NAME = name_dict['y_name']          # Name of outcome variable

VAR_X_NAME_ORD = name_dict['x_name_ord']

VAR_X_NAME_UNORD = name_dict['x_name_unord']

VAR_IV_NAME = name_dict['inst_bin_name']  # Name of instrumental variable

#   Estimation of QIATEs and their standard errors
P_QIATE = True         # True: QIATEs will be estimated. If True, p_iate will
#   always be set to True. Default is False.
P_QIATE_SE = True      # True: SE(QIATE) will be estimated. Default is False.
#   Estimating IATEs and their standard errors may be time consuming
P_QIATE_M_MQIATE = True      # True: QIATE(x) - median(IATE(x)) is estimated,
#   including inference if p_qiate_se == True. Increaes computation time.
#   Default is False.
P_QIATE_M_OPP = True      # True: QIATE(x, q) - QIATE(x, 1-q) is estimated,
#   including inference if p_qiate_se == True. Increaes computation time.
#   Default is False.
P_QIATE_NO_OF_QUANTILES = None   # Number of quantiles for which QIATEs are
#   computed. Default is 99.
P_QIATE_SMOOTH = None           # True: Smooth estimated QIATEs using kernel
#   smoothing. Default is True.
P_QIATE_SMOOTH_BANDWIDTH = None  # Multiplier applied to default bandwidth
#   used for kernel smoothing of QIATE. Default is 1
P_QIATE_BIAS_ADJUST = None       # Bias correction procedure for QIATEs.
#   (see Kutz and Lechner, 2025, for details). Default (or None) is True.
#   If P_QIATE_BIAS_ADJUST is True, P_IATE_SE is set to True as well.

# -----------------------------------------------------------------------------
if not APPLIC_PATH.exists():
    APPLIC_PATH.mkdir(parents=True)

# Modules may sent many irrelevant warnings: Globally ignore them
warnings.filterwarnings('ignore')
# -----------------------------------------------------------------------------
mymcf = ModifiedCausalForest(
    gen_outpath=APPLIC_PATH,
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
                             )

mymcf.train(training_df)  # Returns not used here
results = mymcf.predict(prediction_df)

results_with_cluster_id_df = mymcf.analyse(results)

my_report = McfOptPolReport(mcf=mymcf)

my_report.report()

print('End of computations.\n\nThanks for using ModifiedCausalForest (IV).'
      ' \n\nYours sincerely\nMCF \U0001F600')
