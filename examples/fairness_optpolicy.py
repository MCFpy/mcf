"""Created on Wed Apr  1 15:58:30 2020. -*- coding: utf-8 -*- .

Optimal Policy - Python implementation

Please note that this Optimal Policy (Decision) module is experimental.
It is much less tested than the MCF main module.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).
Appropriate credit must be given.

Michael Lechner & SEW Causal Machine Learning Team
Swiss Institute for Empirical Economics Research
University of St. Gallen, Switzerland

Version: 0.7.0

This is an example to show how to use the OptimalPolicy class of the mcf
module with full specification of all its keywords. It may be seen as an add on
to the published mcf documentation.

"""
import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from mcf.example_data_functions import example_data
from mcf.optpolicy_functions import OptimalPolicy
from mcf.reporting import McfOptPolReport

# ---- This script is experimental !
#      Be careful, it will run for a very long time!


# ------------- NOT passed to OptimalPolicy -----------------------------------
#  Define data to be used in this example
APPLIC_PATH = os.getcwd() + '/example'

# ---------------------- Generate artificial data ------------------------------

# Parameters to generate artificial data (DataFrame) for this example
TRAIN_OBS = 2000        # Number of observations of training data.
#      'best_policy_score': Training data must contain policy scores.
#      'policy tree', 'bps_classifier': Training data must contain policy scores
#                         and features. Default is 1000.

PRED_OBS = 2000         # Number of observations of prediction data.
#                         Prediction data is used to allocate.
#      'best_policy_score': Prediction data must contain policy scores.
#      'policy tree', 'bps_classifier': Prediction data must contain the
#                         features that are used in training. Policy scores are
#                         not required. Default is 1000.

# If True, scores are adjusted for fairness
# considerations a là Bearth, Lechner, Mareckova, and Muny (2024). Requires
# the specification of some additional keywords below
# (var_protected_xxx, fair_xxx).

PREDDATA_IS_TRAINDATA = False  # If True, the same data will be used
#                                for training and prediction (in this case,
#                                PRED_OBS is ignored).

NO_FEATURES = 20         # Number of features. Will generate different types of
#                          features (continuous, dummies, ordered, unordered)

NO_TREATMENTS = 3        # Number of treatments

training_df, prediction_df, name_dict = example_data(
    obs_y_d_x_iate=TRAIN_OBS,
    obs_x_iate=PRED_OBS,
    no_features=NO_FEATURES,
    no_treatments=NO_TREATMENTS,
    seed=12345,
    type_of_heterogeneity='WagerAthey',
    descr_stats=True)

if PREDDATA_IS_TRAINDATA:
    prediction_df = training_df

# ------------- Methods used in Optimal Policy Module --------------------------
# METHODS = ('best_policy_score', 'bps_classifier', 'policy tree',)
# Using 'best_policy_score' requires scores also for the prediction data!
# Therefore, training data is used for the evaluation of 'best_policy_score'.
GEN_METHOD_ALL = ('best_policy_score', 'policy tree', 'bps_classifier',)

# -------- All what follows are parameters of the OptimalPolicy --------
#   Whenever None is specified, parameter will be set to default values.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
GEN_OUTPATH = os.getcwd() + '/example/outputOPT'  # Directory for output.
#   If it does not exist, it will be created. Default is an *.out directory
#   just below to the directory where the programme is run.

WEIGHTPLOT_PATH = os.getcwd() + '/out'
#   Path to put the plots for the different fairness weighting

GEN_OUTFILETEXT = "OptPolicy_0_7_0_Fairness"  # File for text output
#   Default is 'txtFileWithOutput'.
#   *.txt file extension will be added by the programme

# ---------------- Names of variables used ------------------------------------
VAR_ID_NAME = 'id'     # Name of identifier in data. Default is None.

VAR_D_NAME = 'treat'   # Name of discrete treatment.
#   Needed in training data only if 'changers' (different treatment than
#   observed treatment) are analysed and if allocation is compared to observed
#   allocation.

VAR_POLSCORE_NAME = ('y_pot0', 'y_pot1', 'y_pot2',)

#   Treatment specific variables to measure the value of individual treatments,
#   starting with the first, second, third, ..., treatment (e.g. corresponding
#   to treatment values 0, 1, 2, ...).
#   This is usually the estimated potential outcome or any other score related
#   quantity. If Individualized Average Treatment Effects (e.g., of the effect
#   1-0, 2-0, 3-0, etc. are provided instead, make sure that the reference
#   treatment (in this case treatment 0) is included as a vector of zeros
#   (in this example it corresponds to the IATE of 0-0, which is of course 0
#   for all observations).

VAR_POLSCORE_DESC_NAME = (('zero', 'ite1vs0', 'ite2vs0',),
                          (('x_cont0', 'iate1vs0', 'iate2vs0',)))
#   Tuple of tuples. Each tuple of dimension equal to the number of treatments
#   contains treatment specific variables that are used to evaluate the effect
#   of the allocation with respect to those variables. This could be for
#   example policy score not used in training, but which are relevant
#   nevertheless. Default is no variables.

VAR_X_NAME_ORD = ('x_cont0', 'x_cont1', 'x_cont2', 'x_ord0', 'x_ord1', 'x_ord2')
#   Ordered variables used to build policy tree (including dummy variables) and
#   classifier. They are also used to characterise the allocation.

VAR_X_NAME_UNORD = ('x_unord0', 'x_unord1', 'x_unord2',)
#   Unordered variables used to build policy tree and classifier.They are also
#    used to characterise the allocation.

VAR_VI_X_NAME = ('x_cont0', 'x_cont1', 'x_cont2',)
#   Names of variables for which variable importance is computed.
#   Default is None.
VAR_VI_TO_DUMMY_NAME = ('x_unord0',)
#   Names of categorical variables for which variable importance is computed.
#   These variables will be broken up into dummies. Default is None.

# ---------------------- Method specific parameters ---------------------------

# - - - - - - - - - - - - - - - Fairness - - - - - - - - - - - - - - - - - - - -
#   The 'fairscores' method adjusts the policy scores by removing the influence
#   of some features that are specificied in the VAR_PROTECTED_NAME variable.
#   If the 'fairscores' method is not explicitly called, the following variables
#   are not relevant and need not to be specifified.

NO_GRID_VALUES_FAIRNESS_PLOTS = 10
#  Combines the fairness adjusted and the original scores to create a plot
#  which shows welfare loss of when imposing fairness.
#  NO_GRID_VALUES_FAIRNESS_PLOTS is the number of grid points (between 0 and 1)
#  for the respective weighted means.

#   Protected variables (the effect of the policy scores on those variables will
#                        be removed, conditional on the 'materially important'
#                        variables).
VAR_PROTECTED_NAME_ORD = ('x_cont0', 'x_ord4',)
VAR_PROTECTED_NAME_UNORD = ('x_unord0', 'x_unord3',)
#   These variables should NOT be contained in decision variables, i.e.,
#   VAR_X_NAME_ORD and/or VAR_X_NAME_UNORD. If they are included, they will be
#   removed and VAR_X_NAME_ORD and/or VAR_X_NAME_UNORD will be adjusted
#   accordingly. Defaults are None (at least one of them must be specified, if
#   the fairness adjustments are to be used).

#  Materially relavant variables: An effect of the protected variables on the
#  scores is allowed, if captured by these variables (only).
VAR_MATERIAL_NAME_ORD = ('x_cont1', 'x_ord2',)
VAR_MATERIAL_NAME_UNORD = ('x_unord1', 'x_unord2',)

VAR_VI_X_NAME = list(VAR_VI_X_NAME).extend((*VAR_PROTECTED_NAME_ORD,
                                            *VAR_MATERIAL_NAME_ORD))
VAR_VI_TO_DUMMY_NAME = list(VAR_VI_TO_DUMMY_NAME).extend(
    (*VAR_PROTECTED_NAME_UNORD, *VAR_MATERIAL_NAME_UNORD))
#  These variables may (or may not) be included among the decision variables.
#  These variables must (!) not be included among the protected variables.

#  Fairness adjustment methods for policy scores
#     (see Bearth, Lechner, Muny, Mareckova, 2024, for details)
FAIR_TYPE_ALL = ('Quantiled', 'Mean', 'MeanVar', )
#  Method to choose the type of correction for the policy scores.
#  'Mean':  Mean dependence of the policy score on protected var's is removed.
#  'MeanVar':  Mean dependence and heteroscedasticity is removed.
#  'Quantiled': Removing dependence via (an empricial version of) the approach
#               by Strack and Yang (2024).
#  Default (or None) is 'Quantiled'.

#  Method choice when predictions from machine learning are needed
FAIR_REGRESSION_METHOD = None
#  Regression method to adjust scores w.r.t. protected features.
#  Possible choices are scikit-learn's regression methods
#  'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5'
#  'SupportVectorMachine', 'SupportVectorMachineC2', 'SupportVectorMachineC4',
#  'AdaBoost', 'AdaBoost100', 'AdaBoost200',
#  'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12',
#  'LASSO', 'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
#  'Mean'. (Mean is included if other estimators are not useful.)
#  If 'automatic', an optimal methods will be chosen based on 5-fold
#  cross-validation in the training data. If 'automatic', a method is specified
#  it will be used for all scores and all adjustments. If None, every policy
#  score might be adjusted with a different method. 'Mean' is included for cases
#  in which regression methods have no explanatory power.
#  This method is relevant when "fair_type in ('Mean', 'MeanVar')"
#  Default (or None) is 'RandomForest'.

#  Parameters for discretization of features (necessary for 'Quantilized')

FAIR_PROTECTED_DISC_METHOD_ALL = ('Kmeans', 'NoDiscretization',)
#  Whether and how to perform the discretization for
#  protected features (for FAIR_TYPE = 'Quantiled'). Maximum number of groups
#  allowed is set by FAIR_PROTECTED_MAX_GROUPS.
#  'NoDiscretization': Variables are not changed. If one of the features has
#               more different values than FAIR_PROTECTED_MAX_GROUPS, all
#               protected features will formally be treated as continuous. In
#               this case quantile positions will be based on nonparametrically
#               estimated densities. The latter may become unreliable if their
#               dimension is not very small.
#  'EqualCell': Attempts to create equal cells for each feature. Maybe be
#               useful for a very small number of variables with few different
#               values.
#  'Kmeans':    Use Kmeans clustering algorithm to form homogeneous cells.
#  Default (or None) is 'Kmeans'.

FAIR_MATERIAL_DISC_METHOD_ALL = ('Kmeans', 'NoDiscretization',)
#  Whether and how to perform the discretization for materially relevant
#  features (for FAIR_TYPE = 'Quantiled'). Maximum number of groups
#  allowed is set by FAIR_MATERIAL_MAX_GROUPS.
#  'NoDiscretization': Variables are not changed. If one of the features has
#               more different values than FAIR_MATERIAL_MAX_GROUPS, all
#               materially relevant features will formally be treated as
#               continuous. In  this case quantile positions will be based on
#               nonparametrically estimated densities. The latter may become
#                unreliable if their dimension is not very small.
#  'EqualCell': Attempts to create equal cells for each feature. Maybe be
#               useful for a very small number of variables with few different
#               values.
#  'Kmeans:'    Use Kmeans clustering algorithm to form homogeneous cells.
#  Default (or None) is 'Kmeans'.

#  Level of discretization of variables (only if needed)
FAIR_PROTECTED_MAX_GROUPS = None
FAIR_MATERIAL_MAX_GROUPS = None
#  Number of different groups of values of features that are
#  materially relavant /protected (if discretization is used).
#  This is currently only necessary for 'Quantilized'.
#  Its meaning depends on fair_protected_disc_method, fair_material_disc_method:
#  If 'EqualCell': If more than 1 variable is included among the protected
#      features, this restriction is applied to each variable.
#  If 'Kmeans': This is the number of clusters used by Kmeans.
#  Default (or None) is 5.

#  Test for internally consistency.
FAIR_CONSISTENCY_TEST = True
#  The fairness corrections are applied independently to every policy score (
#  which usually is a potential outcome or an IATE(x) for each treatment
#  relative to some base treatment (i.e. comparing 1-0, 2-0, 3-0, etc.).
#  Thus the IATE for the 2-1 comparison can be computed as IATE(2-0)-IATE(1-0).
#  This tests compares two ways to compute a fair score for the 2-1 (and all
#  other comparisons) which should give simular results:
#  a) Difference of two fair (!) scores
#  b) Difference of corresponding scores, subsequently made fair.
#  Note: Depending on the number of treatments, this test may be computationally
#  more expensive than the orginal fairness corrections.
#  Therefore, the default is False.

PT_DEPTH_TREE_1 = 2   # Depth of 1st optimal tree. Default is 3.
#   In this example, this parameter is set to 2 to speed up computation.

PT_DEPTH_TREE_2 = 2   # Depth of 2nd optimal tree. This set is built within
#   the strata obtained from the leaves of the first tree. If set to 0, a second
#   tree is not built. Default is 1 (together with the default for
#   pt_depth_tree_1 this leads to a (not optimal) total tree of level of 4.
#   In this example, this parameter is set to 2.

#   Note that tree depth is defined such that a depth of 1 implies 2 leaves,
#   a depth of 2 implies 4 leaves, a depth of = 3 implies 8 leaves, etc.

# -------------- Other parameters (costs and restrictions) --------------------
OTHER_MAX_SHARES = (1, 1, 1)  # Maximum share allowed for each treatment.
#   This is a tuple with as many elements as treatments.
#   0 <  OTHER_MAX_SHARES <= 1.  (1,...,1) implies unconstrained optimization
#   (default).

# The following is an example on how the fairness adjustment of the
# OptimalPolicy object can be used.
# For convenience the parameters are collected and passed as a dictionary.
# Of course, they can also be passed as single parameters (or not at all, in
# which case default values are used).
params_org = {
    'fair_consistency_test': FAIR_CONSISTENCY_TEST,
    'fair_regression_method': FAIR_REGRESSION_METHOD,
    'fair_protected_max_groups': FAIR_PROTECTED_MAX_GROUPS,
    'fair_material_max_groups': FAIR_MATERIAL_MAX_GROUPS,
    'gen_outfiletext': GEN_OUTFILETEXT,
    'gen_outpath': GEN_OUTPATH,
    'other_max_shares': OTHER_MAX_SHARES,
    'pt_depth_tree_1': PT_DEPTH_TREE_1,
    'pt_depth_tree_2': PT_DEPTH_TREE_2,
    'var_d_name': VAR_D_NAME,
    'var_polscore_desc_name': VAR_POLSCORE_DESC_NAME,
    'var_polscore_name': VAR_POLSCORE_NAME,
    'var_protected_name_ord': VAR_PROTECTED_NAME_ORD,
    'var_protected_name_unord': VAR_PROTECTED_NAME_UNORD,
    'var_material_name_ord': VAR_MATERIAL_NAME_ORD,
    'var_material_name_unord': VAR_MATERIAL_NAME_UNORD,
    'var_vi_x_name': VAR_VI_X_NAME,
    'var_vi_to_dummy_name': VAR_VI_TO_DUMMY_NAME,
    'var_x_name_ord': VAR_X_NAME_ORD,
    'var_x_name_unord': VAR_X_NAME_UNORD,
    }
if not os.path.exists(APPLIC_PATH):
    os.makedirs(APPLIC_PATH)

first_time = True
for fair_type in FAIR_TYPE_ALL:
    params_org['fair_type'] = fair_type
    for fair_material_disc_method in FAIR_MATERIAL_DISC_METHOD_ALL:
        params_org['fair_material_disc_method'] = fair_material_disc_method
        for fair_protected_disc_method in FAIR_PROTECTED_DISC_METHOD_ALL:
            params_org['fair_protected_disc_method'
                       ] = fair_protected_disc_method
            fig, ax = plt.subplots()
            if fair_type in ('Mean', 'MeanVar',):
                liste = ('Fair', fair_type)
            else:
                liste = ('Fair', fair_type,
                         'Mat', fair_material_disc_method,
                         'Prot', fair_protected_disc_method)
            titel = '_'.join(liste)
            titel_plot = ' '.join(liste)

            # --- Fairness correction ------
            params = params_org.copy()
            myoptp_fair = OptimalPolicy(**params)
            training_fair_df, _, _, _ = myoptp_fair.fairscores(
                training_df.copy(), data_title='training')

            # Preparations for plots
            if first_time:
                grid = np.linspace(0, 1, NO_GRID_VALUES_FAIRNESS_PLOTS+1)
                polscore_org_name = [
                    name[:-5] for name in myoptp_fair.var_dict['polscore_name']]
                polscore_fair_name = myoptp_fair.var_dict['polscore_name'].copy(
                    )
                welfare_pot_pred_np = prediction_df[polscore_org_name
                                                    ].to_numpy()
                welfare_pot_train_np = training_df[polscore_org_name].to_numpy()
                first_time = False

            # Different methods for allocation rules
            for idx, gen_method in enumerate(GEN_METHOD_ALL):
                print('Optp_method:', gen_method, titel_plot)
                myoptp = deepcopy(myoptp_fair)
                myoptp.gen_dict['method'] = gen_method
                # ----- Training data ----------
                alloc_train_df, _, _ = myoptp.solve(training_fair_df.copy(),
                                                    data_title='training fair')
                results_eva_train, _ = myoptp.evaluate(
                    alloc_train_df, training_fair_df.copy(),
                    data_title='training fair')
                if gen_method != 'best_policy_score':
                    alloc_df, _ = myoptp.allocate(prediction_df.copy(),
                                                  data_title='prediction')
                    # Evaluate using prediction data
                    results_eva_pred, _ = myoptp.evaluate(
                        alloc_df, prediction_df.copy(), data_title='prediction')
                myoptp.print_time_strings_all_steps()
                my_report = McfOptPolReport(optpol=myoptp,
                                            outputfile='Report_OptP_')
                my_report.report()
                del my_report, myoptp

                # Create combinations of fair and orginal scores -> welfare

                print('Estimating allocations of combinations of fair and '
                      f'original scores ({titel_plot} {gen_method})')
                print('Weight of fair score: ', end=' ')
                welfare = []
                for orginal_weight in grid:
                    print(round(orginal_weight, 2), end=' ', flush=True)
                    polscore_temp_name = [name + '_' + str(orginal_weight)
                                          for name in polscore_org_name]
                    params = params_org.copy()
                    params['var_polscore_name'] = polscore_temp_name
                    params['_int_with_output'] = False
                    params['_int_output_no_new_dir'] = True
                    params['gen_method'] = gen_method
                    myoptp = OptimalPolicy(**params)

                    data_df = training_fair_df.copy()
                    for idx, name in enumerate(polscore_org_name):
                        fair_name = polscore_fair_name[idx]
                        temp_name = polscore_temp_name[idx]
                        data_df[temp_name] = (
                            (1 - orginal_weight) * training_fair_df[name] +
                            orginal_weight * training_fair_df[fair_name]
                            )

                    if gen_method == 'best_policy_score':
                        alloc_df, _, _ = myoptp.solve(data_df)
                    else:
                        myoptp.solve(data_df)
                        alloc_df, _ = myoptp.allocate(prediction_df.copy())
                    new_alloc_np = alloc_df.to_numpy()

                    if new_alloc_np.shape[1] > 1:
                        # Many ways to enforce restrictions, choose one
                        new_alloc_np = new_alloc_np[:, 2]
                    new_alloc_np = new_alloc_np.reshape(-1)
                    row_indices = np.arange(new_alloc_np.shape[0])
                    if gen_method == 'best_policy_score':
                        welfare_np = welfare_pot_train_np[row_indices,
                                                          new_alloc_np]
                    else:
                        welfare_np = welfare_pot_pred_np[row_indices,
                                                         new_alloc_np]
                    welfare.append(np.sum(welfare_np))
                    del myoptp
                welfare_norm = [welf / welfare[0] * 100 for welf in welfare]
                if gen_method == 'best_policy_score':
                    ax.plot(grid, welfare_norm, label=gen_method+'TrainData')
                else:
                    ax.plot(grid, welfare_norm, label=gen_method)

            ax.set_title(titel_plot)
            ax.set_xlabel('Weight of fairness score')
            ax.set_ylabel('Normalized welfare')
            ax.legend()
            plt.savefig(WEIGHTPLOT_PATH + '/' + titel + '.jpeg', format='jpeg')
            plt.savefig(WEIGHTPLOT_PATH + '/' + titel + '.pdf', format='pdf')
            plt.show()
            del myoptp_fair
            if fair_type in ('Mean', 'MeanVar', ):  # Use innerloop only once
                break

        if fair_type in ('Mean', 'MeanVar', ):  # Use innerloop only once
            break

print('End of example estimation.\n\nThanks for using OptimalPolicy. \n\nYours'
      ' sincerely\nExperimental OptimalPolicy MCF modul \U0001F600')
