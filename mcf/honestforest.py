"""Created on Wed Apr  1 15:58:30 2020.

Honest 2-sample Forest - Python implementation
This programme is useful when RF predictions with (weight-based)
inference are needed. Otherwise, use sklearn (which is much faster).

It also contains the two-sample version of the Ordered Forest of
Lechner/Okasa (2019) as a special option.

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

Version: 0.0.0 (based on MCF functions)

-*- coding: utf-8 -*- .

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules.

0.0.9.: Marginal plots added (one variable varies, others set to median value)
0.0.10: No of processes in MP made dependent on free-RAM
0.0.11: Alpha regularity as tuning parameter

"""
import sys
from multiprocessing import freeze_support
# local modules will be loaded below once path is defined -----------

MODULE_PATH = 'd:/mlechner/py/modules'    # path for local general mcf modules
PFAD = 'd:/mlechner/py/applications'  # Path for data, temporary files, output

# ------------ No change below: loading MCF & general modules --------------
sys.path.append(MODULE_PATH)
sys.path.append(PFAD)
import honestforest_functions as hf
# ----------------- No change above --------------


def main():
    """Do the main programme (needed for MP)."""
    outpfad = PFAD + '/output'
    temppfad = PFAD + '/temp'
    datpfad = PFAD + '/data'

    indata = 'unorderedx_sel1_het2clust_n1000_k12_p21'  # csv for estimation
    preddata = 'unorderedx_sel1_het2clust_n1000_k12_p21'  # csv for effects
    outfiletext = indata + "hf.py.0.0.12"

# Define variables in lists; if not relavant use empty list
    # Identifier
    id_name = ['ID']
    # If no identifier -> it will be added the data that is saved for later use
    # Dependent variable
    cluster_name = ['cluster']
    w_name = ['weight']
    y_name = ['y', 'x37']  # ['d']
    y_tree_name = ['y']  # ['d']

    # Features, predictors, independent, confounders variables: ordered
    x_name_ord = [
         'X1', 'x2', 'x3', 'x3', 'x4', 'x5', 'x7', 'X8', 'x9', 'x10', 'X11',
         'x12', 'x13',
         'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'X21', 'x22',
         'x24', 'x25', 'x26', 'x27',  # 'x28', 'x29', 'x30',
         'X11', 'x2'
                ]
    # Features, predictors, independent, confounders variables: unordered
    x_name_unord = ['X34', 'X35',  'x36' ]
    x_name_always_in_ord = []    # ['x4']
    x_name_always_in_unord = []

    x_name_remain_unord = []
    x_name_remain_ord = []

    z_name_mgate = ['x16', 'x24', 'x35']
    # Names of variables for which marginal predictive plots will be computed
    # Variable must be in included x_name_ord or x_name_unord; otherwise
    # variables they will be deleted from list

# Set parameters for estimation
    direct_output_to_file = -1  # if 1, output goes to file (-1: 1)
    save_forest = -1        # Save forest for prediction w/o reestimation -1:0
    #                  file names will be same as indat with extension *.pickle
    verbose = True

    mp_parallel = 3        # number of parallel processes  (>0)
    #                            -1:# of cores*0.9 (reduce if memory problems!)
    #                                 0, 1: no parallel computations
    mp_type_vim = -1        # type of mp for variable importance
    #                           1: variable based (fast, lots of memory)
    #                           2: bootstrap based (slower, less memory)
    #                          -1: 1 if n < 20000, 2 otherwise
    mp_with_ray = -1        # 1: Ray, 0: Concurrent future for Multiprocessing.
    #                         0: may be faster with small samples
    #                         1: (-1): Should be superior with larger samples
    mp_ray_objstore_multiplier = -1  # Increase internal default values for
    #                                  Ray object store above 1 if programme
    #                                  crashes because object store is full
    #                                  (-1: 1)
    boot = 20               # of bootstraps / subsamplings (-1: 1000)
    smaller_sample = 0      # 0<test_only<1: test prog.with smaller sample
    # Ordered forest
    orf = -1                # Ordered forest (ORF): -1: 0
    orf_marg = -1           # Compute marginal effects for ORF: -1: 1
    # data cleaning
    screen_covariates = -1  # 1 (-1): screen covariates (sc)
    check_perfectcorr = -1  # if sc=1: if 1 (-1), then variables that are
    #                         perfectly correlated with others will be deleted
    min_dummy_obs = -1      # if sc=1: dummy variable with obs in one
    #                 category smaller than this value will be deleted (-1: 10)
    clean_data_flag = -1    # if 1 (-1), remove all missing and unnecessary
    #                         variables from data set

    weighted = -1               # 1: use sampling weights,  -1: 0
    # if 1: sampling weights specified in w_name will be used; slows programme

    max_weight_share = -1  # maximum share of any weight, 0 <= 1, -1: 0.05
    # enforced by trimming excess weights and renormalisation for each ate,
    # gate and iate separately; because of renormalising, the final weights be
    # somewhat above this threshold
    subsample_factor = -1   # bootstrap or subsampling
    # 0-1: reduces the default subsample size by 1-subsample_factor
    # -1: 1 ((default=min(0.67,(2*(n^0.8)/n)))
    # n is computed as twice the sample size in the smallest treatment group

    n_min_min = -1  # smallest minimum leaf size
    # scalar or vector: minimum number of observations in leave
    #  (if grid is used, optimal value will be determined by oob)
    #  -2: n**0.4/20, at least 3; -1: n**0.4/10; at least 5
    #  relevant n is 2x the number of observations in smallest treatment group
    n_min_max = -1   # largest minimum leaf size
    #   -2: sqrt(n)/10, at least 3; -1: minc([sqrt(n)/5; at least 5
    #                                 relevant n is twice the number of
    #                                 observations in smallest treatment group
    n_min_grid = -1     # numer of grid values (-1: 2)

    stop_empty = -1     # x: stops splitting the tree if the next x
    # randomly chosen variable did not led to a new leaf
    # 0: new variables will be drawn & splitting continues n times
    # (faster if smaller, but nonzero); (-1:25)

    # grid for number of variables drawn at each new split of tree
    m_min_share = -1
    #       minimum share of variables used for next split (0-1)
    m_max_share = -1
    # maximum share of variables used for next split (0-1)
    # note that if m_max_share = 1, the algorithm corresponds to bagging trees
    # -1: m_min_share = 0.1*azahl_variables; m_max_share =anzahl_variables*0.66
    # -2: m_min_share = 0.2*anzahl_variables; m_max_share =anzahl_variables*0.8
    #     default values reduced to 70% of these values if feature learning
    m_grid = -1  # m_try
    # number of grid values logarithmically spaced including m_min m_max
    m_random_poisson = -1
    # if 1: number of randomly selected variables is stochastic for each split
    # (-1) grid gives mean value of 1+poisson distribution(m-1)
    share_forest_sample = -1
    # 0-1: share of sample used for predicting y given forests (-1: 0.5)
    #      other sample used for building forest

    # Alpha regularity
    # Results of Wager and Athey (2018) suggest alpha_reg < 0.2. A larger value
    # may increase speed of tree building. If grid is used, opt. value by OOB
    alpha_reg_min = -1      # smallest alpha, 0 < alpha < 0.4 (-1: 0.1)
    alpha_reg_max = -1      # , 0 < alpha < 0.5 (-1: 0.2)
    alpha_reg_grid = -1     # number of grid values (-1: 2)

    marg_no_evaluation_points = -1  # Number of evluation points for
#                              continuous variables (marginaleffects) (-1: 50)

    random_thresholds = -1  # 0: no random thresholds
    #               > 0: number of random thresholds used
    #                -1: 10 (using only few thresholds may speed up programme)
    variable_importance_oob = -1
    # 1: (-1:1) computes variable importance based on permuting every single x
    #            in oob prediction; time consuming
    descriptive_stats = -1    # print descriptive stats of input + output files

    # option for weight based inference (only inference useful for aggregates)
    cond_var_flag = -1    # 0: variance estimation uses var(wy)
    #                       1: conditional mean and variances are used (-1: 1)
    knn_flag = -1         # 0: nadaraya-watson estimation (-1)
    #                           1: knn estimation (faster)
    knn_min_k = -1        # k: minimum number of neighbours in
    #                          k-nn estimation(-1: 10)
    knn_const = -1        # constant in number of neighbour
    #                       asymptotic expansion formula of knn (-1: 1)
    nw_bandw = -1         # bandwidth for nw estimation; multiplier
    #                       of silverman's optimal bandwidth (default: 1)
    nw_kern_flag = -1     # kernel for nw estimation:
    #                       1: epanechikov (-1); 2: normal
    panel_data = -1       # 1: if panel data; default (-1) is no panel data
    #                          this activates the clustered standard error,
    #                          does perform only weight based inference
    # use cluster_name to define variable containing identifier for panel unit
    panel_in_rf = -1      # uses the panel structure also when building the
    #                       random samples within the forest procedure
    #                       (default; 1) only if panel == 1
    cluster_std = -1      # 1:clustered standard error; cluster variable in
    #                         variable file; (-1: 0) will be automatically
    #                         set to one if panel data option is activated

    # a priori feature pre-selection by random forest
    fs_yes = -1           # 1: feature selection active
    #                       0: not active (-1: 0)
    # If orf == True: Feature selection will be based on ordered variable
    fs_other_sample = -1  # 0: same sample as for rf estimation used
    #                       1: random sample taken from overall sample (-1: 1)
    fs_other_sample_share = -1  # share of sample to be used for feature
    #                             selection  (-1: 0.2);

    fs_rf_threshold = -1   # rf: threshold in % of loss of variable
    #                            importanance (-1: 0)

    # controls for all figures
    fontsize = -1            # legend, 1 (very small) to 7 (very large);  -1: 2
    dpi = -1                 # > 0: -1: 500
    ci_level = -1            # 0 < 1: confidence level for bounds: -1: 0.90
    no_filled_plot = -1      # use filled plot if more than xx points (-1: 20)
# ----------------------NO CHANGE BELOW ---------------------------------------
    max_cats_cont_vars = -1  # discretising of continuous variables: maximum
#                          number of categories for continuous variables n
#                          values < n speed up programme, -1: not used.
    _with_output = 1
    _max_save_values = 50  # save value of x only if less than 50 (cont. vars)
    _seed_sample_split = 67567885   # seeding is redone when building forest
    _no_ray_in_forest_building = True
    _train_mcf = True
    _pred_mcf = True
# ---------------------------------------------------------------------------
    hf.honestforest(
        PFAD, outpfad, temppfad, datpfad, indata, preddata, outfiletext,
        id_name, y_name, y_tree_name, x_name_ord, x_name_unord,
        x_name_always_in_ord, x_name_always_in_unord, x_name_remain_unord,
        x_name_remain_ord, z_name_mgate,
        cluster_name, w_name, mp_parallel, mp_type_vim,
        direct_output_to_file, screen_covariates,
        n_min_grid, check_perfectcorr, n_min_min, clean_data_flag,
        min_dummy_obs, boot, n_min_max, weighted, subsample_factor,
        m_min_share, m_grid, stop_empty, m_random_poisson, alpha_reg_min,
        alpha_reg_max, alpha_reg_grid,
        random_thresholds, knn_min_k, share_forest_sample, descriptive_stats,
        m_max_share, variable_importance_oob, knn_const,
        nw_kern_flag, cond_var_flag, knn_flag, nw_bandw, panel_data,
        max_cats_cont_vars, cluster_std, fs_yes, fs_other_sample_share,
        fs_other_sample, panel_in_rf, fs_rf_threshold, _with_output,
        _max_save_values, max_weight_share, smaller_sample, _seed_sample_split,
        save_forest, orf, orf_marg, fontsize, dpi, ci_level,
        marg_no_evaluation_points, no_filled_plot, mp_with_ray,
        mp_ray_objstore_multiplier, verbose, _no_ray_in_forest_building,
        _train_mcf, _pred_mcf)


if __name__ == '__main__':
    freeze_support()
    main()
