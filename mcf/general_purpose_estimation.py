"""General purpose procedures.

# -*- coding: utf-8 -*-.
Created on Thu Apr  2 17:55:24 2020

@author: MLechner
"""
import copy
import math
import scipy.stats as sct
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def RandomForest_scikit(
        x_train, y_train, x_pred, x_name=None, y_name='y', boot=1000,
        n_min=2, no_features='sqrt', workers=-1,  max_depth=None, alpha=0,
        max_leaf_nodes=None, pred_p_flag=True, pred_t_flag=False,
        pred_oob_flag=False, with_output=True, variable_importance=False,
        var_im_groups=None, pred_uncertainty=False, pu_ci_level=0.9,
        pu_skew_sym=0.5, var_im_with_output=True,
        variable_importance_oob_flag=False, return_forest_object=False):
    """
    Compute Random Forest predictions with OOB-optimal parameters & var import.

    Based on sklearn, but with many additional features.
    Prediction intervals are based on:
        Zhang, Zimmerman, Nettleton, and Nordman (2020): Random Forest
        Prediction Intervals, THE AMERICAN STATISTICIAN, 2020, VOL. 74, NO. 4,
        392–406: Data Science, https://doi.org/10.1080/00031305.2019.1585288
        (Type III intervals given; symmetric intervals used if skewness of
         distribution is below user specified cut-off-value of skewness)
    Variable importance measures are based on:
        Williamson, Gilbert, Carone, Simon (2020): Nonparametric variable
        importance assessment using machine learning techniques, Biometrics.
        (no crossfitting used, no uncertainty measure given)

    Parameters
    ----------
    x_train : Numy array. Training features.
    y_train : Numpy array. Training outcome.
    x_pred : Numpy array. Features for which to predict outcomes.
    y_name: String. Name of dependende variable. Default is 'y'.
    x_name : List of Str. Name of features used in RF.
    boot : Int, optional. No of trees. The default is 1000.
    n_min : List of Int, optional. Minimum leaf size. The default is [2].
    no_features : List of Int, optional. M_try. The default is ['sqrt'].
    workers : No of parallel processes, optional. The default is -1.
    pred_p_flag : Boolean. Predit with x_pred. Default is True.
    pred_t_flag : Boolean. Predit with x_train. Default is False.
    pred_oob_flag : Boolean. Use OOB for training prediction. Default is False.
    with_output : Boolean. Print output. Default is True.
    max_depth : Int. Depth of tree. Default is None.
    alpha : Float. Minimum share on each side of split. Default is 0.
    max_leaf_nodes : Int. Maximimum number of leafs. Default is None.
    variable_importance : Boolean. OOB based variable importance (mutated x).
                          Default is False.
    var_im_groups : Lists of lists of strings or None.
    pred_uncertainty : Bool. Compute standard error of prediction.
                          Default is False.
    pu_ci_level : Float. Confidence level for prediction intervals.
                         Default = 0.9
    pu_skew_sym : Float. Cut-off of skewness for symetric CIs. Default is 0.5.
    variable_importance_oobx : Bool. Use OOB obs for VI (computational costly).
                                    Default is False.
    return_forest_object: Boolean. Forest object.

    Returns
    -------
    pred_p : 1d Numpy array. Predictions prediction data.
    pred_t: 1d Numpy array. Predictions training data.
    oob_best: Float. OOB based R2
    pu_ci_p: 2d Numpy array. Confidence interval prediction data
    pu_ci_t: 2d Numpy array. Confidence interval training data
    forest: Forest object.

    """
    if with_output:
        var_im_with_output = True
        print('\n')
        print('Computing forests')
    oob_best = -1e15
    pred_p = []
    pred_t = []
    y_1d = y_train.ravel()
    no_of_vars = np.size(x_train, axis=1)
    if no_features is None:
        no_features = no_of_vars
    if isinstance(no_features, str):
        if no_features == 'sqrt':
            no_features = round(math.sqrt(no_of_vars))
        elif no_features == 'auto':
            no_features = no_of_vars
        elif no_features == 'log2':
            no_features = round(math.log2(no_of_vars))
        else:
            no_features = round(no_of_vars/3)
    elif isinstance(no_features, float):
        no_features = round(no_features * no_of_vars)
    if not isinstance(n_min, (list, tuple, np.ndarray)):
        n_min = [n_min]
    if not isinstance(no_features, (list, tuple, np.ndarray)):
        no_features = [no_features]
    if not isinstance(alpha, (list, tuple, np.ndarray)):
        alpha = [alpha]
    for nval in n_min:
        for mval in no_features:
            # add optimisation über alpha --> see honestforests
            for aval in alpha:
                regr = RandomForestRegressor(
                    n_estimators=boot, min_samples_leaf=nval,
                    max_features=mval, bootstrap=True, oob_score=True,
                    random_state=42, n_jobs=workers, max_depth=max_depth,
                    min_weight_fraction_leaf=aval,
                    max_leaf_nodes=max_leaf_nodes)
                regr.fit(x_train, y_1d)
                if with_output and not ((len(no_features) == 1)
                                        and (len(n_min) == 1)
                                        and (len(alpha) == 1)):
                    print(' N_min: {:2}'.format(nval), ' M_Try: {:2}'.format(
                        mval), ' Alpha: {:5.3f}'.format(aval),
                        '|  OOB R2 (in %): {:8.4f}'.format(
                        regr.oob_score_*100))
                if regr.oob_score_ > oob_best:
                    m_opt = copy.copy(mval)
                    n_opt = copy.copy(nval)
                    a_opt = copy.copy(aval)
                    oob_best = copy.copy(regr.oob_score_)
                    regr_best = copy.deepcopy(regr)
    if with_output:
        print('\n')
        print('-' * 80)
        print('Tuning values choosen for forest (if any)')
        print('-' * 80)
        print('Dependent variable: ', y_name[0], 'N_min: {:2}'.format(n_opt),
              'M_Try: {:2}'.format(m_opt), 'Alpha: {:5.3f}'.format(a_opt),
              '|  OOB R2 (in %): {:8.4f}'.format(oob_best * 100))
        print('-' * 80)
    if pred_p_flag:
        pred_p = regr_best.predict(x_pred)
    if variable_importance or pred_uncertainty:
        pred_oob = regr_best.oob_prediction_
    if pred_t_flag:
        if pred_oob_flag:
            if variable_importance or pred_uncertainty:
                pred_t = np.copy(pred_oob)
            else:
                pred_t = regr_best.oob_prediction_
        else:
            pred_t = regr_best.predict(x_train)
    if with_output:
        print(regr_best.get_params())
        print('-' * 80)
        print(y_name, ': ',
              'R2(%) OOB: {:8.4f}'.format(regr_best.oob_score_*100))
        print('-' * 80)
    if variable_importance:
        if variable_importance_oob_flag:
            vi_names_shares = variable_importance_oob(
                x_name, var_im_groups, no_of_vars, y_1d, pred_oob, with_output,
                var_im_with_output, x_train, boot, n_opt, m_opt, workers,
                max_depth, a_opt, max_leaf_nodes)
        else:
            vi_names_shares = variable_importance_testdata(
                x_name, var_im_groups, y_1d, with_output,
                var_im_with_output, x_train, boot, n_opt, m_opt, workers,
                max_depth, a_opt, max_leaf_nodes)
    else:
        vi_names_shares = None
    if pred_uncertainty and (pred_t_flag or pred_p_flag):
        resid_oob = y_1d - pred_oob
        skewness_res_oob = sct.skew(resid_oob)
        if with_output:
            print('Skewness of oob residual:', skewness_res_oob)
        symmetric = np.absolute(skewness_res_oob) < pu_skew_sym
        if symmetric:
            if with_output:
                print('Symmetric intervals used')
            quant = np.quantile(np.absolute(resid_oob), 1-pu_ci_level)
            if pred_t_flag:
                upper_t = pred_t + quant
                lower_t = pred_t - quant
            if pred_p_flag:
                upper_p = pred_p + quant
                lower_p = pred_p - quant
        else:
            if with_output:
                print('Asymmetric intervals used')
            quant = np.quantile(resid_oob, [(1-pu_ci_level)/2,
                                            1-(1-pu_ci_level)/2])
            if pred_t_flag:
                upper_t = pred_t + quant[1]
                lower_t = pred_t + quant[0]
            if pred_p_flag:
                upper_p = pred_p + quant[1]
                lower_p = pred_p + quant[0]
        if pred_t_flag:
            pu_ci_t = (lower_t, upper_t)
        else:
            pu_ci_t = None
        if pred_p_flag:
            pu_ci_p = (lower_p, upper_p)
        else:
            pu_ci_p = None
    else:
        pu_ci_p = None
        pu_ci_t = None
    if return_forest_object:
        forest = regr_best
    else:
        forest = None
    return pred_p, pred_t, oob_best, vi_names_shares, pu_ci_p, pu_ci_t, forest


def variable_importance_testdata(
        x_name, var_im_groups, y_all, with_output, var_im_with_output, x_all,
        boot, n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes,
        share_test_data=0.2):
    """Compute variable importance without out-of-bag predictions."""
    if x_name is None:
        raise Exception('Variable importance needs names of features.')
    if var_im_groups is None:
        var_im_groups = []
    if var_im_groups:
        dummy_names = flatten_list(var_im_groups)
        x_to_check = [x for x in x_name if x not in dummy_names]
    else:
        x_to_check = x_name
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=share_test_data, random_state=42)
    regr_vi = RandomForestRegressor(
        n_estimators=boot, min_samples_leaf=n_opt, max_features=m_opt,
        bootstrap=True, oob_score=True, random_state=42, n_jobs=workers,
        max_depth=max_depth, min_weight_fraction_leaf=a_opt,
        max_leaf_nodes=max_leaf_nodes)
    regr_vi.fit(x_train, y_train)
    r2_full = regr_vi.score(x_test, y_test)
    # Start with single variables (without dummies)
    # indices_of_x = range(no_of_vars)
    loss_in_r2_single = np.empty(len(x_to_check))
    if with_output or var_im_with_output:
        print('Variable importance for ', end='')
    for indx, name_to_randomize in enumerate(x_to_check):
        if with_output or var_im_with_output:
            print(name_to_randomize, end=' ')
        r2_vi = get_r2_test(x_test, y_test, name_to_randomize, x_name, regr_vi)
        loss_in_r2_single[indx] = r2_full - r2_vi
    loss_in_r2_dummy = np.empty(len(var_im_groups))
    if var_im_groups:
        for indx, name_to_randomize in enumerate(var_im_groups):
            if with_output or var_im_with_output:
                print(name_to_randomize, end=' ')
            r2_vi = get_r2_test(x_test, y_test, name_to_randomize, x_name,
                                regr_vi)
            loss_in_r2_dummy[indx] = r2_full - r2_vi
    else:
        loss_in_r2_dummy = 0
    vi_names_shares = print_vi(
        x_to_check, var_im_groups, loss_in_r2_single, loss_in_r2_dummy,
        var_im_with_output)
    return vi_names_shares


def get_r2_test(x_test, y_test, name_to_randomize, x_name, regr_vi):
    """Get R2 for variable importance."""
    x_vi = x_test.copy()
    indices_to_r = find_indices_vi(name_to_randomize, x_name)
    x_to_shuffle = x_vi[:, indices_to_r]
    np.random.shuffle(x_to_shuffle)
    x_vi[:, indices_to_r] = x_to_shuffle
    r2_vi = regr_vi.score(x_vi, y_test)
    return r2_vi


def find_indices_vi(names, x_names):
    """Find indices that correspond to names and return list."""
    x_names = list(x_names)
    if isinstance(names, str):
        names = [names]
    indices = []
    for name in names:
        indices_t = x_names.index(name)
        indices.append(indices_t)
    return indices


def variable_importance_oob(x_name, var_im_groups, no_of_vars, y_1d, pred_oob,
                            with_output, var_im_with_output, x_train, boot,
                            n_opt, m_opt, workers, max_depth, a_opt,
                            max_leaf_nodes):
    """Compute variable importance with out-of-bag predictions."""
    if x_name is None:
        raise Exception('Variable importance needs names of features.')
    if var_im_groups is None:
        var_im_groups = []
    if var_im_groups:
        dummy_names = flatten_list(var_im_groups)
        x_to_check = [x for x in x_name if x not in dummy_names]
    else:
        x_to_check = x_name
    # Start with single variables (without dummies)
    indices_of_x = range(no_of_vars)
    var_y = np.var(y_1d)
    mse_ypred = np.mean((y_1d - pred_oob)**2)
    loss_in_r2_single = np.empty(len(x_to_check))
    if with_output or var_im_with_output:
        print('Variable importance for ', end='')
    for indx, name_to_delete in enumerate(x_to_check):
        if with_output or var_im_with_output:
            print(name_to_delete, end=' ')
        mse_ypred_vi = get_r2_for_vi_oob(
            indices_of_x, x_name, name_to_delete, x_train, y_1d, boot,
            n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes)
        loss_in_r2_single[indx] = (mse_ypred_vi - mse_ypred) / var_y
    loss_in_r2_dummy = np.empty(len(var_im_groups))
    if var_im_groups:
        for indx, name_to_delete in enumerate(var_im_groups):
            if with_output or var_im_with_output:
                print(name_to_delete, end=' ')
            mse_ypred_vi = get_r2_for_vi_oob(
                indices_of_x, x_name, name_to_delete, x_train, y_1d, boot,
                n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes)
            loss_in_r2_dummy[indx] = (mse_ypred_vi - mse_ypred) / var_y
    else:
        loss_in_r2_dummy = 0
    vi_names_shares = print_vi(
        x_to_check, var_im_groups, loss_in_r2_single, loss_in_r2_dummy,
        var_im_with_output)
    return vi_names_shares


def print_vi(x_to_check, var_im_groups, loss_in_r2_single, loss_in_r2_dummy,
             with_output):
    """
    Print variable importance measure.

    Parameters
    ----------
    x_to_check : Non-List of Str. Non-categorial variables.
    var_im_groups : List of lists of Str. Categorical variables as dummies.
    loss_in_r2_single : List. VI measure for non-cat.
    loss_in_r2_dummy : List. VI measure for non-cat.
    with_output : Bool.

    Returns
    -------
    x_names_sorted : List. Sorted Names.
    vim_sorted : List. Sorted VIM.

    """
    if var_im_groups:
        # Get single names and ignore the dummies (last element)
        var_dummies = []
        for name in var_im_groups:
            var_dummies.append(name[-1])
        x_to_check.extend(var_dummies)
        loss_in_r2_single = np.concatenate(
            (loss_in_r2_single, loss_in_r2_dummy))
    var_indices = np.argsort(loss_in_r2_single)
    var_indices = np.flip(var_indices)
    x_names_sorted = np.array(x_to_check, copy=True)
    x_names_sorted = x_names_sorted[var_indices]
    vim_sorted = loss_in_r2_single[var_indices]
    if with_output:
        print('\n')
        print('-' * 80)
        print('Variable importance statistics in %-point of R2 lost',
              'when omitting particular variable')
        print('- ')
        print('Unordered variables are included (in RF and VI) as dummies'
              ' and the variable itself.')
        if var_im_groups:
            print('Here, all elements of these variables are jointly tested ',
                  '(in Table under heading of unordered variable).')
        for idx, vim in enumerate(vim_sorted):
            print('{:<50}: {:>7.2f}'.format(x_names_sorted[idx], vim*100), '%')
        print('-' * 80)
        print('-' * 80, '\n')
    return x_names_sorted, vim_sorted


def get_r2_for_vi_oob(indices_of_x, x_name, name_to_delete, x_train,  y_1d,
                      boot=1000, n_opt=2, m_opt='sqrt', workers=-1,
                      max_depth=None, a_opt=0, max_leaf_nodes=None):
    """
    Estimate r2 for reduced variable set.

    Parameters
    ----------
    indices_of_x : range(no_of_columns_of X).
    x_name: List of Str. All x_names.
    name_to_delete : Str or list of Str. Variables to be deleted.
    x_train : Numpy array. All features.
    boot : Int, optional. No of trees. The default is 1000.
    n_opt : List of Int, optional. Minimum leaf size. The default is 2.
    m_opt : List of Int, optional. M_try. The default is 'sqrt'.
    workers : No of parallel process, optional. The default is -1.
    max_depth : Int. Depth of tree. Default is None.
    a_opt : Float. Minimum share on each side of split. Default is 0..
    max_leaf_nodes : Int. Maximimum number of leafs. Default is None.
    y_1d : 1d-Numpy array.

    Returns
    -------
    mse_ypred_vi : Float. MSE.

    """
    indices_to_delete = find_indices_vi(name_to_delete, x_name)
    indices_to_remain = [i for i in indices_of_x if i not in indices_to_delete]
    regr_vi = RandomForestRegressor(
        n_estimators=boot, min_samples_leaf=n_opt, max_features=m_opt,
        bootstrap=True, oob_score=True, random_state=42, n_jobs=workers,
        max_depth=max_depth, min_weight_fraction_leaf=a_opt,
        max_leaf_nodes=max_leaf_nodes)
    regr_vi.fit(x_train[:, indices_to_remain], y_1d)
    pred_vi = regr_vi.oob_prediction_
    mse_ypred_vi = np.mean((y_1d - pred_vi)**2)
    return mse_ypred_vi


def r_squared(y_dat, y_pred, w_dat=0):
    """
    Compute R2.

    Parameters
    ----------
    y : 1-dim Numpy array. Observed variable.
    y_pred : 1-dim Numpy array. Predicted variable.
    w : 1-dim Numpy array. optional. Weights. The default is 0 (no weighting).

    Returns
    -------
    r2 : Float. R2.

    """
    if np.all(w_dat == 0):
        w_dat = np.ones(len(y_dat))
    else:
        w_dat = w_dat / np.sum(w_dat) * len(w_dat)
        w_dat = w_dat.reshape(-1)
    y_mean = np.average(y_dat, weights=w_dat, axis=0)
    r_2 = 1 - np.dot(w_dat, np.square(y_dat-y_pred)) / np.dot(
        w_dat, np.square(y_dat-y_mean))
    return r_2


def gini_coefficient(x_dat):
    """Compute Gini coefficient of numpy array of values."""
    diffsum = 0
    x_dat = np.sort(x_dat)
    x_mean = np.mean(x_dat)
    if not -1e-15 < x_mean < 1e-15:
        for idx, x_i in enumerate(x_dat[:-1], 1):
            diffsum += np.sum(np.abs(x_i - x_dat[idx:]))
        return diffsum / ((len(x_dat))**2 * x_mean)
    return diffsum


def kernel_density(data, grid, kernel, bandwidth):
    """Compute nonparametric estimate of density of data.

    Parameters
    ----------
    data : Numpy array. Dependent variable.
    grid : grid : Numpy array. Values for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    f_grid : Numpy array. Prediction.

    """
    differ = np.subtract.outer(data, grid)  # This builds a matrix
    y_dach_i = kernel_proc(differ / bandwidth, kernel)
    f_grid = np.mean(y_dach_i, axis=0) / bandwidth
    return f_grid


def kernel_density_y(y_dat, x_dat, grid, kernel, bandwidth):
    """Compute nonparametric estimate of density of data * y.

    Parameters
    ----------
    y_dat : Numpy array. Dependent variable.
    x_dat : Numpy array. Independent variable.
    grid : grid : Numpy array. Values for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    f_grid : Numpy array. Prediction.

    """
    differ = np.subtract.outer(x_dat, grid)  # This builds a matrix
    y_dach_i = kernel_proc(differ / bandwidth, kernel)
    f_grid = np.mean(y_dach_i * np.reshape(y_dat, (len(grid), 1)),
                     axis=0) / bandwidth
    return f_grid


def nadaraya_watson(y_dat, x_dat, grid, kernel, bandwidth):
    """Compute Nadaraya-Watson one dimensional nonparametric regression.

    Parameters
    ----------
    y_dat : Numpy array. Dependent variable.
    x_dat :  Numpy array. Independent variable.
    grid : Numpy array. Values of x for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    estimate : Numpy array. Estimated quantity.

    """
    f_yx = kernel_density_y(y_dat, x_dat, grid, kernel, bandwidth)
    f_x = kernel_density(x_dat, grid, kernel, bandwidth)
    estimate = f_yx / f_x
    return estimate


def kernel_proc(data, kernel):
    """Feed data through kernel for nonparametric estimation.

    This function works for matrices and vectors.

    Parameters
    ----------
    data : Numpy array. Data.
    kernel : Int. 1: Epanechikov  2: Normal.

    Returns
    -------
    y : Numpy array. Kernel applied to data.

    """
    if kernel == 1:
        y_dat = np.zeros(np.shape(data))   # this works with matrices
        smaller_than_1 = np.abs(data) < 1  # abs() seems to be efficient
        y_dat[smaller_than_1] = 3/4 * (1 - (data[smaller_than_1] ** 2))
    if kernel == 2:  # This works for matrices
        y_dat = sct.norm.pdf(data)
    return y_dat


def bandwidth_silverman(data, kernel=1):
    """Compute Silvermans rule of thumb for Epanechikov and normal kernels.

    Silvermans rule of thumb, Cameron, Trivedi, p. 304

    Parameters
    ----------
    data : Numpy array 1D. Data.
    kernel : INT. 1: Epanechikov, 2: Normal

    Returns
    -------
    bandwidth : Float. Optimal bandwidth if data is normally distributed

    """
    obs = len(data)
    if obs < 5:
        raise Exception('Only', len(data), ' observations for',
                        'bandwidth selection.')
    iqr = np.quantile(data, (0.25, 0.75))
    iqr = (iqr[1] - iqr[0]) / 1.349
    std = np.std(data)
    if (std < iqr) or (iqr < 1e-15):
        sss = std
    else:
        sss = iqr
    band = 1.3643 * (obs ** (-0.2)) * sss
    if kernel == 1:  # Epanechikov
        bandwidth = band * 1.7188
    elif kernel == 2:  # Normal distribution
        bandwidth = band * 0.7764
    else:
        raise Exception('Wrong type of kernel in Silverman bandwidth')
    if bandwidth < 1e-15:
        bandwidth = 1
    return bandwidth


def bandwidth_nw_rule_of_thumb(data):
    """Compute rule of thumb for Nadaraya Watson Kernel regression.

    Li & Racine, 2004, Nonparametric Econometrics: Theory & Practice,
                       bottom of page 66

    Parameters
    ----------
    data : Numpy array 1D. Data.

    Returns
    -------
    bandwidth : Float. Optimal bandwidth if data is normally distributed

    """
    obs = len(data)
    if obs < 5:
        raise Exception('Only', len(data), ' observations for',
                        'bandwidth selection.')
    iqr = np.quantile(data, (0.25, 0.75))
    iqr = (iqr[1] - iqr[0]) / 1.349
    std = np.std(data)
    if (std < iqr) or (iqr < 1e-15):
        sss = std
    else:
        sss = iqr
    bandwidth = sss * (obs ** (-0.2))
    if bandwidth < 1e-15:
        bandwidth = 1
    return bandwidth


def moving_avg_mean_var(data, k, mean_and_var=True):
    """Compute moving average of mean and std deviation.

    Parameters
    ----------
    y : Numpy array. Dependent variable. Sorted.
    k : Int. Number of neighbours to be considered.

    Returns
    -------
    y_mean_movaverge : numpy array.
    y_var_movaverge : numpy array.
    """
    var = None
    obs = len(data)
    k = int(k)
    if k >= obs:
        mean = np.full(obs, np.mean(data))
        if mean_and_var:
            var = np.full(obs, np.var(data))
    else:
        weights = np.ones(k) / k
        mean = np.convolve(data, weights, mode='valid')
        half_diff_in_len = (obs - len(mean))/2
        add_dim_first = int(np.ceil(half_diff_in_len))
        add_dim_last = int(np.floor(half_diff_in_len))
        firstvalues = np.full(add_dim_first, mean[0])
        lastvalues = np.full(add_dim_last, mean[-1])
        mean = np.concatenate((firstvalues, mean, lastvalues))
        if mean_and_var:
            data_s = data ** 2
            mean_s = np.convolve(data_s, weights, mode='valid')
            firstvalues = np.full(add_dim_first, mean_s[0])
            lastvalues = np.full(add_dim_last, mean_s[-1])
            mean_s = np.concatenate((firstvalues, mean_s, lastvalues))
            var = (mean_s - mean * mean)
    return mean, var


def weight_var(w0_dat, y0_dat, cl_dat, c_dict, norm=True, w_for_diff=None,
               weights=None, bootstrap=0):
    """Generate the weight-based variance.

    Parameters
    ----------
    w_dat : Numpy array. Weights.
    y_dat : Numpy array. Outcomes.
    cl_dat : Numpy array. Cluster indicator.
    c_dict : Dict. Parameters.
    norm : Boolean. Normalisation. Default is True.
    w_for_diff : Numpy array. weights used for difference wheb clustering.
                 Default is None.
    weights : Numpy array. Sampling weights. Clustering only. Default is None.
    no_agg :   Boolean. No aggregation of weights. Default is False.
    bootstrap: Int. If > 1: Use bootstrap instead for SE estimation.

    Returns
    -------
    variance : Float. Variance.

    """
    w_dat = np.copy(w0_dat)
    y_dat = np.copy(y0_dat)
    if c_dict['cluster_std'] and (cl_dat is not None):
        if not c_dict['w_yes']:
            weights = None
        w_dat, y_dat, _, _ = aggregate_cluster_pos_w(
            cl_dat, w_dat, y_dat, norma=norm, sweights=weights)
    if w_for_diff is not None:
        w_dat = w_dat - w_for_diff
    if norm:
        sum_w_dat = np.abs(np.sum(w_dat))
        if not ((-1e-15 < sum_w_dat < 1e-15)
                or (1-1e-10 < sum_w_dat < 1+1e-10)):
            w_dat = w_dat / sum_w_dat
    w_ret = np.copy(w_dat)
    w_pos = np.abs(w_dat) > 1e-15  # use non-zero only to speed up
    w_dat2 = w_dat[w_pos]
    y_dat = y_dat[w_pos]
    obs = len(w_dat2)
    if obs < 5:
        return 0, 1, w_ret
    est = np.dot(w_dat2, y_dat)
    if bootstrap > 1:
        rng = np.random.default_rng(123345)
        est_b = np.empty(bootstrap)
        for b_idx in range(bootstrap):
            idx = rng.integers(0, high=obs, size=obs)
            w_b = np.copy(w_dat2[idx])
            if norm:
                sum_w_b = np.abs(np.sum(w_b))
                if not ((-1e-15 < sum_w_dat < 1e-15)
                        or (1-1e-10 < sum_w_dat < 1+1e-10)):
                    w_b = w_b / sum_w_b
            est_b[b_idx] = np.dot(w_b, y_dat[idx])
        variance = np.var(est_b)
    else:
        if c_dict['cond_var']:
            sort_ind = np.argsort(w_dat2)
            y_s = y_dat[sort_ind]
            w_s = w_dat2[sort_ind]
            if c_dict['knn']:
                k = int(np.round(c_dict['knn_const'] * np.sqrt(obs) * 2))
                if k < c_dict['knn_min_k']:
                    k = c_dict['knn_min_k']
                if k > obs/2:
                    k = np.floor(obs/2)
                exp_y_cond_w, var_y_cond_w = moving_avg_mean_var(y_s, k)
            else:
                band = bandwidth_nw_rule_of_thumb(w_s) * c_dict['nw_bandw']
                exp_y_cond_w = nadaraya_watson(y_s, w_s, w_s,
                                               c_dict['nw_kern'], band)
                var_y_cond_w = nadaraya_watson((y_s - exp_y_cond_w)**2, w_s,
                                               w_s, c_dict['nw_kern'], band)
            variance = np.dot(w_s**2, var_y_cond_w) + obs * np.var(
                w_s*exp_y_cond_w)
        else:
            variance = len(w_dat2) * np.var(w_dat2 * y_dat)
    return est, variance, w_ret


def aggregate_cluster_pos_w(cl_dat, w_dat, y_dat=None, norma=True, w2_dat=None,
                            sweights=None, y2_compute=False):
    """Aggregate weighted cluster means.

    Parameters
    ----------
    cl_dat : Numpy array. Cluster indicator.
    w_dat : Numpy array. Weights.
    y_dat : Numpy array. Outcomes.
    ...

    Returns
    -------
    w_agg : Numpy array. Aggregated weights. Normalised to one.
    y_agg : Numpy array. Aggregated outcomes.
    w2_agg : Numpy array. Aggregated weights. Normalised to one.

    """
    cluster_no = np.unique(cl_dat)
    # no_cluster = np.size(cluster_no)
    no_cluster = len(cluster_no)
    w_pos = np.abs(w_dat) > 1e-15
    if y_dat is not None:
        if y_dat.ndim == 1:
            y_dat = np.reshape(y_dat, (-1, 1))
        q_obs = np.size(y_dat, axis=1)
        y_agg = np.zeros((no_cluster, q_obs))
    else:
        y_agg = None
    if y2_compute:
        y2_agg = np.copy(y_agg)
    else:
        y2_agg = None
    w_agg = np.zeros(no_cluster)
    if w2_dat is not None:
        w2_agg = np.zeros(no_cluster)
        w2_pos = np.abs(w2_dat) > 1e-15
    else:
        w2_agg = None
    for j, cl_ind in enumerate(cluster_no):
        in_cluster = (cl_dat == cl_ind).reshape(-1)
        in_cluster_pos = in_cluster & w_pos
        if y2_compute:
            in_cluster_pos2 = in_cluster & w2_pos
        if np.any(in_cluster_pos):
            w_agg[j] = np.sum(w_dat[in_cluster_pos])
            if w2_dat is not None:
                w2_agg[j] = np.sum(w2_dat[in_cluster])
            if (y_dat is not None) and np.any(in_cluster_pos):
                for odx in range(q_obs):
                    if sweights is None:
                        y_agg[j, odx] = (np.dot(
                            w_dat[in_cluster_pos], y_dat[in_cluster_pos, odx])
                            / w_agg[j])
                        if y2_compute:
                            y2_agg[j, odx] = (np.dot(
                                w2_dat[in_cluster_pos2],
                                y_dat[in_cluster_pos2, odx]) / w2_agg[j])
                    else:
                        y_agg[j, odx] = (np.dot(
                            w_dat[in_cluster_pos] *
                            sweights[in_cluster_pos].reshape(-1),
                            y_dat[in_cluster_pos, odx]) / w_agg[j])
                        if y2_compute:
                            y2_agg[j, odx] = (np.dot(
                                w2_dat[in_cluster_pos2] *
                                sweights[in_cluster_pos2].reshape(-1),
                                y_dat[in_cluster_pos2, odx]) / w2_agg[j])
    if norma:
        sum_w_agg = np.sum(w_agg)
        if not 1-1e-10 < sum_w_agg < 1+1e-10:
            w_agg = w_agg / sum_w_agg
        if w2_dat is not None:
            sum_w2_agg = np.sum(w2_agg)
            if not 1-1e-10 < sum_w2_agg < 1+1e-10:
                w2_agg = w2_agg / sum_w2_agg
    return w_agg, y_agg, w2_agg, y2_agg


def flatten_list(list_to_flatten):
    """Flatten a list of lists in one single list.

    Parameters
    ----------
    list_to_flatten : List of list. To flatten.

    Returns
    -------
    flatted_list : List. Single list.

    """
    flatted_list = [item for sublist in list_to_flatten for item in sublist]
    return flatted_list
