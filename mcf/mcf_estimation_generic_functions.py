"""
Contains generic functions various generic estimators.

Created on Mon Jun 19 17:50:33 2023.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import copy, deepcopy
from math import sqrt, log2

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, mode
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import tree

from mcf.mcf_print_stats_functions import del_added_chars


# Since pad is not supported by numba, njit does not bring improvements
def moving_avg_mean_var(data, k, mean_and_var=True):
    """Compute moving average of mean and std deviation.

    Parameters
    ----------
    y : Numpy array. Dependent variable. Sorted.
    k : Int. Number of neighbours to be considered.

    Returns
    -------
    mean : numpy array.
    var : numpy array.
    """
    obs = len(data)
    k = int(k)

    if k >= obs:
        mean_ = np.mean(data)
        mean = np.full(obs, mean_)
        if mean_and_var:
            var = np.full(obs, np.var(data, mean=mean_))  # New in 2.0.0)
    else:
        weights = np.ones(k) / k

        # Compute the rolling mean
        mean = np.convolve(data, weights, mode='valid')

        # Compute the rolling variance if needed
        if mean_and_var:
            data_s = data ** 2
            mean_s = np.convolve(data_s, weights, mode='valid')
            var = mean_s - mean**2

        # Pad the results to match the length of the original data
        pad_before = (obs - len(mean)) // 2
        pad_after = obs - len(mean) - pad_before

        mean = np.pad(mean, (pad_before, pad_after), mode='edge')

        if mean_and_var:
            var = np.pad(var, (pad_before, pad_after), mode='edge')
        else:
            var = None

    return mean, var


# Kernel density estimation and kernel regression
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
        raise ValueError(f'Only {obs} observations for bandwidth selection.')
    iqr = np.subtract(*np.percentile(data, [75, 25])) / 1.349
    std = np.std(data)
    sss = min(std, iqr) if iqr > 1e-15 else std
    bandwidth = sss * (obs ** (-0.2))
    if bandwidth < 1e-15:
        bandwidth = 1
    return bandwidth


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
        raise ValueError(f'Only {len(data)} observations for bandwidth '
                         'selection.')
    iqr = np.subtract(*np.percentile(data, [75, 25])) / 1.349
    std = np.std(data)
    sss = min(std, iqr) if iqr > 1e-15 else std
    band = 1.3643 * (obs ** (-0.2)) * sss
    if kernel not in (1, 2):
        raise ValueError('Wrong type of kernel in Silverman bandwidth.')
    if kernel == 1:  # Epanechikov
        bandwidth = band * 1.7188
    elif kernel == 2:  # Normal distribution
        bandwidth = band * 0.7764
    if bandwidth < 1e-15:
        bandwidth = 1
    return bandwidth


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
    return f_yx / f_x


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
    if y_dat.ndim == 2:
        f_grid = np.mean(y_dach_i * y_dat, axis=0) / bandwidth
    else:
        f_grid = np.mean(y_dach_i * np.reshape(y_dat, (len(grid), 1)),
                         axis=0) / bandwidth
    return f_grid


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
        abs_data = np.abs(data)
        y_dat = np.where(abs_data < 1, 3/4 * (1 - abs_data**2), 0)
    elif kernel == 2:  # This works for matrices
        y_dat = norm.pdf(data)
    else:
        raise ValueError('Only Epanechikov and normal kernel supported.')
    return y_dat


def gini_coeff_pos(x_dat):
    """Compute Gini coefficient of numpy array of values with values >= 0."""
    sss = x_dat.sum()
    if sss > 1e-15:                        # Use 'mergesort' for stable sorting
        rrr = np.argsort(-x_dat, kind='mergesort')
        ranks = np.arange(1, len(x_dat) + 1)
        gini = 1 - 2 * (np.sum((ranks - 1) * x_dat[rrr]) + sss) / (len(x_dat)
                                                                   * sss)
        return gini
    return 0


# New optimized function, 4.6.2024
@njit
def quadratic_form(vec, mat):
    """Quadratic form for Numpy: vec'mat*vec.

    Parameters
    ----------
    vec : 1D Numpy Array.
    mat : 2D quadratic Numpy Array.

    Returns
    -------
    Numpy Float. The Quadratic form.
    """
    return np.dot(vec, np.dot(mat, vec))

# Old function that worked fine.
# @njit
# def quadratic_form(vec, mat):
#     """Quadratic form for Numpy: vec'mat*vec.

#     Parameters
#     ----------
#     vec : 1D Numpy Array.
#     mat : 2D quadratic Numpy Array.

#     Returns
#     -------
#     Numpy Float. The Quadratic form.

#     """
#     return vec @ (mat @ vec.T)


def random_forest_scikit(
        x_train, y_train, x_pred, x_name=None, y_name='y', boot=1000,
        n_min=2, no_features='sqrt', workers=-1,  max_depth=None, alpha=0,
        max_leaf_nodes=None, pred_p_flag=True, pred_t_flag=False,
        pred_oob_flag=False, with_output=True, variable_importance=False,
        var_im_groups=None, pred_uncertainty=False, pu_ci_level=0.95,
        pu_skew_sym=0.5, var_im_with_output=True,
        variable_importance_oob_flag=False, return_forest_object=False,
        seed=42):
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
    workers : No of parallel processes, optional. The default is None.
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
                         Default = 0.95
    pu_skew_sym : Float. Cut-off of skewness for symetric CIs. Default is 0.5.
    variable_importance_oobx : Bool. Use OOB obs for VI (computational costly).
                                    Default is False.
    return_forest_object : Boolean. Forest object.
    seed : Int. Seed for random forest.

    Returns
    -------
    pred_p : 1d Numpy array. Predictions prediction data.
    pred_t: 1d Numpy array. Predictions training data.
    oob_best: Float. OOB based R2
    pu_ci_p: 2d Numpy array. Confidence interval prediction data
    pu_ci_t: 2d Numpy array. Confidence interval training data
    forest: Forest object.
    txt : String: Text to print.

    """
    txt = ''
    if with_output:
        var_im_with_output = True
    oob_best = -1e15
    pred_p, pred_t = [], []
    y_1d = y_train.ravel()
    no_of_vars = np.size(x_train, axis=1)
    if no_features is None:
        no_features = no_of_vars
    if isinstance(no_features, str):
        if no_features == 'sqrt':
            no_features = round(sqrt(no_of_vars))
        elif no_features == 'auto':
            no_features = no_of_vars
        elif no_features == 'log2':
            no_features = round(log2(no_of_vars))
        else:
            no_features = round(no_of_vars/3)
    elif isinstance(no_features, float):
        no_features = round(no_features * no_of_vars)
    if not isinstance(no_features, (list, tuple, np.ndarray)):
        no_features = [no_features]
    if not isinstance(alpha, (list, tuple, np.ndarray)):
        alpha = [alpha]
    if not isinstance(n_min, (list, tuple, np.ndarray)):
        n_min = [n_min]
    for nval in n_min:
        for mval in no_features:
            for aval in alpha:
                regr = RandomForestRegressor(
                    n_estimators=boot, min_samples_leaf=round(nval),
                    max_features=mval, bootstrap=True, oob_score=True,
                    random_state=seed, n_jobs=workers, max_depth=max_depth,
                    min_weight_fraction_leaf=aval,
                    max_leaf_nodes=max_leaf_nodes)
                regr.fit(x_train, y_1d)
                if with_output and not ((len(no_features) == 1)
                                        and (len(n_min) == 1)
                                        and (len(alpha) == 1)):
                    txt += (f'\n N_min: {nval:2}  M_Try: {mval:2}'
                            f' Alpha: {aval:5.3f}'
                            f' |  OOB R2 (in %): {regr.oob_score_*100:8.4f}')
                if regr.oob_score_ > oob_best:
                    m_opt, n_opt = copy(mval), copy(nval)
                    a_opt = copy(aval)
                    oob_best = copy(regr.oob_score_)
                    regr_best = deepcopy(regr)
    if with_output:
        txt += '\n' * 2 + '-' * 100
        txt += '\nTuning values choosen for forest (if any)'
        txt += '\n' + '-' * 100
        txt += (f'\nDependent variable: {y_name[0]} N_min: {n_opt:2}'
                f'M_Try: {m_opt:2} Alpha: {a_opt:5.3f}'
                f'|  OOB R2 (in %): {oob_best * 100:8.4f}')
        txt += '\n' + '-' * 100
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
        txt += f'\n{regr_best.get_params()}\n' + '-' * 100
        txt += f'\n{y_name}: R2(%) OOB: {regr_best.oob_score_*100:8.4f}'
        txt += '\n' + '-' * 100
    if variable_importance:
        if variable_importance_oob_flag:
            vi_names_shares, txt_vi = variable_importance_oob(
                x_name, var_im_groups, no_of_vars, y_1d, pred_oob, with_output,
                var_im_with_output, x_train, boot, n_opt, m_opt, workers,
                max_depth, a_opt, max_leaf_nodes)
        else:
            vi_names_shares, txt_vi = variable_importance_testdata(
                x_name, var_im_groups, y_1d, with_output,
                var_im_with_output, x_train, boot, n_opt, m_opt, workers,
                max_depth, a_opt, max_leaf_nodes)
        txt += txt_vi
    else:
        vi_names_shares = None
    if pred_uncertainty and (pred_t_flag or pred_p_flag):
        resid_oob = y_1d - pred_oob
        skewness_res_oob = skew(resid_oob)
        if with_output:
            txt += f'\nSkewness of oob residual: {skewness_res_oob}'
        symmetric = np.absolute(skewness_res_oob) < pu_skew_sym
        if symmetric:
            if with_output:
                txt += '\nSymmetric intervals used'
            quant = np.quantile(np.absolute(resid_oob), 1-pu_ci_level)
            if pred_t_flag:
                upper_t, lower_t = pred_t + quant, pred_t - quant
            if pred_p_flag:
                upper_p, lower_p = pred_p + quant, pred_p - quant
        else:
            if with_output:
                txt += '\nAsymmetric intervals used'
            quant = np.quantile(resid_oob,
                                [(1-pu_ci_level)/2, 1-(1-pu_ci_level)/2])
            if pred_t_flag:
                upper_t, lower_t = pred_t + quant[1], pred_t + quant[0]
            if pred_p_flag:
                upper_p, lower_p = pred_p + quant[1], pred_p + quant[0]
        pu_ci_t = (lower_t, upper_t) if pred_t_flag else None
        pu_ci_p = (lower_p, upper_p) if pred_p_flag else None
    else:
        pu_ci_p, pu_ci_t = None, None
    forest = regr_best if return_forest_object else None
    return (pred_p, pred_t, oob_best, vi_names_shares, pu_ci_p, pu_ci_t,
            forest, txt)


def variable_importance_oob(x_name, var_im_groups, no_of_vars, y_1d, pred_oob,
                            with_output, var_im_with_output, x_train, boot,
                            n_opt, m_opt, workers, max_depth, a_opt,
                            max_leaf_nodes):
    """Compute variable importance with out-of-bag predictions."""
    if x_name is None:
        raise ValueError('Variable importance needs names of features.')
    if var_im_groups is None:
        var_im_groups = []
    if var_im_groups:
        dummy_names = [item for sublist in var_im_groups for item in sublist]
        x_to_check = [x for x in x_name if x not in dummy_names]
    else:
        x_to_check = x_name
    # Start with single variables (without dummies)
    indices_of_x = range(no_of_vars)
    var_y = np.var(y_1d)
    mse_ypred = np.mean((y_1d - pred_oob)**2)
    loss_in_r2_single = np.empty(len(x_to_check))
    txt = ''
    if with_output or var_im_with_output:
        print('Variable importance for ')
    for indx, name_to_delete in enumerate(x_to_check):
        if with_output or var_im_with_output:
            print(f'{name_to_delete}')
        mse_ypred_vi = get_r2_for_vi_oob(
            indices_of_x, x_name, name_to_delete, x_train, y_1d, boot,
            n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes)
        loss_in_r2_single[indx] = (mse_ypred_vi - mse_ypred) / var_y
    loss_in_r2_dummy = np.empty(len(var_im_groups))
    if var_im_groups:
        for indx, name_to_delete in enumerate(var_im_groups):
            if with_output or var_im_with_output:
                print(f'{name_to_delete}')
            mse_ypred_vi = get_r2_for_vi_oob(
                indices_of_x, x_name, name_to_delete, x_train, y_1d, boot,
                n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes)
            loss_in_r2_dummy[indx] = (mse_ypred_vi - mse_ypred) / var_y
    else:
        loss_in_r2_dummy = 0
    x_names_sorted, vim_sorted, txt_vi = print_vi(
        x_to_check, var_im_groups, loss_in_r2_single, loss_in_r2_dummy,
        var_im_with_output)
    vi_names_shares = (x_names_sorted, vim_sorted, txt,)
    txt += txt_vi
    return vi_names_shares, txt


def variable_importance_testdata(
        x_name, var_im_groups, y_all, with_output, var_im_with_output, x_all,
        boot, n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes,
        share_test_data=0.2):
    """Compute variable importance without out-of-bag predictions."""
    if x_name is None:
        raise ValueError('Variable importance needs names of features.')
    if var_im_groups is None:
        var_im_groups = []
    if var_im_groups:
        dummy_names = [item for sublist in var_im_groups for item in sublist]
        x_to_check = [x for x in x_name if x not in dummy_names]
    else:
        x_to_check = x_name
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=share_test_data, random_state=42)
    regr_vi = RandomForestRegressor(
        n_estimators=boot, min_samples_leaf=round(n_opt), max_features=m_opt,
        bootstrap=True, oob_score=True, random_state=42, n_jobs=workers,
        max_depth=max_depth, min_weight_fraction_leaf=a_opt,
        max_leaf_nodes=max_leaf_nodes)
    regr_vi.fit(x_train, y_train)
    r2_full = regr_vi.score(x_test, y_test)
    # Start with single variables (without dummies)
    loss_in_r2_single = np.empty(len(x_to_check))
    rng = np.random.default_rng(12345)
    txt = ''
    if with_output or var_im_with_output:
        print('Variable importance for ')
    for indx, name_to_randomize in enumerate(x_to_check):
        if with_output or var_im_with_output:
            print(name_to_randomize)
        r2_vi = get_r2_test(x_test, y_test, name_to_randomize, x_name, regr_vi,
                            rng)
        loss_in_r2_single[indx] = r2_full - r2_vi
    loss_in_r2_dummy = np.empty(len(var_im_groups))
    if var_im_groups:
        for indx, name_to_randomize in enumerate(var_im_groups):
            if with_output or var_im_with_output:
                print(name_to_randomize)
            r2_vi = get_r2_test(x_test, y_test, name_to_randomize, x_name,
                                regr_vi, rng)
            loss_in_r2_dummy[indx] = r2_full - r2_vi
    else:
        loss_in_r2_dummy = 0
    x_names_sorted, vim_sorted, txt_vi = print_vi(
        x_to_check, var_im_groups, loss_in_r2_single, loss_in_r2_dummy,
        var_im_with_output)
    vi_names_shares = (x_names_sorted, vim_sorted, txt,)
    txt += txt_vi
    return vi_names_shares, txt


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
    workers : No of parallel process, optional. The default is None.
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
        n_estimators=boot, min_samples_leaf=round(n_opt), max_features=m_opt,
        bootstrap=True, oob_score=True, random_state=42, n_jobs=workers,
        max_depth=max_depth, min_weight_fraction_leaf=a_opt,
        max_leaf_nodes=max_leaf_nodes)
    regr_vi.fit(x_train[:, indices_to_remain], y_1d)
    pred_vi = regr_vi.oob_prediction_
    mse_ypred_vi = np.mean((y_1d - pred_vi)**2)
    return mse_ypred_vi


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
    txt : String. Text to print.

    """
    if var_im_groups:
        # Get single names and ignore the dummies (last element)
        var_dummies = [name[-1] for name in var_im_groups]
        x_to_check.extend(var_dummies)
        loss_in_r2_single = np.concatenate(
            (loss_in_r2_single, loss_in_r2_dummy))
    var_indices = np.argsort(loss_in_r2_single)
    var_indices = np.flip(var_indices)
    x_names_sorted = np.array(x_to_check, copy=True)
    x_names_sorted = x_names_sorted[var_indices]
    vim_sorted = loss_in_r2_single[var_indices]
    txt = ''
    if with_output:
        txt += '\n' * 2 + '-' * 100
        txt += ('\nVariable importance statistics in %-point of R2 lost '
                'when omitting particular variable')
        txt += '\n' + '- ' * 50
        txt += ('\nUnordered variables are included (in RF and VI) as dummies'
                ' and the variable itself.')
        if var_im_groups:
            txt += ('\nHere, all elements of these variables are jointly '
                    'tested (in Table under heading of unordered variable).')
        for idx, vim in enumerate(vim_sorted):
            name_ = del_added_chars(x_names_sorted[idx], prime=True)
            txt += f'\n{name_:<50}: {vim:>7.2%}'
        txt += '\n' + '-' * 100
    return x_names_sorted, vim_sorted, txt


def get_r2_test(x_test, y_test, name_to_randomize, x_name, regr_vi, rng=None):
    """Get R2 for variable importance."""
    if rng is None:
        rng = np.random.default_rng()
    x_vi = x_test.copy()
    indices_to_r = find_indices_vi(name_to_randomize, x_name)
    x_to_shuffle = x_vi[:, indices_to_r]
    rng.shuffle(x_to_shuffle)
    x_vi[:, indices_to_r] = x_to_shuffle
    r2_vi = regr_vi.score(x_vi, y_test)
    return r2_vi


def find_indices_vi(names, x_names):
    """Find indices that correspond to names and return list."""
    x_names = list(x_names)
    if isinstance(names, str):
        names = [names]
    indices = [x_names.index(name) for name in names]
    return indices


def best_regression(x_np, y_np, estimator=None, boot=1000, seed=123,
                    max_workers=None, test_share=0.25, cross_validation_k=0,
                    absolute_values_pred=False):
    """Select regression estimator for local centring (no centering of X)."""
    # Initialise estimators
    regress_dic, estimator, key_list = init_scikit(boot, seed, max_workers,
                                                   estimator)

    if estimator != 'automatic':
        return (regress_dic[estimator]['method'],
                regress_dic[estimator]['params'],
                regress_dic[estimator]['label'],
                -1,
                regress_dic[estimator]['transform_x'],
                ' Method determined by user is '
                f'{regress_dic[estimator]["label"]}')

    no_methods, obs = len(regress_dic), len(x_np)
    mse, r_2 = np.zeros(no_methods), np.zeros(no_methods)
    y_pred = np.empty((obs, no_methods))
    if cross_validation_k > 0:
        index = np.arange(obs)       # indices
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(index)
        index_folds = np.array_split(index, cross_validation_k)
        for fold_pred in range(cross_validation_k):
            fold_train = [x for idx, x in enumerate(index_folds)
                          if idx != fold_pred]
            index_train = np.hstack(fold_train)
            index_pred = index_folds[fold_pred]
            y_train, x_train = y_np[index_train], x_np[index_train]
            x_test = x_np[index_pred]
            _, x_train_scaled, x_test_scaled = scale(x_train, x_test)

            for idx, key in enumerate(key_list):
                if regress_dic[key]['obj'] is None:
                    y_pred[index_pred, idx] = np.average(y_train)
                else:
                    if regress_dic[key]['transform_x']:
                        regress_dic[key]['obj'].fit(x_train_scaled, y_train)
                        y_pred[index_pred, idx] = (
                            regress_dic[key]['obj'].predict(x_test_scaled))
                    else:
                        regress_dic[key]['obj'].fit(x_train, y_train)
                        y_pred[index_pred, idx] = (regress_dic[key]
                                                   ['obj'].predict(x_test))
        if absolute_values_pred:
            y_pred = np.abs(y_pred)
        for idx in range(no_methods):
            mse[idx] = mean_squared_error(y_np, y_pred[:, idx])
            r_2[idx] = r2_score(y_np, y_pred[:, idx])
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_np, y_np, test_size=test_share, random_state=seed)
        _, x_train_scaled, x_test_scaled = scale(x_train, x_test)
        for idx, key in enumerate(key_list):
            if regress_dic[key]['transform_x']:
                x_train_adj, x_test_adj = x_train_scaled, x_test_scaled
            else:
                x_train_adj, x_test_adj = x_train, x_test
            # Train the estimator
            if regress_dic[key]['obj'] is None:
                y_pred = np.average(y_train) * np.ones_like(y_test)
            else:
                regress_dic[key]['obj'].fit(x_train_adj, y_train)
                # Test performance
                y_pred = regress_dic[key]['obj'].predict(x_test_adj)
            # Use MSE for comparison of performance
            if absolute_values_pred:
                y_pred = np.abs(y_pred) * np.ones_like(y_test)
            mse[idx] = mean_squared_error(y_test, y_pred)
            r_2[idx] = r2_score(y_test, y_pred)

    # Select best method
    min_indx = np.argmin(mse)
    best_mse = mse[min_indx]
    best_key = key_list[min_indx]
    best_method = regress_dic[best_key]['method']
    best_params = regress_dic[best_key]['params']
    best_lables = regress_dic[best_key]['label']
    best_transform_x = regress_dic[best_key]['transform_x']
    labels = [regress_dic[key]['label'] for key in key_list]
    mse_string = print_performance_mse(labels, mse, r_2, best_lables)
    return (best_method, best_params, best_lables, best_mse,
            best_transform_x, mse_string)


def scale(x_train, x_test=None):
    """Scale featuress."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = None if x_test is None else scaler.transform(x_test)
    return scaler, x_train_scaled, x_test_scaled


def init_scikit(boot, seed, max_workers, estimator):
    """Initialise estimation procedure."""
    regr_dic = {}
    key_list = ('rf', 'rfl5', 'rfls5',
                'svr', 'svr2', 'svr4',
                'ada', 'ada100', 'ada200',
                'gboost', 'gboost6',  'gboost12',
                'lasso',
                'nnet', 'nnet_l1', 'nnet_l2',
                'mean')
    for key in key_list:
        regr_dic[key] = {}

    # Scitkit learn parameters (nonspecified values correspond to defaults)
    regr_dic['rf']['params'] = {
        'n_estimators': boot, 'max_features': 'sqrt',
        'bootstrap': True, 'oob_score': False, 'n_jobs': max_workers,
        'random_state': seed, 'verbose': False, 'min_samples_leaf': 1,
        'min_samples_split': 2}
    regr_dic['rfl5']['params'] = deepcopy(regr_dic['rf']['params'])
    regr_dic['rfl5']['params']['min_samples_leaf'] = 5
    regr_dic['rfls5']['params'] = deepcopy(regr_dic['rf']['params'])
    regr_dic['rfls5']['params']['min_samples_split'] = 5

    regr_dic['svr']['params'] = {
        'kernel': 'rbf', 'gamma': 'scale', 'tol': 1e-3, 'epsilon': 0.1,
        'shrinking': True, 'C': 1.0
        }
    regr_dic['svr2']['params'] = deepcopy(regr_dic['svr']['params'])
    regr_dic['svr2']['params']['C'] = 2
    regr_dic['svr4']['params'] = deepcopy(regr_dic['svr']['params'])
    regr_dic['svr4']['params']['C'] = 4
    regr_dic['ada']['params'] = {
        'estimator': None, 'n_estimators': 50, 'learning_rate': 1.0,
        'loss': 'linear', 'random_state': seed
        }
    regr_dic['ada100']['params'] = deepcopy(regr_dic['ada']['params'])
    regr_dic['ada100']['params']['n_estimators'] = 100
    regr_dic['ada200']['params'] = deepcopy(regr_dic['ada']['params'])
    regr_dic['ada200']['params']['n_estimators'] = 200

    regr_dic['gboost']['params'] = {
        'loss': 'squared_error', 'n_estimators': 200, 'max_depth': 3,
        'random_state': seed
        }
    regr_dic['gboost6']['params'] = deepcopy(regr_dic['gboost']['params'])
    regr_dic['gboost6']['params']['max_depth'] = 6    # Deeper tree
    regr_dic['gboost12']['params'] = deepcopy(regr_dic['gboost']['params'])
    regr_dic['gboost12']['params']['max_depth'] = 12    # Deeper tree

    regr_dic['lasso']['params'] = {
        'max_iter': 1000, 'copy_X': True, 'cv': 5, 'n_jobs': max_workers,
        'random_state': seed
        }
    regr_dic['nnet']['params'] = {
        'alpha': 0.0001, 'max_iter': 2000, 'random_state': seed,
        'hidden_layer_sizes': (100,), 'solver': 'adam',
        'learning_rate_init': 0.001
        }
    regr_dic['nnet_l1']['params'] = deepcopy(regr_dic['nnet']['params'])
    regr_dic['nnet_l1']['params']['hidden_layer_sizes'] = (100, 50, 25)
    regr_dic['nnet_l2']['params'] = deepcopy(regr_dic['nnet']['params'])
    regr_dic['nnet_l2']['params']['hidden_layer_sizes'] = (200, 100, 50, 25, 12)
    regr_dic['mean']['params'] = None

    # Transformation of data needed
    regr_dic['rf']['transform_x'] = False
    regr_dic['rfl5']['transform_x'] = False
    regr_dic['rfls5']['transform_x'] = False
    regr_dic['svr']['transform_x'] = True
    regr_dic['svr2']['transform_x'] = True
    regr_dic['svr4']['transform_x'] = True
    regr_dic['ada']['transform_x'] = False
    regr_dic['ada100']['transform_x'] = False
    regr_dic['ada200']['transform_x'] = False
    regr_dic['gboost']['transform_x'] = False
    regr_dic['gboost6']['transform_x'] = False
    regr_dic['gboost12']['transform_x'] = False
    regr_dic['lasso']['transform_x'] = True
    regr_dic['nnet']['transform_x'] = True
    regr_dic['nnet_l1']['transform_x'] = True
    regr_dic['nnet_l2']['transform_x'] = True
    regr_dic['mean']['transform_x'] = False

    # Name of Methods (as used by other functions)
    regr_dic['rf']['method'] = 'RandomForest'
    regr_dic['rfl5']['method'] = 'RandomForestNminl5'
    regr_dic['rfls5']['method'] = 'RandomForestNminls5'
    regr_dic['svr']['method'] = 'SupportVectorMachine'
    regr_dic['svr2']['method'] = 'SupportVectorMachineC2'
    regr_dic['svr4']['method'] = 'SupportVectorMachineC4'
    regr_dic['ada']['method'] = 'AdaBoost'
    regr_dic['ada100']['method'] = 'AdaBoost100'
    regr_dic['ada200']['method'] = 'AdaBoost200'
    regr_dic['gboost']['method'] = 'GradBoost'
    regr_dic['gboost6']['method'] = 'GradBoostDepth6'
    regr_dic['gboost12']['method'] = 'GradBoostDepth12'
    regr_dic['lasso']['method'] = 'LASSO'
    regr_dic['nnet']['method'] = 'NeuralNet'
    regr_dic['nnet_l1']['method'] = 'NeuralNetLarge'
    regr_dic['nnet_l2']['method'] = 'NeuralNetLarger'
    regr_dic['mean']['method'] = 'Mean'

    # Labels of methods (used for printing)
    regr_dic['rf']['label'] = 'Random Forest'
    regr_dic['rfl5']['label'] = 'Random Forest Min Leaf Size 5'
    regr_dic['rfls5']['label'] = 'Random Forest Min Sample Size 5'
    regr_dic['svr']['label'] = 'Support Vector Machine'
    regr_dic['svr2']['label'] = 'Support Vector Machine C 2'
    regr_dic['svr4']['label'] = 'Support Vector Machine C 4'
    regr_dic['ada']['label'] = 'Ada Boost'
    regr_dic['ada100']['label'] = 'Ada Boost n_estimator 100'
    regr_dic['ada200']['label'] = 'Ada Boost n_estimator 200'
    regr_dic['gboost']['label'] = 'Gradient Boosting'
    regr_dic['gboost6']['label'] = 'Gradient Boosting Depth 6'
    regr_dic['gboost12']['label'] = 'Gradient Boosting Depth 12'
    regr_dic['lasso']['label'] = 'LASSO'
    regr_dic['nnet']['label'] = 'Neural Network'
    regr_dic['nnet_l1']['label'] = 'Neural Network large'
    regr_dic['nnet_l2']['label'] = 'Neural Network larger'
    regr_dic['mean']['label'] = 'Sample Mean'

    # Initialise scikit-learn instances
    estimator_key = None
    for key in key_list:
        if estimator is not None:
            if regr_dic[key]['method'] == estimator:
                estimator_key = key
        if regr_dic[key]['params'] is None:
            regr_dic[key]['obj'] = None
        else:
            regr_dic[key]['obj'] = regress_instance(regr_dic[key]['method'],
                                                    regr_dic[key]['params'])
    if estimator == 'automatic':
        estimator_key = estimator
    return regr_dic, estimator_key, key_list


def print_performance_mse(labels, mse, r_2, best_lables):
    """Create string with mse for all methods."""
    mse_string = ('\nSelection of different methods and tuning parameters\n'
                  '\nOut-of-sample MSE / R2 of ')
    for idx, label in enumerate(labels):
        mse_string += (f'\n... {label}:' + ' ' * (40 - len(label))
                       + f'{mse[idx]:8.4f} / {r_2[idx]:5.2%}')
    mse_string += f'\n\nBest Method: {best_lables}'
    return mse_string


def regress_instance(method, parameters_dict):
    """Initialize regression instance."""
    if method in ('RandomForest', 'RandomForestNminl5', 'RandomForestNminls5'):
        return RandomForestRegressor(**parameters_dict)
    if method in ('SupportVectorMachine', 'SupportVectorMachineC2',
                  'SupportVectorMachineC4'):
        return SVR(**parameters_dict)
    if method in ('AdaBoost', 'AdaBoost100', 'AdaBoost200'):
        return AdaBoostRegressor(**parameters_dict)
    if method in ('GradBoost', 'GradBoostDepth6', 'GradBoostDepth12'):
        return GradientBoostingRegressor(**parameters_dict)
    if method == 'LASSO':
        return LassoCV(**parameters_dict)
    if method in ('NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger'):
        return MLPRegressor(**parameters_dict)
    if method == 'Mean':
        return None
    valid_methods = ('RandomForest', 'RandomForestNmin5',
                     'SupportVectorMachine', 'SupportVectorMachineC2',
                     'SupportVectorMachineC4',
                     'AdaBoost', 'AdaBoost100', 'AdaBoost200',
                     'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12',
                     'LASSO',
                     'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
                     'Mean')
    raise ValueError(f'{method} not considered. Change method. Only '
                     f'{" ".join(valid_methods)} allowed.')


def best_classifier(x_np, y_np, boot=1000, seed=123, max_workers=None,
                    test_share=0.25):
    """Find best classifiers for specific allocation and features."""
    # Find best algorithm (among a very limited choice so far; to be expanded)
    # Data should be scaled when entering algorithm
    while True:
        x_train, x_test, y_train, y_test = train_test_split(
            x_np, y_np, test_size=test_share, random_state=seed)
        # Check if same values are in treated and test data
        unique_train = np.unique(np.round(y_train))
        unique_test = np.unique(np.round(y_test))
        if np.array_equal(unique_train, unique_test):
            break
        test_share += 0.05
        if test_share > 0.5:
            raise ValueError('Test share larger than 50%. Some category '
                             'does not contain all values of treatment.'
                             f'\n{unique_train}, \n{unique_test}')
    params_rf5 = {'n_estimators': boot, 'max_depth': None,
                  'min_samples_split': 5, 'min_samples_leaf': 2,
                  'random_state': seed, 'n_jobs': max_workers}
    params_rf2 = params_rf5.copy()
    params_rf2['min_samples_split'] = 2
    params_mlpc = {'alpha': 1, 'max_iter': 1000, 'random_state': seed}
    params_ada = {'algorithm': "SAMME", 'random_state': seed}

    rfc_5 = RandomForestClassifier(**params_rf5)
    rfc_2 = RandomForestClassifier(**params_rf2)
    neural_net = MLPClassifier(**params_mlpc)
    adda_boost = AdaBoostClassifier(**params_ada)

    methods = ('RandomForestClass5', 'RandomForestClass2', 'NNetClass',
               'AdaClass')
    labels = ('Classification Forest min leaf size = 5',
              'Classification Forest min leaf size = 2',
              'Neural Network Classifier',
              'AddaBoost Classifier')
    params = (params_rf5, params_rf2, params_mlpc, params_ada)
    classif_objects = (rfc_5, rfc_2, neural_net, adda_boost)
    acc_scores = np.empty(len(methods))

    for idx, cl_obj in enumerate(classif_objects):
        # Train the classifiers
        cl_obj.fit(x_train, y_train)
        # Test performance
        y_pred = cl_obj.predict(x_test)
        # Use accuracy score for comparison of performance
        acc_scores[idx] = accuracy_score(y_test, y_pred)

    # Select best method
    max_indx = np.argmax(acc_scores)
    best_score = acc_scores[max_indx]
    best_method = methods[max_indx]
    best_params = params[max_indx]
    best_lables = labels[max_indx]

    return best_method, best_params, best_lables, best_score


def classif_instance(method, parameters_dict):
    """Initialize regressin instance."""
    if method in ('RandomForestClass5', 'RandomForestClass2'):
        return RandomForestClassifier(**parameters_dict)
    if method == 'NNetClass':
        return MLPClassifier(**parameters_dict)
    if method == 'AdaClass':
        return AdaBoostClassifier(**parameters_dict)
    valid_methods = ('RandomForestClass5', 'RandomForestClass2', 'NNetClass',
                     'AdaClass')
    raise ValueError(f'{method} not considered. Change method. Only '
                     f'{" ".join(valid_methods)} allowed.')


def printable_output(label, acc_score):
    """Create string to be printed later on."""
    return f'Best method: {label}      Accuracy score: {acc_score:5.2%}'


def honest_tree_explainable(x_dat, y_dat, tree_type='regression',
                            depth_grid=(2, 3, 4, 5), feature_names=None,
                            title='',  seed=12356):
    """Estimate an honest tree usable for explainability."""
    # Cast x_data in float32 format to avoid trouble later on in honesty part
    x_dat = x_dat.astype(np.float32)

    # Avoid too small leaves
    min_leaf = 0.01 / (max(depth_grid) * 2)  # Min leaf size as share of obs
    # Split into training data and data used to show out-of-sample performance
    x_train, x_oos, y_train, y_oos = train_test_split(
        x_dat, y_dat, test_size=0.25, random_state=seed)

    # Split training data for honest estimation
    x_build, x_fill, y_build, y_fill = train_test_split(
        x_train, y_train, test_size=0.5, random_state=seed)

    plt_all, plt_h_all, fit_all, fit_h_all = [], [], [], []
    for depth in depth_grid:

        tree_parameters = {'random_state': seed,
                           'max_depth': depth,
                           'min_samples_leaf': min_leaf,
                           }
        if tree_type == 'regression':
            tree_parameters['criterion'] = 'squared_error'
            tree_inst = tree.DecisionTreeRegressor(**tree_parameters)
        else:
            tree_parameters['criterion'] = 'gini'
            tree_inst = tree.DecisionTreeClassifier(**tree_parameters)

        # Build tree
        tree_inst.fit(x_build, y_build)
        # Adjust final leaves for honesty
        tree_inst_h = tree_gets_honest(tree_inst, x_fill, y_fill, tree_type)

        # Plots of the tree
        plot = plot_tree_fct(tree_inst, feature_names, depth, y_name=title)
        plt_all.append(plot)
        plot_h = plot_tree_fct(tree_inst_h, feature_names, depth, honest=True,
                               y_name=title)
        plt_h_all.append(plot_h)

        y_pred = tree_inst.predict(x_oos)
        y_pred_h = tree_inst_h.predict(x_oos)

        if tree_type == 'regression':
            fit_all.append(r2_score(y_oos, y_pred))
            fit_h_all.append(r2_score(y_oos, y_pred_h))
        else:
            fit_all.append(accuracy_score(y_oos, y_pred))
            fit_h_all.append(accuracy_score(y_oos, y_pred_h))
    results_dict = {'plots': plt_all, 'plots_h': plt_h_all,
                    'fit': fit_all, 'fit_h': fit_h_all
                    }

    results_dict['fit_title'] = ('R squared' if tree_type == 'regression'
                                 else 'Accuracy score')
    txt = '\n' * 2 + '-' * 100 + '\nTree based evaluation of ' + title
    txt += '\n' + '- ' * 50
    txt += '\nOut-of-sample fit for standard & honest tree '
    txt += '(' + results_dict['fit_title'] + ')'
    txt += '\n' + '- ' * 50
    txt += '\nDepth     Standard tree      Honest tree'
    for idx, depth in enumerate(depth_grid):
        txt += f'\n{depth:5}' + ' ' * 5 + f'{results_dict["fit"][idx]:13.2%}'
        txt += ' ' * 9 + f'{results_dict["fit_h"][idx]:8.2%}'
    return results_dict, txt


def tree_gets_honest(tree_inst, x_dat, y_dat, tree_type):
    """Modify the tree instance w.r.t. leave predictins."""
    # Apply the tree to the leaves data to get the indices of the leaf for
    # each observation
    tree_h = deepcopy(tree_inst)
    leaf_indices = tree_h.apply(x_dat)
    # Map each leaf index to the corresponding outputs and calculate mean
    leaf_values = {}
    leaf_samples = {}
    for leaf in np.unique(leaf_indices):
        mask = leaf_indices == leaf
        # Update value with mean of samples falling into the leaf

        if tree_type == 'regression':
            leaf_values[leaf] = np.mean(y_dat[mask])
        else:
            leaf_values[leaf] = mode(y_dat[mask])[0]
        if np.isnan(leaf_values[leaf]):
            leaf_values[leaf] = np.mean(y_dat)
        # Update samples with count of samples falling into the leaf
        leaf_samples[leaf] = np.sum(mask)

    # Modify tree attributes for each leaf node
    for leaf in np.where(tree_h.tree_.children_left == -1)[0]:  # Find allleaves
        if leaf in leaf_values:
            # Set the node value to the new calculated mean
            tree_h.tree_.value[leaf][0][0] = leaf_values[leaf]
            # Set the samples to the count of samples in that leaf
            tree_h.tree_.n_node_samples[leaf] = leaf_samples[leaf]

    return tree_h


def plot_tree_fct(tree_inst, feature_names, depth, honest=False, y_name=''):
    """Plot the decision tree and return plot instance."""
    # Set a large figure size to accommodate the full tree
    fig, ax = plt.subplots(figsize=(20, 12))
    tree.plot_tree(tree_inst, filled=False, feature_names=feature_names,
                   rounded=True, ax=ax)
    title = f'Decision Tree Structure for {y_name}: Depth {depth}'
    if honest:
        title = 'Honest ' + title
        fig.text(0.5, 0.01,
                 'Note: Only terminal leaves are updated with honest data.',
                 ha='center', va='bottom', fontsize=12, color='black')
    ax.set_title(title)
    return fig
