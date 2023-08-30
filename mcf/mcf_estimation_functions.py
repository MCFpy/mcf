"""
Contains functions for the estimation of various effect.

Created on Mon Jun 19 17:50:33 2023.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import copy, deepcopy
from math import sqrt, log2

import numpy as np
from numba import njit
import scipy.stats as sct
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from mcf import mcf_print_stats_functions as ps


def effect_from_potential(pot_y, pot_y_var, d_values, se_yes=True,
                          continuous=False):
    """Compute effects and stats from potential outcomes.

    Parameters
    ----------
    pot_y_ao : Numpy array. Potential outcomes.
    pot_y_var_ao : Numpy array. Variance of potential outcomes.
    d_values : List. Treatment values.
    se_yes : Bool. Compute standard errors. Default is True.
    continuous: Bool. Continuous treatment. Default is False.

    Returns
    -------
    est : Numpy array. Point estimates.
    se : Numpy array. Standard error.
    t_val : Numpy array. t-value.
    p_val : Numpy array.

    """
    no_of_comparisons = (len(d_values) - 1 if continuous
                         else round(len(d_values) * (len(d_values) - 1) / 2))
    est = np.empty(no_of_comparisons)
    if se_yes:
        stderr = np.empty_like(est)
    comparison = [None] * no_of_comparisons
    j = 0
    for idx, treat1 in enumerate(d_values):
        for jnd, treat2 in enumerate(d_values):
            if jnd <= idx:
                continue
            est[j] = pot_y[jnd] - pot_y[idx]
            if se_yes:
                var = pot_y_var[jnd] + pot_y_var[idx]
                stderr[j] = np.sqrt(var) if var > 0.0000001 else 0.0000001
            comparison[j] = [treat2, treat1]
            j += 1
        if continuous:
            break
    if se_yes:
        t_val = np.abs(est / stderr)
        p_val = sct.t.sf(t_val, 1000000) * 2
    else:
        stderr = t_val = p_val = None
    return est, stderr, t_val, p_val, comparison


def aggregate_pots(mcf_, y_pot_f, y_pot_var_f, txt, effect_dic, fold,
                   pot_is_list=False, title=''):
    """Aggregate the effects from the independent training data folds."""
    first_fold, last_fold = fold == 0, fold == mcf_.cf_dict['folds'] - 1
    if pot_is_list:
        len_list = len(y_pot_f)
    w_text = f'\n\n{title}: Analysis of weights in fold {fold}\n'
    if first_fold:
        if pot_is_list:
            effect_dic = {'y_pot': [0] * len_list, 'y_pot_var': [0] * len_list,
                          'txt_weights': [''] * len_list}
        else:
            effect_dic = {'y_pot': 0, 'y_pot_var': 0, 'txt_weights': ''}
    if pot_is_list:
        y_pot = deepcopy(effect_dic['y_pot'])
        effect_dic['y_pot'] = [y_pot[idx] + val
                               for idx, val in enumerate(y_pot_f)]
        if y_pot_var_f is not None:
            y_pot_var = deepcopy(effect_dic['y_pot_var'])
            effect_dic['y_pot_var'] = [y_pot_var[idx] + val
                                       for idx, val in enumerate(y_pot_var_f)]
    else:
        effect_dic['y_pot'] += y_pot_f
        if y_pot_var_f is not None:
            effect_dic['y_pot_var'] += y_pot_var_f
    if last_fold:   # counting starts with 0
        if pot_is_list:
            y_pot = deepcopy(effect_dic['y_pot'])
            effect_dic['y_pot'] = [x / (fold + 1) for x in y_pot]
            if y_pot_var_f is not None:
                y_pot_var = deepcopy(effect_dic['y_pot_var'])
                effect_dic['y_pot_var'] = [x / ((fold + 1) ** 2)
                                           for x in y_pot_var]
        else:
            effect_dic['y_pot'] /= (fold + 1)
            if y_pot_var_f is not None:
                effect_dic['y_pot_var'] /= ((fold + 1) ** 2)
    if pot_is_list:
        txt_all = deepcopy(effect_dic['txt_weights'])
        effect_dic['txt_weights'] = [txt_all[idx] + w_text + val
                                     for idx, val in enumerate(txt)]
    else:
        effect_dic['txt_weights'] += w_text + txt
    return effect_dic


def add_with_list(pot_all, pot_add):
    """Add potential to previous if both contained in lists."""
    return [pot_all[idx] + val for idx, val in enumerate(pot_add)]


def weight_var(w0_dat, y0_dat, cl_dat, gen_dic, p_dic, norm=True,
               w_for_diff=None, weights=None, bootstrap=0, keep_some_0=False,
               se_yes=True):
    """Generate the weight-based variance.

    Parameters
    ----------
    w_dat : Numpy array. Weights.
    y_dat : Numpy array. Outcomes.
    cl_dat : Numpy array. Cluster indicator.
    p_dic : Dict. Parameters.
    norm : Boolean. Normalisation. Default is True.
    w_for_diff : Numpy array. weights used for difference when clustering.
                 Default is None.
    weights : Numpy array. Sampling weights. Clustering only. Default is None.
    no_agg :   Boolean. No aggregation of weights. Default is False.
    bootstrap: Int. If > 1: Use bootstrap instead for SE estimation.

    Returns
    -------
    est, variance, w_ret
    """
    w_dat, y_dat = w0_dat.copy(), y0_dat.copy()
    if p_dic['cluster_std'] and (cl_dat is not None) and (bootstrap < 1):
        if not gen_dic['weighted']:
            weights = None
        w_dat, y_dat, _, _ = aggregate_cluster_pos_w(
            cl_dat, w_dat, y_dat, norma=norm, sweights=weights)
        if w_for_diff is not None:
            w_dat = w_dat - w_for_diff
    if not p_dic['iate_se']:
        keep_some_0, bootstrap = False, 0
    if norm:
        sum_w_dat = np.abs(np.sum(w_dat))
        if not ((-1e-15 < sum_w_dat < 1e-15)
                or (1-1e-10 < sum_w_dat < 1+1e-10)):
            w_dat = w_dat / sum_w_dat
    w_ret = np.copy(w_dat)
    w_pos = np.abs(w_dat) > 1e-15  # use non-zero only to speed up
    only_copy = np.all(w_pos)
    if keep_some_0 and not only_copy:  # to improve variance estimate
        sum_pos = np.sum(w_pos)
        obs_all = len(w_dat)
        sum_0 = obs_all - sum_pos
        zeros_to_keep = 0.05 * obs_all  # keep to 5% of all obs as zeros
        zeros_to_switch = round(sum_0 - zeros_to_keep)
        if zeros_to_switch <= 2:
            only_copy = True
        else:
            ind_of_0 = np.where(w_pos is False)
            rng = np.random.default_rng(123345)
            ind_to_true = rng.choice(
                ind_of_0[0], size=zeros_to_switch, replace=False)
            w_pos[ind_to_true] = np.invert(w_pos[ind_to_true])
    if only_copy:
        w_dat2 = w_dat.copy()
    else:
        w_dat2, y_dat = w_dat[w_pos], y_dat[w_pos]
    obs = len(w_dat2)
    if obs < 5:
        return 0, 1, w_ret
    est = np.dot(w_dat2, y_dat)
    if se_yes:
        if bootstrap > 1:
            if p_dic['cluster_std'] and (cl_dat is not None) and not only_copy:
                cl_dat = cl_dat[w_pos]
                unique_cl_id = np.unique(cl_dat)
                obs_cl = len(unique_cl_id)
                cl_dat = np.round(cl_dat)
            rng = np.random.default_rng(123345)
            est_b = np.empty(bootstrap)
            for b_idx in range(bootstrap):
                if p_dic['cluster_std'] and (
                        cl_dat is not None and not only_copy):
                    # block bootstrap
                    idx_cl = rng.integers(0, high=obs_cl, size=obs_cl)
                    cl_boot = unique_cl_id[idx_cl]  # relevant indices
                    idx = []
                    for cl_i in np.round(cl_boot):
                        select_idx = cl_dat == cl_i
                        idx_cl_i = np.nonzero(select_idx)
                        idx.extend(idx_cl_i[0])
                else:
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
            if p_dic['cond_var']:
                sort_ind = np.argsort(w_dat2)
                y_s, w_s = y_dat[sort_ind], w_dat2[sort_ind]
                if p_dic['knn']:
                    k = int(np.round(p_dic['knn_const'] * np.sqrt(obs) * 2))
                    if k < p_dic['knn_min_k']:
                        k = p_dic['knn_min_k']
                    if k > obs/2:
                        k = np.floor(obs/2)
                    exp_y_cond_w, var_y_cond_w = moving_avg_mean_var(y_s, k)
                else:
                    band = bandwidth_nw_rule_of_thumb(w_s) * p_dic['nw_bandw']
                    exp_y_cond_w = nadaraya_watson(y_s, w_s, w_s,
                                                   p_dic['nw_kern'], band)
                    var_y_cond_w = nadaraya_watson((y_s - exp_y_cond_w)**2,
                                                   w_s, w_s, p_dic['nw_kern'],
                                                   band)
                variance = np.dot(w_s**2, var_y_cond_w) + obs * np.var(
                    w_s*exp_y_cond_w)
            else:
                variance = len(w_dat2) * np.var(w_dat2 * y_dat)
    else:
        variance = None
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
    w_agg2 : Numpy array. Aggregated weights. Normalised to one.
    y_agg2 : Numpy array. Aggregated outcomes.
    """
    cluster_no = np.unique(cl_dat)
    no_cluster = len(cluster_no)
    w_pos = np.abs(w_dat) > 1e-15
    if y_dat is not None:
        if y_dat.ndim == 1:
            y_dat = np.reshape(y_dat, (-1, 1))
        q_obs = np.size(y_dat, axis=1)
        y_agg = np.zeros((no_cluster, q_obs))
    else:
        y_agg = None
    y2_agg = np.copy(y_agg) if y2_compute else None
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
    var, obs, k = None, len(data), int(k)
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
            var = mean_s - mean * mean
    return mean, var


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
    ass_str = f'Only {len(data)} observations for bandwidth selection.'
    if obs < 5:
        raise ValueError(ass_str)
    iqr = np.quantile(data, (0.25, 0.75))
    iqr = (iqr[1] - iqr[0]) / 1.349
    std = np.std(data)
    sss = std if (std < iqr) or (iqr < 1e-15) else iqr
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
    iqr = np.quantile(data, (0.25, 0.75))
    iqr = (iqr[1] - iqr[0]) / 1.349
    std = np.std(data)
    sss = std if (std < iqr) or (iqr < 1e-15) else iqr
    band = 1.3643 * (obs ** (-0.2)) * sss
    if kernel not in (1, 2):
        raise ValueError('Wrong type of kernel in Silverman bandwidth')
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
    estimate = f_yx / f_x
    return estimate


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
        y_dat = np.zeros(np.shape(data))   # this works with matrices
        smaller_than_1 = np.abs(data) < 1  # abs() seems to be efficient
        y_dat[smaller_than_1] = 3/4 * (1 - (data[smaller_than_1] ** 2))
    if kernel == 2:  # This works for matrices
        y_dat = sct.norm.pdf(data)
    return y_dat


def gini_coeff_pos(x_dat, len_x):
    """Compute Gini coefficient of numpy array of values with values >= 0."""
    sss = x_dat.sum()
    if sss > 1e-15:
        rrr = np.argsort(np.argsort(-x_dat))  # calculates zero based ranks
        return 1 - (2.0 * (rrr * x_dat).sum() + sss) / (len_x * sss)
    return 0


def analyse_weights(weights, title, gen_dic, p_dic, ate=True, continuous=False,
                    no_of_treat_cont=None, d_values_cont=None):
    """Describe the weights.

    Parameters
    ----------
    weights : Numyp array. Weights.
    title : String. Title for output.
    gen_dic, p_dic : Dict. Parameters.
    ate: Boolean. True if Ate is estimated. Default is True.
    continuous : Boolean. Continuous treatment. Default is False.
    no_of_treat_cont : Int. Number of discretized treatments of continuous
                            treatments used for weights. Default is None.
    d_values_cont : Numpy array. Values of discretized treatments of continuous
                                 treatments used for weights. Default is None.

    Returns
    -------
    larger_0 : Numpy array.
    equal_0 : Numpy array.
    mean_pos : Numpy array.
    std_pos : Numpy array.
    gini_all : Numpy array.
    gini_pos : Numpy array.
    share_largest_q : Numpy array.
    sum_larger : Numpy array.
    obs_larger : Numpy array.
    txt : String. Text to print.

    """
    txt = ''
    if ate:
        txt += '\n' * 2 + '=' * 100
        txt += '\nAnalysis of weights (normalised to add to 1): ' + title
    no_of_treat = no_of_treat_cont if continuous else gen_dic['no_of_treat']
    larger_0 = np.empty(no_of_treat, dtype=np.uint32)
    equal_0, mean_pos = np.empty_like(larger_0), np.empty(no_of_treat)
    std_pos, gini_all = np.empty_like(mean_pos), np.empty_like(mean_pos)
    gini_pos = np.empty_like(mean_pos)
    share_largest_q = np.empty((no_of_treat, 3))
    sum_larger = np.empty((no_of_treat, len(p_dic['q_w'])))
    obs_larger = np.empty_like(sum_larger)
    sum_weights = np.sum(weights, axis=1)
    for j in range(no_of_treat):
        if not (((1 - 1e-10) < sum_weights[j] < (1 + 1e-10))
                or (-1e-15 < sum_weights[j] < 1e-15)):
            w_j = weights[j] / sum_weights[j]
        else:
            w_j = weights[j]
        w_pos = w_j[w_j > 1e-15]
        n_pos = len(w_pos)
        larger_0[j] = n_pos
        n_all = len(w_j)
        equal_0[j] = n_all - n_pos
        mean_pos[j], std_pos[j] = np.mean(w_pos), np.std(w_pos)
        gini_all[j] = gini_coeff_pos(w_j, n_all) * 100
        gini_pos[j] = gini_coeff_pos(w_pos, n_pos) * 100
        if n_pos > 5:
            qqq = np.quantile(w_pos, (0.99, 0.95, 0.9))
            for i in range(3):
                share_largest_q[j, i] = np.sum(w_pos[w_pos >=
                                                     (qqq[i] - 1e-15)]) * 100
            for idx, val in enumerate(p_dic['q_w']):
                sum_larger[j, idx] = np.sum(
                    w_pos[w_pos >= (val - 1e-15)]) * 100
                obs_larger[j, idx] = len(
                    w_pos[w_pos >= (val - 1e-15)]) / n_pos * 100
        else:
            share_largest_q = np.empty((no_of_treat, 3))
            sum_larger = np.zeros((no_of_treat, len(p_dic['q_w'])))
            obs_larger = np.zeros_like(sum_larger)
            if gen_dic['with_output']:
                txt += '\nLess than 5 observations in some groups.'
    if ate:
        txt += ps.txt_weight_stat(
            larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger, gen_dic, p_dic,
            continuous=continuous, d_values_cont=d_values_cont)
    return (larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger, txt)


def waldtest(diff, variance):
    """Compute Wald Chi2 statistics.

    Parameters
    ----------
    diff : 1D Numy array.
    variance : 2D Numpy array.

    Returns
    -------
    stat : Numpy float. Test statistic.
    df : Int. Degrees of freedom.
    p : Numpy float. p-value.

    """
    degfr = np.linalg.matrix_rank(variance)
    stat = pval = -1
    if degfr == len(variance):
        if np.all(np.linalg.eigvals(variance) > 1e-15):
            weight = np.linalg.inv(variance)
            stat = quadratic_form(diff, weight)
            pval = sct.chi2.sf(stat, degfr)
    return stat, degfr, pval


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
    return np.dot(vec, mat @ vec)


def random_forest_scikit(
        x_train, y_train, x_pred, x_name=None, y_name='y', boot=1000,
        n_min=2, no_features='sqrt', workers=-1,  max_depth=None, alpha=0,
        max_leaf_nodes=None, pred_p_flag=True, pred_t_flag=False,
        pred_oob_flag=False, with_output=True, variable_importance=False,
        var_im_groups=None, pred_uncertainty=False, pu_ci_level=0.9,
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
                         Default = 0.9
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
        skewness_res_oob = sct.skew(resid_oob)
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
        txt += ('\nVariable importance statistics in %-point of R2 lost'
                'when omitting particular variable')
        txt += '\n' + '- ' * 50
        txt += ('\nUnordered variables are included (in RF and VI) as dummies'
                ' and the variable itself.')
        if var_im_groups:
            txt += ('\nHere, all elements of these variables are jointly '
                    'tested (in Table under heading of unordered variable).')
        for idx, vim in enumerate(vim_sorted):
            txt += f'\n{x_names_sorted[idx]:<50}: {vim:>7.2%}'
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
