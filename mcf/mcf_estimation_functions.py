"""
Contains functions for the estimation of various effect.

Created on Mon Jun 19 17:50:33 2023.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy

import numpy as np
from scipy.stats import t

from mcf import mcf_print_stats_functions as ps
from mcf import mcf_estimation_generic_functions as mcf_est_g


def effect_from_potential(pot_y, pot_y_var, d_values, se_yes=True,
                          continuous=False, return_comparison=True,
                          sequential=False, sequential_dic=None):
    """Compute effects and stats from potential outcomes.

    Parameters
    ----------
    pot_y_ao : Numpy array. Potential outcomes.
    pot_y_var_ao : Numpy array. Variance of potential outcomes.
    d_values : List. Treatment values.
    se_yes : Bool. Compute standard errors. Default is True.
    continuous: Bool. Continuous treatment. Default is False.
    return_comparison: Bool. Return an array indicating the treatments that are
                             compared. Default is True.

    Returns
    -------
    est : Numpy array. Point estimates.
    se : Numpy array. Standard error.
    t_val : Numpy array. t-value.
    p_val : Numpy array.

    """
    if sequential:
        if continuous:
            raise NotImplementedError('QIATEs are not yet implemented for '
                                      'continuous treatments.')
        else:
            # dim of pot_y: (no_of_comparisons, 2)
            est = pot_y[:, 1] - pot_y[:, 0]
            if se_yes:
                var = pot_y_var[:, 1] + pot_y_var[:, 0]
                stderr, t_val, p_val = compute_inference(est, var)
            else:
                stderr = t_val = p_val = None
            if return_comparison:
                no_of_comparisons = round(len(d_values) * (len(d_values) - 1)
                                          / 2)
                comparison = np.empty((no_of_comparisons, 2), dtype=np.int16)
                for _, values in sequential_dic.items():
                    comparison[values[0], 0] = d_values[values[2]]
                    comparison[values[0], 1] = d_values[values[1]]
            else:
                comparison = None
    else:
        if continuous:
            # This legazy code for the continuous case is not yet optimized
            no_of_comparisons = len(d_values) - 1
            est = np.empty(no_of_comparisons)
            if se_yes:
                var = np.empty_like(est)
            comparison = [None for _ in range(no_of_comparisons)]
            j = 0
            for idx, treat1 in enumerate(d_values):
                for jnd, treat2 in enumerate(d_values):
                    if jnd <= idx:
                        continue
                    est[j] = pot_y[jnd] - pot_y[idx]
                    if se_yes:
                        var[j] = pot_y_var[jnd] + pot_y_var[idx]
                    comparison[j] = [treat2, treat1]
                    j += 1
                    break
            if se_yes:
                stderr, t_val, p_val = compute_inference(est, var)
            else:
                stderr = t_val = p_val = None
        else:  # Optimized for discrete case
            idx, jnd = np.triu_indices(len(d_values), k=1)
            est = pot_y[jnd] - pot_y[idx]
            if se_yes:
                var = pot_y_var[jnd] + pot_y_var[idx]
                stderr, t_val, p_val = compute_inference(est, var)
            else:
                stderr = t_val = p_val = None
            d_values = np.array(d_values)
            no_of_comparisons = round(len(d_values) * (len(d_values) - 1) / 2)
            if return_comparison:
                comparison = np.empty((no_of_comparisons, 2), dtype=np.int16)
                comparison[:, 0] = d_values[jnd]
                comparison[:, 1] = d_values[idx]
            else:
                comparison = None
    return est, stderr, t_val, p_val, comparison


def compute_inference(estimate, variance):
    """Compute inference."""
    # This functions requires that RuntimeWarning are turned to exceptions
    # in the calling code. Otherwise it will not use the except block.
    # It is not done here, because of run-time considerations, as when this
    # function is called in large loops it may not be efficient to change
    # warnings too often.
    constant = 0.000001

    # Check if estimate contains NaNs
    mask_est = np.isnan(estimate)
    if mask_est.any():
        estimate = np.where(mask_est,  0, estimate)

    # Check if variance contains NaNs
    mask_var = np.isnan(variance) | (variance < constant)
    if mask_var.any():
        variance = np.where(mask_var,  constant, variance)

    # In case there are still issues with divisions
    try:
        stderr = np.sqrt(variance)
    except RuntimeWarning:
        if len(estimate) == 1:
            try:
                stderr = 100 * np.abs(estimate + constant)
            except RuntimeWarning:
                stderr = 100
        else:
            stderr = np.zeros_like(estimate)
            for idx, est in enumerate(estimate):
                try:
                    stderr[idx] = np.sqrt(variance[idx])
                except RuntimeWarning:
                    stderr[idx] = 100 * np.abs(estimate[idx] + constant)

    try:
        t_val = np.abs(estimate / stderr)
    except RuntimeWarning:
        if len(estimate) == 1:
            t_val = 0
            try:
                stderr = 100 * np.abs(estimate + constant)
            except RuntimeWarning:
                stderr = 100
        else:
            t_val = np.zeros_like(estimate)
            for idx, est in enumerate(estimate):
                try:
                    t_val[idx] = est / stderr[idx]
                except RuntimeWarning:
                    continue
    p_val = t.sf(t_val, 1000000) * 2
    return stderr, t_val, p_val


def aggregate_pots(mcf_, y_pot_f, y_pot_var_f, txt, effect_dic, fold,
                   pot_is_list=False, title=''):
    """Aggregate the effects from the independent training data folds."""
    first_fold = fold == 0
    last_fold = fold == mcf_.cf_dict['folds'] - 1
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
    if last_fold:   # counting folds starts with 0
        if pot_is_list:
            y_pot = deepcopy(effect_dic['y_pot'])
            effect_dic['y_pot'] = [x / (fold + 1) for x in y_pot]
            if y_pot_var_f is not None:
                y_pot_var = deepcopy(effect_dic['y_pot_var'])
                effect_dic['y_pot_var'] = [
                    x / ((fold + 1) ** 2) for x in y_pot_var]
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


def weight_var(w0_dat, y0_dat, cl_dat, gen_dic, p_dic, normalize=True,
               w_for_diff=None, weights=None, bootstrap=0, keep_some_0=False,
               se_yes=True, keep_all=False):
    """Generate the weight-based variance.

    Parameters
    ----------
    w_dat : Numpy array. Weights.
    y_dat : Numpy array. Outcomes.
    cl_dat : Numpy array. Cluster indicator.
    p_dic : Dict. Parameters.
    normalize : Boolean. Normalisation. Default is True.
    w_for_diff : Numpy array. weights used for difference when clustering.
                 Default is None.
    weights : Numpy array. Sampling weights. Clustering only. Default is None.
    no_agg :   Boolean. No aggregation of weights. Default is False.
    bootstrap: Int. If > 1: Use bootstrap instead for SE estimation.
    keep_some_0, se_yes, keep_all: Booleans.

    Returns
    -------
    est, variance, w_ret
    """
    w_dat, y_dat = w0_dat.copy(), y0_dat.copy()
    if p_dic['cluster_std'] and (cl_dat is not None) and (bootstrap < 1):
        if not gen_dic['weighted']:
            weights = None
        w_dat, y_dat, _, _ = aggregate_cluster_pos_w(
            cl_dat, w_dat, y_dat, norma=normalize, sweights=weights)
        if w_for_diff is not None:
            w_dat = w_dat - w_for_diff
    if not p_dic['iate_se']:
        keep_some_0, bootstrap = False, 0
    if normalize:
        sum_w_dat = np.abs(np.sum(w_dat))
        if not ((-1e-15 < sum_w_dat < 1e-15)
                or (1-1e-10 < sum_w_dat < 1+1e-10)):
            w_dat = w_dat / sum_w_dat
    w_ret = np.copy(w_dat)
    if keep_all:
        w_pos = np.ones_like(w_dat, dtype=bool)
    else:
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
                if normalize:
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
                    if k > obs / 2:
                        k = np.floor(obs / 2)
                    exp_y_cond_w, var_y_cond_w = mcf_est_g.moving_avg_mean_var(
                        y_s, k)
                else:
                    band = (mcf_est_g.bandwidth_nw_rule_of_thumb(w_s)
                            * p_dic['nw_bandw'])
                    exp_y_cond_w = mcf_est_g.nadaraya_watson(
                        y_s, w_s, w_s, p_dic['nw_kern'], band)
                    var_y_cond_w = mcf_est_g.nadaraya_watson(
                        (y_s - exp_y_cond_w)**2, w_s, w_s, p_dic['nw_kern'],
                        band)
                variance = (np.dot(w_s**2, var_y_cond_w)
                            + obs * np.var(w_s * exp_y_cond_w))
            else:
                variance = len(w_dat2) * np.var(w_dat2 * y_dat)
            variance *= len(w_dat2) / (len(w_dat2)-1)  # Finite sample adjustm.
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
                            w_dat[in_cluster_pos]
                            * sweights[in_cluster_pos].reshape(-1),
                            y_dat[in_cluster_pos, odx]) / w_agg[j])
                        if y2_compute:
                            y2_agg[j, odx] = (np.dot(
                                w2_dat[in_cluster_pos2]
                                * sweights[in_cluster_pos2].reshape(-1),
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


def analyse_weights(weights, title, gen_dic, p_dic, ate=True, continuous=False,
                    no_of_treat_cont=None, d_values_cont=None, late=False):
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
        if late:
            txt += '\nAnalysis of weights: ' + title
        else:
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
                or (-1e-15 < sum_weights[j] < 1e-15)) and not late:
            w_j = weights[j] / sum_weights[j]
        else:
            w_j = weights[j]
        w_pos = w_j[np.abs(w_j) > 1e-15]
        n_pos = len(w_pos)
        larger_0[j] = n_pos
        n_all = len(w_j)
        equal_0[j] = n_all - n_pos
        mean_pos[j], std_pos[j] = np.mean(w_pos), np.std(w_pos)
        gini_all[j] = mcf_est_g.gini_coeff_pos(w_j) * 100
        gini_pos[j] = mcf_est_g.gini_coeff_pos(w_pos) * 100
        w_pos = np.abs(w_pos)
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
