"""
Contains functions for the estimation of various effect.

Created on Mon Jun 19 17:50:33 2023.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from math import sqrt

import torch

from mcf import mcf_cuda_functions as mcf_c
from mcf import mcf_print_stats_functions as ps


def weight_var_cuda(w0_dat, y0_dat, cl_dat, gen_dic, p_dic, norm=True,
                    w_for_diff=None, weights=None, bootstrap=0,
                    keep_some_0=False, se_yes=True, keep_all=False,
                    precision=32):
    """Generate the weight-based variance (cuda version).

    Parameters
    ----------
    w_dat : Tensor. Weights.
    y_dat : Tensor. Outcomes.
    cl_dat : Tensor. Cluster indicator.
    p_dic : Dict. Parameters.
    norm : Boolean. Normalisation. Default is True.
    w_for_diff : Tensor. Weights used for difference when clustering.
                 Default is None.
    weights : Tensor. Sampling weights. Clustering only. Default is None.
    no_agg :   Boolean. No aggregation of weights. Default is False.
    bootstrap: Int. If > 1: Use bootstrap instead for SE estimation.
    keep_some_0, se_yes, keep_all: Booleans.

    Returns
    -------
    est, variance, w_ret: Tensors.
    """
    w_dat, y_dat = w0_dat.clone(), y0_dat.clone()
    if p_dic['cluster_std'] and (cl_dat is not None) and (bootstrap < 1):
        if not gen_dic['weighted']:
            weights = None
        w_dat, y_dat, _, _ = aggregate_cluster_pos_w_cuda(
            cl_dat, w_dat, y_dat, norma=norm, sweights=weights,
            precision=precision)
        if w_for_diff is not None:
            w_dat = w_dat - w_for_diff
    if not p_dic['iate_se']:
        keep_some_0, bootstrap = False, 0
    if norm:
        sum_w_dat = torch.abs(torch.sum(w_dat))
        if not ((-1e-15 < sum_w_dat < 1e-15)
                or (1-1e-10 < sum_w_dat < 1+1e-10)):
            w_dat = w_dat / sum_w_dat
    w_ret = w_dat.clone()
    if keep_all:
        w_pos = torch.ones_like(w_dat, dtype=mcf_c.tdtype('bool'),
                                device='cuda')
    else:
        w_pos = torch.abs(w_dat) > 1e-15  # use non-zero only to speed up
    only_copy = torch.all(w_pos)
    if keep_some_0 and not only_copy:  # to improve variance estimate
        sum_pos = torch.sum(w_pos)
        obs_all = len(w_dat)
        sum_0 = obs_all - sum_pos
        zeros_to_keep = 0.05 * obs_all  # keep to 5% of all obs as zeros
        zeros_to_switch = torch.round(sum_0 - zeros_to_keep)
        if zeros_to_switch <= 2:
            only_copy = True
        else:
            ind_of_0 = torch.where(w_pos is False)
            g_cuda = torch.Generator(device='cuda')
            g_cuda.manual_seed(123345)
            # ind_to_true = rng.choice(
            #     ind_of_0[0], size=zeros_to_switch, replace=False)
            # Perform random sampling without replacement in PyTorch
            random_indices = torch.randperm(len(ind_of_0), generator=g_cuda
                                            )[:zeros_to_switch]
            # Select corresponding indices from ind_of_0
            ind_to_true = ind_of_0[random_indices]
            w_pos[ind_to_true] = torch.bitwise_not(w_pos[ind_to_true])
    if only_copy:
        w_dat2 = w_dat.clone()
    else:
        w_dat2, y_dat = w_dat[w_pos], y_dat[w_pos]
    obs = len(w_dat2)
    if obs < 5:
        return 0, 1, w_ret
    est = torch.dot(w_dat2, y_dat.reshape(-1))
    if se_yes:
        if bootstrap > 1:
            if p_dic['cluster_std'] and (cl_dat is not None) and not only_copy:
                cl_dat = cl_dat[w_pos]
                unique_cl_id = torch.unique(cl_dat)
                obs_cl = len(unique_cl_id)
                cl_dat = torch.round(cl_dat)
            g_cuda = torch.Generator(device='cuda')
            g_cuda.manual_seed(123345)
            est_b = torch.empty(bootstrap, device='cuda',
                                dtype=mcf_c.tdtype('float', precision))
            for b_idx in range(bootstrap):
                if p_dic['cluster_std'] and (
                        cl_dat is not None and not only_copy):
                    # block bootstrap
                    # idx_cl = rng.integers(0, high=obs_cl, size=obs_cl)
                    idx_cl = torch.randint(0, high=obs_cl, size=(obs_cl,),
                                           generator=g_cuda)
                    cl_boot = unique_cl_id[idx_cl]  # relevant indices
                    idx = []
                    for cl_i in torch.round(cl_boot):
                        select_idx = cl_dat == cl_i
                        idx_cl_i = torch.where(select_idx)
                        idx.extend(idx_cl_i[0])
                else:
                    # idx = rng.integers(0, high=obs, size=obs)
                    idx_cl = torch.randint(0, high=obs, size=(obs,),
                                           generator=g_cuda)
                w_b = w_dat2[idx].clone()
                if norm:
                    sum_w_b = torch.abs(torch.sum(w_b))
                    if not ((-1e-15 < sum_w_dat < 1e-15)
                            or (1-1e-10 < sum_w_dat < 1+1e-10)):
                        w_b = w_b / sum_w_b
                est_b[b_idx] = torch.dot(w_b, y_dat[idx])
            variance = torch.var(est_b)
        else:
            if p_dic['cond_var']:
                sort_ind = torch.argsort(w_dat2)
                y_s, w_s = y_dat[sort_ind], w_dat2[sort_ind]
                if p_dic['knn']:
                    k = int(round(p_dic['knn_const'] * sqrt(obs) * 2))
                    if k < p_dic['knn_min_k']:
                        k = p_dic['knn_min_k']
                    if k > obs / 2:
                        k = torch.floor(obs / 2)
                    exp_y_cond_w, var_y_cond_w = moving_avg_mean_var_cuda(
                        y_s, k, precision=precision)
                else:
                    band = (bandwidth_nw_rule_of_thumb_cuda(w_s)
                            * p_dic['nw_bandw'])
                    exp_y_cond_w = nadaraya_watson_cuda(
                        y_s, w_s, w_s, p_dic['nw_kern'], band)
                    var_y_cond_w = nadaraya_watson_cuda(
                        (y_s - exp_y_cond_w)**2, w_s, w_s, p_dic['nw_kern'],
                        band)
                variance = torch.dot(w_s**2, var_y_cond_w) + obs * torch.var(
                    w_s * exp_y_cond_w)
            else:
                variance = len(w_dat2) * torch.var(w_dat2 * y_dat)
    else:
        variance = None
    return est, variance, w_ret


def aggregate_cluster_pos_w_cuda(cl_dat, w_dat, y_dat=None, norma=True,
                                 w2_dat=None, sweights=None, y2_compute=False,
                                 precision=32):
    """Aggregate weighted cluster means (cuda version).

    Parameters
    ----------
    cl_dat : Tensor. Cluster indicator.
    w_dat : Tensor. Weights.
    y_dat : Tensor. Outcomes.
    ...

    Returns
    -------
    w_agg : Tensor. Aggregated weights. Normalised to one.
    y_agg : Tensor. Aggregated outcomes.
    w_agg2 : Tensor. Aggregated weights. Normalised to one.
    y_agg2 : Tensor. Aggregated outcomes.
    """
    cluster_no = torch.unique(cl_dat)
    no_cluster = len(cluster_no)
    w_pos = torch.abs(w_dat) > 1e-15
    if y_dat is not None:
        if y_dat.ndim == 1:
            y_dat = torch.reshape(y_dat, (-1, 1))
        q_obs = torch.size(y_dat, axis=1)
        y_agg = torch.zeros((no_cluster, q_obs), device='cuda',
                            dtype=mcf_c.tdtype('float', precision))
    else:
        y_agg = None
    y2_agg = y_agg.clone() if y2_compute else None
    w_agg = torch.zeros(no_cluster, device='cuda',
                        dtype=mcf_c.tdtype('float', precision))
    if w2_dat is not None:
        w2_agg = torch.zeros(no_cluster, device='cuda',
                             dtype=mcf_c.tdtype('float', precision))
        w2_pos = torch.abs(w2_dat) > 1e-15
    else:
        w2_agg = None
    for j, cl_ind in enumerate(cluster_no):
        in_cluster = (cl_dat == cl_ind).reshape(-1)
        in_cluster_pos = in_cluster & w_pos
        if y2_compute:
            in_cluster_pos2 = in_cluster & w2_pos
        if torch.any(in_cluster_pos):
            w_agg[j] = torch.sum(w_dat[in_cluster_pos])
            if w2_dat is not None:
                w2_agg[j] = torch.sum(w2_dat[in_cluster])
            if (y_dat is not None) and torch.any(in_cluster_pos):
                for odx in range(q_obs):
                    if sweights is None:
                        y_agg[j, odx] = (torch.dot(
                            w_dat[in_cluster_pos], y_dat[in_cluster_pos, odx])
                            / w_agg[j])
                        if y2_compute:
                            y2_agg[j, odx] = (torch.dot(
                                w2_dat[in_cluster_pos2],
                                y_dat[in_cluster_pos2, odx]) / w2_agg[j])
                    else:
                        y_agg[j, odx] = (torch.dot(
                            w_dat[in_cluster_pos] *
                            sweights[in_cluster_pos].reshape(-1),
                            y_dat[in_cluster_pos, odx]) / w_agg[j])
                        if y2_compute:
                            y2_agg[j, odx] = (torch.dot(
                                w2_dat[in_cluster_pos2] *
                                sweights[in_cluster_pos2].reshape(-1),
                                y_dat[in_cluster_pos2, odx]) / w2_agg[j])
    if norma:
        sum_w_agg = torch.sum(w_agg)
        if not 1-1e-10 < sum_w_agg < 1+1e-10:
            w_agg = w_agg / sum_w_agg
        if w2_dat is not None:
            sum_w2_agg = torch.sum(w2_agg)
            if not 1-1e-10 < sum_w2_agg < 1+1e-10:
                w2_agg = w2_agg / sum_w2_agg
    return w_agg, y_agg, w2_agg, y2_agg


def moving_avg_mean_var_cuda_alternative(data, k, mean_and_var=True,
                                         precision=32):
    """Compute moving average of mean and std deviation (cuda version).

    Parameters
    ----------
    data : Tensor. Dependent variable. Sorted.
    k : Int. Number of neighbours to be considered.

    Returns
    -------
    mean : Tensor.
    var : Tensor.
    """
    obs = len(data)
    k = int(k)
    if k >= obs:
        mean = torch.full(obs, torch.mean(data), device='cuda',
                          dtype=mcf_c.tdtype('float', precision))
        if mean_and_var:
            var = torch.full(obs, torch.var(data), device='cuda',
                             dtype=mcf_c.tdtype('float', precision))
    else:
        weights = torch.ones(k, device='cuda',
                             dtype=mcf_c.tdtype('float', precision)) / k

        # Compute the rolling mean
        # mean = np.convolve(data, weights, mode='valid')
        mean = torch.nn.functional.conv1d(
            data.view(1, 1, -1), weights.view(1, 1, -1), padding=0)
        # Compute the rolling variance if needed
        if mean_and_var:
            data_s = data ** 2
            # mean_s = np.convolve(data_s, weights, mode='valid')
            mean_s = torch.nn.functional.conv1d(
                data_s.view(1, 1, -1), weights.view(1, 1, -1), padding=0)
            var = mean_s - mean**2

        # Pad the results to match the length of the original data
        pad_before = (obs - len(mean)) // 2
        pad_after = obs - len(mean) - pad_before

        mean = torch.nn.functional.pad(mean, (pad_before, pad_after),
                                       mode='replicate')

        if mean_and_var:
            var = torch.nn.functional.pad(var, (pad_before, pad_after),
                                          mode='replicate')
        else:
            var = None

    return mean, var


def moving_avg_mean_var_cuda(data, k, mean_and_var=True, precision=32):
    """Compute moving average of mean and std deviation (cuda version).

    Parameters
    ----------
    data_in: Tensor. Dependent variable. Sorted.
    k: Int. Number of neighbours to be considered.
    mean_and_var: Bool. Whether to compute both mean and variance.

    Returns
    -------
    mean: Tensor.
    var: Tensor if mean_and_var is True, None otherwise.
    """
    obs = len(data)
    k = int(k)

    if k >= obs:
        mean = torch.full((obs,), torch.mean(data), device='cuda',
                          dtype=mcf_c.tdtype('float', precision))
        var = torch.full_like(mean) if mean_and_var else None
    else:
        # Compute the rolling mean
        cumsum = torch.cumsum(data, dim=0)
        cumsum[k:] = cumsum[k:] - cumsum[:-k]
        mean = cumsum[k - 1:] / k

        # Compute the rolling variance if needed
        if mean_and_var:
            data_s = data ** 2
            cumsum_s = torch.cumsum(data_s, dim=0)
            cumsum_s[k:] = cumsum_s[k:] - cumsum_s[:-k]
            mean_s = cumsum_s[k - 1:] / k
            var = mean_s - mean ** 2
        else:
            var = None

        # Pad the results to match the length of the original data
        pad_before = (obs - len(mean)) // 2
        pad_after = obs - len(mean) - pad_before

        mean = torch.cat([mean[0].view(1).repeat(pad_before), mean,
                          mean[-1].view(1).repeat(pad_after)])

        if mean_and_var:
            var = torch.cat([var[0].view(1).repeat(pad_before), var,
                             var[-1].view(1).repeat(pad_after)])

    return mean, var


def bandwidth_nw_rule_of_thumb_cuda(data):
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
    iqr = torch.subtract(*torch.percentile(data, [75, 25])) / 1.349
    std = torch.std(data)
    sss = min(std, iqr) if iqr > 1e-15 else std
    bandwidth = sss * (obs ** (-0.2))
    if bandwidth < 1e-15:
        bandwidth = 1
    return bandwidth


def nadaraya_watson_cuda(y_dat, x_dat, grid, kernel, bandwidth):
    """Compute Nadaraya-Watson one dimensional nonparametric regression (cuda).

    Parameters
    ----------
    y_dat : Tensor. Dependent variable.
    x_dat :  Tensor. Independent variable.
    grid : Tensor. Values of x for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    estimate : Tensor. Estimated quantity.

    """
    f_yx = kernel_density_y_cuda(y_dat, x_dat, grid, kernel, bandwidth)
    f_x = kernel_density_cuda(x_dat, grid, kernel, bandwidth)
    return f_yx / f_x


def kernel_density_cuda(data, grid, kernel, bandwidth):
    """Compute nonparametric estimate of density of data (cuda version).

    Parameters
    ----------
    data : Tensor. Dependent variable.
    grid : Tensor. Values for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    f_grid : Tensor. Prediction.

    """
    differ = torch.subtract.outer(data, grid)  # This builds a matrix
    y_dach_i = kernel_proc_cuda(differ / bandwidth, kernel)
    f_grid = torch.mean(y_dach_i, axis=0) / bandwidth
    return f_grid


def kernel_density_y_cuda(y_dat, x_dat, grid, kernel, bandwidth):
    """Compute nonparametric estimate of density of data * y (cuda version).

    Parameters
    ----------
    y_dat : Tensor. Dependent variable.
    x_dat : Tensor. Independent variable.
    grid : Tensor. Values for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    f_grid : Tensor. Prediction.

    """
    # differ = np.subtract.outer(x_dat, grid)  # This builds a matrix
    differ = x_dat.unsqueeze(1) - grid.unsqueeze(0)
    y_dach_i = kernel_proc_cuda(differ / bandwidth, kernel)
    if y_dat.ndim == 2:
        f_grid = torch.mean(y_dach_i * y_dat, axis=0) / bandwidth
    else:
        f_grid = torch.mean(y_dach_i * torch.reshape(y_dat, (len(grid), 1)),
                            axis=0) / bandwidth
    return f_grid


def kernel_proc_cuda(data, kernel):
    """Feed data through kernel for nonparametric estimation (cuda version).

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
        abs_data = torch.abs(data)
        y_dat = torch.where(abs_data < 1, 3/4 * (1 - abs_data**2), 0)
    elif kernel == 2:  # This works for matrices
        normal_distribution = torch.distributions.Normal(0, 1)
        y_dat = torch.exp(normal_distribution.log_prob(data))
    else:
        raise ValueError('Only Epanechikov and normal kernel supported.')
    return y_dat


def analyse_weights_cuda(weights, title, gen_dic, p_dic, ate=True,
                         continuous=False, no_of_treat_cont=None,
                         d_values_cont=None, precision=32):
    """Describe the weights (cuda version).

    Parameters
    ----------
    weights : Tensor. Weights.
    title : String. Title for output.
    gen_dic, p_dic : Dict. Parameters.
    ate: Boolean. True if Ate is estimated. Default is True.
    continuous : Boolean. Continuous treatment. Default is False.
    no_of_treat_cont : Int. Number of discretized treatments of continuous
                            treatments used for weights. Default is None.
    d_values_cont : Tensor. Values of discretized treatments of continuous
                                 treatments used for weights. Default is None.

    Returns
    -------
    larger_0 : Tensor.
    equal_0 : Tensor.
    mean_pos : Tensor.
    std_pos : Tensor.
    gini_all : Tensor.
    gini_pos : Tensor.
    share_largest_q : Tensor.
    sum_larger : Tensor.
    obs_larger : Tensor.
    txt : String. Text to print.

    """
    txt = ''
    if ate:
        txt += '\n' * 2 + '=' * 100
        txt += '\nAnalysis of weights (normalised to add to 1): ' + title
    no_of_treat = no_of_treat_cont if continuous else gen_dic['no_of_treat']
    larger_0 = torch.empty(no_of_treat, device='cuda',
                           dtype=mcf_c.tdtype('int', precision))
    equal_0 = torch.empty_like(larger_0)
    mean_pos = torch.empty(no_of_treat, device='cuda',
                           dtype=mcf_c.tdtype('float', precision))
    std_pos, gini_all = torch.empty_like(mean_pos), torch.empty_like(mean_pos)
    gini_pos = torch.empty_like(mean_pos)
    share_largest_q = torch.empty((no_of_treat, 3), device='cuda',
                                  dtype=mcf_c.tdtype('float', precision))
    sum_larger = torch.empty((no_of_treat, len(p_dic['q_w'])), device='cuda',
                             dtype=mcf_c.tdtype('float', precision))
    obs_larger = torch.empty_like(sum_larger)
    sum_weights = torch.sum(weights, axis=1)
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
        mean_pos[j], std_pos[j] = torch.mean(w_pos), torch.std(w_pos)
        gini_all[j] = gini_coeff_pos_cuda(w_j) * 100
        gini_pos[j] = gini_coeff_pos_cuda(w_pos) * 100
        if n_pos > 5:
            quantiles = torch.tensor([0.99, 0.95, 0.9],  device='cuda')
            qqq = torch.quantile(w_pos, quantiles)
            for i in range(3):
                share_largest_q[j, i] = torch.sum(
                    w_pos[w_pos >= (qqq[i] - 1e-15)]) * 100
            for idx, val in enumerate(p_dic['q_w']):
                sum_larger[j, idx] = torch.sum(
                    w_pos[w_pos >= (val - 1e-15)]) * 100
                obs_larger[j, idx] = len(
                    w_pos[w_pos >= (val - 1e-15)]) / n_pos * 100
        else:
            share_largest_q = torch.empty(
                (no_of_treat, 3), device='cuda',
                dtype=mcf_c.tdtype('float', precision))
            sum_larger = torch.zeros(
                (no_of_treat, len(p_dic['q_w'])), device='cuda',
                dtype=mcf_c.tdtype('float', precision))
            obs_larger = torch.zeros_like(sum_larger)
            if gen_dic['with_output']:
                txt += '\nLess than 5 observations in some groups.'
    if ate:
        txt += ps.txt_weight_stat(
            larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger, gen_dic, p_dic,
            continuous=continuous, d_values_cont=d_values_cont)
    return (larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger, txt)


def gini_coeff_pos_cuda(x_dat):
    """Compute Gini coefficient of numpy array of values with values >= 0."""
    sss = x_dat.sum()
    if sss > 1e-15:                        # Use 'mergesort' for stable sorting
        rrr = torch.argsort(-x_dat)
        ranks = torch.arange(1, len(x_dat) + 1, device='cuda')
        gini = 1 - 2 * (torch.sum((ranks - 1) * x_dat[rrr]) + sss) / (
            len(x_dat) * sss)
        return gini
    return 0
