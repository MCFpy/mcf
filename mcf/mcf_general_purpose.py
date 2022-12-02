"""General purpose procedures.

Procedures mainly used by the MCF.

# -*- coding: utf-8 -*-.
Created on Thu Apr  2 17:55:24 2020

@author: MLechner
"""
import math

import scipy.stats as sct
import pandas as pd
import numpy as np
import psutil

from mcf import general_purpose as gp
from mcf import general_purpose_system_files as mcf_sys


def stars(pval):
    """Create string with stars for p-values."""
    if pval < 0.001:
        return '****'
    if pval < 0.01:
        return ' ***'
    if pval < 0.05:
        return '  **'
    if pval < 0.1:
        return '   *'
    return '    '


def find_precision(values):
    """Find precision so that all values can be differentiated in printing."""
    len_v = len(np.unique(values))
    precision = 20
    for prec in range(20):
        rounded = np.around(values, decimals=prec)
        if len(set(rounded)) == len_v:  # all unique
            precision = prec            # + 2
            break
    return precision


def print_effect_z(g_r, gm_r, z_values, gate_str, print_output=True,
                   gates_minus_previous=False):
    """Print treatment effects."""
    no_of_effect_per_z = np.size(g_r[0][0])
    if gates_minus_previous:
        print_str = ('- ' * 40 + f'\n                   {gate_str}'
                     + f'                                {gate_str}(change)')
    else:    
        print_str = ('- ' * 40 + f'\n                   {gate_str}'
                     + f'                                {gate_str} - ATE')
    print_str += ('\nComparison      Z      Est         SE  t-val   p-val'
                  + '         Est        SE  t-val  p-val\n' + '- ' * 40
                  + '\n')
    prec = find_precision(z_values)
    if prec == 0:
        z_values = gp.recode_if_all_prime(z_values.copy())
    for j in range(no_of_effect_per_z):
        for zind, z_val in enumerate(z_values):
            treat_s = f'{g_r[zind][4][j][0]:<3} vs {g_r[zind][4][j][1]:>3}'
            val_s = f'{z_val:>7.{prec}f}'
            estse_s = f'{g_r[zind][0][j]:>9.5f}  {g_r[zind][1][j]:>9.5f}'
            t_p_s = f'{g_r[zind][2][j]:>6.2f}  {g_r[zind][3][j]:>6.2%}'
            s_s = stars(g_r[zind][3][j])
            estsem_s = f'{gm_r[zind][0][j]:>9.5f}  {gm_r[zind][1][j]:>9.5f}'
            tm_p_s = f'{gm_r[zind][2][j]:>6.2f}  {gm_r[zind][3][j]:>6.2%}'
            sm_s = stars(gm_r[zind][3][j])
            print_str += (treat_s + val_s + estse_s + t_p_s + s_s + estsem_s
                          + tm_p_s + sm_s + '\n')
        if j < no_of_effect_per_z-1:
            print_str += '- ' * 40 + '\n'
    print_str += '-' * 80
    print_str += ('\nValues of Z may give the order of values'
                  + ' (starting with 0).')
    print_str += '\n' + '-' * 80
    if print_output:
        print(print_str)
    return print_str


def effect_from_potential(pot_y, pot_y_var, d_values,
                          se_yes=True, continuous=False):
    """Compute effects and stats from potential outcomes.

    Parameters
    ----------
    pot_y_ao : Numpy array. Potential outcomes.
    pot_y_var_ao : Numpy array. Variance of potential outcomes.
    d_values : List. Treatment values.
    se_yes : Bool. Compuite standard errors. Default is True.
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
                stderr[j] = np.sqrt(pot_y_var[jnd] + pot_y_var[idx])
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


def statistics_by_treatment(indatei, treat_name, var_name, only_next=False):
    """Descriptive statistics by treatment status.

    Parameters
    ----------
    indatei : String. Input data
    treat_name : String. Name of treatment
    var_name : List of strings. Name of variables to describe
    only_next: Bool. Compare only subsequent treatment pairs

    No Returns
    """
    print('\n-------------- Statistics by treatment status ------------------')
    data = pd.read_csv(filepath_or_buffer=indatei, header=0)
    data = data[treat_name+var_name]
    mean = data.groupby(treat_name).mean(numeric_only=True)
    std = data.groupby(treat_name).std()
    count = data.groupby(treat_name).count()
    count2 = data[treat_name+[var_name[0]]].groupby(treat_name).count()
    with pd.option_context(
            'display.max_rows', 500, 'display.max_columns', 500,
            'display.expand_frame_repr', True, 'display.width', 150,
            'chop_threshold', 1e-13):
        print('\nNumber of observations:')
        print(count2.transpose())
        print('\nMean')
        print(mean.transpose())
        print('\nMedian')
        print(data.groupby(treat_name).median().transpose())
        print('\nStandard deviation')
        print(std.transpose())
        balancing_tests(mean, std, count, only_next)


def balancing_tests(mean, std, count, only_next=False):
    """Compute balancing tests.

    Parameters
    ----------
    mean : Dataframe: Means by treatment groups.
    std : Dataframe: Standard deviation by treatment groups.
    count: Dataframe: obs in treatment
    only_next: Bool. Compare only subsequent treatment pairs

    No Returns.

    """
    std = std.replace(to_replace=0, value=-1)
    value_of_treat = list(reversed(mean.index))
    value_of_treat2 = value_of_treat[:]
    for i in value_of_treat:
        if i >= value_of_treat[-1]:
            value_of_treat2.remove(i)
            for j in value_of_treat2:
                mean_diff = mean.loc[i, :] - mean.loc[j, :]
                std_diff = np.sqrt((std.loc[i, :]**2) / count.loc[i]
                                   + (std.loc[j, :]**2) / count.loc[j])
                t_diff = mean_diff.div(std_diff).abs()
                p_diff = 2 * sct.norm.sf(t_diff) * 100
                stand_diff = (mean_diff / np.sqrt(
                    (std.loc[i, :]**2 + std.loc[j, :]**2) / 2) * 100)
                stand_diff.abs()
                print(f'\nComparing treatments {i:>2} and {j:>2}')
                print('Variable                          Mean       Std',
                      '        t-val   p-val (%) Stand.Difference (%)')
                for jdx, _ in enumerate(mean_diff):
                    print(f'{mean_diff.index[jdx]:30} {mean_diff[jdx]:10.5f}',
                          f'{std_diff[jdx]:10.5f} {t_diff[jdx]:9.2f}',
                          f'{p_diff[jdx]:9.2f} {stand_diff[jdx]:9.2f}')
                if only_next:
                    break


def print_size_weight_matrix(weights, weight_as_sparse, no_of_treat):
    """
    Print size of weight matrix in MB.

    Parameters
    ----------
    weights : Sparse (CSR) or dense 2D Numpy array. Weight matrix.
    weight_as_sparse : Boolean.
    no_of_treat : Int. Number of treatments.

    Returns
    -------
    None.

    """
    total_bytes = mcf_sys.total_size(weights)
    if weight_as_sparse:
        for d_idx in range(no_of_treat):
            total_bytes += (weights[d_idx].data.nbytes
                            + weights[d_idx].indices.nbytes
                            + weights[d_idx].indptr.nbytes)
    print('Size of weight matrix: ', round(total_bytes / (1024 * 1024), 2),
          ' MB')


def no_of_boot_splits_fct(size_of_object_mb, workers, with_output=True):
    """
    Compute size of chunks for MP.

    Parameters
    ----------
    size_of_forest_MB : Float. Size of the object in MB.
    workers : Int. Number of workers in MP.
    with_output : Boolean. optional. Print or not. The default is True.

    Returns
    -------
    no_of_splits : Int. Number of splits.

    """
    basic_size_mb = 53
    _, available, _, _ = mcf_sys.memory_statistics(with_output=False)
    if size_of_object_mb > basic_size_mb:
        multiplier = 1/8 * (14 / workers)
        chunck_size_mb = basic_size_mb * (1 + (available - 33000) / 33000
                                          * multiplier)
        chunck_size_mb = min(chunck_size_mb, 2000)
        chunck_size_mb = max(chunck_size_mb, 10)
        no_of_splits = math.ceil(size_of_object_mb / chunck_size_mb)
    else:
        no_of_splits = 1
        chunck_size_mb = size_of_object_mb
    if with_output:
        print()
        print('Automatic determination of tree batches')
        print(f'Size of object:   {round(size_of_object_mb, 2):6} MB ',
              f'Available RAM: {available:6} MB ',
              f'Number of workers {workers:2} No of splits: {no_of_splits:2} ',
              f'Size of chunk:  {round(chunck_size_mb, 2):6} MB ')
        mcf_sys.memory_statistics()
    return no_of_splits


def bound_norm_weights(weight, max_weight, renormalize=True):
    """Bound and renormalized weights.

    Parameters
    ----------
    weight : 1d Numpy array. Weights.
    max_weight : Scalar Float. Maximum value of any weight
    renormalize : Boolean, optional. If True renormalize the weights that they
               add to 1. The default is True.

    Returns
    -------
    weight_norm : Numpy array of same size as input. Normalized weights.
    no_censored: NP float. Number of censored observations.
    share_censored: NP float. Share of censored observations (0-1).

    """
    weight_norm = weight.flatten()
    too_large = (weight + 1e-15) > max_weight
    if np.any(too_large):
        no_censored = np.count_nonzero(too_large)
        weight_norm[too_large] = max_weight
    else:
        no_censored = 0
    share_censored = no_censored / len(weight)
    if renormalize:
        sum_w = np.sum(weight_norm)
        if not ((-1e-10 < sum_w < 1e-10) or (1-1e-10 < sum_w < 1+1e-10)):
            weight_norm = weight_norm / sum_w
    return weight_norm, no_censored, share_censored


def find_no_of_workers(maxworkers, sys_share=0):
    """
    Find the optimal number of workers for MP such that system does not crash.

    Parameters
    ----------
    maxworkers : Int. Maximum number of workers allowed.

    Returns
    -------
    workers : Int. Workers used.
    sys_share: Float. System share.
    max_cores: Bool. Limit to number of physical(not logical cores)

    """
    share_used = getattr(psutil.virtual_memory(), 'percent') / 100
    if sys_share >= share_used:
        sys_share = 0.9 * share_used
    sys_share = sys_share / 2
    workers = (1-sys_share) / (share_used-sys_share)
    if workers > maxworkers:
        workers = maxworkers
    elif workers < 1.9:
        workers = 1
    else:
        workers = maxworkers
    workers = math.floor(workers + 1e-15)
    return workers
