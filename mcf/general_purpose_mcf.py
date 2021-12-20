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
from mcf import general_purpose_system_files as mcf_sys


def print_effect_z(g_r, gm_r, z_values, gate_str):
    """Print treatment effects.

    Parameters
    ----------
    est : Numpy array. Point estimate.
    se : Numpy array. Standard error.
    t_val : Numpy array. t/z-value.
    p_val : Numpy array. p-value.
    effect_list : List of Int. Treatment values involved in comparison.
    add_title: None or string. Additional title.
    add_info: None or Int. Additional information about parameter.

    Returns
    -------
    None.

    """
    no_of_effect_per_z = np.size(g_r[0][0])
    print('- ' * 40)
    print('                   ', gate_str,
          '                                ', gate_str, '- ATE')
    print('Comparison   Z       Est        SE    t-val   p-val        Est',
          '       SE    t-val  p-val')
    print('- ' * 40)
    for j in range(no_of_effect_per_z):
        for zind, z_val in enumerate(z_values):
            print('{:<3} vs {:>3}'.format(g_r[zind][4][j][0],
                                          g_r[zind][4][j][1]), end=' ')
            print('{:>5.1f}'.format(z_val), end=' ')
            print('{:>8.5f}  {:>8.5f}'.format(g_r[zind][0][j],
                                              g_r[zind][1][j]), end=' ')
            print('{:>5.2f}  {:>5.2f}%'.format(g_r[zind][2][j],
                                               g_r[zind][3][j]*100), end=' ')
            if g_r[zind][3][j] < 0.001:
                print('****', end=' ')
            elif g_r[zind][3][j] < 0.01:
                print(' ***', end=' ')
            elif g_r[zind][3][j] < 0.05:
                print('  **', end=' ')
            elif g_r[zind][3][j] < 0.1:
                print('   *', end=' ')
            else:
                print('    ', end=' ')
            print('{:>8.5f}  {:>8.5f}'.format(gm_r[zind][0][j],
                                              gm_r[zind][1][j]), end=' ')
            print('{:>5.2f}  {:>5.2f}%'.format(gm_r[zind][2][j],
                                               gm_r[zind][3][j]*100), end=' ')
            if gm_r[zind][3][j] < 0.001:
                print('****')
            elif gm_r[zind][3][j] < 0.01:
                print(' ***')
            elif gm_r[zind][3][j] < 0.05:
                print('  **')
            elif gm_r[zind][3][j] < 0.1:
                print('   *')
            else:
                print(' ')
        if j < no_of_effect_per_z-1:
            print('- ' * 40)
    print('-' * 80)
    print('Values of Z may have been recoded into primes.')
    print('-' * 80)


def effect_from_potential(pot_y, pot_y_var, d_values, se_yes=True):
    """Compute effects and stats from potential outcomes.

    Parameters
    ----------
    pot_y_ao : Numpy array. Potential outcomes.
    pot_y_var_ao : Numpy array. Variance of potential outcomes.

    Returns
    -------
    est : Numpy array. Point estimates.
    se : Numpy array. Standard error.
    t_val : Numpy array. t-value.
    p_val : Numpy array.

    """
    no_of_comparisons = round(len(d_values) * (len(d_values) - 1) / 2)
    est = np.empty(no_of_comparisons)
    if se_yes:
        stderr = np.empty(no_of_comparisons)
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
            j = j + 1
    if se_yes:
        t_val = np.abs(est / stderr)
        p_val = sct.t.sf(t_val, 1000000) * 2
    else:
        stderr = t_val = p_val = None
    return est, stderr, t_val, p_val, comparison


def statistics_by_treatment(indatei, treat_name, var_name):
    """Descriptive statistics by treatment status.

    Parameters
    ----------
    indatei : String. Input data
    treat_name : String. Name of treatment
    var_name : List of strings. Name of variables to describe

    No Returns
    """
    print('\n-------------- Statistics by treatment status ------------------')
    data = pd.read_csv(filepath_or_buffer=indatei, header=0)
    data = data[treat_name+var_name]
    mean = data.groupby(treat_name).mean()
    std = data.groupby(treat_name).std()
    count = data.groupby(treat_name).count()
    count2 = data[treat_name+[var_name[0]]].groupby(treat_name).count()
    pd.set_option('display.max_rows', len(data.columns),
                  'display.max_columns', 10)
    print()
    print('Number of observations:')
    print(count2.transpose())
    print('\nMean')
    print(mean.transpose())
    print('\nMedian')
    print(data.groupby(treat_name).median().transpose())
    print('\nStandard deviation')
    print(std.transpose())
    balancing_tests(mean, std, count)


def balancing_tests(mean, std, count):
    """Compute balancing tests.

    Parameters
    ----------
    mean : Dataframe: Means by treatment groups.
    std : Dataframe: Standard deviation by treatment groups.

    No Returns.

    """
    std = std.replace(to_replace=0, value=-1)
    # no_of_treat = mean.shape[0]
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
                stand_diff = (mean_diff / np.sqrt((std.loc[i, :]**2 +
                                                   std.loc[j, :]**2) / 2) *
                              100)
                stand_diff.abs()
                print('\nComparing treatments {0:>2}'.format(i),
                      'and {0:>2}'.format(j))
                print('Variable                          Mean       Std',
                      '        t-val   p-val (%) Stand.Difference (%)')
                for jdx, _ in enumerate(mean_diff):
                    print('{:30}'.format(mean_diff.index[jdx]),
                          '{:10.5f}'.format(mean_diff[jdx]),
                          '{:10.5f}'.format(std_diff[jdx]),
                          '{:9.2f}'.format(t_diff[jdx]),
                          '{:9.2f}'.format(p_diff[jdx]),
                          '{:9.2f}'.format(stand_diff[jdx]))


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
        print('Size of object:   {:6} MB '.format(round(size_of_object_mb, 2)),
              'Available RAM: {:6} MB '.format(available),
              'Number of workers {:2} '.format(workers),
              'No of splits: {:2} '.format(no_of_splits),
              'Size of chunk:  {:6} MB '.format(round(chunck_size_mb, 2)))
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
