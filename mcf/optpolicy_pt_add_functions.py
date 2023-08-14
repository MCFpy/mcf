"""
Provide functions for Black-Box allocations.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from itertools import combinations
from math import factorial

from numba import njit
import numpy as np

from mcf import mcf_print_stats_functions as ps


def describe_tree(optp_, splits_seq, treat, obs=None, score_val=None):
    """Describe leaves of policy tree."""
    gen_dic, ot_dic = optp_.gen_dict, optp_.other_dict
    txt = '\n' + '-' * 100 + '\nLeaf information for estimated policy tree'
    txt += '\n' + '- ' * 50
    for i, splits in enumerate(splits_seq):
        for j in range(2):
            txt += f'\nLeaf {i:d}{j:d}:  '
            for splits_dic in splits[j]:
                txt += f'{splits_dic["x_name"]:4s}'
                if splits_dic['x_type'] == 'unord':
                    if splits_dic['left or right'] == 'left':
                        txt += 'In:     '
                    else:
                        txt += 'Not in: '
                    values_to_print = np.sort(splits_dic['cut-off or set'])
                    for s_i in values_to_print:
                        if isinstance(s_i, int) or (
                                (s_i - np.round(s_i)) < 0.00001):
                            txt += f'{int(np.round(s_i)):2d} '
                        else:
                            txt += f'{s_i:3.1f} '
                else:
                    if splits_dic['left or right'] == 'left':
                        txt += ' <='
                    else:
                        txt += ' > '
                    txt += f'{splits_dic["cut-off or set"]:8.3f} '
            txt += f'\nAlloc Treatment: {treat[i][j]:3d} '
            if score_val is not None:
                txt += (f'Obs: {obs[i][j]:6d}  '
                        f'Avg.score: {score_val[i][j] / obs[i][j]:7.3f} ')
                tmp = (score_val[i][j] / obs[i][j]
                       - ot_dic['costs_of_treat'][treat[i][j]])
                txt += (f'Avg.score-costs: {tmp:7.3f} ')
            txt += ('\n' + '- ' * 50)
    ps.print_mcf(gen_dic, txt, summary=True)


def automatic_cost(optp_, data_df):
    """Compute costs that fulfill constraints."""
    gen_dic, var_dic, ot_dic = optp_.gen_dict, optp_.var_dict, optp_.other_dict
    obs = len(data_df)
    if gen_dic['with_output']:
        print('\nSearching cost values that fulfill constraints')
    data_ps = data_df[var_dic['polscore_name']].to_numpy()
    obs = len(data_ps)
    max_by_treat = np.around(np.array(ot_dic['max_shares']) * obs)
    if any(cost > 0 for cost in ot_dic['costs_of_treat']):
        costs_of_treat = ot_dic['costs_of_treat'].copy()
    else:
        costs_of_treat = np.zeros(gen_dic['no_of_treat'])
    std_ps = np.std(data_ps.reshape(-1))
    step_size = 0.02
    while True:
        treatments = np.argmax(data_ps - costs_of_treat, axis=1)
        values, count = np.unique(treatments, return_counts=True)
        if len(count) == gen_dic['no_of_treat']:
            alloc = count
        else:
            alloc = np.zeros(gen_dic['no_of_treat'])
            for i, j in enumerate(values):
                alloc[j] = count[i]
        diff = alloc - max_by_treat
        diff[diff < 0] = 0
        if not np.any(diff > 0):
            break
        costs_of_treat += diff / obs * std_ps * step_size
    alloc = np.int16(alloc)
    costs_of_treat_update = costs_of_treat * ot_dic['costs_of_treat_mult']
    costs_of_treat_neu = ot_dic['costs_of_treat'].copy()
    for idx, cost in enumerate(costs_of_treat_update):
        if cost > ot_dic['costs_of_treat'][idx]:
            costs_of_treat_neu[idx] = cost
    if gen_dic['with_output']:
        txt = ('\n' + '=' * 100 +
               '\nAutomatic determination of cost that fullfil contraints in'
               ' unconstraint optimation using allocation'
               '\naccording to best best policy score (Black Box)'
               + '\n' + '-' * 100)
        txt += '\nConstraints (share): '
        for j in max_by_treat:
            txt += f'{j / obs:7.2%} '
        txt += '\n' + '- ' * 50
        txt += '\nCost values determined by unconstrained optimization: '
        for j in costs_of_treat:
            txt += f'{j:8.3f}'
        mult_str = [str(mult) for mult in ot_dic['costs_of_treat_mult']]
        mult = ' '.join(mult_str)
        txt += f'\nMultipliers to be used: {mult}'
        txt += '\n' + '- ' * 50 + ('\nAdjusted cost values and allocation in '
                                   'unconstrained optimization')
        for idx, cost in enumerate(costs_of_treat_neu):
            txt += (f'\nCost: {cost:8.3f}    Obs.: {alloc[idx]:6d}    Share:'
                    f' {alloc[idx] / obs:6.2%}')
        txt += '\n' + '-' * 100
        ps.print_mcf(gen_dic, txt, summary=True)
    return costs_of_treat_neu


def combinations_categorical(single_x_np, ps_np_diff, no_of_evalupoints,
                             with_numba, seed=123456):
    """Create all possible combinations of list elements, w/o complements."""
    values = np.unique(single_x_np)
    no_of_values = len(values)
    no_of_combinations = total_sample_splits_categorical(no_of_values)
    if no_of_combinations < no_of_evalupoints:
        combinations_new = all_combinations_no_complements(list(values))
    else:
        values_sorted, no_of_ps = get_values_ordered(
            single_x_np, ps_np_diff, values, no_of_values,
            with_numba=with_numba)
        combinations_t = sorted_values_into_combinations(
            values_sorted, no_of_ps, no_of_values)
        combinations_, _ = drop_complements(combinations_t, list(values))
        len_c = len(combinations_)
        if len_c > no_of_evalupoints:
            rng = np.random.default_rng(seed=seed)
            indx = rng.choice(range(len_c), size=no_of_evalupoints,
                              replace=False).tolist()
            combinations_new = [combinations_[i] for i in indx]
        else:
            combinations_new = combinations_
    return combinations_new


def total_sample_splits_categorical(no_of_values):
    """
    Compute total number of sample splits that can generated by categoricals.

    Parameters
    ----------
    no_of_values : Int.

    Returns
    -------
    no_of_splits: Int.

    """
    no_of_splits = 0
    for i in range(1, no_of_values):
        no_of_splits += factorial(no_of_values) / (factorial(no_of_values-i)
                                                   * factorial(i))
    return no_of_splits/2  # no complements


def all_combinations_no_complements(values):
    """Create all possible combinations of list elements, removing complements.

    Parameters
    ----------
    values : List. Elements to be combined.

    Returns
    -------
    list_without_complements : List of tuples.

    """
    list_all = []
    # This returns a list with tuples of all possible combinations of tuples
    for length in range(1, len(values)):
        list_all.extend(list(combinations(values, length)))
    # Next, the complements to each list will be removed
    list_wo_compl, _ = drop_complements(list_all, values)
    return list_wo_compl


def drop_complements(list_all, values):
    """
    Identify and remove complements.

    Parameters
    ----------
    list_all : List of tuples. Tuples with combinations.
    values : List. All relevant values.

    Returns
    -------
    list_wo_compl : List of Tuples. List_all with complements removed.

    """
    list_w_compl, list_wo_compl = [], []
    for i in list_all:
        if i not in list_w_compl:
            list_wo_compl.append(i)
            compl_of_i = values[:]
            for drop_i in i:
                compl_of_i.remove(drop_i)
            list_w_compl.append(tuple(compl_of_i))
    return list_wo_compl, list_w_compl


def sorted_values_into_combinations(values_sorted, no_of_ps, no_of_values):
    """
    Transfrom sorted values into unique combinations of values.

    Parameters
    ----------
    values_sorted : 2D numpy array. Sorted values for each policy score
    no_of_ps : Int. Number of policy scores.
    no_of_values : Int. Number of values.

    Returns
    -------
    unique_combinations : Unique Tuples to be used for sample splitting.

    """
    unique_combinations = []
    value_idx = np.arange(no_of_values-1)
    for j in range(no_of_ps):
        for i in value_idx:
            next_combi = tuple(values_sorted[value_idx[:i+1], j])
            if next_combi not in unique_combinations:
                unique_combinations.append(next_combi)
    return unique_combinations


def get_values_ordered(single_x_np, ps_np_diff, values, no_of_values,
                       with_numba=True):
    """
    Sort values according policy score differences: NUR Durchlauferhitzer.

    Parameters
    ----------
    single_x_np : 1D numpy array. Covariate.
    ps_np_diff : 2 D numpy array. Policy scores as difference.
    values : 1D numpy array. All unique values of x.
    no_of_values : Int. #  of Unique values of x.
    with_numba : Boolean. Use numba module. Default is True.

    Returns
    -------
    values_sorted : 2D numpy array. Sorted values.

    """
    if with_numba:
        values_sorted, no_of_ps = get_values_ordered_numba(
            single_x_np, ps_np_diff, values, no_of_values)
    else:
        values_sorted, no_of_ps = get_values_ordered_no_numba(
            single_x_np, ps_np_diff, values, no_of_values)
    return values_sorted, no_of_ps


@njit
def get_values_ordered_numba(single_x_np, ps_np_diff, values, no_of_values):
    """
    Sort values according policy score differences.

    Parameters
    ----------
    single_x_np : 1D numpy array. Covariate.
    ps_np_diff : 2 D numpy array. Policy scores as difference.
    values : 1D numpy array. All unique values of x.
    no_of_values : Int. #  of Unique values of x.

    Returns
    -------
    values_sorted : 2D numpy array. Sorted values.

    """
    no_of_ps = np.shape(ps_np_diff)[1]  # wg Numba
    mean_y_by_values = np.empty((no_of_values, no_of_ps))
    for i, val in enumerate(values):
        ps_group = ps_np_diff[np.where(single_x_np == val)]
        for j in range(no_of_ps):  # wg numba
            mean_y_by_values[i, j] = np.mean(ps_group[:, j])
    indices = np.empty((no_of_values, no_of_ps))
    values_sorted = np.empty((no_of_values, no_of_ps))
    for j in range(no_of_ps):
        indices = np.argsort(mean_y_by_values[:, j])
        values_sorted[:, j] = values[indices]
    return values_sorted, no_of_ps


def get_values_ordered_no_numba(single_x_np, ps_np_diff, values, no_of_values):
    """
    Sort values according policy score differences.

    Parameters
    ----------
    single_x_np : 1D numpy array. Covariate.
    ps_np_diff : 2 D numpy array. Policy scores as difference.
    values : 1D numpy array. All unique values of x.
    no_of_values : Int. #  of Unique values of x.

    Returns
    -------
    values_sorted : 2D numpy array. Sorted values.

    """
    no_of_ps = np.size(ps_np_diff, axis=1)
    mean_y_by_values = np.empty((no_of_values, no_of_ps))
    for i, val in enumerate(values):
        ps_group = ps_np_diff[np.where(single_x_np == val)]
        mean_y_by_values[i, :] = np.transpose(np.mean(ps_group, axis=0))
    # indices = np.empty((no_of_values, no_of_ps))
    values_sorted = np.empty((no_of_values, no_of_ps))
    for j in range(no_of_ps):
        indices = np.argsort(mean_y_by_values[:, j])
        values_sorted[:, j] = values[indices]
    return values_sorted, no_of_ps
