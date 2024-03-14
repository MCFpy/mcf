"""
Provide functions for Black-Box allocations.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from itertools import combinations
from math import comb

from numba import njit, prange
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
                        txt += ' In:     '
                    else:
                        txt += ' Not in: '
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
            if obs is not None:
                txt += f'  Obs: {obs[i][j]:6d}  '
            if score_val is not None:
                txt += f'Avg.score: {score_val[i][j] / obs[i][j]:7.3f} '
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
               ' unconstraint optimisation using allocation'
               '\naccording to best best policy score (Black Box)'
               + '\n' + '-' * 100)
        txt += '\nConstraints (share): '
        for j in max_by_treat:
            txt += f'{j / obs:7.2%} '
        txt += '\n' + '- ' * 50
        txt += '\nCost values determined by unconstrained optimization: '
        for j in costs_of_treat:
            txt += f'{j:8.3f}'
        mult_str = [str(round(mult, 3))
                    for mult in ot_dic['costs_of_treat_mult']]
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
        combinations_ = drop_complements(combinations_t, list(values))
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
    Compute total # of sample splits that can be generated by categoricals.

    Parameters
    ----------
    no_of_values : Int.

    Returns
    -------
    no_of_splits: Int.

    """
    no_of_splits = sum(comb(no_of_values, i) for i in range(1, no_of_values))
    return no_of_splits // 2  # no complements


def all_combinations_no_complements(values):
    """Create all possible combinations of list elements, removing complements.

    Parameters
    ----------
    values : List. Elements to be combined.

    Returns
    -------
    list_without_complements : List of tuples.

    """
    all_combinations = [comb for length in range(1, len(values))
                        for comb in combinations(values, length)]
    # Remove complements
    list_without_complements = drop_complements(all_combinations, values)
    return list_without_complements


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
    return list_wo_compl


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
    no_of_ps = np.shape(ps_np_diff)[1]  # because of Numba
    mean_y_by_values = np.empty((no_of_values, no_of_ps))
    for i, val in enumerate(values):
        ps_group = ps_np_diff[single_x_np == val, :]
        for j in range(no_of_ps):  # wg numba
            mean_y_by_values[i, j] = np.mean(ps_group[:, j])
    indices = np.empty((no_of_values, no_of_ps))
    values_sorted = np.empty((no_of_values, no_of_ps))
    for j in range(no_of_ps):
        indices = np.argsort(mean_y_by_values[:, j])
        values_sorted[:, j] = values[indices]
    return values_sorted, no_of_ps


@njit(parallel=True)  # Turns out to increase computation time
def get_values_ordered_numba_prange(
        single_x_np, ps_np_diff, values, no_of_values):
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
    no_of_ps = np.shape(ps_np_diff)[1]  # because of Numba
    mean_y_by_values = np.empty((no_of_values, no_of_ps))
    no_val = len(values)
    for i in prange(no_val):
        val = values[i]
        ps_group = ps_np_diff[single_x_np == val, :]
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
    values_sorted = np.empty((no_of_values, no_of_ps))
    for j in range(no_of_ps):
        indices = np.argsort(mean_y_by_values[:, j])
        values_sorted[:, j] = values[indices]
    return values_sorted, no_of_ps


def adjust_reward(no_by_treat_l, no_by_treat_r, reward_l, reward_r,
                  with_numba, max_by_treat):
    """Adjust rewards if restrictions are violated.

    Parameters
    ----------
    no_by_treat_l : Numpy array.
    no_by_treat_r : Numpy array.
    reward_l : Float.
    reward_r : Float.
    with_numba : Boolean.
    max_by_treat : Int.

    Returns
    -------
    reward_l : Numpy array.
    reward_r : Numpy array.

    """
    if with_numba:
        reward_l, reward_r = adjust_reward_numba(
            no_by_treat_l, no_by_treat_r, reward_l, reward_r, max_by_treat)
    else:
        reward_l, reward_r = adjust_reward_no_numba(
            no_by_treat_l, no_by_treat_r, reward_l, reward_r, max_by_treat)
    return reward_l, reward_r


@njit
def adjust_reward_numba(no_by_treat_l, no_by_treat_r, reward_l, reward_r,
                        max_by_treat):
    """Adjust rewards if restrictions are violated.

    Parameters
    ----------
    no_by_treat_l : Numpy array.
    no_by_treat_r : Numpy array.
    reward_l : Float.
    reward_r : Float.
    max_by_treat : Dict. Parameter.

    Returns
    -------
    reward_l : Numpy array.
    reward_r : Numpy array.

    """
    if no_by_treat_l is not None and no_by_treat_r is not None:
        no_by_treat = no_by_treat_l + no_by_treat_r
        violations = no_by_treat > max_by_treat
        if np.any(violations):
            diff_max = ((no_by_treat - max_by_treat) / max_by_treat).max()
            diff = min(diff_max, 1)
            reward_l = reward_l - diff * np.abs(reward_l)
            reward_r = reward_r - diff * np.abs(reward_r)
    return reward_l, reward_r


def adjust_reward_no_numba(no_by_treat_l, no_by_treat_r, reward_l, reward_r,
                           max_by_treat):
    """Adjust rewards if restrictions are violated.

    Parameters
    ----------
    no_by_treat_l : Numpy array.
    no_by_treat_r : Numpy array.
    reward_l : Float.
    reward_r : Float.
    max_by_treat : List of Int.

    Returns
    -------
    reward_l : Numpy array.
    reward_r : Numpy array.

    """
    if (no_by_treat_l is None) or (no_by_treat_r is None):
        return reward_l, reward_r
    no_by_treat = no_by_treat_l + no_by_treat_r
    if np.any(no_by_treat > max_by_treat):
        diff_max = ((no_by_treat - max_by_treat) / max_by_treat).max()
        diff = min(diff_max, 1)
        reward_l = reward_l - diff * np.abs(reward_l)
        reward_r = reward_r - diff * np.abs(reward_r)
    return reward_l, reward_r


def prepare_data_for_tree_building(optp_, data_df, seed=123456):
    """Prepare data for tree building."""
    int_dic, var_dic = optp_.int_dict, optp_.var_dict
    x_type, x_values = optp_.var_x_type, optp_.var_x_values
    gen_dic, pt_dic = optp_.gen_dict, optp_.pt_dict
    data_ps = data_df[var_dic['polscore_name']].to_numpy()
    data_ps_diff = data_ps[:, 1:] - data_ps[:, 0, np.newaxis]
    no_of_x = len(x_type)
    name_x = [None] * no_of_x
    type_x, values_x = [None] * no_of_x, [None] * no_of_x
    for j, key in enumerate(x_type.keys()):
        name_x[j], type_x[j] = key, x_type[key]
        values_x[j] = (sorted(x_values[key])
                       if x_values[key] is not None else None)
    #                  this None for continuous variables
    data_x = data_df[name_x].to_numpy()
    del data_df
    if gen_dic['x_unord_flag']:
        for m_i in range(no_of_x):
            if type_x[m_i] == 'unord':
                data_x[:, m_i] = np.round(data_x[:, m_i])
                values_x[m_i] = combinations_categorical(
                    data_x[:, m_i], data_ps_diff,
                    pt_dic['no_of_evalupoints'], int_dic['with_numba'],
                    seed=seed)
    return data_x, data_ps, data_ps_diff, name_x, type_x, values_x


def only_1st_tree_fct3(data_ps, costs_of_treat):
    """Find out if further splits make any sense."""
    data = data_ps-costs_of_treat
    no_further_splitting = all_same_max_numba(data)
    return no_further_splitting


@njit
def all_same_max_numba(data):
    """Check same categies have max."""
    ref_val = np.argmax(data[0, :])
    for i in range(1, len(data)):
        opt_treat = np.argmax(data[i, :])
        if ref_val != opt_treat:
            return False
    return True


def evaluate_leaf(data_ps, gen_dic, ot_dic, pt_dic, with_numba=True):
    """Evaluate final value of leaf taking restriction into account.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    gen_dic : Dict. Controls.
    ot_dic, pt_dic : Dict. Controls.

    Returns
    -------
    treat_ind: Int. Index of treatment.
    reward: Int. Value of leaf.
    no_per_treat: Numpy 1D-array of int.

    """
    if with_numba:
        indi, reward_by_treat, obs_all = evaluate_leaf_numba(
            data_ps, gen_dic['no_of_treat'], ot_dic['max_by_treat'],
            ot_dic['restricted'] and pt_dic['enforce_restriction'],
            pt_dic['cost_of_treat_restrict'])
    else:
        indi, reward_by_treat, obs_all = evaluate_leaf_no_numba(
            data_ps, gen_dic['no_of_treat'], ot_dic, pt_dic)
    return indi, reward_by_treat, obs_all


@njit
def evaluate_leaf_numba(data_ps, no_of_treatments, max_by_treat, restricted,
                        costs_of_treat):
    """Evaluate final value of leaf taking restriction into account.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    ...

    Returns
    -------
    treat_ind: Int. Index of treatment.
    reward: Int. Value of leaf.
    no_per_treat: Numpy 1D-array of int.

    """
    obs = len(data_ps)
    obs_all = np.zeros(no_of_treatments)
    indi = np.arange(no_of_treatments)
    if restricted:
        diff_obs = obs - max_by_treat
        treat_not_ok = diff_obs > 0.999
        if np.any(treat_not_ok):
            treat_ok = np.invert(treat_not_ok)
            data_ps_tmp = data_ps[:, treat_ok]
            if data_ps_tmp.size == 0:
                idx = np.argmin(diff_obs)
                treat_ok[idx] = True
                data_ps = data_ps[:, treat_ok]
            else:
                data_ps = data_ps_tmp
            indi = indi[treat_ok]      # Remove obs that violate restriction
            costs_of_treat = costs_of_treat[indi]
    reward_by_treat = np.sum(data_ps, axis=0) - costs_of_treat * obs
    max_i = np.argmax(reward_by_treat)
    obs_all[indi[max_i]] = obs
    return indi[max_i], reward_by_treat[max_i], obs_all


def evaluate_leaf_no_numba(data_ps, no_of_treat, ot_dic, pt_dic):
    """Evaluate final value of leaf taking restriction into account.

    Returns
    -------
    treat_ind: Int. Index of treatment.
    reward: Int. Value of leaf.
    no_per_treat: Numpy 1D-array of int.

    """
    obs_all, obs = np.zeros(no_of_treat), len(data_ps)
    indi = np.arange(no_of_treat)
    costs_of_treat = pt_dic['cost_of_treat_restrict']
    if ot_dic['restricted'] and pt_dic['enforce_restriction']:
        diff_obs = obs - ot_dic['max_by_treat']
        treat_not_ok = diff_obs > 0.999
        if np.any(treat_not_ok):
            treat_ok = np.invert(treat_not_ok)
            data_ps_tmp = data_ps[:, treat_ok]
            if data_ps_tmp.size == 0:
                idx = np.argmin(diff_obs)
                treat_ok[idx] = True
                data_ps = data_ps[:, treat_ok]
            else:
                data_ps = data_ps_tmp
            indi = indi[treat_ok]      # Remove obs that violate restriction
            costs_of_treat = pt_dic['cost_of_treat_restrict'][indi]
    reward_by_treat = data_ps.sum(axis=0) - costs_of_treat * obs
    max_i = np.argmax(reward_by_treat)
    obs_all[indi[max_i]] = obs
    return indi[max_i], reward_by_treat[max_i], obs_all


def get_values_cont_x(data_vector, no_of_evalupoints, with_numba=True):
    """Get cut-off points for tree splitting for continuous variables.

    Parameters
    ----------
    data_vector : Numpy-1D array. Sorted vector
    no_of_evalupoints : Int.
    with_numba : Boolean. Use numba module. Default is True.

    Returns
    -------
    Numpy 1D-array. Sorted cut-off-points

    """
    if with_numba:
        data_vector_new = get_values_cont_x_numba(data_vector,
                                                  no_of_evalupoints)
    else:
        data_vector_new = get_values_cont_x_no_numba(data_vector,
                                                     no_of_evalupoints)
    return data_vector_new


@njit
def get_values_cont_x_numba(data_vector, no_of_evalupoints):
    """Get cut-off points for tree splitting for continuous variables.

    Parameters
    ----------
    data_vector : Numpy-1D array. Sorted vector
    no_of_evalupoints : Int.

    Returns
    -------
    Numpy 1D-array. Sorted cut-off-points

    """
    data_vector = np.unique(data_vector)
    obs = len(data_vector)
    if no_of_evalupoints > (obs - 10):
        data_vector_new = data_vector
    else:
        indices = np.linspace(obs / no_of_evalupoints, obs,
                              no_of_evalupoints+1)
        data_vector_new = np.empty(no_of_evalupoints)
        for i in range(no_of_evalupoints):
            indices_i = np.uint32(indices[i])
            data_vector_new[i] = data_vector[indices_i]
    return data_vector_new


@njit(parallel=True)  # Turns out to increase computation time
def get_values_cont_x_numba_prange(data_vector, no_of_evalupoints):
    """Get cut-off points for tree splitting for continuous variables.

    Parameters
    ----------
    data_vector : Numpy-1D array. Sorted vector
    no_of_evalupoints : Int.

    Returns
    -------
    Numpy 1D-array. Sorted cut-off-points

    """
    data_vector = np.unique(data_vector)
    obs = len(data_vector)
    if no_of_evalupoints > (obs - 10):
        data_vector_new = data_vector
    else:
        indices = np.linspace(obs / no_of_evalupoints, obs,
                              no_of_evalupoints+1)
        data_vector_new = np.empty(no_of_evalupoints)
        for i in prange(no_of_evalupoints):
            indices_i = np.uint32(indices[i])
            data_vector_new[i] = data_vector[indices_i]
    return data_vector_new


def get_values_cont_x_no_numba(data_vector, no_of_evalupoints):
    """Get cut-off points for tree splitting for continuous variables.

       No longer used; only kept if no_numba version would be needed

    Parameters
    ----------
    sorted_data : Numpy-1D array. Sorted vector
    no_of_evalupoints : Int.

    Returns
    -------
    Numpy 1D-array. Sorted cut-off-points

    """
    data_vector = np.unique(data_vector)
    obs = len(data_vector)
    if no_of_evalupoints > (obs - 10):
        return data_vector
    indices = np.uint32(np.linspace(obs / no_of_evalupoints, obs,
                                    no_of_evalupoints, endpoint=False))
    return data_vector[indices]
