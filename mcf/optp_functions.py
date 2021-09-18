"""Created on Fri Jun 26 12:42:02 2020.

Optimal Policy Trees: Functions - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

# -*- coding: utf-8 -*-
"""
import copy
import random
import math
import time
import sys
from concurrent import futures
from multiprocessing import freeze_support
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats as sct
import psutil
from numba import njit
from mcf import general_purpose as gp


def optpoltree(
    indata=None, datpath=None, outpath=None, id_name=None, polscore_name=None,
    x_ord_name=None, x_unord_name=None, effect_vs_0=None, effect_vs_0_se=None,
    output_type=2, outfiletext=None, parallel_processing=True,
    how_many_parallel=None, with_numba=True, screen_covariates=True,
    check_perfectcorr=True, min_dummy_obs=10, clean_data_flag=True,
    no_of_evalupoints=100, depth_of_tree=3, min_leaf_size=None,
    max_shares=None, costs_of_treat=None, costs_of_treat_mult=1,
    only_if_sig_better_vs_0=False, sig_level_vs_0=0.05,
    _smaller_sample=None, _with_output=False):
    """Compute the optimal policy tree."""
    freeze_support()
    time1 = time.time()
    # -------------------------------------------------------------------------
    # use smaller random sample (usually for testing purposes)
    if _smaller_sample is not None:
        if 0 < _smaller_sample < 1:
            np.random.default_rng(seed=9132556)  # Initialise rnd Numpy
            gp.randomsample(datpath, indata + '.csv', 'smaller_indata.csv',
                            _smaller_sample, True)
            indata = 'smaller_indata'

    # set values for control variables
    controls = controls_into_dic(
        how_many_parallel, parallel_processing, output_type, outpath, datpath,
        indata, outfiletext, _with_output, screen_covariates,
        check_perfectcorr, clean_data_flag, min_dummy_obs, no_of_evalupoints,
        max_shares, depth_of_tree, costs_of_treat, costs_of_treat_mult,
        with_numba, min_leaf_size, only_if_sig_better_vs_0, sig_level_vs_0)
    variables = variable_dict(id_name, polscore_name, x_ord_name,
                              x_unord_name, effect_vs_0, effect_vs_0_se)

    # Set defaults for many control variables of the MCF & define variables
    c_dict, v_dict = get_controls(controls, variables)

    # Some descriptive stats of input and redirection of output file
    if c_dict['with_output']:
        if c_dict['print_to_file']:
            orig_stdout = sys.stdout
            gp.delete_file_if_exists(c_dict['outfiletext'])
            if c_dict['print_to_terminal']:
                sys.stdout = gp.OutputTerminalFile(c_dict['outfiletext'])
            else:
                outfiletext = open(c_dict['outfiletext'], 'w')
                sys.stdout = outfiletext

    if c_dict['with_output']:
        print('\nParameter for Optimal Policy Tree:')
        gp.print_dic(c_dict)
        print('\nVariables used:')
        gp.print_dic(v_dict)
        gp.print_descriptive_stats_file(
            c_dict['indata'], to_file=c_dict['print_to_file'])
        names_to_check = (v_dict['id_name'] + v_dict['polscore_name']
                          + v_dict['x_ord_name'] + v_dict['x_unord_name'])
        if c_dict['only_if_sig_better_vs_0']:
            names_to_check = names_to_check + v_dict['effect_vs_0'] + v_dict[
                'effect_vs_0_se']
    else:
        c_dict['print_to_file'] = False

    # Prepare data
    # Remove missings and keep only variables needed for further analysis
    if c_dict['clean_data_flag']:
        indata2 = gp.clean_reduce_data(
            c_dict['indata'], c_dict['indata_temp'], names_to_check,
            c_dict['with_output'], c_dict['with_output'],
            c_dict['print_to_file'])
    else:
        indata2 = c_dict['indata']
    # Remove variables that do not have enough independent variation
    if c_dict['screen_covariates']:
        x_variables_in, _ = gp.screen_variables(
            indata2, v_dict['x_ord_name'] + v_dict['x_unord_name'],
            c_dict['check_perfectcorr'], c_dict['min_dummy_obs'],
            c_dict['with_output'])
    x_type, x_value, c_dict = classify_var_for_pol_tree(
        indata2, v_dict, c_dict, list(x_variables_in))
    time2 = time.time()
    if isinstance(c_dict['costs_of_treat'], int):
        if c_dict['costs_of_treat'] == -1:
            c_dict = automatic_cost(indata2, v_dict, c_dict)
    optimal_tree, _, _ = optimal_tree_proc(
        indata2, x_type, x_value, v_dict, c_dict)

    # Prozedur um den Output darzustellen
    if c_dict['with_output']:
        # if __name__ == '__main__':  # ohne das geht Multiprocessing nicht
        descr_policy_tree(indata2, optimal_tree, x_type, x_value,
                          v_dict, c_dict)
    time3 = time.time()

    # Print timing information
    time_string = ['Data preparation: ',
                   'Tree building     ',
                   'Total time:       ']
    time_difference = [time2 - time1, time3 - time2, time3 - time1]
    if c_dict['with_output']:
        if c_dict['parallel']:
            print('\nMultiprocessing')
        else:
            print('\nNo parallel processing')
        gp.print_timing(time_string, time_difference)  # print to file
        if c_dict['print_to_file']:
            if c_dict['print_to_terminal']:
                sys.stdout.output.close()
            else:
                outfiletext.close()
            sys.stdout = orig_stdout


def automatic_cost(datafile_name, v_dict, c_dict):
    """Compute costs that fulfill constraints.

    Parameters
    ----------
    data_file : string. Input data
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.

    Returns
    -------
    c_dict : Dict. Parameters with modified cost variable.

    """
    if c_dict['with_output']:
        print('Searching cost values that fulfill constrains')
    if c_dict['only_if_sig_better_vs_0']:
        data_ps, data_df = adjust_policy_score(datafile_name, c_dict, v_dict)
    else:
        data_df = pd.read_csv(datafile_name)
        data_ps = data_df[v_dict['polscore_name']].to_numpy()
    obs = len(data_ps)
    max_by_treat = np.array(c_dict['max_by_treat'])
    costs_of_treat = np.zeros(c_dict['no_of_treatments'])
    # std_ps = np.std(data_ps.flatten())
    std_ps = np.std(data_ps.reshape(-1))
    step_size = 0.02
    while True:
        treatments = np.argmax(data_ps - costs_of_treat, axis=1)
        values, count = np.unique(treatments, return_counts=True)
        # if np.size(count) == c_dict['no_of_treatments']:
        if len(count) == c_dict['no_of_treatments']:
            # alloc = count.copy()
            alloc = count
        else:
            alloc = np.zeros(c_dict['no_of_treatments'])
            for i, j in enumerate(values):
                # alloc[j] = count[i].copy()
                alloc[j] = count[i]
        diff = alloc - max_by_treat
        diff[diff < 0] = 0
        if not np.any(diff > 0):
            break
        costs_of_treat += diff / obs * std_ps * step_size
    alloc = np.uint32(alloc)
    if c_dict['with_output']:
        print('-' * 80)
        print('Cost values and allocation in unconstrained optimization')
        for i, j in enumerate(costs_of_treat):
            print('{:8.3f}  {:6d}  {:6.2f}'.format(j, alloc[i],
                                                   alloc[i] / obs * 100))
        print('Multiplier to be used: {:8.3f}'.format(c_dict['costs_mult']))
        print('-' * 80)
    c_dict['costs_of_treat'] = costs_of_treat * c_dict['costs_mult']
    return c_dict


def subsample_leaf(polscore_df, x_df, split):
    """Reduces dataframes to data in leaf.

    Parameters
    ----------
    polscore_df : Dataframe. Policyscores.
    x_df : Dataframe. Policy variables.
    split : dict. Split information.

    Returns
    -------
    polscore_df_red : Dataframe. Reduced.
    x_df_red : Dataframe. Reduced

    """
    if split['x_type'] == 'unord':
        if split['left or right'] == 'left':  # x in set
            condition = x_df[split['x_name']].isin(split['cut-off or set'])
        else:
            condition = ~x_df[split['x_name']].isin(split['cut-off or set'])
    else:  # x not in set
        if split['left or right'] == 'left':
            condition = x_df[split['x_name']] <= split['cut-off or set']
        else:
            condition = x_df[split['x_name']] > split['cut-off or set']
    polscore_df_red = polscore_df[condition]
    x_df_red = x_df[condition]
    return polscore_df_red, x_df_red


def final_leaf_dict(leaf, left_right):
    """Generate a dictionary used in evaluating the policy tree.

    Parameters
    ----------
    leaf : List.
    left_right : string.

    Returns
    -------
    return_dic :dict.

    """
    return_dic = {'x_name': leaf[5],
                  'x_type': leaf[6],
                  'cut-off or set': leaf[7],
                  'left or right': left_right}
    return return_dic


def two_leafs_info(tree, polscore_df, x_df, leaf):
    """Compute the information contained in two adjacent leaves.

    Parameters
    ----------
    tree : List of lists.
    polscore_df : Dataframe. Policyscore.
    x_df : Dataframe. Policy variables.
    leaf : List. Terminal leaf under investigation.

    Raises
    ------
    Exception : If inconsistent leaf numbers.

    Returns
    -------
    leaf_splits : Tuple of List of dict. All splits that lead to left leaf.
    score : Tuple of Float. Final value of left leaf. (left, right)
    obs : Tuple of Int. Number of observations in leaf.

    """
    # Collect decision path that the two final leaves
    leaf_splits_pre = []
    parent_nr = leaf[1]
    current_nr = leaf[0]
    if len(tree) > 1:
        while parent_nr is not None:
            for leaf_i in tree:
                if leaf_i[0] == parent_nr:
                    if current_nr == leaf_i[2]:
                        left_right = 'left'
                    elif current_nr == leaf_i[3]:
                        left_right = 'right'
                    else:
                        raise Exception('Leaf numbers are inconsistent.')
                    new_dic = final_leaf_dict(leaf_i, left_right)
                    leaf_splits_pre.append(new_dic.copy())
                    try:
                        parent_nr = leaf_i[1].copy()
                    except AttributeError:
                        parent_nr = leaf_i[1]
                    try:
                        current_nr = leaf_i[0].copy()
                    except AttributeError:
                        current_nr = leaf_i[0]
        leaf_splits_pre = list(reversed(leaf_splits_pre))  # Starting 1st split
        # compute policy score
        for split in leaf_splits_pre:
            polscore_df, x_df = subsample_leaf(polscore_df, x_df, split)
    final_dict_l = final_leaf_dict(leaf, 'left')
    final_dict_r = final_leaf_dict(leaf, 'right')
    polscore_df_l = subsample_leaf(polscore_df, x_df, final_dict_l)[0]
    polscore_df_r = subsample_leaf(polscore_df, x_df, final_dict_r)[0]
    leaf_splits_r = copy.deepcopy(leaf_splits_pre)
    leaf_splits_l = leaf_splits_pre   # one copy is enough
    leaf_splits_l.append(final_dict_l)
    leaf_splits_r.append(final_dict_r)
    leaf_splits = (leaf_splits_l, leaf_splits_r)
    obs = (polscore_df_l.shape[0], polscore_df_r.shape[0])
    polscore_df_l = polscore_df_l.iloc[:, leaf[8][0]]
    polscore_df_r = polscore_df_r.iloc[:, leaf[8][1]]
    score = (polscore_df_l.sum(axis=0), polscore_df_r.sum(axis=0))
    return leaf_splits, score, obs, tuple(leaf[8])


def descr_policy_tree(datafile_name, tree, x_name, x_type, v_dict, c_dict):
    """Describe optimal policy tree and parameters used.

    Content of tree for each node:
    0: Node identifier (INT: 0-...)
    1: Parent knot
    2: Child node left
    3: Child node right
    4: Type of node (2: Active -> will be further splitted or made terminal
                    1: Terminal node, no further splits
                    0: previous node that lead already to further splits)
    5: String: Name of variable used for decision of next split
    6: x_type of variable (policy categorisation, maybe different from MCF)
    7: If x_type = 'unordered': Set of values that goes to left daughter
    7: If x_type = 0: Cut-off value (larger goes to right daughter)
    8: List of Treatment state for both daughters [left, right]

    Parameters
    ----------
    datafile_name: String.
    tree : List of lists.
    x_type : Dict. Type information of variables.
    v_dict : Dict. Variables.
    c_dict : Dict. Controls.

    Returns
    -------
    None.

    """
    data_df = pd.read_csv(datafile_name)
    data_df_ps = data_df[v_dict['polscore_name']]
    data_df_x = data_df[x_name]
    if tree is None:
        raise Exception('Not enough evaluation points for current depth.' +
                        'Reduce depth or increase variable and / or ' +
                        'evaluation points for continuous variables.')
    length = len(tree)
    ids = [None] * length
    terminal_leafs = []
    for leaf_i in range(length):
        ids[leaf_i] = tree[leaf_i][0]
        if tree[leaf_i][4] == 1:
            terminal_leafs.append(tree[leaf_i])
    depth = int(np.round(math.log(len(terminal_leafs) * 2) / math.log(2)))
    if not len(set(ids)) == len(ids):
        raise Exception('Some leafs IDs are identical. Rerun programme.')
    print('\n' + ('=' * 80))
    print('Descriptive statistic of estimated policy tree')
    if c_dict['only_if_sig_better_vs_0']:
        print('While tree building, policy scores not significantly',
              'different from zero are set to zero. Below, orginal scores',
              'are used.')
        print('Significance level used for recoding: {:6.4f} %'.format(
            c_dict['sig_level_vs_0'] * 100))
    print('-' * 80)
    print('Policy scores: ', end=' ')
    for i in v_dict['polscore_name']:
        print(i, end=' ')
    print('\nDecision variables: ', end=' ')
    for i in x_type.keys():
        print(i, end=' ')
    print('\nDepth of tree:   {:d} '.format(depth))
    print('Minimum leaf size: {:d} '.format(c_dict['min_leaf_size']))
    print('- ' * 40)
    splits_seq = [None] * len(terminal_leafs)
    score_val = [None] * len(terminal_leafs)
    obs = [None] * len(terminal_leafs)
    treat = [None] * len(terminal_leafs)
    for i, leaf in enumerate(terminal_leafs):
        splits_seq[i], score_val[i], obs[i], treat[i] = two_leafs_info(
            tree, data_df_ps, data_df_x, leaf)
    total_obs = 0
    total_obs_by_treat = np.zeros((data_df_ps.shape[1]))
    total_score = 0
    total_cost = 0
    for i, obs_i in enumerate(obs):
        total_obs += obs_i[0] + obs_i[1]
        total_obs_by_treat[treat[i][0]] += obs_i[0]
        total_obs_by_treat[treat[i][1]] += obs_i[1]
        total_score += score_val[i][0] + score_val[i][1]
    total_cost = np.sum(c_dict['costs_of_treat'] * total_obs_by_treat)
    print('Total score:        {:14.4f} '.format(total_score),
          '  Average score:        {:14.4f}'.format(total_score / total_obs))
    print('Total cost:         {:14.4f} '.format(total_cost),
          '  Average cost:         {:14.4f}'.format(total_cost / total_obs))
    print('Total score - cost: {:14.4f} '.format(total_score-total_cost),
          '  Average score - cost: {:14.4f}'.format(
              (total_score-total_cost) / total_obs))
    print('- ' * 40)
    print('Total number of observations: {:d}'.format(int(total_obs)))
    print('Treatments:                           ', end=' ')
    for i, j in enumerate(v_dict['polscore_name']):
        print('{:6d} '.format(i), end=' ')
    print('\nObservations allocated, by treatment: ', end=' ')
    for i in total_obs_by_treat:
        print('{:6d} '.format(int(i)), end=' ')
    print('\nObservations targeted, by treatment:  ', end=' ')
    for i in c_dict['max_by_treat']:
        print('{:6d} '.format(int(i)), end=' ')
    print('\nCost per treatment:                   ', end=' ')
    for i in c_dict['costs_of_treat']:
        print('{:6.2f} '.format(i), end=' ')
    print('\n' + '-' * 80)
    for i, splits in enumerate(splits_seq):
        for j in range(2):
            print('Leaf {:d}{:d}:  '.format(i, j), end=' ')
            for splits_dic in splits[j]:
                print('{:4s}'.format(splits_dic['x_name']), end=' ')
                if splits_dic['x_type'] == 'unord':
                    if splits_dic['left or right'] == 'left':
                        print('In:     ', end='')
                    else:
                        print('Not in: ', end='')
                    values_to_print = np.sort(splits_dic['cut-off or set'])
                    for s_i in values_to_print:
                        if isinstance(s_i, int) or (
                                (s_i - np.round(s_i)) < 0.00001):
                            print('{:2d} '.format(
                                int(np.round(s_i))), end=' ')
                        else:
                            print('{:3.1f} '.format(s_i), end=' ')
                else:
                    if splits_dic['left or right'] == 'left':
                        print('<=', end='')
                    else:
                        print('> ', end='')
                    print('{:8.3f} '.format(splits_dic['cut-off or set']),
                          end=' ')
            print()
            print('Alloc Treatment: {:3d} '.format(treat[i][j]), end='')
            print('Obs: {:6d}  Avg.score: {:7.3f} '.format(
                obs[i][j], score_val[i][j] / obs[i][j]), end=' ')
            print('Avg.score-costs: {:7.3f} '.format(
                score_val[i][j] / obs[i][j]
                - c_dict['costs_of_treat'][treat[i][j]]))
            print('- ' * 40)
    print('=' * 80)


def combinations_categorical(single_x_np, ps_np_diff, c_dict):
    """Create all possible combinations of list elements, removing complements.

    Parameters
    ----------
    single_x_np : 1D Numpy array. Features.
    ps_np_diff : 2D Numpy array. Policy scores as difference.
    c_dict : Dict. Controls.

    Returns
    -------
    combinations : List of tuples with values for each split.

    """
    values = np.unique(single_x_np)
    # no_of_values = np.size(values)
    no_of_values = len(values)
    no_of_combinations = gp.total_sample_splits_categorical(no_of_values)
    if no_of_combinations < c_dict['no_of_evalupoints']:
        combinations = gp.all_combinations_no_complements(list(values))
    else:
        values_sorted, no_of_ps = get_values_ordered(
            single_x_np, ps_np_diff, values, no_of_values,
            with_numba=c_dict['with_numba'])
        combinations_t = sorted_values_into_combinations(
            values_sorted, no_of_ps, no_of_values)
        if len(combinations_t) > c_dict['no_of_evalupoints']:
            combinations_t = random.sample(
                combinations_t, c_dict['no_of_evalupoints'])
        combinations, _ = gp.drop_complements(combinations_t, list(values))
    return combinations


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
    indices = np.empty((no_of_values, no_of_ps))
    values_sorted = np.empty((no_of_values, no_of_ps))
    for j in range(no_of_ps):
        indices = np.argsort(mean_y_by_values[:, j])
        values_sorted[:, j] = values[indices]
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


def get_values_cont_x(data_vector, no_of_evalupoints, with_numba=True):
    """Get cut-off points for tree splitting for continuous variables.

    Parameters
    ----------
    data_vector : Numpy-1D array. Sorted vector
    no_of_evalupoints : Int.   c_dict['no_of_evalupoints']
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
    no_of_evalupoints : Int.   c_dict['no_of_evalupoints']

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


def get_values_cont_x_no_numba(data_vector, no_of_evalupoints):
    """Get cut-off points for tree splitting for continuous variables.

       No longer used; only kept if no_numba version would be needed

    Parameters
    ----------
    sorted_data : Numpy-1D array. Sorted vector
    no_of_evalupoints : Int.   c_dict['no_of_evalupoints']

    Returns
    -------
    Numpy 1D-array. Sorted cut-off-points

    """
    data_vector = np.unique(data_vector)
    obs = len(data_vector)
    if no_of_evalupoints > (obs - 10):
        return data_vector
    indices = np.linspace(obs / no_of_evalupoints, obs,
                          no_of_evalupoints, endpoint=False)
    indices = np.uint32(indices)
    return data_vector[indices]


def merge_trees(tree_l, tree_r, name_x_m, type_x_m, val_x, treedepth):
    """Merge trees and add new split.

    0: Node identifier (INT: 0-...)
    1: Parent knot
    2: Child node left
    3: Child node right
    4: Type of node (1: Terminal node, no further splits
                    0: previous node that lead already to further splits)
    5: String: Name of variable used for decision of next split
    6: x_type of variable (policy categorisation, maybe different from MCF)
    7: If x_type = 'unordered': Set of values that goes to left daughter
    7: If x_type = 0: Cut-off value (larger goes to right daughter)
    8: List of Treatment state for both daughters [left, right]

    Parameters
    ----------
    tree_l : List of lists. Left tree.
    tree_r : List of lists. Right tree.
    name_x_m : String. Name of variables used for splitting.
    type_x_m : String. Type of variables used for splitting.
    val_x : Float, Int, or set of Int. Values used for splitting.
    treedepth : Int. Current level of tree. 1: final level.

    Returns
    -------
    new_tree : List of lists. The merged trees.

    """
    leaf = [None] * 9
    leaf[0] = random.randrange(100000)
    leaf[1] = None
    leaf[5] = name_x_m
    leaf[6] = type_x_m
    leaf[7] = val_x
    if treedepth == 2:  # Final split (defines 2 final leaves)
        leaf[4] = 1
        leaf[2] = None
        leaf[3] = None
        leaf[8] = [tree_l, tree_r]  # For 1st tree --> treatment states
        new_tree = [leaf]
    else:
        leaf[4] = 0
        leaf[2] = tree_l[0][0]
        leaf[3] = tree_r[0][0]
        tree_l[0][1] = leaf[0]
        tree_r[0][1] = leaf[0]
        new_tree = [None] * (1 + 2 * len(tree_l))
        new_tree[0] = leaf
        i = 1
        for i_l in tree_l:
            new_tree[i] = i_l
            i += 1
        for i_r in tree_r:
            new_tree[i] = i_r
            i += 1
    return new_tree


def evaluate_leaf(data_ps, c_dict):
    """Evaluate final value of leaf taking restriction into account.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    c_dict : Dict. Controls.

    Returns
    -------
    treat_ind: Int. Index of treatment.
    reward: Int. Value of leaf.
    no_per_treat: Numpy 1D-array of int.

    """
    if c_dict['with_numba']:
        indi, reward_by_treat, obs_all = evaluate_leaf_numba(
            data_ps, c_dict['no_of_treatments'], c_dict['max_by_treat'],
            c_dict['restricted'], c_dict['costs_of_treat'])
    else:
        indi, reward_by_treat, obs_all = evaluate_leaf_no_numba(data_ps,
                                                                c_dict)
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
    obs_all = np.zeros(no_of_treatments)
    obs = len(data_ps)
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
    reward_by_treat = data_ps.sum(axis=0) - costs_of_treat * obs
    max_i = np.argmax(reward_by_treat)
    obs_all[indi[max_i]] = obs
    return indi[max_i], reward_by_treat[max_i], obs_all


def evaluate_leaf_no_numba(data_ps, c_dict):
    """Evaluate final value of leaf taking restriction into account.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    max_per_treat : Tuple of int. Maximum number of obs in treatment.

    Returns
    -------
    treat_ind: Int. Index of treatment.
    reward: Int. Value of leaf.
    no_per_treat: Numpy 1D-array of int.

    """
    obs_all = np.zeros(c_dict['no_of_treatments'])
    obs = len(data_ps)
    indi = np.arange(c_dict['no_of_treatments'])
    if c_dict['restricted']:
        diff_obs = obs - c_dict['max_by_treat']
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
            costs_of_treat = c_dict['costs_of_treat'][indi]
        else:
            costs_of_treat = c_dict['costs_of_treat']
    else:
        costs_of_treat = c_dict['costs_of_treat']
    reward_by_treat = data_ps.sum(axis=0) - costs_of_treat * obs
    max_i = np.argmax(reward_by_treat)
    obs_all[indi[max_i]] = obs
    return indi[max_i], reward_by_treat[max_i], obs_all


def tree_search(data_ps, data_ps_diff, data_x, name_x, type_x, values_x,
                c_dict, treedepth, no_further_splits=False):
    """Build tree.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    data_ps_diff : Numpy array. Policy scores as differences.
    data_x : Numpy array. Policy variables.
    ind_sort_x : Numpy array. Sorted Indices with respect to cols. of x
    ind_leaf: Numpy array. Remaining data in leaf.
    name_x : List of strings. Name of policy variables.
    type_x : List of strings. Type of policy variable.
    values_x : List of sets. Values of x for non-continuous variables.
    c_dict : Dict. Parameters.
    treedepth : Int. Current depth of tree.
    no_further_splits : Boolean.
        Further splits do not matter. Take next (1st) split as final. Default
        is False.

    Returns
    -------
    tree : List of lists. Current tree.
    reward : Float. Total reward that comes from this tree.
    no_by_treat : List of int. Number of treated by treatment state (0-...)

    """
    if treedepth == 1:  # Evaluate tree
        tree, reward, no_by_treat = evaluate_leaf(data_ps, c_dict)
    else:
        if not no_further_splits and (treedepth < c_dict['depth_of_tree']):
            no_further_splits = only_1st_tree_fct3(data_ps, c_dict)
        min_leaf_size = c_dict['min_leaf_size'] * 2**(treedepth - 2)
        no_of_x = len(type_x)
        reward = -math.inf  # minus infinity
        tree = None
        no_by_treat = None
        for m_i in range(no_of_x):
            if c_dict['with_output']:
                if treedepth == c_dict['depth_of_tree']:
                    print('{:20s} '.format(name_x[m_i]),
                          '{:4.1f}%'.format(m_i / no_of_x * 100),
                          'of variables completed')
            if type_x[m_i] == 'cont':
                values_x_to_check = get_values_cont_x(
                    data_x[:, m_i], c_dict['no_of_evalupoints'],
                    with_numba=c_dict['with_numba'])
            elif type_x[m_i] == 'disc':
                values_x_to_check = values_x[m_i][:]
            else:
                if treedepth < c_dict['depth_of_tree']:
                    values_x_to_check = combinations_categorical(
                        data_x[:, m_i], data_ps_diff, c_dict)
                else:
                    values_x_to_check = values_x[m_i][:]
            for val_x in values_x_to_check:
                if type_x[m_i] == 'unord':
                    left = np.isin(data_x[:, m_i], val_x)
                else:
                    left = data_x[:, m_i] <= (val_x + 1e-15)
                obs_left = np.count_nonzero(left)
                if not (min_leaf_size <= obs_left
                        <= (len(left) - min_leaf_size)):
                    continue
                right = np.invert(left)
                tree_l, reward_l, no_by_treat_l = tree_search(
                    data_ps[left, :], data_ps_diff[left, :], data_x[left, :],
                    name_x, type_x, values_x, c_dict, treedepth - 1,
                    no_further_splits)
                tree_r, reward_r, no_by_treat_r = tree_search(
                    data_ps[right, :], data_ps_diff[right, :],
                    data_x[right, :], name_x, type_x, values_x, c_dict,
                    treedepth - 1, no_further_splits)
                if c_dict['restricted']:
                    reward_l, reward_r = adjust_reward(
                        no_by_treat_l, no_by_treat_r, reward_l, reward_r,
                        c_dict)
                if reward_l + reward_r > reward:
                    reward = reward_l + reward_r
                    no_by_treat = no_by_treat_l + no_by_treat_r
                    tree = merge_trees(tree_l, tree_r, name_x[m_i],
                                       type_x[m_i], val_x, treedepth)
                if no_further_splits:
                    return tree, reward, no_by_treat
    return tree, reward, no_by_treat


def only_1st_tree_fct(data_ps, c_dict):
    """Find out if further splits make any sense. NOT USED."""
    no_further_splitting = True
    for i, _ in enumerate(data_ps):
        if i == 0:
            ref_val = np.argmax(data_ps[i]-c_dict['costs_of_treat'])
        else:
            opt_treat = np.argmax(data_ps[i]-c_dict['costs_of_treat'])
            if ref_val != opt_treat:
                no_further_splitting = False
                break
    return no_further_splitting


def only_1st_tree_fct2(data_ps, c_dict):
    """Find out if further splits make any sense.NOT USED."""
    no_further_splitting = True
    opt_treat = np.argmax(data_ps-c_dict['costs_of_treat'], axis=1)
    no_further_splitting = np.all(opt_treat == opt_treat[0])
    return no_further_splitting


def only_1st_tree_fct3(data_ps, c_dict):
    """Find out if further splits make any sense."""
    data = data_ps-c_dict['costs_of_treat']
    no_further_splitting = all_same_max_numba(data)
    return no_further_splitting


@njit
def all_same_max_numba(data):
    """Check same categies have max."""
    no_further_splitting = True
    for i in range(len(data)):
        if i == 0:
            ref_val = np.argmax(data[i, :])
        else:
            opt_treat = np.argmax(data[i, :])
            if ref_val != opt_treat:
                no_further_splitting = False
                break
    return no_further_splitting


def tree_search_multip_single(data_ps, data_ps_diff, data_x, name_x, type_x,
                              values_x, c_dict, treedepth, m_i):
    """Build tree. Only first level. For multiprocessing only.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    data_ps_diff : Numpy array. Policy scores relative to reference category.
    data_x : Numpy array. Policy variables.
    ind_sort_x : Numpy array. Sorted Indices with respect to cols. of x
    ind_leaf: Numpy array. Remaining data in leaf.
    name_x : List of strings. Name of policy variables.
    type_x : List of strings. Type of policy variable.
    values_x : List of sets. Values of x for non-continuous variables.
    c_dict : Dict. Parameters.
    treedepth : Current depth of tree.

    Returns
    -------
    tree : List of lists. Current tree.
    reward : Float. Total reward that comes from this tree.
    no_by_treat : List of int. Number of treated by treatment state (0-...)

    """
    if treedepth == 1:  # Evaluate tree
        raise Exception('This should not happen in Multiprocessing.')
    reward = -math.inf  # minus infinity
    tree = None
    no_by_treat = None
    # data_x_m_i = data_x[:, m_i].copy()
    if type_x[m_i] == 'cont':
        values_x_to_check = get_values_cont_x(
            data_x[:, m_i], c_dict['no_of_evalupoints'],
            with_numba=c_dict['with_numba'])
    elif type_x[m_i] == 'disc':
        values_x_to_check = values_x[m_i][:]
    else:
        if treedepth < c_dict['depth_of_tree']:
            values_x_to_check = combinations_categorical(
                data_x[:, m_i], data_ps_diff, c_dict)
        else:
            values_x_to_check = values_x[m_i][:]
    for val_x in values_x_to_check:
        if type_x[m_i] == 'unord':
            left = np.isin(data_x[:, m_i], val_x)
        else:
            left = data_x[:, m_i] <= (val_x + 1e-15)
        obs_left = np.count_nonzero(left)
        if not (c_dict['min_leaf_size'] <= obs_left
                <= (len(left)-c_dict['min_leaf_size'])):
            continue
        # if np.all(left):
        #     continue
        # if not np.any(left):
        #     continue
        right = np.invert(left)
        tree_l, reward_l, no_by_treat_l = tree_search(
            data_ps[left, :], data_ps_diff[left, :], data_x[left, :],
            name_x, type_x, values_x, c_dict, treedepth - 1)
        tree_r, reward_r, no_by_treat_r = tree_search(
            data_ps[right, :], data_ps_diff[right, :], data_x[right, :],
            name_x, type_x, values_x, c_dict, treedepth - 1)
        if c_dict['restricted']:
            reward_l, reward_r = adjust_reward(
                no_by_treat_l, no_by_treat_r, reward_l, reward_r, c_dict)
        if reward_l + reward_r > reward:
            reward = reward_l + reward_r
            no_by_treat = no_by_treat_l + no_by_treat_r
            tree = merge_trees(tree_l, tree_r, name_x[m_i],
                               type_x[m_i], val_x, treedepth)
    return tree, reward, no_by_treat


def adjust_reward(no_by_treat_l, no_by_treat_r, reward_l, reward_r, c_dict):
    """Adjust rewards if restrictions are violated.

    Parameters
    ----------
    no_by_treat_l : Numpy array.
    no_by_treat_r : Numpy array.
    reward_l : Float.
    reward_r : Float.
    c_dict : Dict. Parameter.

    Returns
    -------
    reward_l : Numpy array.
    reward_r : Numpy array.

    """
    if c_dict['with_numba']:
        reward_l, reward_r = adjust_reward_numba(
            no_by_treat_l, no_by_treat_r, reward_l, reward_r,
            c_dict['max_by_treat'])
    else:
        reward_l, reward_r = adjust_reward_no_numba(
            no_by_treat_l, no_by_treat_r, reward_l, reward_r,
            c_dict['max_by_treat'])
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
    c_dict : Dict. Parameter.

    Returns
    -------
    reward_l : Numpy array.
    reward_r : Numpy array.

    """
    if not ((no_by_treat_l is None) or (no_by_treat_r is None)):
        violations = (no_by_treat_l + no_by_treat_r) > max_by_treat
        if np.any(violations):
            diff = (no_by_treat_l + no_by_treat_r - max_by_treat)
            diff = diff / max_by_treat
            diff = diff.max()
            diff = min(diff, 1)
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
        # no_by_treat_l = -math.inf
        # no_by_treat_r = -math.inf
        return reward_l, reward_r
    if np.any(no_by_treat_l + no_by_treat_r > max_by_treat):
        diff = (no_by_treat_l + no_by_treat_r - max_by_treat)
        diff = diff / max_by_treat
        diff = diff.max()
        diff = min(diff, 1)
        reward_l = reward_l - diff * np.abs(reward_l)
        reward_r = reward_r - diff * np.abs(reward_r)
    return reward_l, reward_r


def adjust_policy_score(datafile_name, c_dict, v_dict):
    """
    Adjust the policy score to account for insignificant effects.

    Parameters
    ----------
    datafile_name (str): Name of data file.
    c_dict (dict): Dictionary with controls.
    v_dict (dict): Dictionary with variables.

    Returns
    -------
    data_ps (numpy array, N x no of treatment): Policy scores.

    """
    data_df = pd.read_csv(datafile_name)
    data_ps = data_df[v_dict['polscore_name']].to_numpy(copy=True)
    data_ps_vs_0 = data_df[v_dict['effect_vs_0']].to_numpy()
    data_ps_vs_0_se = data_df[v_dict['effect_vs_0_se']].to_numpy()
    p_val = sct.t.sf(np.abs(data_ps_vs_0 / data_ps_vs_0_se), 1000000)  # 1sided
    no_of_recoded = 0
    for i in range(len(data_ps)):
        for idx, _ in enumerate(v_dict['effect_vs_0']):
            if (data_ps_vs_0[i, idx] > 0) and (
                    p_val[i, idx] > c_dict['sig_level_vs_0']):
                data_ps[i, idx+1] = data_ps[i, 0] - 1e-8  # a bit smaller
                no_of_recoded += 1
    if c_dict['with_output']:
        print()
        print('{:5d} policy scores recoded'.format(no_of_recoded))
    return data_ps, data_df


def optimal_tree_proc(datafile_name, x_type, x_value, v_dict, c_dict):
    """Build optimal policy tree.

    This function is for multiprocessing only.

    Parameters
    ----------
    datafile_name: String.
    x_type : Dict. Type information of variables.
    x_value : Dict. Value information of variables.
    v_dict: Dict. Variables.
    c_dict : Dict. Parameters.

    Returns
    -------
    optimal_tree :  List of lists.
    optimal_reward: Int.

    """
    if c_dict['with_output']:
        print('Building policy/decision tree')
    if c_dict['only_if_sig_better_vs_0']:
        data_ps, data_df = adjust_policy_score(datafile_name, c_dict, v_dict)
    else:
        data_df = pd.read_csv(datafile_name)
        data_ps = data_df[v_dict['polscore_name']].to_numpy()
    data_ps_diff = data_ps[:, 1:] - data_ps[:, 0, np.newaxis]
    no_of_x = len(x_type)
    name_x = [None] * no_of_x
    type_x = [None] * no_of_x
    values_x = [None] * no_of_x
    for j, key in enumerate(x_type.keys()):
        # name_x[j] = copy.copy(key)
        name_x[j] = key
        type_x[j] = x_type[key]
        if x_value[key] is not None:
            values_x[j] = sorted(x_value[key])
        else:
            values_x[j] = None
    data_x = data_df[name_x].to_numpy()
    del data_df
    if c_dict['x_unord_flag']:
        for m_i in range(no_of_x):
            if type_x[m_i] == 'unord':
                data_x[:, m_i] = np.round(data_x[:, m_i])
                values_x[m_i] = combinations_categorical(
                    data_x[:, m_i], data_ps_diff, c_dict)
    optimal_tree = None
    x_trees = []
    if c_dict['parallel']:
        para = c_dict['no_parallel']
        with futures.ProcessPoolExecutor(max_workers=para) as fpp:
            trees = {fpp.submit(tree_search_multip_single, data_ps,
                                data_ps_diff, data_x, name_x, type_x, values_x,
                                c_dict, c_dict['depth_of_tree'], m_i):
                     m_i for m_i in range(len(type_x))}
            for idx, val in enumerate(futures.as_completed(trees)):
                if c_dict['with_output']:
                    gp.share_completed(idx, len(type_x))
                x_trees.append(val.result())
        optimal_reward = np.empty(len(type_x))
        for idx, tree in enumerate(x_trees):
            optimal_reward[idx] = tree[1]
        max_i = np.argmax(optimal_reward)
        optimal_reward = optimal_reward[max_i]
        optimal_tree = x_trees[max_i][0]
        obs_total = x_trees[max_i][2]
    else:
        optimal_tree, optimal_reward, obs_total = tree_search(
            data_ps, data_ps_diff, data_x, name_x, type_x, values_x, c_dict,
            c_dict['depth_of_tree'])
    return optimal_tree, optimal_reward, obs_total


def structure_of_node_tabl_poltree():
    """Info about content of NODE_TABLE.

    Returns
    -------
    decription : STR. Information on node table with inital node

    """
    description = """Trees are fully saved in Node_Table (list of lists)
    Structure des Node_table
      - Each knot is one list that contains further lists
    This is the position and information for a given node
    The following items will be filled in the first sample
    0: Node identifier (INT: 0-...)
    1: Parent knot
    2: Child node left
    3: Child node right
    4: Type of node (2: Active -> will be further splitted or made terminal
                    1: Terminal node, no further splits
                    0: previous node that lead already to further splits)
    5: String: Name of variable used for decision of next split
    6: x_type of variable (policy categorisation, maybe different from MCF)
    7: If x_type = 'unordered': Set of values that goes to left daughter
    7: If x_type = 0: Cut-off value (larger goes to right daughter,
                                    equal and smaller to left daughter)

    """
    print("\n", description)


def init_node_table_poltree():
    """Initialise Node table for first leaf.

    Parameters
    ----------
    n_tr : Int. Number of observation in training subsample.

    Returns
    -------
    node_table : List of lists. First init_node_table

    """
    node_table = [None] * 8
    node_table[0] = 0
    node_table[1] = 0
    node_table[2] = 1
    node_table[3] = 2
    node_table[4] = 2
    return [node_table]


def classify_var_for_pol_tree(datafile_name, v_dict, c_dict, all_var_names):
    """Classify variables as most convenient for policy trees building.

    Parameters
    ----------
    datafile_name : String.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    all_var_names : List of strings. All variables available.

    Returns
    -------
    x_type_dict : Dict. Key is variable name. Value is type of variable.
    x_value_dict : Dict. Key is variable name. Value is list of values.

    """
    if not all_var_names:
        raise Exception('No variables left.')
    data = pd.read_csv(datafile_name)
    x_continuous = False
    x_ordered = False
    x_unordered = False
    x_type_dict = {}
    x_value_dict = {}
    for var in all_var_names:
        values = np.unique(data[var].to_numpy())  # Sorted values
        if var in v_dict['x_ord_name']:
            if len(values) > c_dict['no_of_evalupoints']:
                x_type_dict.update({var: 'cont'})
                x_value_dict.update({var: None})
                x_continuous = True
            else:
                x_type_dict.update({var: 'disc'})
                x_value_dict.update({var: values.tolist()})
                x_ordered = True
        elif var in v_dict['x_unord_name']:
            values_round = np.round(values)
            if np.sum(np.abs(values-values_round)) > 1e-10:
                raise Exception('Categorical variables must be coded as',
                                'integers.')
            x_type_dict.update({var: 'unord'})
            x_value_dict.update({var: values.tolist()})
            x_unordered = True
        else:
            raise Exception(var + 'is neither ordered nor unordered.')
    c_dict.update({'x_cont_flag': x_continuous,
                   'x_ord_flag': x_ordered,
                   'x_unord_flag': x_unordered})
    return x_type_dict, x_value_dict, c_dict


def check_adjust_varnames(variable_name, datafile_name):
    """Adjust variable name to its spelling in file. Check if in file.

    Parameters
    ----------
    variable_name : List of strings.
    file_name : String. CSV file.

    Returns
    -------
    new_variable_name : List of strings.

    """
    data = pd.read_csv(datafile_name)
    variable_in_file = list(data)   # List of variable names
    new_variable_name = variable_name[:]
    variable_in_file_upper = [s.upper() for s in variable_in_file]
    missing_variables = []
    for name_i, name in enumerate(variable_name):
        name_upper = name.upper()
        if name_upper in variable_in_file_upper:
            posi = variable_in_file_upper.index(name_upper)
            new_variable_name[name_i] = variable_in_file[posi]
        else:
            missing_variables.append(name)
    if not missing_variables:       # List is empty
        return new_variable_name
    print('The following variables are not in the data: ',
          missing_variables)
    print('The data consists only of these variables:', variable_in_file)
    raise Exception('Programme stopped because of missing variables.')


def get_controls(c_dict, v_dict):
    """Update defaults for controls and variables.

    Parameters
    ----------
    c : Dict. Parameters.
    v : Dict. Variables.

    Returns
    -------
    con : Dict. Parameters.
    var : Dict. Variables.

    """
    path_programme_run = str(Path(__file__).parent.absolute())
    if c_dict['datpfad'] is None:
        c_dict['datpfad'] = path_programme_run
    if c_dict['outpfad'] is None:
        c_dict['outpfad'] = path_programme_run + '/out'
    if os.path.isdir(c_dict['outpfad']):
        if not c_dict['with_output'] is False:
            print("Directory for output %s already exists" % c_dict['outpfad'])
    else:
        try:
            os.mkdir(c_dict['outpfad'])
        except OSError as error:
            raise Exception("Creation of the directory %s failed"
                            % c_dict['outpfad']) from error
        else:
            if not c_dict['with_output'] is False:
                print("Successfully created the directory %s"
                      % c_dict['outpfad'])
    if c_dict['indata'] is None:
        raise Exception('Filename of indata must be specified')
    if c_dict['outfiletext'] is None:
        c_dict['outfiletext'] = c_dict['indata']
    temppfad = c_dict['outpfad'] + '/_tempoptp_'
    if os.path.isdir(temppfad):
        file_list = os.listdir(temppfad)
        if file_list:
            for temp_file in file_list:
                os.remove(os.path.join(temppfad, temp_file))
        if not c_dict['with_output'] is False:
            print("Temporary directory  %s already exists" % temppfad)
            if file_list:
                print('All files deleted.')
    else:
        try:
            os.mkdir(temppfad)
        except OSError as error:
            raise Exception("Creation of the directory %s failed" % temppfad
                            ) from error
        else:
            if not c_dict['with_output'] is False:
                print("Successfully created the directory %s" % temppfad)

    c_dict['outfiletext'] = (c_dict['outpfad'] + '/' + c_dict['outfiletext']
                             + '.txt')
    c_dict['indata'] = c_dict['datpfad'] + '/' + c_dict['indata'] + '.csv'
    indata_temp = temppfad + '/' + 'indat_temp' + '.csv'

    if c_dict['output_type'] is None:
        c_dict['output_type'] = 2
    if c_dict['output_type'] == 0:
        print_to_file = False
        print_to_terminal = True
    elif c_dict['output_type'] == 1:
        print_to_file = True
        print_to_terminal = False
    else:
        print_to_file = True
        print_to_terminal = True

    if c_dict['parallel'] is not False:
        c_dict['parallel'] = True
    if c_dict['parallel'] == 0:
        c_dict['parallel'] = False
    else:
        c_dict['parallel'] = True
    if c_dict['no_parallel'] is None:
        c_dict['no_parallel'] = 0
    if c_dict['no_parallel'] < 0.5:
        c_dict['no_parallel'] = psutil.cpu_count()
    else:
        c_dict['no_parallel'] = round(c_dict['no_parallel'])
    if c_dict['screen_covariates'] is not False:
        c_dict['screen_covariates'] = True
    if c_dict['check_perfectcorr'] is not True:
        c_dict['check_perfectcorr'] = False
    if c_dict['min_dummy_obs'] is None:
        c_dict['min_dummy_obs'] = 0
    if c_dict['min_dummy_obs'] < 1:
        c_dict['min_dummy_obs'] = 10
    else:
        c_dict['min_dummy_obs'] = round(c_dict['min_dummy_obs'])
    if c_dict['clean_data_flag'] is not False:
        c_dict['clean_data_flag'] = True
    if c_dict['no_of_evalupoints'] is None:
        c_dict['no_of_evalupoints'] = 0
    if c_dict['no_of_evalupoints'] < 5:
        c_dict['no_of_evalupoints'] = 100
    else:
        c_dict['no_of_evalupoints'] = round(c_dict['no_of_evalupoints'])
    if c_dict['max_shares'] is None:
        c_dict['max_shares'] = [1 for i in range(len(v_dict['polscore_name']))]
    if len(c_dict['max_shares']) != len(v_dict['polscore_name']):
        raise Exception('# of policy scores different from # of restrictions.')
    if c_dict['depth_of_tree'] is None:
        c_dict['depth_of_tree'] = 0
    if c_dict['depth_of_tree'] < 1:  # 'normal definition of depth + 1'
        c_dict['depth_of_tree'] = 4
    else:
        c_dict['depth_of_tree'] = int(round(c_dict['depth_of_tree']) + 1)
    zeros = 0
    for i in c_dict['max_shares']:
        if not 0 <= i <= 1:
            raise Exception('Restrictions not between 0 and 1.')
        if i == 0:
            zeros += 1
    if zeros == len(c_dict['max_shares']):
        raise Exception('All restrictions are zero. No allocation possible.')
    if sum(c_dict['max_shares']) < 1:
        raise Exception('Sum of restrictions < 1. No allocation possible.')
    restricted = bool(np.any(np.array(c_dict['max_shares']) < 1))
    if c_dict['costs_of_treat'] is None:
        c_dict['costs_of_treat'] = 0
    if isinstance(c_dict['costs_of_treat'], (int, float)):
        if ((c_dict['costs_of_treat'] == 0) or
                np.all(np.array(c_dict['max_shares']) >= 1)):
            c_dict['costs_of_treat'] = np.zeros(len(c_dict['max_shares']))
        else:
            c_dict['costs_of_treat'] = -1
    else:
        if len(c_dict['costs_of_treat']) != len(c_dict['max_shares']):
            c_dict['costs_of_treat'] = np.zeros(len(c_dict['max_shares']))
        else:
            c_dict['costs_of_treat'] = np.array(c_dict['costs_of_treat'])
    if c_dict['costs_mult'] is None:
        c_dict['costs_mult'] = 1
    if isinstance(c_dict['costs_mult'], (int, float)):
        if c_dict['costs_mult'] < 0:
            c_dict['costs_mult'] = 1
    else:
        if len(c_dict['costs_mult']) != len(c_dict['max_shares']):
            c_dict['costs_mult'] = 1
        else:
            c_dict['costs_mult'] = np.array(c_dict['costs_mult'])
            c_dict['costs_mult'][c_dict['costs_mult'] < 0] = 0
    if c_dict['with_numba'] is not False:
        c_dict['with_numba'] = True

    v_dict_new = copy.deepcopy(v_dict)
    if not c_dict['clean_data_flag']:  # otherwise all vars are capital
        for key in v_dict.keys():   # Check if variables are ok
            v_dict_new[key] = check_adjust_varnames(v_dict[key],
                                                    c_dict['indata'])
    with open(c_dict['indata']) as file:
        n_of_obs = sum(1 for line in file)-1
    max_by_treat = np.ceil(n_of_obs * np.array(c_dict['max_shares']))
    for i, val in enumerate(max_by_treat):
        if val > n_of_obs:
            max_by_treat = n_of_obs
    no_of_treatments = len(v_dict_new['polscore_name'])
    if c_dict['min_leaf_size'] is None:
        c_dict['min_leaf_size'] = 0
    if c_dict['min_leaf_size'] < 1:
        c_dict['min_leaf_size'] = int(
            0.1 * n_of_obs / (2 ** c_dict['depth_of_tree']))
    else:
        c_dict['min_leaf_size'] = int(c_dict['min_leaf_size'])
    if c_dict['only_if_sig_better_vs_0'] is not True:
        c_dict['only_if_sig_better_vs_0'] = False
    if c_dict['only_if_sig_better_vs_0']:
        if c_dict['sig_level_vs_0'] is None:
            c_dict['sig_level_vs_0'] = 0.05
        if not 0 < c_dict['sig_level_vs_0'] < 1:
            c_dict['sig_level_vs_0'] = 0.05
            if len(v_dict['effect_vs_0']) != (no_of_treatments-1):
                raise Exception('Wrong dimension of variables effect_vs_0')
            if len(v_dict['effect_vs_0_se']) != (no_of_treatments-1):
                raise Exception('Wrong dimension of variables effect_vs_0_se')
    add_c = {'temppfad': temppfad,
             'indata_temp': indata_temp,
             'print_to_file': print_to_file,
             'print_to_terminal': print_to_terminal,
             'max_by_treat': max_by_treat,
             'no_of_treatments': no_of_treatments,
             'restricted': restricted}
    c_dict.update(add_c)
    return c_dict, v_dict_new


def variable_dict(id_name, polscore_name, x_ord_name, x_unord_name,
                  effect_vs_0, effect_vs_0_se):
    """Pack variable names into a dictionary.

    Parameters
    ----------
    id_name : Tuple of string.
    polscore : Tuple of strings.
    x_ord_name : Tuple of strings.
    x_unord_name : Tuple of strings.

    Returns
    -------
    var : Dictionary. Variable names

    """
    def capital_letter_and_list(string_list):
        if not isinstance(string_list, list):
            string_list = list(string_list)
        string_list = [s.upper() for s in string_list]
        return string_list

    if id_name is None:
        id_name = []
    else:
        id_name = capital_letter_and_list(id_name)
    if (polscore_name is None) or (polscore_name == []):
        raise Exception('Policy Score must be specified.')
    polscore_name = capital_letter_and_list(polscore_name)
    if x_ord_name is None:
        x_ord_name = []
    else:
        x_ord_name = capital_letter_and_list(x_ord_name)
    if x_unord_name is None:
        x_unord_name = []
    else:
        x_unord_name = capital_letter_and_list(x_unord_name)
    if (x_ord_name == []) and (x_unord_name):
        raise Exception('x_ord_name or x_unord_name must contain names.')
    if effect_vs_0 is None:
        effect_vs_0 = []
    else:
        effect_vs_0 = capital_letter_and_list(effect_vs_0)
    if effect_vs_0_se is None:
        effect_vs_0_se = []
    else:
        effect_vs_0_se = capital_letter_and_list(effect_vs_0_se)
    var = {'id_name': id_name,
           'polscore_name': polscore_name,
           'x_ord_name': x_ord_name,
           'x_unord_name': x_unord_name,
           'effect_vs_0': effect_vs_0,
           'effect_vs_0_se': effect_vs_0_se}
    return var


def controls_into_dic(how_many_parallel, parallel_processing,
                      output_type, outpfad, datpfad,
                      indata, outfiletext, with_output, screen_covariates,
                      check_perfectcorr, clean_data_flag, min_dummy_obs,
                      no_of_evalupoints, max_shares, depth_of_tree,
                      costs_of_treat, costs_of_treat_mult, with_numba,
                      min_leaf_size, only_if_sig_better_vs_0,
                      sig_level_vs_0):
    """Build dictionary containing control parameters for later easier use.

    Parameters
    ----------
    how_many_parallel : int.
    parallel_processing : int.
    direct_output_to_file : int.
    outpfad : string.
    temppfad : string.
    datpfad : string.
    indata : string.
    outfiletext : string.
    with_output: boolean.
    screen_covariates : int.
    check_perfectcorr : int.
    clean_data_flag : int.
    min_dummy_obs : int.
    no_of_evalupoints : int.
    max_shares: tuple of float.
    ...

    Returns
    -------
    control_dic : Dict. Parameters.

    """
    control_dic = {'no_parallel': how_many_parallel,
                   'parallel': parallel_processing,
                   'output_type': output_type,
                   'outpfad': outpfad,
                   'datpfad': datpfad,
                   'indata': indata,
                   'with_output': with_output,
                   'outfiletext': outfiletext,
                   'screen_covariates': screen_covariates,
                   'check_perfectcorr': check_perfectcorr,
                   'clean_data_flag': clean_data_flag,
                   'min_dummy_obs': min_dummy_obs,
                   'no_of_evalupoints': no_of_evalupoints,
                   'max_shares': max_shares,
                   'costs_of_treat': costs_of_treat,
                   'costs_mult': costs_of_treat_mult,
                   'depth_of_tree': depth_of_tree,
                   'with_numba': with_numba,
                   'min_leaf_size': min_leaf_size,
                   'only_if_sig_better_vs_0': only_if_sig_better_vs_0,
                   'sig_level_vs_0': sig_level_vs_0
                   }
    return control_dic
