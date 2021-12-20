"""Created on Fri Jun 26 12:42:02 2020.

Optimal Policy Trees: Functions - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

# -*- coding: utf-8 -*-
"""
import copy
import math
import time
import sys
from multiprocessing import freeze_support
from pathlib import Path
import os
import pandas as pd
import numpy as np
from psutil import cpu_count
from mcf import opt_tree_functions as opttf
from mcf import general_purpose as gp


def optpoltree(
    indata=None, preddata=None, datpath=None, outpath=None,
    save_pred_to_file=None, id_name=None, polscore_name=None, x_ord_name=None,
    x_unord_name=None, effect_vs_0=None, effect_vs_0_se=None, output_type=2,
    outfiletext=None, mp_with_ray=True, parallel_processing=True,
    how_many_parallel=None, with_numba=True, screen_covariates=True,
    check_perfectcorr=True, min_dummy_obs=10, clean_data_flag=True,
    ft_yes=True, ft_no_of_evalupoints=100, ft_depth=3, ft_min_leaf_size=None,
    st_yes=True, st_depth=5, st_min_leaf_size=None, max_shares=None,
    costs_of_treat=None, costs_of_treat_mult=1, only_if_sig_better_vs_0=False,
        sig_level_vs_0=0.05, _smaller_sample=None, _with_output=False):
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
        indata, preddata, outfiletext, _with_output, screen_covariates,
        check_perfectcorr, clean_data_flag, min_dummy_obs,
        ft_no_of_evalupoints,
        max_shares, ft_depth, costs_of_treat, costs_of_treat_mult,
        with_numba, ft_min_leaf_size, only_if_sig_better_vs_0, sig_level_vs_0,
        save_pred_to_file, mp_with_ray, st_yes, st_depth, st_min_leaf_size,
        ft_yes)
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
        if c_dict['indata'] != c_dict['preddata']:
            gp.print_descriptive_stats_file(
                c_dict['preddata'], to_file=c_dict['print_to_file'])
    else:
        c_dict['print_to_file'] = False
    names_to_check_train = (v_dict['id_name'] + v_dict['polscore_name']
                            + v_dict['x_ord_name'] + v_dict['x_unord_name'])
    if c_dict['only_if_sig_better_vs_0']:
        names_to_check_train = names_to_check_train + v_dict[
            'effect_vs_0'] + v_dict['effect_vs_0_se']
    if c_dict['indata'] != c_dict['preddata'] and c_dict['with_output']:
        gp.check_all_vars_in_data(
            c_dict['preddata'], v_dict['x_ord_name'] + v_dict['x_unord_name'])
        gp.print_descriptive_stats_file(
            c_dict['preddata'], varnames=v_dict['x_ord_name']
            + v_dict['x_unord_name'], to_file=c_dict['print_to_file'])
    # Prepare data
    # Remove missings and keep only variables needed for further analysis
    if c_dict['clean_data_flag']:
        indata2 = gp.clean_reduce_data(
            c_dict['indata'], c_dict['indata_temp'], names_to_check_train,
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
    if c_dict['ft_yes']:
        optimal_tree, _, _ = opttf.optimal_tree_proc(
            indata2, x_type, x_value, v_dict, c_dict)
    else:
        optimal_tree = None
    if c_dict['st_yes']:
        sequential_tree, _, _ = opttf.sequential_tree_proc(
            indata2, x_type, x_value, v_dict, c_dict)
    else:
        sequential_tree = None
    # Prozedur um den Output darzustellen
    if c_dict['with_output'] and c_dict['ft_yes']:
        print('=' * 80)
        print('OPTIMAL Policy Tree')
        print('=' * 80)
        descr_policy_tree(indata2, optimal_tree, x_type, x_value, v_dict,
                          c_dict)
    if c_dict['with_output'] and c_dict['st_yes']:
        print('=' * 80)
        print('SEQUENTIAL Policy Tree')
        print('=' * 80)
        descr_policy_tree(indata2, sequential_tree, x_type, x_value,
                          v_dict, c_dict)
    if c_dict['ft_yes']:
        opt_alloc_pred = pred_policy_allocation(
            optimal_tree, x_value, v_dict, c_dict,
            len(v_dict['polscore_name']))
    else:
        opt_alloc_pred = None
    if c_dict['st_yes']:
        seq_alloc_pred = pred_policy_allocation(
            sequential_tree, x_value, v_dict, c_dict,
            len(v_dict['polscore_name']))
    else:
        sequential_tree = None
    time3 = time.time()

    # Print timing information
    time_string = ['Data preparation: ', 'Tree building     ',
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
    return opt_alloc_pred, seq_alloc_pred


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
        data_ps, data_df = opttf.adjust_policy_score(datafile_name, c_dict,
                                                     v_dict)
    else:
        data_df = pd.read_csv(datafile_name)
        data_ps = data_df[v_dict['polscore_name']].to_numpy()
    obs = len(data_ps)
    max_by_treat = np.array(c_dict['max_by_treat'])
    costs_of_treat = np.zeros(c_dict['no_of_treatments'])
    std_ps = np.std(data_ps.reshape(-1))
    step_size = 0.02
    while True:
        treatments = np.argmax(data_ps - costs_of_treat, axis=1)
        values, count = np.unique(treatments, return_counts=True)
        if len(count) == c_dict['no_of_treatments']:
            alloc = count
        else:
            alloc = np.zeros(c_dict['no_of_treatments'])
            for i, j in enumerate(values):
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


def subsample_leaf(any_df, x_df, split):
    """Reduces dataframes to data in leaf.

    Parameters
    ----------
    any_df : Dataframe. Policyscores, indices or any other of same length as x.
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
    any_df_red = any_df[condition]
    x_df_red = x_df[condition]
    return any_df_red, x_df_red


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
    if leaf[5] is None or leaf[6] is None or leaf[7] is None or (
            left_right is None):
        print(leaf)
        raise Exception('No valid entries in final leaf.')
    return_dic = {'x_name': leaf[5], 'x_type': leaf[6],
                  'cut-off or set': leaf[7], 'left or right': left_right}
    return return_dic


def two_leafs_info(tree, polscore_df, x_df, leaf, polscore_is_index=False):
    """Compute the information contained in two adjacent leaves.

    Parameters
    ----------
    tree : List of lists.
    polscore_df : Dataframe. Policyscore or index.
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
    # Collect decision path of the two final leaves
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
    obs = (polscore_df_l.shape[0], polscore_df_r.shape[0])
    if polscore_is_index:
        # Policy score contains index of observation
        score = (0, 0)
    else:
        polscore_df_l = polscore_df_l.iloc[:, leaf[8][0]]
        polscore_df_r = polscore_df_r.iloc[:, leaf[8][1]]
        score = (polscore_df_l.sum(axis=0), polscore_df_r.sum(axis=0))
    leaf_splits_r = copy.deepcopy(leaf_splits_pre)
    leaf_splits_l = leaf_splits_pre   # one copy is enough
    leaf_splits_l.append(final_dict_l)
    leaf_splits_r.append(final_dict_r)
    leaf_splits = (leaf_splits_l, leaf_splits_r)
    polscore_df_lr = (polscore_df_l, polscore_df_r)
    # leaf 8 contains treatment information in final leaf
    return leaf_splits, score, obs, tuple(leaf[8]), polscore_df_lr


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
    print('Descriptive statistic of estimated policy tree: Training sample')
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
    print('Minimum leaf size: {:d} '.format(c_dict['ft_min_leaf_size']))
    print('- ' * 40)
    splits_seq = [None] * len(terminal_leafs)
    score_val = [None] * len(terminal_leafs)
    obs = [None] * len(terminal_leafs)
    treat = [None] * len(terminal_leafs)
    for i, leaf in enumerate(terminal_leafs):
        splits_seq[i], score_val[i], obs[i], treat[i], _ = two_leafs_info(
            tree, data_df_ps, data_df_x, leaf, polscore_is_index=False)
    total_obs_by_treat = np.zeros((data_df_ps.shape[1]))
    total_obs = total_score = total_cost = 0
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


def pred_policy_allocation(tree, x_name, v_dict, c_dict, no_of_treat):
    """Describe optimal policy tree in prediction sample and get predictions.

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
    no_of_treat: Int. Number of treated.

    Returns
    -------
    None.

    """
    data_df = pd.read_csv(c_dict['preddata'])
    x_name = [x.upper() for x in x_name]
    data_df.columns = [x.upper() for x in data_df.columns]
    data_df_x = data_df[x_name]
    total_obs = len(data_df_x)
    x_indx = pd.DataFrame(data=range(len(data_df_x)), columns=('Sorter',))
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
    if not len(set(ids)) == len(ids):
        raise Exception('Some leafs IDs are identical. Rerun programme.')
    if c_dict['with_output']:
        print('\n' + ('=' * 80))
        print('Descriptive statistic of estimated policy tree: prediction',
              ' sample')
    splits_seq = [None] * len(terminal_leafs)
    obs = [None] * len(terminal_leafs)
    treat = [None] * len(terminal_leafs)
    indx_in_leaf = [None] * len(terminal_leafs)
    for i, leaf in enumerate(terminal_leafs):
        splits_seq[i], _, obs[i], treat[i], indx_in_leaf[i] = two_leafs_info(
            tree, x_indx, data_df_x, leaf, polscore_is_index=True)
    predicted_treatment = pred_treat_fct(treat, indx_in_leaf, total_obs)
    if c_dict['save_pred_to_file']:
        pd_df = pd.DataFrame(data=predicted_treatment, columns=('Alloctreat',))
        datanew = pd.concat([pd_df, data_df], axis=1)
        gp.delete_file_if_exists(c_dict['pred_save_file'])
        datanew.to_csv(c_dict['pred_save_file'], index=False)
    if c_dict['with_output']:
        total_obs_by_treat = np.zeros(no_of_treat)
        total_obs_temp = total_cost = 0
        for i, obs_i in enumerate(obs):
            total_obs_temp += obs_i[0] + obs_i[1]
            total_obs_by_treat[treat[i][0]] += obs_i[0]
            total_obs_by_treat[treat[i][1]] += obs_i[1]
        total_cost = np.sum(c_dict['costs_of_treat'] * total_obs_by_treat)
        if int(total_obs_temp) != total_obs:
            print('Total observations in x:                       ', total_obs)
            print('Total observations in with treatment allocated:',
                  total_obs_temp)
            raise Exception('Some observations did get a treatment allocation.'
                            )
        print('- ' * 40)
        print('Total cost:         {:14.4f} '.format(total_cost),
              '  Average cost:         {:14.4f}'.format(total_cost/total_obs))
        print('- ' * 40)
        print('Total number of observations: {:d}'.format(int(total_obs)))
        print('Treatments:                           ', end=' ')
        for i, j in enumerate(v_dict['polscore_name']):
            print('{:6d} '.format(i), end=' ')
        print('\nObservations allocated, by treatment: ', end=' ')
        for i in total_obs_by_treat:
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
                print('- ' * 40)
        print('=' * 80)
    return predicted_treatment


def pred_treat_fct(treat, indx_in_leaf, total_obs):
    """Collect data and bring in the same as original data."""
    pred_treat = np.zeros((total_obs, 2), dtype=np.int64)
    idx_start = 0
    for idx, idx_leaf in enumerate(indx_in_leaf):
        idx_treat = treat[idx]
        idx_end = idx_start + len(idx_leaf[0])
        pred_treat[idx_start:idx_end, 0] = idx_leaf[0].to_numpy().flatten()
        pred_treat[idx_start:idx_end, 1] = idx_treat[0]
        idx_start = idx_end
        idx_end = idx_start + len(idx_leaf[1])
        pred_treat[idx_start:idx_end, 0] = idx_leaf[1].to_numpy().flatten()
        pred_treat[idx_start:idx_end, 1] = idx_treat[1]
        idx_start = idx_end
    pred_treat = pred_treat[pred_treat[:, 0].argsort()]  # sort on indices
    pred_treat = pred_treat[:, 1]
    return pred_treat


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
    x_continuous = x_ordered = x_unordered = False
    x_type_dict = {}
    x_value_dict = {}
    for var in all_var_names:
        values = np.unique(data[var].to_numpy())  # Sorted values
        if var in v_dict['x_ord_name']:
            if len(values) > c_dict['ft_no_of_evalupoints']:
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
    if c_dict['preddata'] is None:
        c_dict['preddata'] = c_dict['indata']
    else:
        c_dict['preddata'] = c_dict['datpfad'] + '/' + c_dict['preddata'
                                                              ] + '.csv'
    indata_temp = temppfad + '/' + 'indat_temp' + '.csv'
    c_dict['save_pred_to_file'] = c_dict['save_pred_to_file'] is not False
    c_dict['pred_save_file'] = c_dict['outfiletext'][:-4] + 'OptTreat.csv'

    if c_dict['output_type'] is None:
        c_dict['output_type'] = 2
    if c_dict['output_type'] == 0:
        print_to_file = False
        print_to_terminal = True
    elif c_dict['output_type'] == 1:
        print_to_file = True
        print_to_terminal = False
    else:
        print_to_file = print_to_terminal = True

    if c_dict['parallel'] is not False:
        c_dict['parallel'] = True
    c_dict['parallel'] = not c_dict['parallel'] == 0
    if c_dict['no_parallel'] is None:
        c_dict['no_parallel'] = 0
    if c_dict['no_parallel'] < 0.5:
        c_dict['no_parallel'] = cpu_count(logical=False) - 1
    else:
        c_dict['no_parallel'] = round(c_dict['no_parallel'])
    c_dict['mp_with_ray'] = c_dict['mp_with_ray'] is not False
    if c_dict['screen_covariates'] is not False:
        c_dict['screen_covariates'] = True
    if c_dict['check_perfectcorr'] is not True:
        c_dict['check_perfectcorr'] = False
    if c_dict['min_dummy_obs'] is None:
        c_dict['min_dummy_obs'] = 0
    c_dict['min_dummy_obs'] = 10 if c_dict['min_dummy_obs'] < 1 else round(
        c_dict['min_dummy_obs'])
    if c_dict['clean_data_flag'] is not False:
        c_dict['clean_data_flag'] = True

    if c_dict['ft_yes'] is not False:
        c_dict['ft_yes'] = True
    else:
        c_dict['ft_yes'] = False
    if c_dict['st_yes'] is not False:
        c_dict['st_yes'] = True
    else:
        c_dict['st_yes'] = False
    if c_dict['ft_no_of_evalupoints'] is None:
        c_dict['ft_no_of_evalupoints'] = 0
    c_dict['ft_no_of_evalupoints'] = 100 if c_dict[
        'ft_no_of_evalupoints'] < 5 else round(c_dict['ft_no_of_evalupoints'])
    if c_dict['max_shares'] is None:
        c_dict['max_shares'] = [1 for i in range(len(v_dict['polscore_name']))]
    if len(c_dict['max_shares']) != len(v_dict['polscore_name']):
        raise Exception('# of policy scores different from # of restrictions.')
    if c_dict['ft_depth'] is None:
        c_dict['ft_depth'] = 0
    c_dict['ft_depth'] = 4 if c_dict['ft_depth'] < 1 else int(
        round(c_dict['ft_depth']) + 1)
    if c_dict['st_depth'] is None:
        c_dict['st_depth'] = 0
    c_dict['st_depth'] = 3 if c_dict['st_depth'] < 1 else int(
        round(c_dict['st_depth']))
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

    if c_dict['ft_min_leaf_size'] is None:
        c_dict['ft_min_leaf_size'] = 0
    if c_dict['ft_min_leaf_size'] < 1:
        c_dict['ft_min_leaf_size'] = int(
            0.1 * n_of_obs / (2 ** c_dict['ft_depth']))
    else:
        c_dict['ft_min_leaf_size'] = int(c_dict['ft_min_leaf_size'])
    if c_dict['st_min_leaf_size'] is None:
        c_dict['st_min_leaf_size'] = 0
    if c_dict['st_min_leaf_size'] < 1:
        c_dict['st_min_leaf_size'] = int(
            0.1 * n_of_obs / (2 ** c_dict['st_depth']))
    else:
        c_dict['st_min_leaf_size'] = int(c_dict['st_min_leaf_size'])
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
    add_c = {
        'temppfad': temppfad, 'indata_temp': indata_temp,
        'print_to_file': print_to_file, 'print_to_terminal': print_to_terminal,
        'max_by_treat': max_by_treat, 'no_of_treatments': no_of_treatments,
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
    effect_vs_0 : None or list.
    effect_vs_0_se :  None or list.

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
    var = {
        'id_name': id_name, 'polscore_name': polscore_name,
        'x_ord_name': x_ord_name, 'x_unord_name': x_unord_name,
        'effect_vs_0': effect_vs_0, 'effect_vs_0_se': effect_vs_0_se}
    return var


def controls_into_dic(how_many_parallel, parallel_processing,
                      output_type, outpfad, datpfad, indata, preddata,
                      outfiletext, with_output, screen_covariates,
                      check_perfectcorr, clean_data_flag, min_dummy_obs,
                      ft_no_of_evalupoints, max_shares, ft_depth,
                      costs_of_treat, costs_of_treat_mult, with_numba,
                      ft_min_leaf_size, only_if_sig_better_vs_0,
                      sig_level_vs_0, save_pred_to_file, mp_with_ray,
                      st_yes, st_depth, st_min_leaf_size, ft_yes):
    """Build dictionary containing control parameters for later easier use."""
    control_dic = {
        'no_parallel': how_many_parallel, 'parallel': parallel_processing,
        'output_type': output_type, 'outpfad': outpfad, 'datpfad': datpfad,
        'save_pred_to_file': save_pred_to_file, 'indata': indata,
        'preddata': preddata, 'with_output': with_output,
        'outfiletext': outfiletext, 'screen_covariates': screen_covariates,
        'check_perfectcorr': check_perfectcorr, 'min_dummy_obs': min_dummy_obs,
        'clean_data_flag': clean_data_flag, 'mp_with_ray': mp_with_ray,
        'with_numba': with_numba, 'max_shares': max_shares,
        'costs_of_treat': costs_of_treat, 'costs_mult': costs_of_treat_mult,
        'only_if_sig_better_vs_0': only_if_sig_better_vs_0,
        'sig_level_vs_0': sig_level_vs_0, 'ft_yes': ft_yes,
        'ft_depth': ft_depth, 'ft_no_of_evalupoints': ft_no_of_evalupoints,
        'ft_min_leaf_size': ft_min_leaf_size,  'st_yes': st_yes,
        'st_min_leaf_size': st_min_leaf_size, 'st_depth': st_depth
                   }
    return control_dic
