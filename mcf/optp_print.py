"""Created on April, 4, 2022.

Optimal Policy - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

# -*- coding: utf-8 -*-

Print functions

"""
import math

import numpy as np
import pandas as pd

from mcf import optp_tree_add_functions as optp_ta


def print_alloc_other_outcomes(results, with_boot=False):
    """Describe allocation using additional estimated potential outcomes."""
    var_name, obs = results['name_add'], results['obs']
    number_var = round(len(var_name) / results['no_of_treat'])
    all_idx = np.reshape(np.arange(len(var_name)),
                         (number_var, results['no_of_treat']))
    print('-' * 80)
    for var_idx, idx in enumerate(all_idx):
        score_add = results['score_add'][var_idx]
        score_change_add = results['score_change_add'][var_idx]
        if results['obs_change'] > 0:
            score_add_m_obs = results['score_add_m_obs'][var_idx]
            score_change_add_m_obs = results['score_change_add_m_obs'][var_idx]
        if with_boot:
            score_add_std = results['score_add_std'][var_idx]
            score_add_q = results['score_add_q'][var_idx]
            if results['obs_change'] > 0:
                score_add_m_obs_std = results['score_add_m_obs_std'][var_idx]
                score_add_m_obs_q = results['score_add_m_obs_q'][var_idx]
                score_change_add_std = results['score_change_add_std'][var_idx]
                score_change_add_q = results['score_change_add_q'][var_idx]
                score_change_add_m_obs_std = results[
                    'score_change_add_m_obs_std'][var_idx]
                score_change_add_m_obs_q = results[
                    'score_change_add_m_obs_q'][var_idx]
        var_name_idx = [var_name[i] for i in idx]
        print('Scores: ', end=' ')
        for i in var_name_idx:
            print(i, end=' ')
        print('\n' + '-   ' * 20)
        print(f'Total score (level):   {score_add:.4f} ',
              f'     Average score (level):  {score_add / obs:.4f}')
        if with_boot:
            print('-   ' * 20)
            print_boot_info(score_add_std, score_add_q, results['quants'], obs)
        if results['obs_change'] > 0:
            print('-   ' * 20)
            print_changers_info(results['obs_change'], score_change_add,
                                print_line=False)
            if with_boot:
                print('-   ' * 20)
                print_boot_info(score_change_add_std, score_change_add_q,
                                results['quants'], results['obs_change'])
            print('- ' * 40)
            print('Relative to observed allocation', '\n' + '-   ' * 20)
            print(f'Total score - observed alloc: {score_add_m_obs:.4f} ',
                  f'     Average score:        {score_add_m_obs / obs:.4f}')
            if with_boot:
                print('-   ' * 20)
                print_boot_info(score_add_m_obs_std, score_add_m_obs_q,
                                results['quants'], obs)
            print('-   ' * 20)
            print_changers_info(
                results['obs_change'], score_change_add_m_obs,
                print_line=False)
            if with_boot:
                print('-   ' * 20)
                print_boot_info(score_change_add_m_obs_std,
                                score_change_add_m_obs_q,
                                results['quants'], results['obs_change'])
        print('-' * 80)


def print_changers_info(obs_changers, score_changers, print_line=True):
    """Print changers info."""
    if obs_changers > 0:
        print(f'Total score (changers): {score_changers:12.4f} ',
              ' Average score (changers):',
              f'  {score_changers / obs_changers:12.4f}')
        if print_line:
            print('-' * 80)


def print_boot_info(std, quant_val, quants, obs):
    """Print bootstrap results."""
    print(f'Bootstrap std:         {std: .4f}     ',
          f'Bootstrap std of mean: {std/obs: .4f}')
    print('Bootstrap quantiles:       ', end='')
    for q_idx, quant in enumerate(quants):
        print(f'{quant: 2.0%}: {quant_val[q_idx]: .4f}', end='')
    print()
    print('Bootstrap quantiles of mean', end='')
    for q_idx, quant in enumerate(quants):
        print(f'{quant: 2.0%}: {quant_val[q_idx]/obs: .4f}', end='')
    print()


def bb_allocation_stats(allocation, c_dict, v_dict, data_file):
    """
    Show descriptive stats for the various black-box allocations.

    Parameters
    ----------
    allocation : Dict
        Dictionary of allocations.
    c_dict : Dict
        Controls.
    v_dict : Dict
        Variables.
    data_file : String.
        File name of file containing the data.
    treatment : Numpy 1D array.
        Treatment information. Default is None.

    Returns
    -------
    None.

    """
    with_boot = c_dict['bb_bootstraps'] > 0
    print('=' * 80, '\n' + 'Black-Box approaches', '\n' + '-' * 80)
    if c_dict['only_if_sig_better_vs_0']:
        print('While tree building, policy scores not significantly',
              'different from zero are set to zero.')
        print('Significance level used for recoding:',
              f' {c_dict["sig_level_vs_0"] * 100:6.4f} %', '\n' + '- ' * 40)
    print(data_file)
    print('- ' * 40)
    if c_dict['bb_stochastic']:
        print('Stochastic simulations used for allocation.', '\n' + '- ' * 40)
    if c_dict['bb_bootstraps'] > 0:
        print('Number of bootstrap replications:',
              f' {c_dict["bb_bootstraps"]: 5d}', '\n' + '- ' * 40)
    print('Policy scores: ', end=' ')
    for i in v_dict['polscore_name']:
        print(i, end=' ')
    print('\n' + '-' * 80)
    for alloc in allocation:
        results = alloc['results']
        print('*' * 80, '\nBlack Box ' + alloc['type'], '\n' + '-' * 80)
        print_allocation(results['score'], results['obs'], v_dict, c_dict,
                         results['obs_by_treat'])
        print_boot_info(results['score_std'], results['score_q'],
                        results['quants'], results['obs'])
        if results['obs_change'] > 0:
            print('-   ' * 20)
            print(f'Number of changers: {results["obs_change"]:d}',
                  f'({results["obs_change"]/results["obs"]:.4%})')
            print('-   ' * 20)
            print_changers_info(results['obs_change'], results['score_change'],
                                print_line=False)
            if with_boot:
                print_boot_info(results['score_change_std'],
                                results['score_change_q'], results['quants'],
                                results['obs_change'])
            print('\n' + '- ' * 40 + '\nRelative to observed alloction:',
                  '\n' + '-   ' * 20)
            print('Total score - observed alloc:   ',
                  f'{results["score_m_obs"]:14.4f}    Average score:      ',
                  f'  {results["score_m_obs"] / results["obs"]:14.4f}')
            print_boot_info(results['score_m_obs_std'],
                            results['score_m_obs_q'], results['quants'],
                            results['obs'])
            print('-   ' * 20)
            print_changers_info(results['obs_change'],
                                results['score_change_m_obs'],
                                print_line=False)
            if with_boot:
                print_boot_info(results['score_change_m_obs_std'],
                                results['score_change_m_obs_q'],
                                results['quants'], results['obs_change'])
        else:
            print('-   ' * 20, '\nEither there are no changers or ',
                  'treatment variable is not available.')
        if v_dict['polscore_desc_name'] is not None:
            print_alloc_other_outcomes(results, with_boot)
    print('For relative-to-population scores, score of changers and',
          'population should be the same')
    print('-' * 80)


def print_allocation(score, obs, v_dict, c_dict, obs_by_treat,
                     prediction=False):
    """Show total policy score and observations by treatment."""
    cost = np.sum(c_dict['costs_of_treat'] * obs_by_treat)
    if score is not None:
        print(f'Total score:        {score:14.4f} ',
              f'  Average score:        {score / obs:14.4f}')
    print(f'Total cost:         {cost:14.4f} ',
          f'  Average cost:         {cost / obs:14.4f}')
    if score is not None:
        print(f'Total score - cost: {score-cost:14.4f} ',
              '  Average score - cost:',
              f'{(score-cost)/obs:14.4f}', '\n' + '- ' * 40)
    print(f'Total number of observations: {int(obs):d}')
    print('Treatments:                           ', end=' ')
    for i, _ in enumerate(v_dict['polscore_name']):
        print(f'{i:6d} ', end=' ')
    print('\nObservations allocated, by treatment: ', end=' ')
    for i in obs_by_treat:
        print(f'{int(i):6d} ', end=' ')
    if not prediction:
        print('\nObservations allowed,   by treatment: ', end=' ')
        for i in c_dict['max_by_treat']:
            print(f'{int(i):6d} ', end=' ')
    print('\n' + '- ' * 17, 'Shares in %', '- ' * 17)
    print('Observations allocated, by treatment: ', end=' ')
    for i in obs_by_treat:
        print(f'{(i/obs*100):6.2f} ', end=' ')
    if not prediction:
        print('\nObservations allowed,   by treatment: ', end=' ')
        for i in c_dict['max_by_treat']:
            print(f'{(i/obs*100):6.2f} ', end=' ')
    print('\nCost per treatment:                   ', end=' ')
    for i in c_dict['costs_of_treat']:
        print(f'{i:6.2f} ', end=' ')
    print('\n' + '-' * 80)


def print_split_info(splits_seq, treat, c_dict, obs=None, score_val=None):
    """Print leaf information."""
    for i, splits in enumerate(splits_seq):
        for j in range(2):
            print(f'Leaf {i:d}{j:d}:  ', end=' ')
            for splits_dic in splits[j]:
                print(f'{splits_dic["x_name"]:4s}', end=' ')
                if splits_dic['x_type'] == 'unord':
                    if splits_dic['left or right'] == 'left':
                        print('In:     ', end='')
                    else:
                        print('Not in: ', end='')
                    values_to_print = np.sort(splits_dic['cut-off or set'])
                    for s_i in values_to_print:
                        if isinstance(s_i, int) or (
                                (s_i - np.round(s_i)) < 0.00001):
                            print(f'{int(np.round(s_i)):2d} ', end=' ')
                        else:
                            print(f'{s_i:3.1f} ', end=' ')
                else:
                    if splits_dic['left or right'] == 'left':
                        print('<=', end='')
                    else:
                        print('> ', end='')
                    print(f'{splits_dic["cut-off or set"]:8.3f} ', end=' ')
            print()
            print(f'Alloc Treatment: {treat[i][j]:3d} ', end='')
            if score_val is not None:
                print(f'Obs: {obs[i][j]:6d}  ',
                      f'Avg.score: {score_val[i][j] / obs[i][j]:7.3f} ',
                      end=' ')
                tmp = (score_val[i][j] / obs[i][j]
                       - c_dict['costs_of_treat'][treat[i][j]])
                print(f'Avg.score-costs: {tmp:7.3f} ')
            print('\n' + '- ' * 40)
    print('=' * 80)


def describe_alloc_other_outcomes_tree(var_name, data_df, no_of_treat,
                                       terminal_leafs, tree, data_df_x):
    """Describe additional outcomes variables in optimal policy tree."""
    data_df_ps_desc = data_df[var_name]
    number_var = len(var_name) / no_of_treat
    total_obs = len(data_df_ps_desc)
    txt = f'Wrong dimensions of additional outcome variable {var_name}'
    assert np.int64(number_var * no_of_treat) == data_df_ps_desc.shape[1], txt
    all_idx_tmp = np.arange(data_df_ps_desc.shape[1])
    all_idx = np.reshape(all_idx_tmp,
                         (np.int64(data_df_ps_desc.shape[1] / no_of_treat),
                          no_of_treat))
    print()
    for idx in all_idx:
        po_this_var = data_df_ps_desc.iloc[:, idx]
        total_score = 0
        for i, leaf in enumerate(terminal_leafs):
            _, score_val, _, _, _ = optp_ta.two_leafs_info(
                tree, po_this_var, data_df_x, leaf, polscore_is_index=False)
            total_score += score_val[0] + score_val[1]
        print('-' * 80)
        print('Scores: ', end=' ')
        var_name_idx = [var_name[i] for i in idx]
        for i in var_name_idx:
            print(i, end=' ')
        print(f'\nTotal score (level): {total_score:.4f} ',
              f'  Average score (level): {total_score / total_obs:.4f}')
    print('-' * 80)


def describe_alloc_other_outcomes(var_name, po_np, no_of_treat, alloc,
                                  changers=None, changers_only=False,
                                  alloc_act=None):
    """Describe allocation using additional estimated potential outcomes."""
    number_var = round(len(var_name) / no_of_treat)
    obs = len(po_np)
    if changers is not None:
        obs_changers = np.sum(changers)
    txt = f'Wrong dimensions of additional outcome variable {var_name}'
    assert np.int64(number_var * no_of_treat) == po_np.shape[1], txt
    all_idx_tmp = np.arange(po_np.shape[1])
    all_idx = np.reshape(all_idx_tmp, (number_var, no_of_treat))
    for idx in all_idx:
        po_this_var = po_np[:, idx]
        score = score_changers = score_act = score_changers_act = 0
        for i, _ in enumerate(po_np):
            score += po_this_var[i, alloc[i]]
            if alloc_act is not None:
                score_act += po_this_var[i, alloc_act[i]]
            if changers is not None:
                if changers[i]:
                    if alloc_act is not None:
                        score_changers += po_this_var[i, alloc[i]]
                        score_changers_act += po_this_var[i, alloc_act[i]]
        print('Scores: ', end=' ')
        var_name_idx = [var_name[i] for i in idx]
        for i in var_name_idx:
            print(i, end=' ')
        if changers_only:
            print('\n' + '- ' * 40)
            if changers is not None:
                print('Number of changers', obs_changers)
        else:
            print(f'\nTotal score (level):     {score:.4f} ',
                  f' Average score (level):       {score / obs:.4f}')
            print('- ' * 40)
        if changers is not None:
            print_changers_info(obs_changers, score_changers, print_line=False)
        if alloc_act is not None:
            print('- ' * 40, '\n' + 'Relative to observed allocation',
                  '\n' + '-   ' * 20)
            print(f'Total score (level):  {score-score_act:.4f}   ',
                  f'  Average score (level): {(score-score_act) / obs:.4f}')
            if changers is not None:
                print_changers_info(obs_changers,
                                    score_changers-score_changers_act)
            else:
                print('- ' * 40)
    print('-' * 80)


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
    assert len(set(ids)) == len(ids), ('Some leafs IDs are identical.'
                                       + ' Rerun programme.')
    print('\n' + '=' * 80)
    print('Descriptive statistic of estimated policy tree: Training sample')
    if c_dict['only_if_sig_better_vs_0']:
        print('While tree building, policy scores not significantly',
              'different from zero are set to zero. Below, orginal scores',
              'are used.')
        print('Significance level used for recoding:',
              f' {c_dict["sig_level_vs_0"] * 100:6.4f} %')
    print('-' * 80, '\nPolicy scores: ', end=' ')
    for i in v_dict['polscore_name']:
        print(i, end=' ')
    print('\nDecision variables: ', end=' ')
    for i in x_type.keys():
        print(i, end=' ')
    print(f'\nDepth of tree:   {depth:d} ')
    print(f'Minimum leaf size: {c_dict["ft_min_leaf_size"]:d} ',
          '\n' + '- ' * 40)
    len_tl = len(terminal_leafs)
    splits_seq, score_val = [None] * len_tl, [None] * len_tl
    obs, treat = [None] * len_tl, [None] * len_tl
    for i, leaf in enumerate(terminal_leafs):
        (splits_seq[i], score_val[i], obs[i], treat[i], _
         ) = optp_ta.two_leafs_info(
            tree, data_df_ps, data_df_x, leaf, polscore_is_index=False)
    total_obs_by_treat = np.zeros((data_df_ps.shape[1]))
    total_obs = total_score = 0
    for i, obs_i in enumerate(obs):
        total_obs += obs_i[0] + obs_i[1]
        total_obs_by_treat[treat[i][0]] += obs_i[0]
        total_obs_by_treat[treat[i][1]] += obs_i[1]
        total_score += score_val[i][0] + score_val[i][1]
    print_allocation(total_score, total_obs, v_dict, c_dict,
                     total_obs_by_treat)
    print_split_info(splits_seq, treat, c_dict, obs, score_val)
    if v_dict['polscore_desc_name'] is not None:
        describe_alloc_other_outcomes_tree(
            v_dict['polscore_desc_name'], data_df, data_df_ps.shape[1],
            terminal_leafs, tree, data_df_x)
