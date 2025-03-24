"""
Provide functions for old policy tree allocations.

Most functions are no longer used.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy
from math import inf
from random import randrange

import numpy as np
import pandas as pd
import ray

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import optpolicy_pt_add_functions as opt_pt_add
from mcf import optpolicy_pt_eff_functions as opt_pt_eff
from mcf import mcf_general_sys as mcf_sys


def policy_tree_allocation(optp_, data_df):
    """Compute optimal policy tree brute force."""
    var_dic, ot_dic, pt_dic = optp_.var_dict, optp_.other_dict, optp_.pt_dict
    po_np = data_df[var_dic['polscore_name']].to_numpy()
    no_obs = len(po_np)
    if k := len(optp_.var_x_type) > 30:
        txt_warning = ('\n\nWARNING\n'
                       f'WARNING \n{k} features specified for optimal policy '
                       'tree. '
                       '\nWARNING Consider reducing this number by substantive '
                       'reasoning or checking variable importance statistics '
                       '\nWARNING\n\n')
    else:
        txt_warning = ''
    if ot_dic['restricted']:
        max_by_treat = np.int64(
            np.floor(no_obs * np.array(ot_dic['max_shares'])))
        if all((share >= 1) for share in ot_dic['max_shares']):
            raise ValueError('All shares larger than 1, but claimed to be'
                             ' restricted specification.'
                             f'{ot_dic["max_shares"]}')
    else:
        max_by_treat = np.int64(no_obs * np.ones(len(ot_dic['max_shares'])))
    optp_.other_dict['max_by_treat'] = max_by_treat
    if ot_dic['restricted']:
        costs_of_treat_update = opt_pt_add.automatic_cost(optp_, data_df)
    else:
        costs_of_treat_update = ot_dic['costs_of_treat']
    # Do not update cost information in attribute but keep it separate
    optp_.pt_dict['cost_of_treat_restrict'] = np.array(costs_of_treat_update)

    tree_levels = 2 if pt_dic['depth_tree_2'] > 1 else 1
    data_df_list = [data_df]   # Put df in list for use of iterators
    level_2_dic = {'level_1_tree': None, 'level_2_tree_list': []}
    for tree_number in range(tree_levels):  # Iterate over levels
        for idx, data_tmp_df in enumerate(data_df_list):  # Iterate over leaves
            optp_local = get_adjusted_optp(optp_, tree_number, tree_levels)
            best_tree = None
            level_str = 'first' if tree_number == 0 else 'second'
            txt = f'\nCreating {level_str} level optimal policy tree'
            if tree_number == 1:
                txt += f'. Leaf of first tree: {idx+1}'
            if optp_.gen_dict['with_output']:
                mcf_ps.print_mcf(optp_.gen_dict, txt, summary=False)
            while best_tree is None:
                if optp_.gen_dict['method'] == 'policy tree':
                    best_tree, _, _ = opt_pt_eff.optimal_tree_eff_proc(
                        optp_local, data_tmp_df, seed=12345)
                elif optp_.gen_dict['method'] == 'policy tree old':
                    best_tree, _, _ = optimal_tree_proc(optp_local, data_tmp_df,
                                                        seed=12345)
                if best_tree is None:
                    if optp_.gen_dict['with_output']:
                        txt = ('No tree obtained for depth '
                               f'{optp_local.pt_dict["depth"]}. Depth is '
                               'reduced.\nIf results are desired for original '
                               'depth level, try reducing the minimum leaf size'
                               ' or increasing the # of evaluation points.')
                        mcf_ps.print_mcf(optp_.gen_dict, txt, summary=True)
                    optp_local.pt_dict['depth'] -= 1
                    if optp_local.pt_dict['depth'] == 1:
                        break
            if tree_number == 1:
                level_2_dic['level_2_tree_list'].append(best_tree)
        if tree_levels == 1:
            optp_.pt_dict['policy_tree'] = best_tree
        elif tree_levels == 2:
            if tree_number == 0:
                level_2_dic['level_1_tree'] = best_tree
                # create new data_tmp_df_list from trainings data in leaf
                data_df_list, leaf_id_df_list = get_data_from_tree(
                    optp_local, data_df, best_tree)
            elif tree_number == 1:
                # collect all trees and form new tree from the two tree levels
                optp_.pt_dict['policy_tree'] = merge_tree_levels(
                    leaf_id_df_list, level_2_dic)

    pt_alloc_np, _, _, pt_alloc_txt, tree_info_dic = pred_policy_allocation(
        optp_, data_df)
    allocation_df = pd.DataFrame(data=pt_alloc_np, columns=('Policy Tree',))
    if pt_alloc_txt is None:
        return allocation_df, '', tree_info_dic

    return allocation_df, pt_alloc_txt + txt_warning, tree_info_dic


def policy_tree_prediction_only(optp_, data_df):
    """Predict allocation with policy tree from potentially new data."""
    # Check if all variables used from training are included
    x_type, x_values = optp_.var_x_type, optp_.var_x_values
    # Some consistency check of the data
    var_low = [name.casefold() for name in data_df.columns]
    data_df.columns = var_low
    all_included = all(var in var_low for var in x_type.keys())
    if not all_included:
        miss_var = [var for var in x_type.keys() if var not in var_low]
        raise ValueError('Not all features used for training included in '
                         f'Missing variables{miss_var} ')
    for name, val in enumerate(x_type):
        if val in (1, 2):  # Unordered variable
            values = data_df[name].unique()
            no_in_val = [val for val in values if val not in x_values[name]]
            if not no_in_val:
                raise ValueError(f'The following values of {name} were not '
                                 'contained in the prediction data:'
                                 f' {no_in_val}')
    pt_alloc_np, _, _, pt_alloc_txt, _ = pred_policy_allocation(optp_, data_df)
    allocation_df = pd.DataFrame(data=pt_alloc_np, columns=('Policy Tree',))
    return allocation_df, pt_alloc_txt


def pred_policy_allocation(optp_, data_df):
    """Get predictions of optimal policy tree.

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

    """
    def pred_treat_fct(treat, indx_in_leaf, total_obs):
        """Collect data and bring in the same order as original data."""
        pred_treat = np.zeros((total_obs, 2), dtype=np.int64)
        idx_start = 0
        for idx_leaf, idx_treat in zip(indx_in_leaf, treat):
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

    pt_dic, x_values = optp_.pt_dict, optp_.var_x_values
    gen_dic = optp_.gen_dict
    x_name = list(x_values.keys())
    data_df_x = data_df[x_name]   # pylint: disable=E1136
    total_obs = len(data_df_x)
    x_indx = pd.DataFrame(data=range(len(data_df_x)), columns=('Sorter',))
    length = len(pt_dic['policy_tree'])
    ids = [None for _ in range(length)]
    terminal_leafs = []
    terminal_leafs_id = []
    for leaf_i in range(length):
        ids[leaf_i] = pt_dic['policy_tree'][leaf_i][0]
        if pt_dic['policy_tree'][leaf_i][4] == 1:
            terminal_leafs.append(pt_dic['policy_tree'][leaf_i])
            terminal_leafs_id.append(pt_dic['policy_tree'][leaf_i][0])
    assert len(set(ids)) == len(ids), ('Some leafs IDs are identical.' +
                                       'Rerun programme.')
    splits_seq = [None for _ in range(len(terminal_leafs))]
    obs = [None for _ in range(len(terminal_leafs))]
    treat = [None for _ in range(len(terminal_leafs))]
    indx_in_leaf = [None for _ in range(len(terminal_leafs))]
    for i, leaf in enumerate(terminal_leafs):
        splits_seq[i], _, obs[i], treat[i], indx_in_leaf[i] = two_leafs_info(
            pt_dic['policy_tree'], x_indx, data_df_x, leaf,
            polscore_is_index=True)
    predicted_treatment = pred_treat_fct(treat, indx_in_leaf, total_obs)
    text, tree_dic = (opt_pt_add.describe_tree(optp_, splits_seq, treat, obs)
                      if gen_dic['with_output'] else (None, None))

    return predicted_treatment, indx_in_leaf, terminal_leafs_id, text, tree_dic


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
    parent_nr, current_nr = leaf[1], leaf[0]
    if len(tree) > 1:
        while parent_nr is not None:
            for leaf_i in tree:
                if leaf_i[0] == parent_nr:
                    if current_nr == leaf_i[2]:
                        left_right = 'left'
                    elif current_nr == leaf_i[3]:
                        left_right = 'right'
                    else:
                        raise RuntimeError('Leaf numbers are inconsistent.')
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
    leaf_splits_r = deepcopy(leaf_splits_pre)
    leaf_splits_l = leaf_splits_pre   # one copy is enough
    leaf_splits_l.append(final_dict_l)
    leaf_splits_r.append(final_dict_r)
    leaf_splits = (leaf_splits_l, leaf_splits_r)
    polscore_df_lr = (polscore_df_l, polscore_df_r)
    # leaf 8 contains treatment information in final leaf
    return leaf_splits, score, obs, tuple(leaf[8]), polscore_df_lr


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


def optimal_tree_proc(optp_, data_df, seed=12345):
    """Build optimal policy tree."""
    gen_dic, pt_dic, int_dic = optp_.gen_dict, optp_.pt_dict, optp_.int_dict
    ot_dic = optp_.other_dict
    if gen_dic['with_output']:
        print('\nBuilding optimal policy / decision tree')
    (data_x, data_ps, data_ps_diff, name_x, type_x, values_x
     ) = opt_pt_add.prepare_data_for_tree_building(optp_, data_df, seed=seed)
    optimal_tree, x_trees = None, []
    if gen_dic['mp_parallel'] > 1.5:
        if not ray.is_initialized():
            mcf_sys.init_ray_with_fallback(
                gen_dic['mp_parallel'], int_dic, gen_dic,
                ray_err_txt='Ray does not start up in policy tree estimation.'
                )
        data_x_ref = ray.put(data_x)
        data_ps_ref = ray.put(data_ps)
        data_ps_diff_ref = ray.put(data_ps_diff)
        still_running = [ray_tree_search_multip_single.remote(
            data_ps_ref, data_ps_diff_ref, data_x_ref, name_x, type_x,
            values_x, gen_dic, pt_dic, ot_dic, pt_dic['depth'], m_i,
            int_dic['with_numba'], m_i**3)
            for m_i in range(len(type_x))]
        idx, x_trees = 0, [None for _ in range(len(type_x))]
        while len(still_running) > 0:
            finished, still_running = ray.wait(still_running)
            finished_res = ray.get(finished)
            for ret_all_i in finished_res:
                if gen_dic['with_output']:
                    mcf_gp.share_completed(idx+1, len(type_x))
                x_trees[idx] = ret_all_i
                idx += 1
        optimal_reward = np.empty(len(type_x))
        for idx, tree in enumerate(x_trees):
            optimal_reward[idx] = tree[1]
        max_i = np.argmax(optimal_reward)
        optimal_reward = optimal_reward[max_i]
        optimal_tree, obs_total = x_trees[max_i][0], x_trees[max_i][2]
    else:
        optimal_tree, optimal_reward, obs_total = tree_search(
            data_ps, data_ps_diff, data_x, name_x, type_x, values_x, pt_dic,
            gen_dic, ot_dic, pt_dic['depth'], with_numba=int_dic['with_numba'],
            seed=seed)
    return optimal_tree, optimal_reward, obs_total


def tree_search(data_ps, data_ps_diff, data_x, name_x, type_x, values_x,
                pt_dic, gen_dic, ot_dic, treedepth, no_further_splits=False,
                with_numba=True, seed=12345):
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
    pt_dic, gen_dic : Dict's. Parameters.
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
        tree, reward, no_by_treat = opt_pt_add.evaluate_leaf(
            data_ps, gen_dic, ot_dic, pt_dic, with_numba=with_numba)
    else:
        if not no_further_splits and (treedepth < pt_dic['depth']):
            no_further_splits = opt_pt_add.only_1st_tree_fct3(
                data_ps, pt_dic['cost_of_treat_restrict'])
        min_leaf_size = pt_dic['min_leaf_size'] * 2**(treedepth - 2)
        no_of_x, reward = len(type_x), -inf
        tree = no_by_treat = None
        for m_i in range(no_of_x):
            if gen_dic['with_output']:
                if treedepth == pt_dic['depth']:
                    txt = (f'{name_x[m_i]:20s}  {m_i / no_of_x * 100:4.1f}%'
                           ' of variables completed')
                    mcf_ps.print_mcf(gen_dic, txt, summary=False)
            if type_x[m_i] == 'cont':
                values_x_to_check = opt_pt_add.get_values_cont_x(
                    data_x[:, m_i], pt_dic['no_of_evalupoints'],
                    with_numba=with_numba)
            elif type_x[m_i] == 'disc':
                values_x_to_check = values_x[m_i][:]
            else:
                if treedepth < pt_dic['depth']:
                    values_x_to_check = opt_pt_add.combinations_categorical(
                        data_x[:, m_i], data_ps_diff,
                        pt_dic['no_of_evalupoints'], with_numba,
                        seed=seed)
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
                    name_x, type_x, values_x, pt_dic, gen_dic, ot_dic,
                    treedepth - 1, no_further_splits, with_numba=with_numba,
                    seed=seed+1)
                tree_r, reward_r, no_by_treat_r = tree_search(
                    data_ps[right, :], data_ps_diff[right, :],
                    data_x[right, :], name_x, type_x, values_x, pt_dic,
                    gen_dic, ot_dic, treedepth - 1, no_further_splits,
                    with_numba=with_numba, seed=seed+1)
                if ot_dic['restricted'] and pt_dic['enforce_restriction']:
                    reward_l, reward_r = opt_pt_add.adjust_reward(
                        no_by_treat_l, no_by_treat_r, reward_l, reward_r,
                        with_numba, ot_dic['max_by_treat'])
                if reward_l + reward_r > reward:
                    reward = reward_l + reward_r
                    no_by_treat = no_by_treat_l + no_by_treat_r
                    tree = merge_trees(tree_l, tree_r, name_x[m_i],
                                       type_x[m_i], val_x, treedepth)
                if no_further_splits:
                    return tree, reward, no_by_treat
    return tree, reward, no_by_treat


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
    leaf = [None for _ in range(9)]
    leaf[0], leaf[1] = randrange(100000), None
    leaf[5], leaf[6], leaf[7] = name_x_m, type_x_m, val_x
    if treedepth == 2:  # Final split (defines 2 final leaves)
        leaf[2], leaf[3], leaf[4] = None, None, 1
        leaf[8] = [tree_l, tree_r]  # For 1st tree --> treatment states
        new_tree = [leaf]
    else:
        leaf[2], leaf[3], leaf[4] = tree_l[0][0], tree_r[0][0], 0
        tree_l[0][1], tree_r[0][1] = leaf[0], leaf[0]
        new_tree = [None for _ in range(1 + 2 * len(tree_l))]
        new_tree[0] = leaf
        i = 1
        for i_l in tree_l:
            new_tree[i] = i_l
            i += 1
        for i_r in tree_r:
            new_tree[i] = i_r
            i += 1
    return new_tree


@ray.remote
def ray_tree_search_multip_single(data_ps, data_ps_diff, data_x, name_x,
                                  type_x, values_x, gen_dic, pt_dic, ot_dic,
                                  treedepth, m_i, with_numba=True,
                                  seed=123456):
    """Prepare function for Ray."""
    return tree_search_multip_single(data_ps, data_ps_diff, data_x, name_x,
                                     type_x, values_x, gen_dic, pt_dic, ot_dic,
                                     treedepth, m_i, with_numba=with_numba,
                                     seed=seed)


def tree_search_multip_single(data_ps, data_ps_diff, data_x, name_x, type_x,
                              values_x, gen_dic, pt_dic, ot_dic, treedepth,
                              m_i, with_numba=True, seed=12345):
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
    pt_dic : Dict. Parameters.
    gen_dic : Dict. Parameters.
    treedepth : Int. Current depth of tree.
    seed: Int. Seed for combinatorical.

    Returns
    -------
    tree : List of lists. Current tree.
    reward : Float. Total reward that comes from this tree.
    no_by_treat : List of int. Number of treated by treatment state (0-...)

    """
    assert treedepth != 1, 'This should not happen in Multiprocessing.'
    reward, tree, no_by_treat = -inf, None, None
    if type_x[m_i] == 'cont':
        values_x_to_check = opt_pt_add.get_values_cont_x(
            data_x[:, m_i], pt_dic['no_of_evalupoints'], with_numba=with_numba)
    elif type_x[m_i] == 'disc':
        values_x_to_check = values_x[m_i][:]
    else:
        if treedepth < pt_dic['depth']:
            values_x_to_check = opt_pt_add.combinations_categorical(
                data_x[:, m_i], data_ps_diff, pt_dic['no_of_evalupoints'],
                with_numba, seed=seed)
        else:
            values_x_to_check = values_x[m_i][:]
    for val_x in values_x_to_check:
        if type_x[m_i] == 'unord':
            left = np.isin(data_x[:, m_i], val_x)
        else:
            left = data_x[:, m_i] <= (val_x + 1e-15)
        obs_left = np.count_nonzero(left)
        if not (pt_dic['min_leaf_size'] <= obs_left
                <= (len(left)-pt_dic['min_leaf_size'])):
            continue
        right = np.invert(left)
        tree_l, reward_l, no_by_treat_l = tree_search(
            data_ps[left, :], data_ps_diff[left, :], data_x[left, :],
            name_x, type_x, values_x, pt_dic, gen_dic, ot_dic, treedepth - 1,
            with_numba=with_numba, seed=seed+1)
        tree_r, reward_r, no_by_treat_r = tree_search(
            data_ps[right, :], data_ps_diff[right, :], data_x[right, :],
            name_x, type_x, values_x, pt_dic, gen_dic, ot_dic, treedepth - 1,
            with_numba=with_numba, seed=seed+1)
        if ot_dic['restricted'] and pt_dic['enforce_restriction']:
            reward_l, reward_r = opt_pt_add.adjust_reward(
                no_by_treat_l, no_by_treat_r, reward_l, reward_r,
                with_numba, ot_dic['max_by_treat'])
        if reward_l + reward_r > reward:
            reward = reward_l + reward_r
            no_by_treat = no_by_treat_l + no_by_treat_r
            tree = merge_trees(tree_l, tree_r, name_x[m_i],
                               type_x[m_i], val_x, treedepth)
    return tree, reward, no_by_treat


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
        raise ValueError('No valid entries in final leaf.')
    return_dic = {'x_name': leaf[5], 'x_type': leaf[6],
                  'cut-off or set': leaf[7], 'left or right': left_right}
    return return_dic


def get_adjusted_optp(optp_, tree_number, tree_levels):
    """Copy and adjust instance of policy tree."""
    optp_local = deepcopy(optp_)
    if tree_number == 0:
        optp_local.pt_dict['depth'] = optp_.pt_dict['depth_tree_1']
        if tree_levels == 2:
            # Adjust minimum leaf size --> speeds up computation
            optp_local.pt_dict['min_leaf_size'] = round(
                optp_.pt_dict['min_leaf_size']
                * 2**(optp_.pt_dict['depth_tree_2'] - 1))
    elif tree_number == 1:
        optp_local.pt_dict['depth'] = optp_.pt_dict['depth_tree_2']
    else:
        raise NotImplementedError('Not yet (?) implemented for more than 2 '
                                  'levels of optimal trees.')
    return optp_local


def get_data_from_tree(optp_local, data_df, tree):
    """Get the data for the final leafs as list."""
    leaf_data_df_list, leaf_id_list = [], []
    optp_local.pt_dict[('policy_tree')] = tree
    optp_local.gen_dict['with_output'] = False
    _, idx_in_leaf, leaf_id, _, _ = pred_policy_allocation(optp_local, data_df)
    for idx, id_ in zip(idx_in_leaf, leaf_id):
        for idx_left_right in idx:
            leaf_id_list.append(id_)
            leaf_data_df_list.append(
                data_df.iloc[idx_left_right['Sorter'].tolist()])
    return leaf_data_df_list, leaf_id_list


def merge_tree_levels(leaf_id_list, level_2_dic):
    """Merge second level trees to first level tree."""
    no_final_double_leafs = round(len(leaf_id_list) / 2)
    tree = deepcopy(level_2_dic['level_1_tree'])
    for idx in range(no_final_double_leafs):
        # Find leaf in 1st tree that has to be adapted
        leaf_1_idx = find_leaf_one_idx(tree, leaf_id_list[idx*2])
        # Relevant trees of second round
        tree_2_l = deepcopy(level_2_dic['level_2_tree_list'][idx*2])
        tree_2_r = deepcopy(level_2_dic['level_2_tree_list'][idx*2+1])
        if tree_2_l is not None and tree_2_r is not None:
            # Recode parent leave in 1st tree for both second level trees
            tree_2_l[0][1] = tree[leaf_1_idx][0]
            # Recode final leaf in 1st tree with information of first leaf 2nd
            # level
            tree[leaf_1_idx][2] = tree_2_l[0][0]    # Add ID of daughter leave
            tree_2_r[0][1] = tree[leaf_1_idx][0]
            tree[leaf_1_idx][3] = tree_2_r[0][0]
            tree[leaf_1_idx][4] = 0     # Not a final leaf anymore
            tree = tree + tree_2_l + tree_2_r
    return tree


def find_leaf_one_idx(tree, leaf_id):
    """Find id of a leaf in another tree."""
    for idx, leaf in enumerate(tree):
        if leaf_id == leaf[0]:
            return idx
    raise ValueError('Leaf id not found.')
