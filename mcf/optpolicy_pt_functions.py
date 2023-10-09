"""
Provide functions for Black-Box allocations.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy
from math import inf
import random

import numpy as np
import pandas as pd
import ray

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as ps
from mcf import optpolicy_pt_add_functions as opt_pt_add
from mcf import optpolicy_pt_eff_functions as opt_pt_eff


def policy_tree_allocation(optp_, data_df):
    """Compute optimal policy tree brute force."""
    var_dic, ot_dic = optp_.var_dict, optp_.other_dict
    po_np = data_df[var_dic['polscore_name']].to_numpy()
    no_obs = len(po_np)
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
    best_tree = None
    while best_tree is None:
        if optp_.gen_dict['method'] == 'policy tree':
            best_tree, _, _ = optimal_tree_proc(optp_, data_df, seed=12345)
        elif optp_.gen_dict['method'] == 'policy tree eff':
            best_tree, _, _ = opt_pt_eff.optimal_tree_eff_proc(optp_, data_df,
                                                               seed=12345)
        if best_tree is None:
            if optp_.gen_dict['with_output']:
                txt = (f'No tree obtained for depth {optp_.pt_dict["depth"]}.'
                       ' Depth is reduced. If results are desired for original'
                       ' depth level, try reducing the minimum leaf size or'
                       ' in increase the number of evaluation points.')
                optp_.pt_dict['depth'] -= 1
                ps.print_mcf(optp_.gen_dict, txt, summary=True)
    optp_.pt_dict['policy_tree'] = best_tree
    pt_alloc_np = pred_policy_allocation(optp_, data_df)
    allocation_df = pd.DataFrame(data=pt_alloc_np, columns=('Policy Tree',))
    return allocation_df


def policy_tree_prediction_only(optp_, data_df):
    """Predict allocation with policy tree from potentially new data."""
    # Check if all variables used from training are included
    x_type, x_values = optp_.var_x_type, optp_.var_x_values
    # Some consistency check of the data
    var_up = [name.upper() for name in data_df.columns]
    data_df.columns = var_up
    all_included = all(var in var_up for var in x_type.keys())
    if not all_included:
        miss_var = [var for var in x_type.keys() if var not in var_up]
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
    pt_alloc_np = pred_policy_allocation(optp_, data_df)
    allocation_df = pd.DataFrame(data=pt_alloc_np, columns=('Policy Tree',))
    return allocation_df


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

    pt_dic, x_values = optp_.pt_dict, optp_.var_x_values
    gen_dic = optp_.gen_dict
    x_name = list(x_values.keys())
    data_df_x = data_df[x_name]   # pylint: disable=E1136
    total_obs = len(data_df_x)
    x_indx = pd.DataFrame(data=range(len(data_df_x)), columns=('Sorter',))
    length = len(pt_dic['policy_tree'])
    ids = [None] * length
    terminal_leafs = []
    for leaf_i in range(length):
        ids[leaf_i] = pt_dic['policy_tree'][leaf_i][0]
        if pt_dic['policy_tree'][leaf_i][4] == 1:
            terminal_leafs.append(pt_dic['policy_tree'][leaf_i])
    assert len(set(ids)) == len(ids), ('Some leafs IDs are identical.' +
                                       'Rerun programme.')
    splits_seq = [None] * len(terminal_leafs)
    obs = [None] * len(terminal_leafs)
    treat = [None] * len(terminal_leafs)
    indx_in_leaf = [None] * len(terminal_leafs)
    for i, leaf in enumerate(terminal_leafs):
        splits_seq[i], _, obs[i], treat[i], indx_in_leaf[i] = two_leafs_info(
            pt_dic['policy_tree'], x_indx, data_df_x, leaf,
            polscore_is_index=True)
    predicted_treatment = pred_treat_fct(treat, indx_in_leaf, total_obs)
    if gen_dic['with_output']:
        opt_pt_add.describe_tree(optp_, splits_seq, treat, obs)
    return predicted_treatment


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
    if int_dic['parallel_processing']:
        if not ray.is_initialized():
            ray.init(num_cpus=int_dic['mp_parallel'], include_dashboard=False)
        data_x_ref = ray.put(data_x)
        data_ps_ref = ray.put(data_ps)
        data_ps_diff_ref = ray.put(data_ps_diff)
        still_running = [ray_tree_search_multip_single.remote(
            data_ps_ref, data_ps_diff_ref, data_x_ref, name_x, type_x,
            values_x, gen_dic, pt_dic, ot_dic, pt_dic['depth'], m_i,
            int_dic['with_numba'], m_i**3)
            for m_i in range(len(type_x))]
        idx, x_trees = 0, [None] * len(type_x)
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
                    ps.print_mcf(gen_dic, txt, summary=False)
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
    leaf = [None] * 9
    leaf[0], leaf[1] = random.randrange(100000), None
    leaf[5], leaf[6], leaf[7] = name_x_m, type_x_m, val_x
    if treedepth == 2:  # Final split (defines 2 final leaves)
        leaf[2], leaf[3], leaf[4] = None, None, 1
        leaf[8] = [tree_l, tree_r]  # For 1st tree --> treatment states
        new_tree = [leaf]
    else:
        leaf[2], leaf[3], leaf[4] = tree_l[0][0], tree_r[0][0], 0
        tree_l[0][1], tree_r[0][1] = leaf[0], leaf[0]
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
