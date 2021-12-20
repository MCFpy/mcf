"""
Created on Wed Dec  8 15:20:07 2021.

Optimal Policy Trees: Tree Functions - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

# -*- coding: utf-8 -*-
"""
import random
import math
from concurrent import futures
import pandas as pd
import numpy as np
import ray
import scipy.stats as sct
from numba import njit
from mcf import general_purpose as gp


def combinations_categorical(single_x_np, ps_np_diff, c_dict, ft_yes=True):
    """
    Create all possible combinations of list elements, removing complements.

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
    no_of_values = len(values)
    no_of_combinations = gp.total_sample_splits_categorical(no_of_values)
    if no_of_combinations < c_dict['ft_no_of_evalupoints']:
        combinations = gp.all_combinations_no_complements(list(values))
    else:
        values_sorted, no_of_ps = get_values_ordered(
            single_x_np, ps_np_diff, values, no_of_values,
            with_numba=c_dict['with_numba'])
        combinations_t = sorted_values_into_combinations(
            values_sorted, no_of_ps, no_of_values)
        if (len(combinations_t) > c_dict['ft_no_of_evalupoints']) and ft_yes:
            combinations_t = random.sample(
                combinations_t, c_dict['ft_no_of_evalupoints'])
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
        leaf[2] = leaf[3] = None
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


def seq_tree_search(data_ps, data_ps_diff, data_x, name_x, type_x, values_x,
                    c_dict):
    """Build sequential tree.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    data_ps_diff : Numpy array. Policy scores as differences to cat 0.
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
    9: Level (0-c_dict['st_depth'])
    10: Indices of data: Numpy series
    11: Treatment of leaf

    """
    def add_leaves_to_tree(tree, best_treat_l, best_treat_r, best_name_x,
                           best_type_x, best_val_x, best_left, best_right,
                           indices, level, final, min_leaf_size, parent_leaf):
        # Check if any split, if not remove last leaf
        if best_treat_l is None or best_treat_r is None:   # status --> final
            index_of_grandparent = index_from_leaf_id(tree, parent_leaf[1])
            tree[index_of_grandparent][4] = 1
            index_of_parent_l = index_from_leaf_id(
                tree, tree[index_of_grandparent][2])
            index_of_parent_r = index_from_leaf_id(
                tree, tree[index_of_grandparent][3])
            del tree[index_of_parent_l]
            del tree[index_of_parent_r]
            return tree
        # Create and assign to left daughter
        daughter_left = [None] * 12
        daughter_left[0] = random.randrange(100000)
        daughter_left[1] = parent_leaf[0]
        daughter_left[4] = 2
        daughter_left[9] = level + 1
        daughter_left[10] = indices[best_left]
        daughter_left[11] = best_treat_l
        # Create and assign to right daughter
        daughter_right = [None] * 12
        daughter_right[0] = random.randrange(100000)
        daughter_right[1] = parent_leaf[0]
        daughter_right[4] = 2
        daughter_right[9] = level + 1
        daughter_right[10] = indices[best_right]
        daughter_right[11] = best_treat_r
        # Change values in parent leaf
        parent_leaf[2] = daughter_left[0]
        parent_leaf[3] = daughter_right[0]
        if final:
            parent_leaf[4] = 1
        else:
            parent_leaf[4] = 0
        parent_leaf[5] = best_name_x
        parent_leaf[6] = best_type_x
        parent_leaf[7] = best_val_x
        parent_leaf[8] = [best_treat_l, best_treat_r]
        # Exchange the parent leaf in the tree
        index_of_parent = index_from_leaf_id(tree, parent_leaf[0])
        tree[index_of_parent] = parent_leaf.copy()
        if not final:
            tree.append(daughter_left)
            tree.append(daughter_right)
        return tree

    def index_from_leaf_id(tree, leaf_id):
        for leaf_no, leaf in enumerate(tree):
            if leaf[0] == leaf_id:
                return leaf_no
        raise Exception('Leaf_id not found in tree.')

    def list_of_leaves_f(level, tree):
        list_of_leaves = []
        for leaf in tree:
            if leaf[9] == level and leaf[4] == 2:  # active leaves
                list_of_leaves.append(leaf)
        if not list_of_leaves:
            print('Level: ', level)
            raise Exception('No leaves to investigate')
        return list_of_leaves

    def initiale_node_table(obs):
        leaf = [None] * 12
        leaf[0] = random.randrange(100000)
        leaf[1] = None
        leaf[4] = 2
        leaf[9] = 0
        leaf[10] = np.arange(obs)
        return [leaf]

    def get_leaf_data(data_x, data_ps_diff, data_ps, current_leaf):
        indices_l = current_leaf[10]
        return (data_x[indices_l], data_ps_diff[indices_l], data_ps[indices_l],
                current_leaf[10])

    tree = initiale_node_table(len(data_ps))
    no_of_x = len(type_x)
    for level in range(c_dict['st_depth']):
        min_leaf_size = c_dict['st_min_leaf_size'] * 2**(
            c_dict['st_depth'] - level)
        list_of_leaves = list_of_leaves_f(level, tree)
        final = (c_dict['st_depth'] - (level + 1)) == 0
        for parent_leaf in list_of_leaves:
            reward = -math.inf  # minus infinity
            (data_x_leaf, data_ps_diff_leaf, data_ps_leaf, indices_leaf
             ) = get_leaf_data(data_x, data_ps_diff, data_ps, parent_leaf)
            obs_leaf = len(indices_leaf)
            best_treat_l = best_treat_r = best_name_x = best_type_x = None
            best_val_x = best_left = best_right = None
            for m_i in range(no_of_x):
                if type_x[m_i] == 'cont':
                    values_x_to_check = get_values_cont_x(
                        data_x_leaf[:, m_i], obs_leaf,
                        with_numba=c_dict['with_numba'])
                elif type_x[m_i] == 'disc':
                    values_x_to_check = values_x[m_i][:]
                else:
                    values_x_to_check = combinations_categorical(
                            data_x_leaf[:, m_i], data_ps_diff_leaf, c_dict)
                for val_x in values_x_to_check:
                    if type_x[m_i] == 'unord':
                        left = np.isin(data_x_leaf[:, m_i], val_x)
                    else:
                        left = data_x_leaf[:, m_i] <= (val_x + 1e-15)
                    obs_left = np.count_nonzero(left)
                    if not (min_leaf_size <= obs_left
                            <= (len(left) - min_leaf_size)):
                        continue
                    right = np.invert(left)
                    treat_l, reward_l, no_by_treat_l = evaluate_leaf(
                        data_ps_leaf[left], c_dict)
                    treat_r, reward_r, no_by_treat_r = evaluate_leaf(
                        data_ps_leaf[right], c_dict)
                    if reward_r + reward_l > reward:
                        reward = reward_l + reward_r
                        no_by_treat = no_by_treat_l + no_by_treat_r
                        best_treat_l = treat_l
                        best_treat_r = treat_r
                        best_left = left.copy()
                        best_right = right.copy()
                        best_name_x = name_x[m_i]
                        best_type_x = type_x[m_i]
                        best_val_x = val_x
            tree = add_leaves_to_tree(
                tree, best_treat_l, best_treat_r, best_name_x, best_type_x,
                best_val_x, best_left, best_right, indices_leaf, level, final,
                min_leaf_size, parent_leaf)
    return tree, reward, no_by_treat


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
        if not no_further_splits and (treedepth < c_dict['ft_depth']):
            no_further_splits = only_1st_tree_fct3(data_ps, c_dict)
        min_leaf_size = c_dict['ft_min_leaf_size'] * 2**(treedepth - 2)
        no_of_x = len(type_x)
        reward = -math.inf  # minus infinity
        tree = no_by_treat = None
        for m_i in range(no_of_x):
            if c_dict['with_output']:
                if treedepth == c_dict['ft_depth']:
                    print('{:20s} '.format(name_x[m_i]),
                          '{:4.1f}%'.format(m_i / no_of_x * 100),
                          'of variables completed')
            if type_x[m_i] == 'cont':
                values_x_to_check = get_values_cont_x(
                    data_x[:, m_i], c_dict['ft_no_of_evalupoints'],
                    with_numba=c_dict['with_numba'])
            elif type_x[m_i] == 'disc':
                values_x_to_check = values_x[m_i][:]
            else:
                if treedepth < c_dict['ft_depth']:
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
    if type_x[m_i] == 'cont':
        values_x_to_check = get_values_cont_x(
            data_x[:, m_i], c_dict['ft_no_of_evalupoints'],
            with_numba=c_dict['with_numba'])
    elif type_x[m_i] == 'disc':
        values_x_to_check = values_x[m_i][:]
    else:
        if treedepth < c_dict['ft_depth']:
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
        if not (c_dict['ft_min_leaf_size'] <= obs_left
                <= (len(left)-c_dict['ft_min_leaf_size'])):
            continue
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


def prepare_data_for_tree_builddata(datafile_name, c_dict, v_dict, x_type,
                                    x_value):
    """Prepare data for tree building."""
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
    return data_x, data_ps, data_ps_diff, name_x, type_x, values_x


def sequential_tree_proc(datafile_name, x_type, x_value, v_dict, c_dict):
    """Build sequential policy tree.

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
    optimal_reward: Float. Rewards of tree.
    obs_total: Int. Number of observations.

    """
    if c_dict['with_output']:
        print('Building sequential policy / decision tree')
        print('No multiprocessing for sequential tree building (not yet).')
    (data_x, data_ps, data_ps_diff, name_x, type_x, values_x
     ) = prepare_data_for_tree_builddata(datafile_name, c_dict, v_dict, x_type,
                                         x_value)
    seq_tree, seq_reward, obs_total = seq_tree_search(
        data_ps, data_ps_diff, data_x, name_x, type_x, values_x, c_dict)
    return seq_tree, seq_reward, obs_total


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
    optimal_reward: Float. Rewards of tree.
    obs_total: Int. Number of observations.

    """
    if c_dict['with_output']:
        print('Building optimal policy / decision tree')
    (data_x, data_ps, data_ps_diff, name_x, type_x, values_x
     ) = prepare_data_for_tree_builddata(datafile_name, c_dict, v_dict, x_type,
                                         x_value)
    optimal_tree = None
    x_trees = []
    if c_dict['parallel']:
        maxworkers = c_dict['no_parallel']
        if c_dict['mp_with_ray']:
            if not ray.is_initialized():
                ray.init(num_cpus=maxworkers, include_dashboard=False)
            data_x_ref = ray.put(data_x)
            data_ps_ref = ray.put(data_ps)
            data_ps_diff_ref = ray.put(data_ps_diff)
            still_running = [ray_tree_search_multip_single.remote(
                data_ps_ref, data_ps_diff_ref, data_x_ref, name_x, type_x,
                values_x, c_dict, c_dict['ft_depth'], m_i)
                for m_i in range(len(type_x))]
            idx = 0
            x_trees = [None] * len(type_x)
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for ret_all_i in finished_res:
                    if c_dict['with_output']:
                        gp.share_completed(idx+1, len(type_x))
                    x_trees[idx] = ret_all_i
                    idx += 1
        else:
            with futures.ProcessPoolExecutor(max_workers=maxworkers) as fpp:
                trees = {fpp.submit(tree_search_multip_single, data_ps,
                                    data_ps_diff, data_x, name_x, type_x,
                                    values_x, c_dict,
                                    c_dict['ft_depth'], m_i):
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
            c_dict['ft_depth'])
    return optimal_tree, optimal_reward, obs_total


@ray.remote
def ray_tree_search_multip_single(data_ps, data_ps_diff, data_x, name_x,
                                  type_x, values_x, c_dict, treedepth, m_i):
    """Prepare function for Ray."""
    return tree_search_multip_single(data_ps, data_ps_diff, data_x, name_x,
                                     type_x, values_x, c_dict, treedepth, m_i)


def structure_of_node_tabl_poltree():
    """Info about content of NODE_TABLE.

    Returns
    -------
    decription : STR. Information on node table with inital node.

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
    8: If x_type = 0: Cut-off value (larger goes to right daughter,
                                    equal and smaller to left daughter)

    """
    print("\n", description)
