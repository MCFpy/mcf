"""
Provide functions for Policy Tree allocations.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
# from itertools import combinations

from functools import lru_cache
from itertools import count
from numbers import Real  # For type checking only
from math import inf
from random import randrange
from typing import TYPE_CHECKING, Sequence, TypeVar

from numba import njit
import numpy as np
import ray
from pandas import DataFrame

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import optpolicy_pt_add_functions as opt_pt_add
from mcf import mcf_general_sys as mcf_sys

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy
T = TypeVar('T')


def optimal_tree_eff_proc(optp_: 'OptimalPolicy',
                          data_df: DataFrame,
                          seed: int = 12345
                          ) -> tuple[list[list, ...], np.ndarray, int]:
    """Build optimal policy tree."""
    gen_dic, pt_dic, int_dic = optp_.gen_dict, optp_.pt_dict, optp_.int_dict
    ot_dic = optp_.other_dict
    if gen_dic['with_output']:
        print('\nBuilding optimal policy / decision tree')
    (data_x, data_ps, data_ps_diff, data_opt_treat, name_x, type_x, values_x,
     values_comp_all
     ) = prepare_data_for_tree_building_eff(optp_, data_df, seed=seed)
    # All values_x are already sorted before this step
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
        data_opt_treat_ref = ray.put(data_opt_treat)
        if int_dic['xtr_parallel']:
            # Variable and single tree are in single worker
            values_x_xtr_p = prepare_ordered_for_xtr_splits(type_x, values_x)
            still_running = [ray_tree_search_eff_multip_single.remote(
                data_ps_ref, data_ps_diff_ref, data_opt_treat_ref, data_x_ref,
                name_x, type_x, values_x, values_comp_all,
                gen_dic, ot_dic, pt_dic, pt_dic['depth'],
                m_i, int_dic['with_numba'], m_i**3,
                values_x_xtr_p[m_i][split_mi]
                ) for m_i in range(len(type_x))
                for split_mi in range(len(values_x_xtr_p[m_i]))]
        else:
            # Variable and single tree are in single worker
            still_running = [ray_tree_search_eff_multip_single.remote(
                data_ps_ref, data_ps_diff_ref, data_x_ref, name_x, type_x,
                values_x, values_comp_all,
                gen_dic, ot_dic, pt_dic, pt_dic['depth'],
                m_i, int_dic['with_numba'], m_i**3
                )
                for m_i in range(len(type_x))]
        total_threads = len(still_running)
        idx, x_trees = 0, []
        while len(still_running) > 0:
            finished, still_running = ray.wait(still_running)
            finished_res = ray.get(finished)
            for ret_all_i in finished_res:
                if gen_dic['with_output']:
                    mcf_gp.share_completed(idx+1, total_threads)
                x_trees.append(ret_all_i)
                idx += 1
        optimal_reward = np.empty(len(x_trees))
        for idx, tree in enumerate(x_trees):
            optimal_reward[idx] = tree[1]
        max_i = np.argmax(optimal_reward)
        optimal_reward = optimal_reward[max_i]
        optimal_tree, obs_total = x_trees[max_i][0], x_trees[max_i][2]
    else:
        (optimal_tree, optimal_reward, obs_total, values_comp_all
         ) = tree_search_eff(
            data_ps, data_ps_diff, data_opt_treat, data_x,
            name_x, type_x, values_x,
            values_comp_all, pt_dic, gen_dic, ot_dic, pt_dic['depth'],
            with_numba=int_dic['with_numba'], seed=seed)

    return optimal_tree, optimal_reward, obs_total


def tree_search_eff(data_ps: np.ndarray,
                    data_ps_diff: np.ndarray,
                    data_opt_treat: np.ndarray,
                    data_x: np.ndarray,
                    name_x: list[str, ...],
                    type_x: list[str, ...],
                    values_x:  list[set, ...],
                    values_comp_all: list[list, ...],
                    pt_dic: dict,
                    gen_dic: dict,
                    ot_dic: dict,
                    treedepth: int,
                    no_further_splits: bool = False,
                    with_numba: bool = True,
                    seed: int = 12345
                    ) -> tuple[list[list, ...],
                               float,
                               list[int],
                               list[list, ...]
                               ]:
    """Build tree EFF.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    data_ps_diff : Numpy array. Policy scores as differences.
    data_x : Numpy array. Policy variables.
    name_x : List of strings. Name of policy variables.
    type_x : List of strings. Type of policy variable.
    values_x : List of sets. Values of x in initial data.
    values_comp_all : Categorical values.
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
    values_comp_all : list of lists. Values and combinations - unordered vars.
    """
    if treedepth == 1:  # Evaluate tree, end of splitting
        tree, reward, no_by_treat = opt_pt_add.evaluate_leaf(
            data_ps, gen_dic, ot_dic, pt_dic, with_numba=with_numba)
    else:
        if not no_further_splits and (treedepth < pt_dic['depth']):
            no_further_splits = opt_pt_add.only_1st_tree_fct3(
                data_ps, pt_dic['cost_of_treat_restrict'])

        min_leaf_size = pt_dic['min_leaf_size'] * 2**(treedepth - 2)
        no_of_x, reward = len(type_x), -inf
        tree = no_by_treat = None

        # Check if there is any gain by additional splits. If not, choose
        # arbitray split point --> implemented such that function
        # get_val_to_check_eff will return only single value
        # For very shallow trees that are subject to restrictions, this is not
        # enforced as it is more likely to lead to violations of the
        # restrictions.
        if treedepth < 4 and (pt_dic['depth'] > 2 or not ot_dic['restricted']):
            no_gain_splitting = no_further_gain_split_funct(data_opt_treat)
        else:
            no_gain_splitting = False

        # Iterate over the single variables
        no_variation = np.ptp(data_x, axis=0) < 1e-15
        for m_i in range(no_of_x):
            if no_variation[m_i]:
                continue
            data_x_m_i = data_x[:, m_i]
            type_x_mi = type_x[m_i]

            if gen_dic['with_output']:
                if treedepth == pt_dic['depth']:
                    txt = (f'{name_x[m_i]:20s}  {m_i / no_of_x * 100:4.1f}%'
                           ' of variables completed')
                    mcf_ps.print_mcf(gen_dic, txt, summary=False)

            values_x_to_check, values_comp_all[m_i] = get_val_to_check_eff(
                type_x[m_i], values_x[m_i][:], values_comp_all[m_i],
                data_x_m_i, data_ps_diff, pt_dic['no_of_evalupoints'],
                select_values_cat=pt_dic['select_values_cat'],
                eva_cat_mult=pt_dic['eva_cat_mult'],
                with_numba=with_numba,
                seed=seed,
                no_gain_split=no_gain_splitting
                )
            obs_all = data_x_m_i.shape[0]

            for val_x in values_x_to_check:
                if type_x_mi == 'unord':
                    left = np.isin(data_x_m_i, val_x)
                else:
                    left = data_x_m_i <= (val_x + 1e-15)
                obs_left = left.sum()

                if obs_left < min_leaf_size:
                    continue
                if obs_all - obs_left < min_leaf_size:  # Obs right
                    if type_x_mi == 'unord':
                        continue
                    # Because x are ordered, rights will get only smaller
                    break

                right = ~left

                (tree_l, reward_l, no_by_treat_l, values_comp_all
                 ) = tree_search_eff(
                    data_ps[left, :], data_ps_diff[left, :],
                    data_opt_treat[left], data_x[left, :],
                    name_x, type_x, values_x, values_comp_all,
                    pt_dic, gen_dic, ot_dic,
                    treedepth-1,
                    no_further_splits,
                    with_numba=with_numba,
                    seed=seed+1
                    )
                (tree_r, reward_r, no_by_treat_r, values_comp_all
                 ) = tree_search_eff(
                    data_ps[right, :], data_ps_diff[right, :],
                    data_opt_treat[right], data_x[right, :],
                    name_x, type_x, values_x, values_comp_all,
                    pt_dic, gen_dic, ot_dic,
                    treedepth-1,
                    no_further_splits,
                    with_numba=with_numba,
                    seed=seed+1
                    )
                if ot_dic['restricted'] and pt_dic['enforce_restriction']:
                    reward_l, reward_r = opt_pt_add.adjust_reward(
                        no_by_treat_l, no_by_treat_r, reward_l, reward_r,
                        with_numba, ot_dic['max_by_treat'])

                if reward_l + reward_r > reward:
                    reward = reward_l + reward_r
                    no_by_treat = no_by_treat_l + no_by_treat_r
                    tree = merge_trees_eff(tree_l, tree_r, name_x[m_i],
                                           type_x[m_i], val_x, treedepth)

                if no_further_splits:

                    return tree, reward, no_by_treat, values_comp_all

    return tree, reward, no_by_treat, values_comp_all


# def merge_trees_eff(tree_l: list[list, ...],
#                     tree_r: list[list, ...],
#                     name_x_m: str,
#                     type_x_m: str,
#                     val_x: float,
#                     treedepth: int,
#                     _id_gen=count(1),
#                     ) -> tuple[list[list, ...]]:
#     """Merge trees and add new split (optimized version-limited gain).

#     0: Node identifier (INT: 0-...)
#     1: Parent knot
#     2: Child node left
#     3: Child node right
#     4: Type of node (1: Terminal node, no further splits
#                     0: previous node that lead already to further splits)
#     5: String: Name of variable used for decision of next split
#     6: x_type of variable (policy categorisation, maybe different from MCF)
#     7: If x_type = 'unordered': Set of values that goes to left daughter
#     7: If x_type = 0: Cut-off value (larger goes to right daughter)
#     8: List of Treatment state for both daughters [left, right]
#     _id_gen: (global) counter (from itertools)

#     Parameters
#     ----------
#     tree_l : List of lists. Left tree.
#     tree_r : List of lists. Right tree.
#     name_x_m : String. Name of variables used for splitting.
#     type_x_m : String. Type of variables used for splitting.
#     val_x : Float, Int, or set of Int. Values used for splitting.
#     treedepth : Int. Current level of tree. 1: final level.

#     Returns
#     -------
#     new_tree : List of lists. The merged trees.

#     """
#     note_id = next(_id_gen)
#     leaf = [None] * 9
#     leaf[0], leaf[1] = note_id, None
#     leaf[5], leaf[6], leaf[7] = name_x_m, type_x_m, val_x

#     if treedepth == 2:  # Final split (defines 2 final leaves)
#         leaf[2], leaf[3], leaf[4] = None, None, 1
#         leaf[8] = [tree_l, tree_r]  # For 1st tree --> treatment states
#         new_tree = [leaf]
#     else:
#         leaf[2], leaf[3], leaf[4] = tree_l[0][0], tree_r[0][0], 0
#         tree_l[0][1], tree_r[0][1] = note_id, note_id
#         # new_tree = [None] * (1 + 2 * len(tree_l))
#         # new_tree[0] = leaf
#         # tree = tree_l[:]
#         # tree.extend(tree_r[:])
#         # new_tree[1:] = tree
#         new_tree = [leaf] + tree_l + tree_r

#     return new_tree


def merge_trees_eff(tree_l: list[list, ...],
                    tree_r: list[list, ...],
                    name_x_m: str,
                    type_x_m: str,
                    val_x: float,
                    treedepth: int,
                    _id_gen=count(1),
                    ) -> tuple[list[list, ...]]:
    """Merge trees and add new split (optimized version-limited gain).

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
    _id_gen: (global) counter (from itertools)

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
    note_id = next(_id_gen) + randrange(100000)  # Counter lives as a default
#   argument—no module-level globals. Random number added as this is used in
#   parallel threads.
    if treedepth == 2:
        leaf = [
            note_id, None, None, None, 1,
            name_x_m, type_x_m, val_x,
            [tree_l, tree_r]
        ]
        return [leaf]

    # Copy so we don’t mutate inputs
    left = [node.copy() for node in tree_l]
    right = [node.copy() for node in tree_r]
    left[0][1] = right[0][1] = note_id

    leaf = [
        note_id, None,
        left[0][0], right[0][0], 0,
        name_x_m, type_x_m, val_x,
        None
    ]
    return [leaf] + left + right


@ray.remote
def ray_tree_search_eff_multip_single(
        data_ps: np.ndarray,
        data_ps_diff: np.ndarray,
        data_opt_treat: np.ndarray,
        data_x: np.ndarray,
        name_x: list[str, ...],
        type_x: list[str, ...],
        values_x: list[set, ...],
        values_comp_all: list[list, ...],
        gen_dic: dict, ot_dic: dict, pt_dic: dict,
        treedepth: int,
        m_i: int,
        with_numba: bool = True,
        seed: int = 123456,
        first_split_idx: Real | None = None
        ) -> tuple[list[list, ...], float, list[int]]:
    """Prepare function for Ray."""
    return tree_search_eff_multip_single(
        data_ps, data_ps_diff, data_opt_treat, data_x,
        name_x, type_x, values_x,
        values_comp_all, gen_dic, ot_dic, pt_dic, treedepth, m_i,
        with_numba=with_numba, seed=seed, first_split_idx=first_split_idx
        )


def tree_search_eff_multip_single(
        data_ps: np.ndarray,
        data_ps_diff: np.ndarray,
        data_opt_treat: np.ndarray,
        data_x: np.ndarray,
        name_x: list[str, ...],
        type_x: list[str, ...],
        values_x: list[set, ...],
        values_comp_all: list[list, ...],
        gen_dic: dict,  ot_dic: dict, pt_dic: dict,
        treedepth: int,
        m_i: int,
        with_numba: bool = True,
        seed: int = 123456,
        first_split_idx: Real | None = None
        ) -> tuple[list[list, ...], float, list[int]]:
    """Build tree. Only first level. For multiprocessing only."""
    assert treedepth != 1, 'This should not happen in Multiprocessing.'
    reward, tree, no_by_treat = -inf, None, None

    min_leaf_size = pt_dic['min_leaf_size']
    x_col = data_x[:, m_i]

    values_x_to_check, values_comp_all[m_i] = get_val_to_check_eff(
        type_x[m_i], values_x[m_i][:], values_comp_all[m_i], x_col,
        data_ps_diff, pt_dic['no_of_evalupoints'],
        select_values_cat=pt_dic['select_values_cat'],
        eva_cat_mult=pt_dic['eva_cat_mult'],
        with_numba=with_numba,
        seed=seed,
        first_split_idx=first_split_idx)

    type_x_m_i, name_x_m_i = type_x[m_i], name_x[m_i]
    obs_all = x_col.shape[0]
    for val_x in values_x_to_check:
        if type_x_m_i == 'unord':
            left = np.isin(x_col, val_x)
        else:
            left = x_col <= (val_x + 1e-15)

        # Fast check of stopping criteria: Leaf size too small
        obs_left = left.sum()
        obs_right = obs_all - obs_left
        if obs_left < min_leaf_size or obs_right < min_leaf_size:
            continue

        right = ~left

        tree_l, reward_l, no_by_treat_l, values_comp_all = tree_search_eff(
            data_ps[left, :], data_ps_diff[left, :],
            data_opt_treat[left], data_x[left, :],
            name_x, type_x, values_x, values_comp_all, pt_dic, gen_dic, ot_dic,
            treedepth - 1,
            with_numba=with_numba,
            seed=seed + 1
            )
        tree_r, reward_r, no_by_treat_r, values_comp_all = tree_search_eff(
            data_ps[right, :], data_ps_diff[right, :],
            data_opt_treat[right], data_x[right, :],
            name_x, type_x, values_x, values_comp_all, pt_dic, gen_dic, ot_dic,
            treedepth - 1,
            with_numba=with_numba,
            seed=seed + 1
            )
        if ot_dic['restricted'] and pt_dic['enforce_restriction']:
            reward_l, reward_r = opt_pt_add.adjust_reward(
                no_by_treat_l, no_by_treat_r, reward_l, reward_r,
                with_numba, ot_dic['max_by_treat'])
        if reward_l + reward_r > reward:
            reward = reward_l + reward_r
            no_by_treat = no_by_treat_l + no_by_treat_r
            tree = merge_trees_eff(tree_l, tree_r, name_x_m_i,
                                   type_x_m_i, val_x, treedepth)
    return tree, reward, no_by_treat


def get_val_to_check_eff(type_x_m_i: str,
                         values_x_m_i: Real,
                         values_comp_all_m_i: list[tuple, ...],
                         data_x_m_i: np.ndarray,
                         data_ps_diff: np.ndarray,
                         no_of_evalupoints: int,
                         select_values_cat: bool = False,
                         eva_cat_mult: int = 1,
                         with_numba: bool = True,
                         seed: int = 1234,
                         first_split_idx: Real | None = None,
                         no_gain_split: bool = False
                         ) -> tuple[list, list]:
    """Get the values to check for next splits of leaf."""
    if type_x_m_i in ('cont', 'disc'):
        if no_gain_split:
            max_x, min_x = data_x_m_i.max(), data_x_m_i.min()
            values_x_to_check = (min_x + 0.5 * (max_x - min_x),)
        else:
            values_x_to_check_tmp = get_values_cont_ord_x_eff(data_x_m_i,
                                                              values_x_m_i)
            if first_split_idx is not None and len(values_x_to_check_tmp) > 2:
                if not isinstance(first_split_idx, set):
                    first_split_idx_set = set(first_split_idx)

                values_x_to_check = [val for val in values_x_to_check_tmp
                                     if val in first_split_idx_set]
                if not values_x_to_check:
                    values_x_to_check = values_x_to_check_tmp
            else:
                values_x_to_check = values_x_to_check_tmp
    elif type_x_m_i == 'unord':
        # Take the pre-computed values of the splitting points that fall into
        # the range of the data
        if no_gain_split:
            no_of_evalupoints = 1
            eva_cat_mult = 1
        values_x_to_check, values_comp_all_m_i = combinations_categorical_eff(
            data_x_m_i, data_ps_diff, values_comp_all_m_i, no_of_evalupoints,
            select_values=select_values_cat,
            factor=eva_cat_mult,
            with_numba=with_numba,
            seed=seed)
    else:
        raise ValueError('Wrong data type')
    return values_x_to_check, values_comp_all_m_i


def combinations_categorical_eff(single_x_np: np.ndarray,
                                 ps_np_diff: np.ndarray,
                                 values_comp_all: list[tuple, ...] | None,
                                 no_of_evalupoints: int,
                                 select_values: bool = False,
                                 factor: Real = 1,
                                 with_numba: bool = True,
                                 seed: int = 123456
                                 ) -> tuple[list, list]:
    """Create all possible combinations of list elements, w/o complements."""
    values = np.unique(single_x_np)
    no_of_values = len(values)
    no_eva_point = int(no_of_evalupoints * factor)

    if values_comp_all is not None:
        for hist in values_comp_all:
            if len(hist[0]) == len(values) and np.array_equal(hist[0], values):
                return hist[1], values_comp_all  # No need to compute new

    if with_numba:
        no_of_combinations = total_sample_splits_categorical_eff_numba(
            no_of_values)
    else:
        no_of_combinations = total_sample_splits_categorical_eff(no_of_values)

    if no_of_combinations < no_eva_point:
        combinations_new = all_combinations_no_complements_eff(values)
    else:
        if select_values:
            no_of_evalupoints_new = round(find_evapoints_cat(
                len(values), no_eva_point, with_numba=with_numba))

            rng = np.random.default_rng(seed=seed)
            # indx = rng.choice(range(len(values)), size=no_of_evalupoints_new,
            #                   replace=False)
            indx = rng.choice(len(values), size=no_of_evalupoints_new,
                              replace=False)
            combinations_new = all_combinations_no_complements_eff(values[indx])
        else:
            # Sort values according to policy score differences
            values_sorted, no_of_ps = opt_pt_add.get_values_ordered(
                single_x_np, ps_np_diff, values, no_of_values,
                with_numba=with_numba)
            combinations_t = sorted_values_into_combinations_eff(
                values_sorted, no_of_ps, no_of_values)
            combinations_ = drop_complements_eff(combinations_t, values,
                                                 sublist=False)
            len_c = len(combinations_)

            if len_c > no_eva_point:
                if no_eva_point == 1:
                    indx = (len_c // 2,)
                else:
                    rng = np.random.default_rng(seed=seed)
                    # indx = rng.choice(len_c, size=no_eva_point,
                    #                   replace=False).tolist()
                    indx = rng.choice(len_c, size=no_eva_point, replace=False)
                combinations_new = [combinations_[i] for i in indx]
            elif len_c < no_eva_point:
                # Fill with some random combinations previously omitted.
                # This case can happen because of the ordering used above.
                combinations_new = add_combis(combinations_, values_sorted,
                                              no_eva_point)
            else:
                combinations_new = combinations_
    if values_comp_all is None:
        values_comp_all = []
    values_comp_all.append((values, combinations_new))

    return combinations_new, values_comp_all


def add_combis(combinations_: list,
               values_sorted: list,
               no_values_to_add: int
               ) -> list:
    """Add combinations."""
    no_to_add = min(no_values_to_add, len(values_sorted) // 2)

    combinations_.extend(map(tuple, values_sorted[:no_to_add]))

    return combinations_


@lru_cache(maxsize=None)
def find_evapoints_cat(no_values: int,
                       no_of_evalupoints: int,
                       with_numba: bool = True
                       ) -> int:
    """Find number of categories that find to no_of_evaluation points."""
    if no_values < 6:
        return no_values

    for vals in range(6, no_values):
        if with_numba and vals < 20:
            no_of_combinations = total_sample_splits_categorical_eff(vals)
        else:
            no_of_combinations = opt_pt_add.total_sample_splits_categorical(
                vals)
        if no_of_combinations > no_of_evalupoints:
            no_values = vals
            break

    return no_values


@njit
def comb_numba(n: int, k: int) -> int:
    """Scipy function comb for numba."""
    if k > n or k < 0:
        return 0

    if k in (0, n):
        return 1

    result = np.int64(1)
    for i in range(1, k + 1):
        result = result * (n - k + i) // i

    return result


@njit    # Optimization check July, 1, 2025
def total_sample_splits_categorical_eff_numba(no_of_values: int) -> int:
    """Compute the total # of unique sample splits for categorical variables.

    (excluding complementary splits)

    Parameters
    ----------
    no_of_values : int
        Number of unique categorical values.

    Returns
    -------
    int
        Total number of unique sample splits.
    """
    if no_of_values < 2:
        return 0  # No possible splits if less than 2 categories
    return (1 << (no_of_values - 1)) - 1  # Equivalent to 2**(no_of_values-1)-1


def total_sample_splits_categorical_eff(no_of_values: int) -> int:
    """Compute the total # of unique sample splits for categorical variables.

    (excluding complementary splits)

    Parameters
    ----------
    no_of_values : int
        Number of unique categorical values.

    Returns
    -------
    int
        Total number of unique sample splits.
    """
    if no_of_values < 2:
        return 0  # No possible splits if less than 2 categories
    return (1 << (no_of_values - 1)) - 1  # Equivalent to 2**(no_of_values-1)-1


# Optimized with ChatGPT-4.0.mini-high, July, 1, 2025
def all_combinations_no_complements_eff(values: Sequence[T]
                                        ) -> list[tuple[T, ...]]:
    """
    Only emit subsets whose 'highest' element (values[n-1]) is absent.

    That gives exactly (2^{n-1}-1) non‐empty, non‐full subsets.
    """
    n = len(values)
    if n < 2:
        return []
    out: list[tuple[T, ...]] = []
    # iterate masks 1 .. 2^{n-1}-1
    limit = 1 << (n - 1)
    for mask in range(1, limit):
        combo = []
        m = mask
        i = 0
        while m:
            if m & 1:
                combo.append(values[i])
            i += 1
            m >>= 1
        out.append(tuple(combo))

    return out


# def all_combinations_no_complements_eff(values: list[Real, ...]
#                                         ) -> list[Real, ...]:
#     """Create all possible combinations of list elements,removing complements.

#     Parameters
#     ----------
#     values : List. Elements to be combined.

#     Returns
#     -------
#     list_without_complements : List of tuples.

#     """
#     # This returns a list with tuples of all possible combinations of tuples
#     list_all = [combinations(values, length) for length
#                 in range(1, len(values))]
#     # Next, the complements to each list will be removed
#     list_wo_compl = drop_complements_eff(list_all, values)
#     return list_wo_compl


def drop_complements_eff(list_all: list[tuple, ...],
                         values: list[Real, ...],
                         sublist: bool = True
                         ) -> list:
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
    values_set = set(values)
    list_w_compl, list_wo_compl = set(), []

    if sublist:
        for sub1 in list_all:
            for i_i in sub1:
                i = tuple(i_i)
                if i not in list_w_compl:
                    list_wo_compl.append(i)
                    compl_of_i = tuple(values_set - set(i))
                    list_w_compl.add(compl_of_i)
    else:
        for i in list_all:
            if i not in list_w_compl:
                list_wo_compl.append(i)
                compl_of_i = tuple(values_set - set(i))
                list_w_compl.add(compl_of_i)

    return list_wo_compl


# Appears to be optimized as much as possible. 5.6.2024
def sorted_values_into_combinations_eff(values_sorted: np.ndarray,
                                        no_of_ps: int,
                                        no_of_values: int
                                        ) -> list[tuple, ...]:
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
    all_combs = [
        tuple(values_sorted[:i, j])
        for j in range(no_of_ps)
        for i in range(1, no_of_values)
        ]
    unique_combinations = set(all_combs)

    return list(unique_combinations)


def prepare_data_for_tree_building_eff(optp_: 'OptimalPolicy',
                                       data_df: DataFrame,
                                       seed: int = 123456
                                       ) -> tuple[np.ndarray, np.ndarray,
                                                  list, list, list,  list, list
                                                  ]:
    """Prepare data for tree building."""
    x_type, x_values = optp_.var_x_type, optp_.var_x_values
    # x_values is numpy array
    data_ps = data_df[optp_.var_dict['polscore_name']].to_numpy()
    data_ps_diff = data_ps[:, 1:] - data_ps[:, 0, np.newaxis]
    data_opt_treat = np.argmax(data_ps, axis=1)
    no_of_x = len(x_type)
    name_x = [None for _ in range(no_of_x)]
    type_x = [None for _ in range(no_of_x)]
    values_x = [None for _ in range(no_of_x)]
    values_comp_all = [None for _ in range(no_of_x)]
    for j, key in enumerate(x_type.keys()):
        name_x[j], type_x[j] = key, x_type[key]
        vals = np.array(x_values[key])  # Values are sorted - all vars
        if type_x[j] == 'cont':
            obs = len(vals)
            start = obs / optp_.pt_dict['no_of_evalupoints'] / 2
            stop = obs - 1 - start
            indi = np.linspace(start, stop,
                               num=optp_.pt_dict['no_of_evalupoints'],
                               dtype=np.int32)
            values_x[j] = vals[indi]
        else:
            values_x[j] = vals.copy()
    data_x = data_df[name_x].to_numpy()
    del data_df
    if optp_.gen_dict['x_unord_flag']:
        for m_i in range(no_of_x):
            if type_x[m_i] == 'unord':
                data_x[:, m_i] = np.round(data_x[:, m_i])
                (values_x[m_i], values_comp_all[m_i]
                 ) = combinations_categorical_eff(
                    data_x[:, m_i], data_ps_diff, None,
                    optp_.pt_dict['no_of_evalupoints'],
                    select_values=optp_.pt_dict['select_values_cat'],
                    factor=optp_.pt_dict['eva_cat_mult'],
                    with_numba=optp_.int_dict['with_numba'],
                    seed=seed)

    return (data_x, data_ps, data_ps_diff, data_opt_treat,
            name_x, type_x, values_x, values_comp_all
            )


# Seems to be optimized. 5.6.2024; function is not critical
def prepare_ordered_for_xtr_splits(type_x: list, values_x: list) -> list:
    """Prepare data for additional parallelization."""
    values_x_xtr_p = values_x.copy()
    max_splits = 4
    for idx, type_ in enumerate(type_x):
        if type_ == 'unord':
            values_x_xtr_p[idx] = [values_x[idx]]    # Make it a list
        else:
            values_to_split = values_x[idx]
            number_of_splits = min(max_splits, len(values_to_split) // 2)
            if number_of_splits > 1:
                values_x_xtr_p[idx] = np.array_split(values_to_split,
                                                     number_of_splits)
            else:
                values_x_xtr_p[idx] = [values_to_split]    # Make it a list
    return values_x_xtr_p


# Optimized version
def get_values_cont_ord_x_eff(data_vector: np.ndarray,
                              x_values: np.ndarray
                              ) -> np.ndarray:
    """Get cut-off points for tree splitting for single continuous variables.

    Parameters
    ----------
    data_vector : Numpy-1D array.
    x_values : Numpy array. Potential cut-off points.

    Returns
    -------
    Numpy 1D-array. Sorted cut-off-points.
    """
    len_data_vector = data_vector.shape[0]

    if len_data_vector < (x_values.shape[0] / 2):
        return np.unique(data_vector)

    min_x, max_x = np.min(data_vector), np.max(data_vector)

    split_values_mask = (x_values >= min_x) & (x_values < max_x)
    split_x_values = x_values[split_values_mask]

    no_vals = split_x_values.shape[0]
    if no_vals < 2:
        return np.array([min_x + 0.5 * (max_x - min_x)])

    if len_data_vector < (no_vals / 2):
        return np.unique(data_vector)

    return split_x_values


def no_further_gain_split_funct(opt_treat: np.ndarray) -> bool:
    """Define if additional leaf splitting can lead to improvements."""
    no_further_gain_split = np.all(opt_treat == opt_treat[0])

    return no_further_gain_split
