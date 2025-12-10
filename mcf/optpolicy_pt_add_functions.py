"""
Provide functions for Black-Box allocations.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from itertools import combinations
from math import comb
from typing import Any, TYPE_CHECKING

from numba import njit, prange
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from mcf import mcf_print_stats_functions as mcf_ps
if TYPE_CHECKING:
    from mcf.optpolicy_methods import OptimalPolicy


def describe_tree(optp_: 'OptimalPolicy',
                  splits_seq: list,
                  treat: list,
                  obs: list | None = None,
                  score_val: list | None = None,
                  indx_in_leaf: list | None = None,
                  fairness_corrected: bool = False,
                  ) -> tuple[str, dict]:
    """Describe leaves of policy tree."""
    gen_cfg, other_cfg = optp_.gen_cfg, optp_.other_cfg
    txt = '\n' + '-' * 100 + '\nDetails of the estimated policy tree '
    txt += '\n' + '- ' * 50
    txt += (f'\nDepth of 1st tree: {optp_.pt_cfg.depth_tree_1_adj}, '
            'depth of 2nd tree (inside the leaves of the 1st tree): '
            f'{optp_.pt_cfg.depth_tree_2_adj}, '
            f'total depth: {optp_.pt_cfg.total_depth_adj}'
            )
    txt += '\n' + '- ' * 50
    tree_dic = {}
    if fairness_corrected:
        fair_cfg, var_cfg = optp_.fair_cfg, optp_.var_cfg
        # For all leaves: describe
        strata_txt, _ = describe_protected_material_strata(fair_cfg, var_cfg)

        txt += strata_txt
    txt += '\n\nAnalysing the single leaves of the policy tree\n' + '- ' * 50
    for i, splits in enumerate(splits_seq):
        for j in range(2):
            # This is a final leaf
            leaf_key = 'Leaf' + str(i) + str(j)
            tree_dic[leaf_key] = {}
            txt += f'\nLeaf {i:d}{j:d}:  '
            if fairness_corrected:
                variables_leaf_name = []

            for splits_dic in splits[j]:  # Iterate over leaf information
                tree_dic[leaf_key]['splitvariable'] = splits_dic['x_name']
                if fairness_corrected:
                    variables_leaf_name.append(splits_dic['x_name'])

                tree_dic[leaf_key]['splitvariable_type'] = splits_dic['x_type']
                txt += f'{splits_dic["x_name"]:4s}'
                if splits_dic['x_type'] == 'unord':
                    if splits_dic['left or right'] == 'left':
                        txt += ' In:     '
                        tree_dic[leaf_key]['split_type'] = 'values are in set'
                    else:
                        txt += ' Not in: '
                        tree_dic[leaf_key]['split_type'] = (
                            'values are not in set')
                    values_to_print = np.sort(splits_dic['cut-off or set'])
                    tree_dic[leaf_key]['cut-off or set'] = values_to_print
                    for s_i in values_to_print:
                        if isinstance(s_i, int) or (
                                (s_i - np.round(s_i)) < 0.00001):
                            txt += f'{int(np.round(s_i)):2d} '
                        else:
                            txt += f'{s_i:3.1f} '
                else:
                    if splits_dic['left or right'] == 'left':
                        txt += ' <='
                        tree_dic[leaf_key]['split_type'] = 'values are <= than'
                    else:
                        txt += ' > '
                        tree_dic[leaf_key]['split_type'] = 'values are > than'
                    txt += f'{splits_dic["cut-off or set"]:8.3f} '
                    tree_dic[leaf_key]['cut-off or set'] = splits_dic[
                        'cut-off or set']
            txt += f'\nAlloc Treatment: {treat[i][j]:3d} '
            if obs is not None:
                txt += f'  Obs: {obs[i][j]:6d}  '
            if score_val is not None:
                txt += f'Avg.score: {score_val[i][j] / obs[i][j]:7.3f} '
                tmp = (score_val[i][j] / obs[i][j]
                       - other_cfg.costs_of_treat[treat[i][j]])
                txt += (f'Avg.score-costs: {tmp:7.3f} ')

            if fairness_corrected:
                # Indices of obs in leaf
                variables_leaf_name = list(set(variables_leaf_name))
                idx = indx_in_leaf[i][j].to_numpy().reshape(-1)
                strata_indices_leaf_np = fair_cfg.fair_strata.to_numpy()[idx]

                variables_leaf_org_name = [
                    fair_cfg.decision_vars_fair_org_name[var]
                    for var in variables_leaf_name]

                x_data_leaf_np = fair_cfg.decision_vars_org_df[
                    variables_leaf_org_name].iloc[idx].to_numpy()

                txt_strata, _ = strata_specific_stats(
                    variables_leaf_org_name,
                    fair_cfg.x_ord_org_name,
                    strata_indices_leaf_np,
                    x_data_leaf_np,
                    var_width=20, return_strata_indices=False
                    )
                txt += txt_strata
            txt += ('\n' + '- ' * 50)
    txt += ('\nNote: Splitpoints displayed for ordered variables are '
            'midpoints between observable values '
            '\n(e.g., 0.5 for a variable with values of 0 and 1).'
            ) + '\n' + '-' * 100
    mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    return txt, tree_dic


def describe_protected_material_strata(fair_cfg: Any,
                                       var_cfg: Any,
                                       ) -> tuple[str, list[NDArray[Any]]]:
    """Describe the strata generated by the protected and mat.rel. variables."""
    # Get the necessary data from the fairness directory
    mat_rel_name = [var for var in fair_cfg.protected_matrel.columns
                    if var in var_cfg.material_ord_name
                    or var in var_cfg.protected_ord_name
                    or var in var_cfg.prot_mat_no_dummy_name
                    ]
    strata_indices_np = fair_cfg.fair_strata.to_numpy()
    mat_rel_np = fair_cfg.protected_matrel[mat_rel_name].to_numpy()
    if len(strata_indices_np) != len(mat_rel_np):
        raise ValueError('Array with indices of stratification variables '
                         'and array with protected variables have different '
                         f'length ({len(strata_indices_np)} vs '
                         f'{len(mat_rel_np)}. Probably, them come from '
                         'different datasets and are not compatible.'
                         )

    mat_rel_ord_name = [var for var in mat_rel_name
                        if var in var_cfg.material_ord_name
                        or var in var_cfg.protected_ord_name
                        ]

    # Some descriptive stats(min-max, values) by stratum
    strata_no = np.unique(strata_indices_np)

    txt = '\n' + '- ' * 50
    txt = (f'\nInformation on the {len(strata_no)} strata build '
           'by the quantilization (based on the protected and materially '
           'relevant variables, if specified).'
           )
    VAR_WIDTH = 20

    txt_strata, strata_indices = strata_specific_stats(
        mat_rel_name,
        mat_rel_ord_name,
        strata_indices_np,
        mat_rel_np,
        var_width=VAR_WIDTH, return_strata_indices=True
        )
    txt += txt_strata + '\n' + '- ' * 50
    txt += (f'\nNote: Strata formed using {fair_cfg.protected_disc_method} '
            'for protected variables.'
            )
    if var_cfg.material_ord_name or var_cfg.material_unord_name:
        txt += (f'Strata formed using {fair_cfg.material_disc_method} '
                'for materially relevant variables.'
                )
    txt += '\n' + '- ' * 50

    return txt, strata_indices


def strata_specific_stats(var_name: list[str],
                          var_ord_name: list[str],
                          strata_indices_np: NDArray[Any],
                          var_np: NDArray[Any],
                          var_width: int = 20,
                          return_strata_indices: bool = True,
                          ) -> tuple[str, list[NDArray[Any]]]:
    """Describe specific variables by stratum."""
    strata_no = np.unique(strata_indices_np)
    all_indices = np.arange(len(strata_indices_np))
    strata_indices = [] if return_strata_indices else None

    var_name_padded = [s.rjust(var_width) for s in var_name]
    txt = '\n\nVariable' + ' '.join(var_name_padded)
    subheader = []
    for name in var_name:
        sub_name = ('Min'.rjust(int(var_width/2))
                    + 'Max'.rjust(int(var_width/2))
                    if name in var_ord_name else 'Values'.center(var_width)
                    )
        subheader.append(sub_name)
    txt += '\nStratum obs ' + ' '.join(subheader)
    for stratum in np.int16(strata_no):
        idx = all_indices[(strata_indices_np == stratum).reshape(-1)]
        if return_strata_indices:
            strata_indices.append(idx)
        var_stratum = var_np[idx]
        txt += f'\n{stratum:<5d} {len(var_stratum):>5d}'
        for jdx, name in enumerate(var_name):
            if name in var_ord_name:
                min_ = np.min(var_stratum[:, jdx])
                max_ = np.max(var_stratum[:, jdx])
                if isinteger(min_):
                    txt += f' {round(min_):>10d}'
                else:
                    txt += f' {min_:>10.3f}'
                if isinteger(max_):
                    txt += f' {round(max_):>9d}'
                else:
                    txt += f' {max_:>9.3f}'
            else:
                unique_val = np.unique(var_stratum[:, jdx])
                if isinteger(unique_val):
                    uni_str = [f'{round(s):5>d}' for s in unique_val]
                else:
                    uni_str = [f'{s:>5.2f}' for s in unique_val]
                txt += f'|{(" ".join(uni_str)).center(var_width)}'

    return txt, strata_indices


def isinteger(floating: NDArray[Any]) -> bool:
    """Decide if float is close enough to integer."""
    return np.all(np.abs(floating - np.round(floating)) < 1e-8)


def automatic_cost(optp_: 'OptimalPolicy',
                   data_df: DataFrame
                   ) -> list[float]:
    """Compute costs that fulfill constraints."""
    gen_cfg, var_cfg = optp_.gen_cfg, optp_.var_cfg
    other_cfg = optp_.other_cfg

    if gen_cfg.with_output:
        print('\nSearching cost values that fulfill constraints')
    data_ps = data_df[var_cfg.polscore_name].to_numpy()
    obs = data_ps.shape[0]

    max_by_treat = np.around(np.array(other_cfg.max_shares) * obs)
    if any(cost > 0 for cost in other_cfg.costs_of_treat):
        costs_of_treat = other_cfg.costs_of_treat.copy()
    else:
        costs_of_treat = np.zeros(gen_cfg.no_of_treat)
    std_ps = np.std(data_ps.reshape(-1))
    step_size = 0.02
    max_iterations = 1000

    for iterations in range(max_iterations):
        treatments = np.argmax(data_ps - costs_of_treat, axis=1)
        values, count = np.unique(treatments, return_counts=True)
        if len(count) == gen_cfg.no_of_treat:
            alloc = count
        else:
            alloc = np.zeros(gen_cfg.no_of_treat)
            for i, j in enumerate(values):
                alloc[j] = count[i]
        diff = alloc - max_by_treat

        diff[diff < 0] = 0
        if not np.any(diff > 0):
            break

        if iterations % 100 == 0:
            step_size /= 2
            if gen_cfg.with_output:
                iter_string = ' '.join(f'({int(s):3d})' for s in diff)
                print(f'Iterations: {iterations}, {iter_string}')

        costs_of_treat += diff / obs * std_ps * step_size

    alloc = alloc.astype(np.int32)
    costs_of_treat_update = costs_of_treat * other_cfg.costs_of_treat_mult
    costs_of_treat_neu = other_cfg.costs_of_treat.copy()
    for idx, cost in enumerate(costs_of_treat_update):
        if cost > other_cfg.costs_of_treat[idx]:
            costs_of_treat_neu[idx] = cost

    if gen_cfg.with_output:
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
                    for mult in other_cfg.costs_of_treat_mult
                    ]
        mult = ' '.join(mult_str)
        txt += f'\nMultipliers to be used: {mult}'
        txt += '\n' + '- ' * 50 + ('\nAdjusted cost values and allocation in '
                                   'unconstrained optimization')
        for idx, cost in enumerate(costs_of_treat_neu):
            txt += (f'\nCost: {cost:8.3f}    Obs.: {alloc[idx]:6d}    Share:'
                    f' {alloc[idx] / obs:6.2%}')
        txt += '\n' + '-' * 100
        mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    return costs_of_treat_neu


# No longer used, more efficient version available
def combinations_categorical(single_x_np: NDArray[Any],
                             ps_np_diff: NDArray[Any],
                             no_of_evalupoints: int,
                             with_numba: bool,
                             seed: int = 123456
                             ) -> list[range]:
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


# No longer needed, more efficient version available
def total_sample_splits_categorical(no_of_values: int) -> int:
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


# No longer needed, more efficient version available
def all_combinations_no_complements(values: list) -> list[tuple]:
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


# No longer needed, more efficient version available
def drop_complements(list_all: list[tuple],
                     values: list
                     ) -> list[tuple]:
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


# No longer needed, more efficient version available
def sorted_values_into_combinations(values_sorted: NDArray[Any],
                                    no_of_ps: int,
                                    no_of_values: int
                                    ) -> list[tuple]:
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


def get_values_ordered(single_x_np: NDArray[Any],
                       ps_np_diff: NDArray[Any],
                       values: NDArray[Any],
                       no_of_values: int,
                       with_numba: bool = True
                       ) -> tuple[NDArray[Any], int]:
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
            single_x_np, ps_np_diff, values, no_of_values
            )
    else:
        values_sorted, no_of_ps = get_values_ordered_no_numba(
            single_x_np, ps_np_diff, values, no_of_values
            )
    return values_sorted, no_of_ps


@njit   # Optimized by ChatGPT-4.o mini-high
def get_values_ordered_numba(single_x_np:  NDArray[Any],
                             ps_np_diff:   NDArray[Any],
                             values:       NDArray[Any],
                             no_of_values: int,
                             ) -> tuple[NDArray[Any], int]:
    """
    Sort values by mean of ps_np_diff within each group defined by single_x_np.

    This version:
      - Accumulates sums & counts in one pass (O(N * P))
      - Computes means in-place (O(V * P))
      - Sorts each column (O(P * V log V))
      - Never allocates big boolean masks or intermediate slices.

    Parameters
    ----------
    single_x_np   : 1D array of length N, group labels (must match `values`)
    ps_np_diff    : 2D array of shape (N, P), policy-score differences
    values        : 1D array of length V, all unique group labels
    no_of_values  : int, = V

    Returns
    -------
    values_sorted : 2D array of shape (V, P), the `values` array reordered by
                    ascending mean ps_np_diff in each column
    no_of_ps      : int, = P
    """
    n = single_x_np.shape[0]
    no_of_ps = ps_np_diff.shape[1]

    # accumulators
    sums = np.zeros((no_of_values, no_of_ps), dtype=np.float64)
    counts = np.zeros(no_of_values,           dtype=np.int64)

    # one‐pass: group sums & counts
    for r in range(n):
        x = single_x_np[r]
        # linear search for the group index i where values[i] == x
        # (if values is sorted, you could binary‐search here)
        for i in range(no_of_values):
            if values[i] == x:
                break

        counts[i] += 1
        for j in range(no_of_ps):
            sums[i, j] += ps_np_diff[r, j]

    # compute group means
    for i in range(no_of_values):
        cnt = counts[i]
        if cnt > 0:
            inv = 1.0 / cnt
            for j in range(no_of_ps):
                sums[i, j] *= inv
        else:
            # optional: define what you want when a group is empty
            for j in range(no_of_ps):
                sums[i, j] = 0.0

    # allocate output and sort each column
    values_sorted = np.empty((no_of_values, no_of_ps), dtype=values.dtype)
    for j in range(no_of_ps):
        idx = np.argsort(sums[:, j])  # 1D argsort is supported in nopython
        for k in range(no_of_values):
            values_sorted[k, j] = values[idx[k]]

    return values_sorted, no_of_ps


@njit(parallel=True)  # Turns out to increase computation time, not used
def get_values_ordered_numba_prange(
        single_x_np: NDArray[Any],
        ps_np_diff: NDArray[Any],
        values: NDArray[Any],
        no_of_values: int,
        ) -> tuple[NDArray[Any], int]:
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


def get_values_ordered_no_numba(single_x_np: NDArray[Any],
                                ps_np_diff: NDArray[Any],
                                values: NDArray[Any],
                                no_of_values: int
                                ) -> tuple[NDArray[Any], int]:
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


def adjust_reward(no_by_treat_l: NDArray[Any],
                  no_by_treat_r: NDArray[Any],
                  reward_l: float,
                  reward_r: float,
                  with_numba: bool,
                  max_by_treat: int
                  ) -> tuple[NDArray[Any], NDArray[Any]]:
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


# Check for optimization, June, 5, 2024
@njit
def adjust_reward_numba(no_by_treat_l: NDArray[Any],
                        no_by_treat_r: NDArray[Any],
                        reward_l: float,
                        reward_r: float,
                        max_by_treat: int
                        ) -> tuple[NDArray[Any], NDArray[Any]]:
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


def adjust_reward_no_numba(no_by_treat_l: NDArray[Any],
                           no_by_treat_r: NDArray[Any],
                           reward_l: float,
                           reward_r: float,
                           max_by_treat: int
                           ) -> tuple[NDArray[Any], NDArray[Any]]:
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


# No longer needed, more efficient version available
def prepare_data_for_tree_building(
        optp_: 'OptimalPolicy',
        data_df: DataFrame,
        seed: int = 123456
        ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], list, list, list]:
    """Prepare data for tree building."""
    int_cfg, var_cfg = optp_.int_cfg, optp_.var_cfg
    x_type, x_values = optp_.var_x_type, optp_.var_x_values
    gen_cfg, pt_cfg = optp_.gen_cfg, optp_.pt_cfg
    data_ps = data_df[var_cfg.polscore_name].to_numpy()
    data_ps_diff = data_ps[:, 1:] - data_ps[:, 0, np.newaxis]
    no_of_x = len(x_type)
    name_x = [None for _ in range(no_of_x)]
    type_x = [None for _ in range(no_of_x)]
    values_x = [None for _ in range(no_of_x)]
    for j, key in enumerate(x_type.keys()):
        name_x[j], type_x[j] = key, x_type[key]
        values_x[j] = (sorted(x_values[key])
                       if x_values[key] is not None else None)
    #                  this None for continuous variables
    data_x = data_df[name_x].to_numpy()
    del data_df
    if gen_cfg.x_unord_flag:
        for m_i in range(no_of_x):
            if type_x[m_i] == 'unord':
                data_x[:, m_i] = np.round(data_x[:, m_i])
                values_x[m_i] = combinations_categorical(
                    data_x[:, m_i], data_ps_diff,
                    pt_cfg.no_of_evalupoints, int_cfg.with_numba,
                    seed=seed)
    return data_x, data_ps, data_ps_diff, name_x, type_x, values_x


# Version optimized by ChatGPT-4o.mini-high, 2025, July, 1
@njit
def only_1st_tree_fct3(data_ps: NDArray[Any],
                       costs_of_treat: NDArray[Any]
                       ) -> bool:
    """Find out if further splits make any sense.

    Return True if every row of (data_ps - costs_of_treat)
    has its maximum in the *same* column.
    """
    n_rows, n_cols = data_ps.shape

    # --- Find argmax in row 0 ---------------------------------
    ref_idx = 0
    # first element
    best_val0 = data_ps[0, 0] - costs_of_treat[0]
    for j in range(1, n_cols):
        tmp = data_ps[0, j] - costs_of_treat[j]
        if tmp > best_val0:
            best_val0 = tmp
            ref_idx = j

    # --- Check all other rows --------------------------------
    for i in range(1, n_rows):
        # find argmax in row i
        best_val = data_ps[i, 0] - costs_of_treat[0]
        best_j = 0
        for j in range(1, n_cols):
            tmp = data_ps[i, j] - costs_of_treat[j]
            if tmp > best_val:
                best_val = tmp
                best_j = j

        # early exit on first mismatch
        if best_j != ref_idx:
            return False

    return True


def evaluate_leaf(data_ps: NDArray[Any],
                  gen_cfg: Any,
                  other_cfg: Any,
                  pt_cfg: Any,
                  with_numba: bool = True
                  ) -> tuple[int, np.floating, int]:
    """Evaluate final value of leaf taking restriction into account.

    Parameters
    ----------
    data_ps : Numpy array. Policy scores.
    gen_cfg : GenCfg dataclass. Controls.
    other_cfg, pt_cfg : Dataclass. Controls.

    Returns
    -------
    treat_ind: Int. Index of treatment.
    reward: Int. Value of leaf.
    no_per_treat: Numpy 1D-array of int.

    """
    if with_numba:
        indi, reward_by_treat, obs_all = evaluate_leaf_numba(
            data_ps, gen_cfg.no_of_treat, other_cfg.max_by_treat,
            other_cfg.restricted and pt_cfg.enforce_restriction,
            pt_cfg.cost_of_treat_restrict
            )
    else:
        indi, reward_by_treat, obs_all = evaluate_leaf_no_numba(
            data_ps, gen_cfg.no_of_treat, other_cfg, pt_cfg
            )
    return indi, reward_by_treat, obs_all


# Version optimized by ChatGPT-4o.mini-high, 2025, July, 1
@njit(parallel=False)
def evaluate_leaf_numba(data_ps: NDArray[Any],
                        no_of_treatments: int,
                        max_by_treat: NDArray[Any],
                        restricted: bool,
                        costs_of_treat: NDArray[Any],
                        ) -> tuple[int, np.floating[Any], NDArray[Any]]:
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
    obs = data_ps.shape[0]

    # 1) compute column sums in pure Python loops
    col_sums = np.empty(no_of_treatments, dtype=np.float64)
    for j in range(no_of_treatments):
        s = 0.0
        for i in range(obs):
            s += data_ps[i, j]
        col_sums[j] = s

    # 2) find best valid treatment
    best_reward = -np.inf
    best_treat = -1
    for j in range(no_of_treatments):
        if restricted and obs > max_by_treat[j]:
            # this treatment would exceed its own max_by_treat[j]
            continue
        # evaluate reward
        r = col_sums[j] - costs_of_treat[j] * obs
        if r > best_reward:
            best_reward = r
            best_treat = j

    # 3) fallback: if all were restricted out, pick the one with smallest
    #              over-limit
    if restricted and best_treat == -1:
        # obs - max_by_treat[j] is > 0 for every j; pick j with minimal
        # (obs - max_by_treat[j])
        min_diff = obs - max_by_treat[0]
        best_treat = 0
        for j in range(1, no_of_treatments):
            diff = obs - max_by_treat[j]
            if diff < min_diff:
                min_diff = diff
                best_treat = j
        best_reward = col_sums[best_treat] - costs_of_treat[best_treat] * obs

    # 4) build the obs_all vector
    obs_all = np.zeros(no_of_treatments, dtype=np.int64)
    obs_all[best_treat] = obs

    return best_treat, best_reward, obs_all


def evaluate_leaf_no_numba(data_ps: NDArray[Any],
                           no_of_treat: int,
                           other_cfg: Any,
                           pt_cfg: Any,
                           ) -> tuple[int, np.floating[Any], int]:
    """Evaluate final value of leaf taking restriction into account.

    Returns
    -------
    treat_ind: Int. Index of treatment.
    reward: Int. Value of leaf.
    no_per_treat: Numpy 1D-array of int.

    """
    obs_all, obs = np.zeros(no_of_treat), len(data_ps)
    indi = np.arange(no_of_treat)
    costs_of_treat = pt_cfg.cost_of_treat_restrict
    if other_cfg.restricted and pt_cfg.enforce_restriction:
        diff_obs = obs - other_cfg.max_by_treat
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
            costs_of_treat = pt_cfg.cost_of_treat_restrict[indi]
    reward_by_treat = data_ps.sum(axis=0) - costs_of_treat * obs
    max_i = np.argmax(reward_by_treat)
    obs_all[indi[max_i]] = obs

    return indi[max_i], reward_by_treat[max_i], obs_all


def get_values_cont_x(data_vector: NDArray[Any],
                      no_of_evalupoints: int,
                      with_numba: bool = True,
                      ) -> NDArray[Any]:
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
def get_values_cont_x_numba(data_vector: NDArray[Any],
                            no_of_evalupoints: int
                            ) -> NDArray[Any]:
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
        return data_vector

    indices = np.linspace(obs / no_of_evalupoints, obs, no_of_evalupoints + 1)
    data_vector_new = np.empty(no_of_evalupoints)

    for i in range(no_of_evalupoints):
        indices_i = int(indices[i])
        data_vector_new[i] = data_vector[indices_i]

    return data_vector_new


@njit(parallel=True)  # Turns out to increase computation time, not used
def get_values_cont_x_numba_prange(data_vector: NDArray[Any],
                                   no_of_evalupoints: int
                                   ) -> NDArray[Any]:
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


def get_values_cont_x_no_numba(data_vector: NDArray[Any],
                               no_of_evalupoints: int
                               ) -> NDArray[Any]:
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
