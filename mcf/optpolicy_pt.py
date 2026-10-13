"""
Provide functions for old policy tree allocations.

Most functions are no longer used.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy
from itertools import count
from numbers import Real  # For type checking only
from math import inf, ceil
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
try:
    import ray
except (ImportError, OSError):
    ray = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats as mcf_ps
from mcf import optpolicy_pt_add as opt_pt_add
from mcf import optpolicy_pt_eff as opt_pt_eff
from mcf.mcfoptp_parallel_backend_ray_classical import init_ray_with_fallback, ray_put_all
from mcf.mcfoptp_parallel_backend_forest_executor import make_forest_executor
from mcf.mcfoptp_parallel_backends_base import (batched_tasks, batch_size_fct, SequentialExecutor,
                                                TaskSpec,
                                                )
if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy


def policy_tree_allocation(optp_: 'OptimalPolicy',
                           data_df: DataFrame
                           ) -> tuple[DataFrame, str, dict]:
    """Compute optimal policy tree brute force (recursive algorithm)."""
    other_cfg, pt_cfg = optp_.other_cfg, optp_.pt_cfg
    no_of_features = len(optp_.var_x_type)
    if no_of_features > 30:
        txt_warning = ('\n\nWARNING\nWARNING\n'
                       f'{no_of_features} features specified for optimal policy tree. '
                       '\nWARNING Consider reducing this number by substantive reasoning or '
                       'checking variable importance statistics.\nWARNING\n\n'
                       )
    else:
        txt_warning = ''

    if other_cfg.restricted:
        costs_of_treat_update = opt_pt_add.automatic_cost(optp_, data_df)
    else:
        costs_of_treat_update = other_cfg.costs_of_treat
    # Do not update cost information in attribute but keep it separate
    optp_.pt_cfg.cost_of_treat_restrict = np.array(costs_of_treat_update)

    tree_levels = 2 if pt_cfg.depth_tree_2 > 1 else 1
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
            if optp_.gen_cfg.with_output:
                mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=False)
            while best_tree is None:
                if optp_.gen_cfg.method == 'policy_tree':
                    best_tree, _, _ = opt_pt_eff.optimal_tree_eff_proc(
                        optp_local, data_tmp_df, seed=12345, zero_tol=optp_.int_cfg.zero_tol,
                        )
                elif optp_.gen_cfg.method == 'policy tree old':
                    best_tree, _, _ = optimal_tree_proc(optp_local, data_tmp_df, seed=12345)
                if best_tree is None:
                    if optp_.gen_cfg.with_output:
                        txt = ('No tree obtained for depth '
                               f'{optp_local.pt_cfg.depth-1}. Depth is '
                               'reduced.\nIf results are desired for original '
                               'depth level, try reducing the minimum leaf size'
                               ' or increasing the # of evaluation points.'
                               )
                        mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=True)
                    optp_local.pt_cfg.depth -= 1
                    if optp_local.pt_cfg.depth == 1:
                        break
            if best_tree is None:
                if tree_number == 0:
                    raise RuntimeError('No feasible first-level tree could be constructed, even '
                                       'with a depth of one. Reduce pt_min_leaf_size or provide '
                                       'policy variables with sufficient variation.'
                                       )
                level_2_dic['level_2_tree_list'].append(None)
                continue

            if tree_number == 1:
                level_2_dic['level_2_tree_list'].append(best_tree)

        if tree_levels == 1:
            optp_.pt_cfg.policy_tree = best_tree
        elif tree_levels == 2:
            if tree_number == 0:
                level_2_dic['level_1_tree'] = best_tree
                # create new data_tmp_df_list from trainings data in leaf
                data_df_list, leaf_id_df_list = get_data_from_tree(optp_local, data_df, best_tree)
            elif tree_number == 1:
                # collect all trees and form new tree from the two tree levels
                optp_.pt_cfg.policy_tree = merge_tree_levels(leaf_id_df_list, level_2_dic)

    optp_.pt_cfg.policy_tree = renumber_tree(optp_.pt_cfg.policy_tree)
    pt_alloc_np, _, _, pt_alloc_txt, tree_info_dic = pred_policy_allocation(optp_, data_df)
    allocation_df = DataFrame(data=pt_alloc_np, columns=('Policy Tree',))
    if pt_alloc_txt is None:
        return allocation_df, '', tree_info_dic

    return allocation_df, pt_alloc_txt + txt_warning, tree_info_dic


def policy_tree_prediction_only(optp_: 'OptimalPolicy',
                                data_df: DataFrame
                                ) -> tuple[DataFrame, str]:
    """Predict allocation with policy tree from potentially new data."""
    # Check if all variables used from training are included
    x_type, x_values = optp_.var_x_type, optp_.var_x_values
    # Some consistency check of the data

    # Use lower-case column names without modifying the caller's dataframe.
    data_df = data_df.copy(deep=False)
    data_df.columns = [name.casefold() for name in data_df.columns]
    missing_vars = [name for name in x_type if name not in data_df.columns]

    if missing_vars:
        raise ValueError('Not all features used for training are included. '
                         f'\nMissing variables: {" ".join(missing_vars)}')
    try:
        x_pred_np = data_df[list(x_type)].to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('All policy-tree prediction features must be numeric.') from exc

    if not np.all(np.isfinite(x_pred_np)):
        raise ValueError(
            'Policy-tree prediction features must contain only finite values.'
            )

    for name, var_type in x_type.items():
        if var_type != 'unord':
            continue

        values = data_df[name].unique()
        rounded_values = np.round(values)
        if np.sum(np.abs(values - rounded_values)) > optp_.int_cfg.sum_tol:
            raise TypeError('Categorical variables must be coded as integers.')
        unseen_values = np.setdiff1d(rounded_values, np.round(np.asarray(x_values[name])))
        if unseen_values.size:
            raise ValueError(f'The following values of {name} were not contained in the training '
                             f'data: {unseen_values.tolist()}'
                             )
    pt_alloc_np, _, _, pt_alloc_txt, _ = pred_policy_allocation(optp_, data_df)
    allocation_df = DataFrame(data=pt_alloc_np, columns=('Policy Tree',))

    return allocation_df, pt_alloc_txt


def pred_policy_allocation(optp_: 'OptimalPolicy',
                           data_df: DataFrame,
                           ) -> tuple[NDArray[Any], list, list, str, dict]:
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
    pt_cfg = optp_.pt_cfg
    gen_cfg = optp_.gen_cfg
    policy_tree = pt_cfg.policy_tree

    if not policy_tree:
        raise RuntimeError('No fitted policy tree is available for prediction.')

    tree_by_id = {leaf[0]: leaf for leaf in policy_tree}
    if len(tree_by_id) != len(policy_tree):
        raise ValueError('Some leaf IDs are identical. Rerun programme.')

    root_nodes = [leaf for leaf in policy_tree if leaf[1] is None]
    if len(root_nodes) != 1:
        raise ValueError(f'Expected one root node, found {len(root_nodes)}.')

    terminal_leafs = [leaf for leaf in policy_tree if leaf[2] is None or leaf[3] is None]
    terminal_leafs_id = [leaf[0] for leaf in terminal_leafs]
    terminal_positions = {leaf[0]: idx for idx, leaf in enumerate(terminal_leafs)}

    split_types = {leaf[5]: leaf[6] for leaf in policy_tree}
    x_arrays = {name: data_df[name].to_numpy(copy=False) for name in split_types}
    for name, var_type in split_types.items():
        if var_type == 'unord':
            x_arrays[name] = np.round(x_arrays[name])

    total_obs = len(data_df)
    predicted_treatment = np.empty(total_obs, dtype=np.int64)
    assigned = np.zeros(total_obs, dtype=bool)

    splits_seq = [None] * len(terminal_leafs)
    obs = [None] * len(terminal_leafs)
    treat = [None] * len(terminal_leafs)
    indx_in_leaf = [None] * len(terminal_leafs)

    all_indices = np.arange(total_obs, dtype=np.intp)
    stack = [(root_nodes[0], all_indices, [])]

    while stack:
        leaf, indices, path = stack.pop()
        values = x_arrays[leaf[5]][indices]

        if leaf[6] == 'unord':
            goes_left = np.isin(values, tuple(leaf[7]))
        else:
            goes_left = values <= leaf[7]

        left_indices = indices[goes_left]
        right_indices = indices[~goes_left]

        left_split = final_leaf_dict(leaf, 'left')
        right_split = final_leaf_dict(leaf, 'right')

        indices_by_side = (left_indices, right_indices)
        splits_by_side = (left_split, right_split)
        terminal_sides = [side for side in (0, 1) if leaf[side + 2] is None]

        if terminal_sides:
            position = terminal_positions[leaf[0]]
            splits_seq[position] = tuple(path + [splits_by_side[side]] for side in terminal_sides)
            obs[position] = tuple(len(indices_by_side[side]) for side in terminal_sides)
            treat[position] = tuple(leaf[8][side] for side in terminal_sides)
            indx_in_leaf[position] = tuple(
                DataFrame(data=indices_by_side[side], index=indices_by_side[side],
                          columns=('Sorter',))
                for side in terminal_sides
                )
            for side in terminal_sides:
                predicted_treatment[indices_by_side[side]] = leaf[8][side]
                assigned[indices_by_side[side]] = True

        for side in (1, 0):
            child_id = leaf[side + 2]
            if child_id is not None:
                stack.append((tree_by_id[child_id], indices_by_side[side],
                              path + [splits_by_side[side]]))

    if not assigned.all():
        missing = np.flatnonzero(~assigned)
        raise RuntimeError('Policy-tree prediction left observations '
                           f'unassigned: {missing[:10].tolist()}'
                           )
    if gen_cfg.with_output:
        fairness_xvariable_correction = (
            optp_.fair_cfg.solvefair_used
            and optp_.fair_cfg.adjust_target in ('xvariables', 'scores_xvariables')
            )
        text, tree_dic = opt_pt_add.describe_tree(optp_, splits_seq, treat,
                                                  obs=obs, indx_in_leaf=indx_in_leaf,
                                                  fairness_corrected=fairness_xvariable_correction,
                                                  )
    else:
        text = tree_dic = None

    return predicted_treatment, indx_in_leaf, terminal_leafs_id, text, tree_dic


def two_leafs_info(tree: list[list, ...],
                   polscore_df: DataFrame,
                   x_df: DataFrame,
                   leaf: list,
                   polscore_is_index: bool = False,
                   ) -> tuple[tuple[list, ...], tuple[Any, ...], tuple[int, ...],
                              tuple[int, ...], tuple[Any, ...]]:
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
    # leaf[8] contains treatment information in final leaf

    terminal_sides = [side for side in (0, 1) if leaf[side + 2] is None]
    return (tuple(leaf_splits[side] for side in terminal_sides),
            tuple(score[side] for side in terminal_sides),
            tuple(obs[side] for side in terminal_sides),
            tuple(leaf[8][side] for side in terminal_sides),
            tuple(polscore_df_lr[side] for side in terminal_sides),
            )


def subsample_leaf(any_df: DataFrame,
                   x_df: DataFrame,
                   split: dict,
                   ) -> tuple[DataFrame, DataFrame]:
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
        condition = x_df[split['x_name']].round().isin(split['cut-off or set'])
        if split['left or right'] == 'right':
            condition = ~condition
    else:  # x not in set
        if split['left or right'] == 'left':
            condition = x_df[split['x_name']] <= split['cut-off or set']
        else:
            condition = x_df[split['x_name']] > split['cut-off or set']
    try:
        any_df_red = any_df[condition]
        x_df_red = x_df[condition]
    except RuntimeError as rte:
        raise RuntimeError('Most likely something went wrong with the indexing of the data. '
                           'It is likely that decision variables and scores have different indices.'
                           ) from rte
    return any_df_red, x_df_red


# No longer used part of old policy tree
def optimal_tree_proc(optp_: 'OptimalPolicy',
                      data_df: DataFrame,
                      seed: int = 12345,
                      ) -> tuple[list, NDArray[Any], int]:
    """Build optimal policy tree."""
    gen_cfg, pt_cfg, int_cfg = optp_.gen_cfg, optp_.pt_cfg, optp_.int_cfg
    other_cfg = optp_.other_cfg
    if gen_cfg.with_output:
        print('\nBuilding optimal policy / decision tree')
    use_tqdm, output_clean = mcf_gp.tqdm_setup(tqdm, gen_cfg.with_output)

    (data_x, data_ps, data_ps_diff, name_x, type_x, values_x
     ) = opt_pt_add.prepare_data_for_tree_building(optp_, data_df, seed=seed)

    if gen_cfg.mp_parallel > 1.5:
        ray_err_txt='Ray does not start up in policy tree estimation.'
        if int_cfg.mp_use_old_ray:
            if ray is None or ray_tree_search_multip_single is None:
                raise ImportError('int_cfg.mp_use_old_ray=True, but ray is not installed. '
                                  'Install ray or use the new backend with '
                                  'mp_use_old_ray=False.'
                                  )
            if not ray.is_initialized():
                init_ray_with_fallback(gen_cfg.mp_parallel, gen_cfg, ray_err_txt=ray_err_txt)
            data_x_ref, data_ps_ref, data_ps_diff_ref = ray_put_all(data_x, data_ps, data_ps_diff)
            still_running = [ray_tree_search_multip_single.remote(
                data_ps_ref,
                data_ps_diff=data_ps_diff_ref, data_x=data_x_ref, name_x=name_x, type_x=type_x,
                values_x=values_x, gen_cfg=gen_cfg, pt_cfg=pt_cfg, other_cfg=other_cfg,
                treedepth=pt_cfg.depth, m_i=m_i, with_numba=int_cfg.with_numba, seed=int(m_i**3),
                zero_tol=int_cfg.zero_tol,
                )
                for m_i in range(len(type_x))
                ]
            task_positions = {task_ref: position for position, task_ref in enumerate(still_running)}
            idx = 0
            x_trees = [None] * len(type_x)
            pbar = tqdm(total=len(type_x), desc='Policy tree') if use_tqdm else None

            try:
                while still_running:
                    finished, still_running = ray.wait(still_running, num_returns=1)
                    finished_res = ray.get(finished)
                    for task_ref, ret_all_i in zip(finished, finished_res):
                        if pbar is not None:
                            pbar.update()
                        mcf_gp.progress_clean_memory(output=output_clean, current_idx=idx + 1,
                                                     total=len(type_x),
                                                     )
                        x_trees[task_positions[task_ref]] = ret_all_i
                        idx += 1
            finally:
                if pbar is not None:
                    pbar.close()

        else:
            maxworkers_local = gen_cfg.mp_parallel

            try:
                executor = make_forest_executor(int_cfg=int_cfg,
                                                maxworkers=maxworkers_local,
                                                ray_err_txt=ray_err_txt,
                                                )
            except RuntimeError as exc:
                if int_cfg.mp_backend == 'ray':
                    executor = SequentialExecutor()
                else:
                    raise RuntimeError('Failed to make executor in policy tree estimation.'
                                       ) from exc

            try:
                policy_tree_data_handle_final = executor.put_shared({'data_x': data_x,
                                                                     'data_ps': data_ps,
                                                                     'data_ps_diff': data_ps_diff,
                                                                     },
                                                                    name='policy_tree_data_final',
                                                                    )
                tasks = [TaskSpec(func=tree_search_multip_single_backend_policy_final,
                                  kwargs={'data': policy_tree_data_handle_final,
                                          'name_x': name_x,
                                          'type_x': type_x,
                                          'values_x': values_x,
                                          'gen_cfg': gen_cfg,
                                          'pt_cfg': pt_cfg,
                                          'other_cfg': other_cfg,
                                          'int_cfg': int_cfg,
                                          'm_i': m_i,
                                          },
                                  name=f'policy_tree_final_{m_i}',
                                  )
                         for m_i in range(len(type_x))
                         ]
                batch_size = batch_size_fct(no_batches=int_cfg.mp_batches,
                                            no_workers=gen_cfg.mp_parallel,
                                            no_tasks=len(tasks),
                                            )
                idx = 0
                x_trees = [None for _ in range(len(type_x))]


                task_iter = batched_tasks(tasks, batch_size=batch_size)
                if use_tqdm:
                    task_iter = tqdm(task_iter,
                                     total=ceil(len(tasks) / batch_size),
                                     desc='Policy tree batches',
                                     )
                for task_batch in task_iter:
                    batch_results = executor.map(task_batch)

                    for ret_all_i in batch_results:
                        mcf_gp.progress_clean_memory(output=output_clean,
                                                     current_idx=idx + 1,
                                                     total=len(type_x),
                                                     )
                        x_trees[idx] = ret_all_i
                        idx += 1

            finally:
                executor.shutdown()


        optimal_reward = np.empty(len(type_x))
        for idx, tree in enumerate(x_trees):
            optimal_reward[idx] = tree[1]
        max_i = np.argmax(optimal_reward)
        optimal_reward = optimal_reward[max_i]
        optimal_tree, obs_total = x_trees[max_i][0], x_trees[max_i][2]
    else:
        optimal_tree, optimal_reward, obs_total = tree_search(data_ps,
                                                              data_ps_diff=data_ps_diff,
                                                              data_x=data_x, name_x=name_x,
                                                              type_x=type_x, values_x=values_x,
                                                              pt_cfg=pt_cfg, gen_cfg=gen_cfg,
                                                              other_cfg=other_cfg,
                                                              treedepth=pt_cfg.depth,
                                                              with_numba=int_cfg.with_numba,
                                                              seed=seed, zero_tol=int_cfg.zero_tol,
                                                              )
    return optimal_tree, optimal_reward, obs_total


def tree_search_multip_single_backend_policy_final(*, data: dict[str, Any],
                                                   name_x: Any,
                                                   type_x: Any,
                                                   values_x: Any,
                                                   gen_cfg: Any,
                                                   pt_cfg: Any,
                                                   other_cfg: Any,
                                                   int_cfg: Any,
                                                   m_i: int,
                                                   ) -> Any:
    """Backend-agnostic wrapper for one policy-tree search task."""
    return tree_search_multip_single(
        data['data_ps'],
        data_ps_diff=data['data_ps_diff'],
        data_x=data['data_x'],
        name_x=name_x,
        type_x=type_x,
        values_x=values_x,
        gen_cfg=gen_cfg,
        pt_cfg=pt_cfg,
        other_cfg=other_cfg,
        treedepth=pt_cfg.depth,
        m_i=m_i,
        with_numba=int_cfg.with_numba,
        seed=int(m_i ** 3),
        zero_tol=int_cfg.zero_tol,
    )



# No longer used, part of old policy tree (therefore no annotations)
def tree_search(data_ps, *,
                data_ps_diff, data_x, name_x, type_x, values_x,
                pt_cfg, gen_cfg, other_cfg, treedepth, no_further_splits=False,
                with_numba=True, seed=12345, zero_tol: float = 1e-10,
                ):
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
    pt_cfg, gen_cfg : Dict's. Parameters.
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
            data_ps, gen_cfg, other_cfg, pt_cfg, with_numba=with_numba)
    else:
        if not no_further_splits and (treedepth < pt_cfg.depth):
            no_further_splits = opt_pt_add.only_1st_tree_fct3(
                data_ps, pt_cfg.cost_of_treat_restrict
                )
        min_leaf_size = pt_cfg.min_leaf_size * 2**(treedepth - 2)
        no_of_x, reward = len(type_x), -inf
        tree = no_by_treat = None
        for m_i in range(no_of_x):
            if gen_cfg.with_output:
                if treedepth == pt_cfg.depth:
                    txt = (f'{name_x[m_i]:20s}  {m_i / no_of_x * 100:4.1f}%'
                           ' of variables completed')
                    mcf_ps.print_mcf(gen_cfg, txt, summary=False)
            match type_x[m_i]:
                case 'cont':
                    values_x_to_check = opt_pt_add.get_values_cont_x(
                        data_x[:, m_i], pt_cfg.no_of_evalupoints,
                        with_numba=with_numba
                        )
                case 'disc':
                    values_x_to_check = values_x[m_i][:]
                case _:
                    if treedepth < pt_cfg.depth:
                        values_x_to_check = opt_pt_add.combinations_categorical(
                            data_x[:, m_i], data_ps_diff,
                            pt_cfg.no_of_evalupoints, with_numba,
                            seed=seed, zero_tol=zero_tol,
                            )
                    else:
                        values_x_to_check = values_x[m_i][:]
            for val_x in values_x_to_check:
                if type_x[m_i] == 'unord':
                    left = np.isin(data_x[:, m_i], val_x)
                else:
                    left = data_x[:, m_i] <= val_x
                obs_left = np.count_nonzero(left)
                if not (min_leaf_size <= obs_left
                        <= (len(left) - min_leaf_size)):
                    continue
                right = np.invert(left)
                tree_l, reward_l, no_by_treat_l = tree_search(data_ps[left, :],
                                                             data_ps_diff=data_ps_diff[left, :],
                                                             data_x=data_x[left, :], name_x=name_x,
                                                             type_x=type_x, values_x=values_x,
                                                             pt_cfg=pt_cfg, gen_cfg=gen_cfg,
                                                             other_cfg=other_cfg,
                                                             treedepth=treedepth - 1,
                                                             no_further_splits=no_further_splits,
                                                             with_numba=with_numba, seed=seed+1,
                                                             zero_tol=zero_tol,
                                                             )
                tree_r, reward_r, no_by_treat_r = tree_search(data_ps[right, :],
                                                              data_ps_diff=data_ps_diff[right, :],
                                                              data_x=data_x[right, :],
                                                              name_x= name_x, type_x=type_x,
                                                              values_x=values_x, pt_cfg=pt_cfg,
                                                              gen_cfg=gen_cfg, other_cfg=other_cfg,
                                                              treedepth=treedepth - 1,
                                                              no_further_splits=no_further_splits,
                                                              with_numba=with_numba, seed=seed+1,
                                                              zero_tol=zero_tol,
                                                              )
                if other_cfg.restricted and pt_cfg.enforce_restriction:
                    reward_l, reward_r = opt_pt_add.adjust_reward(
                        no_by_treat_l=no_by_treat_l, no_by_treat_r=no_by_treat_r, reward_l=reward_l,
                        reward_r=reward_r, with_numba=with_numba,
                        max_by_treat=other_cfg.max_by_treat,
                        )
                if reward_l + reward_r > reward:
                    reward = reward_l + reward_r
                    no_by_treat = no_by_treat_l + no_by_treat_r
                    tree = merge_trees(tree_l=tree_l, tree_r=tree_r, name_x_m=name_x[m_i],
                                       type_x_m=type_x[m_i], val_x=val_x, treedepth=treedepth
                                       )
                if no_further_splits and tree is not None:
                    return tree, reward, no_by_treat

    return tree, reward, no_by_treat


def merge_trees(*, tree_l: list[list],
                tree_r: list[list],
                name_x_m: str,
                type_x_m: str,
                val_x: Real | set[Real],
                treedepth: int,
                _id_gen=count(1),
                ) -> list[list]:
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
    leaf[0], leaf[1] = next(_id_gen), None
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


# No longer used, more efficient code available (no annotations)
# @ray.remote
def _ray_tree_search_multip_single(data_ps, *,
                                   data_ps_diff, data_x,
                                   name_x, type_x, values_x,
                                   gen_cfg, pt_cfg, other_cfg,
                                   treedepth, m_i,
                                   with_numba=True, seed=123456,
                                   zero_tol: float = 1e-10,
                                   ):
    """Prepare function for Ray."""
    return tree_search_multip_single(data_ps,
                                     data_ps_diff=data_ps_diff, data_x=data_x, name_x=name_x,
                                     type_x=type_x, values_x=values_x, gen_cfg=gen_cfg,
                                     pt_cfg=pt_cfg, other_cfg=other_cfg, treedepth=treedepth,
                                     m_i=m_i, with_numba=with_numba, seed=seed, zero_tol=zero_tol,
                                     )

ray_tree_search_multip_single = (ray.remote(_ray_tree_search_multip_single) if ray is not None
                                 else None
                                 )
# No longer used, more efficient code available (no annotations)
def tree_search_multip_single(data_ps, *,
                              data_ps_diff, data_x, name_x, type_x,
                              values_x, gen_cfg, pt_cfg, other_cfg, treedepth,
                              m_i, with_numba=True, seed=12345,
                              zero_tol: float = 1e-10,
                              ):
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
    pt_cfg : PtCfg dataclass. Parameters.
    gen_cfg : gen_cfg dataclass. Parameters.
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
            data_x[:, m_i], pt_cfg.no_of_evalupoints, with_numba=with_numba
            )
    elif type_x[m_i] == 'disc':
        values_x_to_check = values_x[m_i][:]
    else:
        if treedepth < pt_cfg.depth:
            values_x_to_check = opt_pt_add.combinations_categorical(
                data_x[:, m_i], data_ps_diff, pt_cfg.no_of_evalupoints,
                with_numba, seed=seed, zero_tol=zero_tol,
                )
        else:
            values_x_to_check = values_x[m_i][:]
    for val_x in values_x_to_check:
        if type_x[m_i] == 'unord':
            left = np.isin(data_x[:, m_i], val_x)
        else:
            left = data_x[:, m_i] <= val_x
        obs_left = np.count_nonzero(left)
        if not (pt_cfg.min_leaf_size <= obs_left
                <= (len(left)-pt_cfg.min_leaf_size)
                ):
            continue
        right = np.invert(left)
        tree_l, reward_l, no_by_treat_l = tree_search(data_ps[left, :],
                                                      data_ps_diff=data_ps_diff[left, :],
                                                      data_x=data_x[left, :], name_x=name_x,
                                                      type_x=type_x, values_x=values_x,
                                                      pt_cfg=pt_cfg, gen_cfg=gen_cfg,
                                                      other_cfg=other_cfg, treedepth=treedepth-1,
                                                      with_numba=with_numba, seed=seed+1,
                                                      zero_tol=zero_tol,
                                                      )
        tree_r, reward_r, no_by_treat_r = tree_search(data_ps[right, :],
                                                      data_ps_diff=data_ps_diff[right, :],
                                                      data_x=data_x[right, :], name_x=name_x,
                                                      type_x=type_x, values_x=values_x,
                                                      pt_cfg=pt_cfg, gen_cfg=gen_cfg,
                                                      other_cfg=other_cfg, treedepth=treedepth - 1,
                                                      with_numba=with_numba, seed=seed+1,
                                                      zero_tol=zero_tol,
                                                      )
        if other_cfg.restricted and pt_cfg.enforce_restriction:
            reward_l, reward_r = opt_pt_add.adjust_reward(no_by_treat_l=no_by_treat_l,
                                                          no_by_treat_r=no_by_treat_r,
                                                          reward_l=reward_l, reward_r=reward_r,
                                                          with_numba=with_numba,
                                                          max_by_treat=other_cfg.max_by_treat,
                                                          )
        if reward_l + reward_r > reward:
            reward = reward_l + reward_r
            no_by_treat = no_by_treat_l + no_by_treat_r
            tree = merge_trees(tree_l=tree_l, tree_r=tree_r, name_x_m=name_x[m_i],
                               type_x_m=type_x[m_i], val_x=val_x, treedepth=treedepth
                               )
    return tree, reward, no_by_treat


def final_leaf_dict(leaf: list, left_right: str) -> dict:
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
                  'cut-off or set': leaf[7], 'left or right': left_right
                  }
    return return_dic


def get_adjusted_optp(optp_: 'OptimalPolicy',
                      tree_number: int,
                      tree_levels: int,
                      ) -> 'OptimalPolicy':
    """Copy and adjust instance of policy tree."""
    optp_local = deepcopy(optp_)
    match tree_number:
        case 0:
            optp_local.pt_cfg.depth = optp_.pt_cfg.depth_tree_1
            if tree_levels == 2:
                # Adjust minimum leaf size --> speeds up computation
                optp_local.pt_cfg.min_leaf_size = round(
                    optp_.pt_cfg.min_leaf_size * 2**(optp_.pt_cfg.depth_tree_2 - 1)
                    )
        case 1:
            optp_local.pt_cfg.depth = optp_.pt_cfg.depth_tree_2
        case _:
            raise NotImplementedError('Not yet (?) implemented for more than 2 '
                                      'levels of optimal trees.'
                                      )
    return optp_local


def get_data_from_tree(optp_local: 'OptimalPolicy',
                       data_df: DataFrame,
                       tree: list[list]
                       ) -> tuple[list[DataFrame], list[int]]:
    """Get the data for the final leafs as list."""
    leaf_data_df_list, leaf_id_list = [], []
    optp_local.pt_cfg.policy_tree = tree
    optp_local.gen_cfg.with_output = False
    _, idx_in_leaf, leaf_id, _, _ = pred_policy_allocation(optp_local, data_df)
    for idx, id_ in zip(idx_in_leaf, leaf_id):
        for idx_left_right in idx:
            leaf_id_list.append(id_)
            leaf_data_df_list.append(data_df.iloc[idx_left_right['Sorter'].tolist()])

    return leaf_data_df_list, leaf_id_list


def renumber_tree(tree: list[list], first_id: int = 0) -> list[list]:
    """Assign consecutive IDs while preserving all tree connections."""
    old_ids = [node[0] for node in tree]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError('Policy tree contains duplicate node IDs before renumbering.')

    id_map = {old_id: first_id + idx for idx, old_id in enumerate(old_ids)}
    for node in tree:
        node[0] = id_map[node[0]]
        for position in (1, 2, 3):
            if node[position] is not None:
                node[position] = id_map[node[position]]

    return tree


def merge_tree_levels(leaf_id_list: list[int],
                      level_2_dic: dict,
                      ) -> list[list]:
    """Merge second level trees to first level tree."""
    no_final_double_leafs = round(len(leaf_id_list) / 2)
    tree = deepcopy(level_2_dic['level_1_tree'])
    next_id = max(node[0] for node in tree) + 1
    for idx in range(no_final_double_leafs):
        # Find leaf in 1st tree that has to be adapted
        leaf_1_idx = find_leaf_one_idx(tree, leaf_id_list[idx*2])
        # Relevant trees of second round
        tree_2_l = deepcopy(level_2_dic['level_2_tree_list'][idx*2])
        tree_2_r = deepcopy(level_2_dic['level_2_tree_list'][idx*2+1])
        for side, subtree in enumerate((tree_2_l, tree_2_r)):
            if subtree is None:
                continue
            subtree = renumber_tree(subtree, next_id)
            next_id += len(subtree)
            subtree[0][1] = tree[leaf_1_idx][0]
            tree[leaf_1_idx][side + 2] = subtree[0][0]
            tree[leaf_1_idx][4] = 0
            tree.extend(subtree)

    return tree


def find_leaf_one_idx(tree: list[list], leaf_id: int) -> int:
    """Find id of a leaf in another tree."""
    for idx, leaf in enumerate(tree):
        if leaf_id == leaf[0]:
            return idx

    raise ValueError('Leaf id not found.')
