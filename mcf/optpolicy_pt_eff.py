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
from typing import Any, TYPE_CHECKING, Sequence, TypeVar

from numba import njit
import numpy as np
from numpy.typing import NDArray
try:
    import ray
except (ImportError, OSError):
    ray = None
from pandas import DataFrame
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats as mcf_ps
from mcf import optpolicy_pt_add as opt_pt_add
from mcf.mcfoptp_parallel_backend_ray_classical import init_ray_with_fallback, ray_put_all
from mcf.mcfoptp_parallel_backend_forest_executor import forest_executor_context, map_task_batches
from mcf.mcfoptp_parallel_backends_base import TaskSpec

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.optpolicy_init import GenCfg, OtherCfg, PtCfg
T = TypeVar('T')


def optimal_tree_eff_proc(optp_: 'OptimalPolicy',
                          data_df: DataFrame,
                          seed: int = 12345,
                          zero_tol: float = 1e-10,
                          ) -> tuple[list[list], NDArray[Any], int]:
    """Build optimal policy tree."""
    gen_cfg, pt_cfg, int_cfg = optp_.gen_cfg, optp_.pt_cfg, optp_.int_cfg
    other_cfg = optp_.other_cfg
    if gen_cfg.with_output:
        print('\nBuilding optimal policy / decision tree')
    use_tqdm, output_clean = mcf_gp.tqdm_setup(tqdm, gen_cfg.with_output)

    (data_x, data_ps, data_ps_diff, data_opt_treat, name_x, type_x, values_x, values_comp_all
     ) = prepare_data_for_tree_building_eff(optp_, data_df, seed=seed, zero_tol=zero_tol)
    # All values_x are already sorted before this step

    kw_args = {'pt_cfg': pt_cfg, 'gen_cfg': gen_cfg, 'other_cfg': other_cfg,
               'treedepth': pt_cfg.depth, 'with_numba': int_cfg.with_numba,
               'zero_tol': int_cfg.zero_tol, 'name_x': name_x, 'type_x': type_x, 
               'values_x': values_x, 'values_comp_all': values_comp_all,
               }
    if gen_cfg.mp_parallel > 1.5:
        ray_err_txt='Ray does not start up in policy tree estimation.'
        if int_cfg.mp_use_old_ray:
            if ray is None or ray_tree_search_eff_multip_single is None:
                raise ImportError('int_cfg.mp_use_old_ray=True, but ray is not installed. '
                                  'Install ray or use the new backend with '
                                  'mp_use_old_ray=False.'
                                  )
            if not ray.is_initialized():
                init_ray_with_fallback(gen_cfg.mp_parallel, gen_cfg,
                                       ray_err_txt=ray_err_txt,
                                       )
            (data_x_ref, data_ps_ref, data_ps_diff_ref, data_opt_treat_ref
             ) = ray_put_all(data_x, data_ps, data_ps_diff, data_opt_treat)

            if int_cfg.xtr_parallel:
                # Variable and single tree are in single worker
                values_x_xtr_p = prepare_ordered_for_xtr_splits(type_x, values_x)
                still_running = [ray_tree_search_eff_multip_single.remote(
                    data_ps_ref,
                    data_ps_diff=data_ps_diff_ref, data_opt_treat=data_opt_treat_ref,
                    data_x=data_x_ref,
                    **kw_args,
                    m_i=m_i, seed=int(m_i**3), first_split_idx=values_x_xtr_p[m_i][split_mi],
                    ) for m_i in range(len(type_x))
                    for split_mi in range(len(values_x_xtr_p[m_i]))
                    ]
            else:
                # Variable and single tree are in single worker
                still_running = [ray_tree_search_eff_multip_single.remote(
                    data_ps_ref,
                    data_ps_diff=data_ps_diff_ref, data_opt_treat=data_opt_treat_ref,
                    data_x=data_x_ref,
                    **kw_args,
                    m_i=m_i, seed=int(m_i**3),
                    )
                    for m_i in range(len(type_x))]

            total_threads = len(still_running)
            task_positions = {task_ref: position for position, task_ref in enumerate(still_running)}
            idx = 0
            x_trees = [None] * total_threads
            pbar = tqdm(total=total_threads, desc='Policy tree') if use_tqdm else None

            try:
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running, num_returns=1)
                    finished_res = ray.get(finished)
                    for task_ref, ret_all_i in zip(finished, finished_res):
                        if pbar is not None:
                            pbar.update()
                        mcf_gp.progress_clean_memory(output=output_clean, current_idx=idx + 1,
                                                     total=total_threads,
                                                     )
                        x_trees[task_positions[task_ref]] = ret_all_i
                        idx += 1

            finally:
                if pbar is not None:
                    pbar.close()
        else:
            with forest_executor_context(int_cfg=int_cfg, maxworkers=gen_cfg.mp_parallel,
                                         ray_err_txt=ray_err_txt,
                                         fail_txt='Failed to make executor in tree search.',
                                         ) as (executor, maxworkers):
                data_x_handle = executor.put_shared(data_x, name='tree_search_data_x')
                data_ps_handle = executor.put_shared(data_ps, name='tree_search_data_ps')
                data_ps_diff_handle = executor.put_shared(data_ps_diff,
                                                              name='tree_search_data_ps_diff',
                                                              )
                data_opt_treat_handle = executor.put_shared(data_opt_treat,
                                                                name='tree_search_data_opt_treat',
                                                                )
                if int_cfg.xtr_parallel:
                    values_x_xtr_p = prepare_ordered_for_xtr_splits(type_x, values_x)
                    tasks = [TaskSpec(func=tree_search_eff_multip_single_backend_xtr,
                                      kwargs={'data_ps': data_ps_handle,
                                              'data_ps_diff': data_ps_diff_handle,
                                              'data_opt_treat': data_opt_treat_handle,
                                              'data_x': data_x_handle,
                                              'kw_args': kw_args,
                                              'm_i': m_i,
                                              'seed': int(m_i ** 3),
                                              'first_split_idx': values_x_xtr_p[m_i][split_mi],
                                              },
                                      name=f'tree_search_xtr_{m_i}_{split_mi}',
                                      )
                             for m_i in range(len(type_x))
                             for split_mi in range(len(values_x_xtr_p[m_i]))
                             ]
                else:
                    tasks = [TaskSpec(func=tree_search_eff_multip_single_backend_xtr,
                                      kwargs={'data_ps': data_ps_handle,
                                              'data_ps_diff': data_ps_diff_handle,
                                              'data_opt_treat': data_opt_treat_handle,
                                              'data_x': data_x_handle,
                                              'kw_args': kw_args,
                                              'm_i': m_i,
                                              'seed': int(m_i ** 3),
                                              },
                                      name=f'tree_search_xtr_{m_i}',
                                      )
                             for m_i in range(len(type_x))
                             ]
                total_threads = len(tasks)
                idx = 0
                x_trees = []

                result_iter = map_task_batches(executor=executor,
                                               tasks=tasks,
                                               int_cfg=int_cfg,
                                               maxworkers=maxworkers,
                                               min_worker_waves=(4 if int_cfg.mp_backend == 'joblib'
                                                                 else 1
                                                                 ),
                                               )
                if use_tqdm:
                    result_iter = tqdm(result_iter, total=total_threads, desc='Policy tree')

                for ret_all_i in result_iter:
                    mcf_gp.progress_clean_memory(output=output_clean, current_idx=idx + 1,
                                                 total=total_threads,
                                                 )
                    x_trees.append(ret_all_i)
                    idx += 1

        optimal_reward = np.empty(len(x_trees))
        for idx, tree in enumerate(x_trees):
            optimal_reward[idx] = tree[1]
        max_i = np.argmax(optimal_reward)
        optimal_reward = optimal_reward[max_i]
        optimal_tree, obs_total = x_trees[max_i][0], x_trees[max_i][2]
    else:
        (optimal_tree, optimal_reward, obs_total, _,
         ) = tree_search_eff(data_ps,
                             data_ps_diff=data_ps_diff, data_opt_treat=data_opt_treat,
                             data_x=data_x,
                             **kw_args,
                             seed=seed,
                             )

    return optimal_tree, optimal_reward, obs_total


def tree_search_eff_multip_single_backend_xtr(*, data_ps: Any,
                                              data_ps_diff: Any,
                                              data_opt_treat: Any,
                                              data_x: Any,
                                              kw_args: dict[str, Any],
                                              m_i: int,
                                              seed: int,
                                              first_split_idx: Any | None = None,
                                              ) -> Any:
    """Backend-agnostic wrapper for one XTR tree-search task."""
    kwargs = dict(kw_args)
    kwargs['m_i'] = m_i
    kwargs['seed'] = seed
    if first_split_idx is not None:
        kwargs['first_split_idx'] = first_split_idx

    return tree_search_eff_multip_single(data_ps,
                                         data_ps_diff=data_ps_diff, data_opt_treat=data_opt_treat,
                                         data_x=data_x,
                                         **kwargs,
                                         )


def evaluate_terminal_leaves_eff(score_sums: NDArray[Any],
                                 observations: NDArray[Any],
                                 costs: NDArray[Any], *,
                                 other_cfg: 'OtherCfg',
                                 pt_cfg: 'PtCfg',
                                 ) -> tuple[NDArray[Any], NDArray[Any]]:
    """Select the best feasible treatment for every candidate leaf."""
    rewards = score_sums - observations[:, None] * costs

    if other_cfg.restricted and pt_cfg.enforce_restriction:
        max_by_treat = np.asarray(other_cfg.max_by_treat)
        feasible = observations[:, None] <= max_by_treat[None, :]
        any_feasible = np.any(feasible, axis=1)
        rewards = np.where(feasible, rewards, -np.inf)

        if not np.all(any_feasible):
            fallback_treatment = int(np.argmax(max_by_treat))
            no_feasible = ~any_feasible
            rewards[no_feasible, fallback_treatment] = (
                score_sums[no_feasible, fallback_treatment]
                - observations[no_feasible] * costs[fallback_treatment]
                )

    treatments = np.argmax(rewards, axis=1)
    selected_rewards = rewards[np.arange(len(rewards)), treatments]

    return treatments, selected_rewards


@njit
def cumulative_scores_at_splits_numba(data_ps: NDArray[Any],
                                      order: NDArray[Any],
                                      split_positions: NDArray[Any],
                                      ) -> tuple[NDArray[Any], NDArray[Any]]:
    """Accumulate scores in sorted order and retain the required prefix sums."""
    no_splits = len(split_positions)
    no_treatments = data_ps.shape[1]
    scores_left = np.empty((no_splits, no_treatments), dtype=np.float64)
    scores_total = np.zeros(no_treatments, dtype=np.float64)
    split_order = np.argsort(split_positions)
    next_split = 0

    for position in range(len(order)):
        row = order[position]
        for treatment in range(no_treatments):
            scores_total[treatment] += data_ps[row, treatment]

        while next_split < no_splits:
            split_idx = split_order[next_split]
            if split_positions[split_idx] != position + 1:
                break
            for treatment in range(no_treatments):
                scores_left[split_idx, treatment] = scores_total[treatment]
            next_split += 1

    return scores_left, scores_total


def evaluate_terminal_splits_eff(data_ps: NDArray[Any],
                                 data_x: NDArray[Any],
                                 values_to_check: Sequence, *,
                                 unordered: bool,
                                 min_leaf_size: int,
                                 gen_cfg: 'GenCfg',
                                 other_cfg: 'OtherCfg',
                                 pt_cfg: 'PtCfg',
                                 with_numba: bool = True,
                                 ) -> tuple[Any, float, NDArray[Any], int, int] | None:
    """Evaluate all candidate terminal splits simultaneously."""
    obs_total = len(data_x)

    if unordered:
        categorical_values, category_codes = np.unique(data_x, return_inverse=True)
        no_categories = len(categorical_values)
        category_counts = np.bincount(category_codes, minlength=no_categories)

        scores_by_category = np.empty((no_categories, data_ps.shape[1]), dtype=float)
        for treat_idx in range(data_ps.shape[1]):
            scores_by_category[:, treat_idx] = np.bincount(
                category_codes, weights=data_ps[:, treat_idx], minlength=no_categories
                )
        candidate_left = np.zeros((len(values_to_check), no_categories), dtype=bool)
        if len(values_to_check):
            candidate_sizes = np.fromiter((len(value) for value in values_to_check),
                                          dtype=np.intp, count=len(values_to_check),
                                          )
            candidate_rows = np.repeat(np.arange(len(values_to_check)), candidate_sizes)
            selected_codes = np.searchsorted(categorical_values, np.concatenate(values_to_check))
            candidate_left[candidate_rows, selected_codes] = True

        obs_left = candidate_left @ category_counts
        valid = (obs_left >= min_leaf_size) & (obs_total - obs_left >= min_leaf_size)
        if not np.any(valid):
            return None

        values_valid = [values_to_check[idx] for idx in np.flatnonzero(valid)]
        obs_left = obs_left[valid]
        scores_left = candidate_left[valid] @ scores_by_category
        scores_total = scores_by_category.sum(axis=0)

    else:
        values_to_check = np.asarray(values_to_check)
        order = np.argsort(data_x, kind='quicksort')
        data_x_sorted = data_x[order]

        obs_left = np.searchsorted(data_x_sorted, values_to_check, side='right')
        valid = (obs_left >= min_leaf_size) & (obs_total - obs_left >= min_leaf_size)
        if not np.any(valid):
            return None

        values_valid = values_to_check[valid]
        obs_left = obs_left[valid]
        if with_numba and data_ps.dtype in (np.float32, np.float64):
            scores_left, scores_total = cumulative_scores_at_splits_numba(data_ps, order, obs_left)
        else:
            scores_cumsum = np.cumsum(data_ps[order], axis=0, dtype=np.float64)
            scores_left = scores_cumsum[obs_left - 1]
            scores_total = scores_cumsum[-1]

    obs_right = obs_total - obs_left
    scores_right = scores_total - scores_left

    costs = np.asarray(pt_cfg.cost_of_treat_restrict, dtype=float)

    treat_left, reward_left = evaluate_terminal_leaves_eff(scores_left, obs_left, costs,
                                                           other_cfg=other_cfg, pt_cfg=pt_cfg,
                                                           )
    treat_right, reward_right = evaluate_terminal_leaves_eff(scores_right, obs_right, costs,
                                                             other_cfg=other_cfg, pt_cfg=pt_cfg,
                                                             )
    if other_cfg.restricted and pt_cfg.enforce_restriction:
        no_by_treat = np.zeros((len(values_valid), gen_cfg.no_of_treat), dtype=np.int64)
        rows = np.arange(len(values_valid))
        no_by_treat[rows, treat_left] += obs_left
        no_by_treat[rows, treat_right] += obs_right

        max_by_treat = np.asarray(other_cfg.max_by_treat)
        violations = no_by_treat > max_by_treat
        relative_excess = np.zeros_like(no_by_treat, dtype=float)
        positive_capacity = violations & (max_by_treat > 0)

        np.divide(no_by_treat - max_by_treat,
                  max_by_treat,
                  out=relative_excess,
                  where=positive_capacity,
                  )
        penalty = np.max(relative_excess, axis=1)
        zero_capacity_violation = np.any(violations & (max_by_treat <= 0), axis=1)
        penalty[zero_capacity_violation] = 1
        penalty = np.minimum(penalty, 1)

        reward_left -= penalty * np.abs(reward_left)
        reward_right -= penalty * np.abs(reward_right)

    rewards = reward_left + reward_right
    best_idx = int(np.argmax(rewards))

    if other_cfg.restricted and pt_cfg.enforce_restriction:
        best_no_by_treat = no_by_treat[best_idx]
    else:
        best_no_by_treat = np.zeros(gen_cfg.no_of_treat, dtype=np.int64)
        best_no_by_treat[treat_left[best_idx]] += obs_left[best_idx]
        best_no_by_treat[treat_right[best_idx]] += obs_right[best_idx]

    return (values_valid[best_idx],
            float(rewards[best_idx]),
            best_no_by_treat,
            int(treat_left[best_idx]),
            int(treat_right[best_idx]),
            )


@njit
def no_variation_float_numba(data_x: NDArray[Any],
                             zero_tol: float,
                             ) -> NDArray[np.bool_]:
    """Check feature ranges, stopping as soon as sufficient variation is found."""
    no_obs, no_features = data_x.shape
    if no_obs == 0:
        raise ValueError('Cannot check variation without observations.')
    no_variation = np.empty(no_features, dtype=np.bool_)

    for feature in range(no_features):
        minimum = maximum = data_x[0, feature]
        for observation in range(1, no_obs):
            value = data_x[observation, feature]
            minimum = np.minimum(minimum, value)
            maximum = np.maximum(maximum, value)
            if maximum - minimum >= zero_tol:
                break
        no_variation[feature] = maximum - minimum < zero_tol

    return no_variation


def tree_search_eff(data_ps: NDArray[Any], *,
                    data_ps_diff: NDArray[Any],
                    data_opt_treat: NDArray[Any],
                    data_x: NDArray[Any],
                    name_x: list[str],
                    type_x: list[str],
                    values_x:  list[set],
                    values_comp_all: list[list],
                    pt_cfg: 'PtCfg',
                    gen_cfg: 'GenCfg',
                    other_cfg: 'OtherCfg',
                    treedepth: int,
                    no_further_splits: bool = False,
                    with_numba: bool = True,
                    seed: int = 12345,
                    zero_tol: float=1e-10,
                    ) -> tuple[list[list], float, list[int], list[list]]:
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
    pt_cfg, gen_cfg : DCs, Dict's. Parameters.
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
            data_ps, gen_cfg, other_cfg, pt_cfg, with_numba=with_numba)
    else:
        if not no_further_splits and treedepth < pt_cfg.depth:
            no_further_splits = no_further_gain_split_funct(data_opt_treat)

        min_leaf_size = pt_cfg.min_leaf_size * 2**(treedepth - 2)
        no_of_x, reward = len(type_x), -inf
        tree = no_by_treat = None

        # Check if there is any gain by additional splits. If not, choose
        # arbitray split point --> implemented such that function
        # get_val_to_check_eff will return only single value
        # For very shallow trees that are subject to restrictions, this is not
        # enforced as it is more likely to lead to violations of the
        # restrictions.
        if treedepth < 4 and (pt_cfg.depth > 2 or not other_cfg.restricted):
            if treedepth < pt_cfg.depth:
                no_gain_splitting = no_further_splits
            else:
                no_gain_splitting = no_further_gain_split_funct(data_opt_treat)
        else:
            no_gain_splitting = False

        # Iterate over the single variables

        if with_numba and data_x.dtype in (np.float32, np.float64):
            comparison_dtype = np.result_type(data_x, zero_tol)
            tolerance = np.asarray(zero_tol, dtype=comparison_dtype).item()
            no_variation = no_variation_float_numba(data_x, tolerance)
        elif np.issubdtype(data_x.dtype, np.integer):
            no_variation = np.min(data_x, axis=0) == np.max(data_x, axis=0)
        else:
            no_variation = np.ptp(data_x, axis=0) < zero_tol

        for m_i in range(no_of_x):
            if no_variation[m_i]:
                continue
            data_x_m_i = data_x[:, m_i]
            type_x_mi = type_x[m_i]

            if gen_cfg.with_output:
                if treedepth == pt_cfg.depth:
                    txt = (f'{name_x[m_i]:20s}  {m_i / no_of_x * 100:4.1f}%'
                           ' of variables completed')
                    mcf_ps.print_mcf(gen_cfg, txt, summary=False)

            values_x_to_check, values_comp_all[m_i] = get_val_to_check_eff(
                type_x[m_i],
                values_x_m_i=values_x[m_i][:], values_comp_all_m_i=values_comp_all[m_i],
                data_x_m_i=data_x_m_i, data_ps_diff=data_ps_diff,
                no_of_evalupoints=pt_cfg.no_of_evalupoints,
                select_values_cat=pt_cfg.select_values_cat, eva_cat_mult=pt_cfg.eva_cat_mult,
                with_numba=with_numba, seed=seed, no_gain_split=no_gain_splitting,
                zero_tol=zero_tol,
                )
            obs_all = data_x_m_i.shape[0]

            if treedepth == 2 and not no_further_splits:
                terminal_result = evaluate_terminal_splits_eff(data_ps,
                                                               data_x_m_i,
                                                               values_x_to_check,
                                                               unordered=type_x_mi == 'unord',
                                                               min_leaf_size=min_leaf_size,
                                                               gen_cfg=gen_cfg,
                                                               other_cfg=other_cfg,
                                                               pt_cfg=pt_cfg,
                                                               with_numba=with_numba,
                                                               )
                if terminal_result is not None:
                    (val_terminal, reward_terminal, no_by_treat_terminal, treat_left, treat_right,
                     ) = terminal_result

                    if reward_terminal > reward:
                        reward = reward_terminal
                        no_by_treat = no_by_treat_terminal
                        tree = merge_trees_eff(tree_l=treat_left,
                                               tree_r=treat_right,
                                               name_x_m=name_x[m_i],
                                               type_x_m=type_x_mi,
                                               val_x=val_terminal,
                                               treedepth=treedepth,
                                               )
                continue

            if type_x_mi == 'unord':
                categorical_values, category_codes = np.unique(data_x_m_i, return_inverse=True)

            for val_x in values_x_to_check:
                if type_x_mi == 'unord':
                    selected_codes = np.searchsorted(
                        categorical_values, np.asarray(val_x)
                        )
                    category_left = np.zeros(
                        len(categorical_values), dtype=bool
                        )
                    category_left[selected_codes] = True
                    left = category_left[category_codes]
                else:
                    left = data_x_m_i <= val_x

                obs_left = left.sum()

                if obs_left < min_leaf_size:
                    continue
                if obs_all - obs_left < min_leaf_size:  # Obs right
                    if type_x_mi == 'unord':
                        continue
                    # Because x are ordered, rights will get only smaller
                    break

                right = ~left
                if treedepth == 2:
                    tree_l, reward_l, no_by_treat_l = opt_pt_add.evaluate_leaf(
                        data_ps[left, :], gen_cfg, other_cfg, pt_cfg, with_numba=with_numba
                        )
                    tree_r, reward_r, no_by_treat_r = opt_pt_add.evaluate_leaf(
                        data_ps[right, :], gen_cfg, other_cfg, pt_cfg, with_numba=with_numba
                        )
                else:
                    (tree_l, reward_l, no_by_treat_l, values_comp_all
                     ) = tree_search_eff(data_ps[left, :],
                                         data_ps_diff=data_ps_diff[left, :],
                                         data_opt_treat=data_opt_treat[left],
                                         data_x=data_x[left, :],
                                         name_x=name_x, type_x=type_x, values_x=values_x,
                                         values_comp_all=values_comp_all, pt_cfg=pt_cfg,
                                         gen_cfg=gen_cfg, other_cfg=other_cfg,
                                         treedepth=treedepth-1,
                                         no_further_splits=no_further_splits,
                                         with_numba=with_numba,
                                         seed=seed+1, zero_tol=zero_tol,
                                         )
                    (tree_r, reward_r, no_by_treat_r, values_comp_all
                     ) = tree_search_eff(data_ps[right, :],
                                         data_ps_diff=data_ps_diff[right, :],
                                         data_opt_treat=data_opt_treat[right],
                                         data_x=data_x[right, :],
                                         name_x=name_x, type_x=type_x, values_x=values_x,
                                         values_comp_all=values_comp_all, pt_cfg=pt_cfg,
                                         gen_cfg=gen_cfg, other_cfg=other_cfg,
                                         treedepth=treedepth-1,
                                         no_further_splits=no_further_splits, with_numba=with_numba,
                                         seed=seed+1, zero_tol=zero_tol,
                                         )
                if other_cfg.restricted and pt_cfg.enforce_restriction:
                    reward_l, reward_r = opt_pt_add.adjust_reward(no_by_treat_l=no_by_treat_l,
                                                                  no_by_treat_r=no_by_treat_r,
                                                                  reward_l=reward_l,
                                                                  reward_r=reward_r,
                                                                  with_numba=with_numba,
                                                                  max_by_treat=
                                                                  other_cfg.max_by_treat
                                                                  )
                if reward_l + reward_r > reward:
                    reward = reward_l + reward_r
                    no_by_treat = no_by_treat_l + no_by_treat_r
                    tree = merge_trees_eff(tree_l=tree_l, tree_r=tree_r, name_x_m=name_x[m_i],
                                           type_x_m=type_x[m_i], val_x=val_x, treedepth=treedepth,
                                           )
                if no_further_splits and tree is not None:
                    return tree, reward, no_by_treat, values_comp_all

    return tree, reward, no_by_treat, values_comp_all


def merge_trees_eff(*, tree_l: list[list],
                    tree_r: list[list],
                    name_x_m: str,
                    type_x_m: str,
                    val_x: float,
                    treedepth: int,
                    _id_gen=count(1),
                    ) -> list[list]:
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
    note_id = next(_id_gen)
    if treedepth == 2:
        leaf = [note_id, None, None, None, 1, name_x_m, type_x_m, val_x, [tree_l, tree_r]]

        return [leaf]

    # Copy so we don’t mutate inputs
    left = [node.copy() for node in tree_l]
    right = [node.copy() for node in tree_r]
    left[0][1] = right[0][1] = note_id

    leaf = [note_id, None, left[0][0], right[0][0], 0, name_x_m, type_x_m, val_x, None]

    return [leaf] + left + right


# @ray.remote
def _ray_tree_search_eff_multip_single(data_ps: NDArray[Any], *,
                                      data_ps_diff: NDArray[Any],
                                      data_opt_treat: NDArray[Any],
                                      data_x: NDArray[Any],
                                      name_x: list[str],
                                      type_x: list[str],
                                      values_x: list[set],
                                      values_comp_all: list[list],
                                      gen_cfg: 'GenCfg',
                                      other_cfg: 'OtherCfg',
                                      pt_cfg: 'PtCfg',
                                      treedepth: int,
                                      m_i: int,
                                      with_numba: bool = True,
                                      seed: int = 123456,
                                      first_split_idx: Real | None = None,
                                      zero_tol: float = 1e-10,
                                      ) -> tuple[list[list], float, list[int]]:
    """Prepare function for Ray."""
    return tree_search_eff_multip_single(data_ps,
                                         data_ps_diff=data_ps_diff, data_opt_treat=data_opt_treat,
                                         data_x=data_x, name_x=name_x, type_x=type_x,
                                         values_x=values_x, values_comp_all=values_comp_all,
                                         gen_cfg=gen_cfg, other_cfg=other_cfg, pt_cfg=pt_cfg,
                                         treedepth=treedepth, m_i=m_i, with_numba=with_numba,
                                         seed=seed, first_split_idx=first_split_idx,
                                         zero_tol=zero_tol,
                                         )


ray_tree_search_eff_multip_single = (ray.remote(_ray_tree_search_eff_multip_single)
                                     if ray is not None else None
                                     )


def tree_search_eff_multip_single(data_ps: NDArray[Any], *,
                                  data_ps_diff: NDArray[Any],
                                  data_opt_treat: NDArray[Any],
                                  data_x: NDArray[Any],
                                  name_x: list[str],
                                  type_x: list[str],
                                  values_x: list[set],
                                  values_comp_all: list[list],
                                  gen_cfg: 'GenCfg',
                                  other_cfg: 'OtherCfg',
                                  pt_cfg: 'PtCfg',
                                  treedepth: int,
                                  m_i: int,
                                  with_numba: bool = True,
                                  seed: int = 123456,
                                  first_split_idx: Real | None = None,
                                  zero_tol: float = 1e-10,
                                  ) -> tuple[list[list], float, list[int]]:
    """Build tree. Only first level. For multiprocessing only."""
    assert treedepth != 1, 'This should not happen in Multiprocessing.'
    reward, tree, no_by_treat = -inf, None, None

    min_leaf_size = pt_cfg.min_leaf_size * 2**(treedepth - 2)
    x_col = data_x[:, m_i]

    values_x_to_check, values_comp_all[m_i] = get_val_to_check_eff(
        type_x[m_i],
        values_x_m_i=values_x[m_i][:], values_comp_all_m_i=values_comp_all[m_i], data_x_m_i=x_col,
        data_ps_diff=data_ps_diff, no_of_evalupoints=pt_cfg.no_of_evalupoints,
        select_values_cat=pt_cfg.select_values_cat, eva_cat_mult=pt_cfg.eva_cat_mult,
        with_numba=with_numba,seed=seed, first_split_idx=first_split_idx, zero_tol=zero_tol,
        )
    type_x_m_i, name_x_m_i = type_x[m_i], name_x[m_i]

    obs_all = x_col.shape[0]

    if treedepth == 2:
        terminal_result = evaluate_terminal_splits_eff(data_ps,
                                                       x_col,
                                                       values_x_to_check,
                                                       unordered=type_x_m_i == 'unord',
                                                       min_leaf_size=min_leaf_size,
                                                       gen_cfg=gen_cfg,
                                                       other_cfg=other_cfg,
                                                       pt_cfg=pt_cfg,
                                                       with_numba=with_numba,
                                                       )
        if terminal_result is None:
            return None, -inf, None

        val_x, reward, no_by_treat, treat_left, treat_right = terminal_result
        tree = merge_trees_eff(tree_l=treat_left, tree_r=treat_right, name_x_m=name_x_m_i,
                               type_x_m=type_x_m_i, val_x=val_x, treedepth=treedepth,
                               )
        return tree, reward, no_by_treat

    if type_x_m_i == 'unord':
        categorical_values, category_codes = np.unique(x_col, return_inverse=True)

    for val_x in values_x_to_check:
        if type_x_m_i == 'unord':
            selected_codes = np.searchsorted(categorical_values, np.asarray(val_x))
            category_left = np.zeros(len(categorical_values), dtype=bool)
            category_left[selected_codes] = True
            left = category_left[category_codes]
        else:
            left = x_col <= val_x

        # Fast check of stopping criteria: Leaf size too small
        obs_left = left.sum()
        obs_right = obs_all - obs_left
        if obs_left < min_leaf_size or obs_right < min_leaf_size:
            continue

        right = ~left
        if treedepth == 2:
            tree_l, reward_l, no_by_treat_l = opt_pt_add.evaluate_leaf(
                data_ps[left, :], gen_cfg, other_cfg, pt_cfg, with_numba=with_numba
                )
            tree_r, reward_r, no_by_treat_r = opt_pt_add.evaluate_leaf(
                data_ps[right, :], gen_cfg, other_cfg, pt_cfg, with_numba=with_numba
                )
        else:
            tree_l, reward_l, no_by_treat_l, values_comp_all = tree_search_eff(
                data_ps[left, :],
                data_ps_diff=data_ps_diff[left, :], data_opt_treat=data_opt_treat[left],
                data_x=data_x[left, :], name_x=name_x, type_x=type_x, values_x=values_x,
                values_comp_all=values_comp_all, pt_cfg=pt_cfg, gen_cfg=gen_cfg,
                other_cfg=other_cfg,
                treedepth=treedepth-1, with_numba=with_numba, seed=seed+1, zero_tol=zero_tol,
                )
            tree_r, reward_r, no_by_treat_r, values_comp_all = tree_search_eff(
                data_ps[right, :],
                data_ps_diff=data_ps_diff[right, :], data_opt_treat=data_opt_treat[right],
                data_x=data_x[right, :], name_x=name_x, type_x=type_x, values_x=values_x,
                values_comp_all=values_comp_all, pt_cfg=pt_cfg, gen_cfg=gen_cfg,
                other_cfg=other_cfg,
                treedepth=treedepth-1, with_numba=with_numba, seed=seed+1, zero_tol=zero_tol,
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
            tree = merge_trees_eff(tree_l=tree_l, tree_r=tree_r, name_x_m=name_x_m_i,
                                   type_x_m=type_x_m_i, val_x=val_x, treedepth=treedepth
                                   )
    return tree, reward, no_by_treat


def get_val_to_check_eff(type_x_m_i: str, *,
                         values_x_m_i: Real,
                         values_comp_all_m_i: list[tuple],
                         data_x_m_i: NDArray[Any],
                         data_ps_diff: NDArray[Any],
                         no_of_evalupoints: int,
                         select_values_cat: bool = False,
                         eva_cat_mult: int = 1,
                         with_numba: bool = True,
                         seed: int = 1234,
                         first_split_idx: Real | None = None,
                         no_gain_split: bool = False,
                         zero_tol: float = 1e-10,
                         ) -> tuple[list, list]:
    """Get the values to check for next splits of leaf."""
    match type_x_m_i:
        case 'cont' | 'disc':
            if no_gain_split:
                values, counts = np.unique(data_x_m_i, return_counts=True)
                obs_left = np.cumsum(counts)
                split_idx = np.argmin(np.abs(2 * obs_left - data_x_m_i.size))
                values_x_to_check = (values[split_idx],)
            else:
                values_x_to_check_tmp = get_values_cont_ord_x_eff(data_x_m_i, values_x_m_i)
                if first_split_idx is not None and len(values_x_to_check_tmp) > 2:
                    first_split_idx_set = (first_split_idx if isinstance(first_split_idx, set)
                                           else set(first_split_idx)
                                           )
                    values_x_to_check = [val for val in values_x_to_check_tmp
                                         if val in first_split_idx_set
                                         ]
                    if not values_x_to_check:
                        values_x_to_check = values_x_to_check_tmp
                else:
                    values_x_to_check = values_x_to_check_tmp
        case 'unord':
            # Take pre-computed values of the splitting points that fall into the range of the data
            (values_x_to_check, values_comp_all_m_i) = combinations_categorical_eff(
                data_x_m_i, data_ps_diff, values_comp_all_m_i,
                no_of_evalupoints,
                select_values=select_values_cat,
                factor=eva_cat_mult,
                with_numba=with_numba,
                seed=seed,
                zero_tol=zero_tol,
                )
        case _:
            raise ValueError('Wrong data type')

    return values_x_to_check, values_comp_all_m_i


def combinations_categorical_eff(single_x_np: NDArray[Any],
                                 ps_np_diff: NDArray[Any],
                                 values_comp_all: list[tuple] | None,
                                 no_of_evalupoints: int, *,
                                 select_values: bool = False,
                                 factor: Real = 1,
                                 with_numba: bool = True,
                                 seed: int = 123456,
                                 zero_tol: float = 1e-10,
                                 ) -> tuple[list, list]:
    """Create all possible combinations of list elements, w/o complements."""
    values = np.unique(single_x_np)
    no_of_values = len(values)
    no_eva_point = max(1, int(no_of_evalupoints * factor))

    if values_comp_all is not None:
        for hist in values_comp_all:
            if len(hist[0]) == len(values) and np.array_equal(hist[0], values):
                return hist[1], values_comp_all  # No need to compute new

    if with_numba and no_of_values < 64:
        no_of_combinations = total_sample_splits_categorical_eff_numba(no_of_values)
    else:
        no_of_combinations = total_sample_splits_categorical_eff(no_of_values)

    exhaustive = no_of_combinations <= no_eva_point

    if exhaustive and values_comp_all is not None:
        for hist in values_comp_all:
            if len(hist[0]) == len(values) and np.array_equal(hist[0], values):
                return hist[1], values_comp_all  # No need to compute new

    if exhaustive:
        combinations_new = all_combinations_no_complements_eff(values)
    else:
        if select_values:
            no_of_evalupoints_new = round(find_evapoints_cat(
                len(values), no_eva_point, with_numba=with_numba))

            rng = np.random.default_rng(seed=seed)
            indx = rng.choice(len(values), size=no_of_evalupoints_new, replace=False)

            combinations_new = all_combinations_no_complements_eff(values[indx])
            if len(combinations_new) > no_eva_point:
                combination_idx = rng.choice(len(combinations_new), size=no_eva_point, replace=False
                                             )
                combinations_new = [combinations_new[idx] for idx in combination_idx]
        else:
            # Sort values according to policy score differences
            values_sorted, no_of_ps = opt_pt_add.get_values_ordered(
                single_x_np, ps_np_diff, values, no_of_values,
                with_numba=with_numba, zero_tol=zero_tol,
                )
            combinations_t = sorted_values_into_combinations_eff(values_sorted, no_of_ps,
                                                                 no_of_values,
                                                                 )
            combinations_ = drop_complements_eff(combinations_t, values,sublist=False)
            len_c = len(combinations_)

            if len_c > no_eva_point:
                if no_eva_point == 1:
                    indx = (len_c // 2,)
                else:
                    rng = np.random.default_rng(seed=seed)
                    indx = rng.choice(len_c, size=no_eva_point, replace=False)
                combinations_new = [combinations_[i] for i in indx]
            elif len_c < no_eva_point:
                # Fill with some random combinations previously omitted.
                # This case can happen because of the ordering used above.
                combinations_new = add_combis(combinations_, values, no_eva_point, seed=seed)
            else:
                combinations_new = combinations_
    if values_comp_all is None:
        values_comp_all = []
    if exhaustive:
        values_comp_all.append((values, combinations_new))

    return combinations_new, values_comp_all


@lru_cache(maxsize=32)
def random_category_subsets(no_values: int,
                            target_size: int,
                            seed: int,
                            ) -> tuple[tuple[int, ...], ...]:
    """Cache distinct random subsets of category positions, excluding the last."""
    rng = np.random.default_rng(seed=seed)
    subsets = []
    seen = set()

    while len(subsets) < target_size:
        selected = rng.integers(0, 2, size=no_values - 1, dtype=np.int8).astype(bool)
        positions = tuple(np.flatnonzero(selected).tolist())
        if not positions or positions in seen:
            continue
        subsets.append(positions)
        seen.add(positions)

    return tuple(subsets)


def add_combis(combinations_: list[tuple],
               values: NDArray[Any],
               target_size: int, *,
               seed: int,
               ) -> list[tuple]:
    """Add randomly selected combinations not represented so far."""
    values = np.asarray(values)
    values_set = frozenset(values.tolist())
    last_value = values[-1]

    def canonical(combination: tuple) -> frozenset:
        """Represent complementary splits by the same set."""
        combination_set = frozenset(combination)
        if last_value in combination_set:
            combination_set = values_set - combination_set

        return combination_set

    combinations_new = []
    combinations_set = set()
    for combination in combinations_:
        combination_key = canonical(combination)
        if combination_key not in combinations_set:
            combinations_new.append(tuple(combination))
            combinations_set.add(combination_key)

    if len(combinations_new) >= target_size:
        return combinations_new

    for positions in random_category_subsets(len(values), target_size, seed):
        combination = tuple(values[position] for position in positions)
        combination_key = frozenset(combination)
        if combination_key in combinations_set:
            continue

        combinations_new.append(combination)
        combinations_set.add(combination_key)
        if len(combinations_new) >= target_size:
            break

    return combinations_new


@lru_cache(maxsize=None)
def find_evapoints_cat(no_values: int, no_of_evalupoints: int, with_numba: bool = True) -> int:
    """Find number of categories that find to no_of_evaluation points."""
    if no_values < 6:
        return no_values

    for vals in range(6, no_values):
        if with_numba and vals < 20:
            no_of_combinations = total_sample_splits_categorical_eff(vals)
        else:
            no_of_combinations = opt_pt_add.total_sample_splits_categorical(vals)
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


@njit
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


def all_combinations_no_complements_eff(values: Sequence[T]) -> list[tuple[T, ...]]:
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


def drop_complements_eff(list_all: list[tuple], values: list[Real], sublist: bool = True) -> list:
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
def sorted_values_into_combinations_eff(values_sorted: NDArray[Any],
                                        no_of_ps: int,
                                        no_of_values: int,
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
    all_combs = [tuple(values_sorted[:i, j])
                 for j in range(no_of_ps) for i in range(1, no_of_values)
                 ]
    unique_combinations = set(all_combs)

    return list(unique_combinations)


def prepare_data_for_tree_building_eff(optp_: 'OptimalPolicy',
                                       data_df: DataFrame, *,
                                       seed: int = 123456,
                                       zero_tol: float = 1e-10,
                                       ) -> tuple[NDArray[Any], NDArray[Any],
                                                  list, list, list,  list, list,
                                                  ]:
    """Prepare data for tree building."""
    x_type, x_values = optp_.var_x_type, optp_.var_x_values

    data_ps = data_df[optp_.var_cfg.polscore_name].to_numpy(copy=False)
    data_opt_treat = np.argmax(data_ps - np.asarray(optp_.pt_cfg.cost_of_treat_restrict), axis=1
                               ).astype(np.int32, copy=False)

    if optp_.gen_cfg.x_unord_flag:
        data_ps_diff = data_ps[:, 1:] - data_ps[:, 0, np.newaxis]
    else:
        data_ps_diff = data_ps[:, :0]

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
            start = obs / optp_.pt_cfg.no_of_evalupoints / 2
            stop = obs - 1 - start
            indi = np.linspace(start, stop, num=optp_.pt_cfg.no_of_evalupoints, dtype=np.int32)
            values_x[j] = vals[indi]
        else:
            values_x[j] = vals.copy()
    data_x = data_df[name_x].to_numpy(copy=optp_.gen_cfg.x_unord_flag)
    del data_df
    if optp_.gen_cfg.x_unord_flag:
        for m_i in range(no_of_x):
            if type_x[m_i] == 'unord':
                data_x[:, m_i] = np.round(data_x[:, m_i])
                (values_x[m_i], values_comp_all[m_i],
                 ) = combinations_categorical_eff(data_x[:, m_i], data_ps_diff, None,
                                                  optp_.pt_cfg.no_of_evalupoints,
                                                  select_values=optp_.pt_cfg.select_values_cat,
                                                  factor=optp_.pt_cfg.eva_cat_mult,
                                                  with_numba=optp_.int_cfg.with_numba, seed=seed,
                                                  zero_tol=zero_tol,
                                                  )
    return data_x, data_ps, data_ps_diff, data_opt_treat, name_x, type_x, values_x, values_comp_all


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
                values_x_xtr_p[idx] = np.array_split(values_to_split, number_of_splits)
            else:
                values_x_xtr_p[idx] = [values_to_split]    # Make it a list

    return values_x_xtr_p


# Optimized version
def get_values_cont_ord_x_eff(data_vector: NDArray[Any], x_values: NDArray[Any]) -> NDArray[Any]:
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
    if no_vals == 0:
        return np.array([min_x + 0.5 * (max_x - min_x)])

    if len_data_vector < (no_vals / 2):
        return np.unique(data_vector)

    return split_x_values


def no_further_gain_split_funct(opt_treat: NDArray[Any]) -> bool:
    """Define if additional leaf splitting can lead to improvements."""
    no_further_gain_split = np.all(opt_treat == opt_treat[0])

    return no_further_gain_split
