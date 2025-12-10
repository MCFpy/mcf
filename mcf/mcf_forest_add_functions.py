"""
Contains functions for building the forest.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from typing import Any, TYPE_CHECKING

from numba import njit
import numpy as np
from numpy.typing import NDArray
import ray
from pandas import DataFrame

from mcf import mcf_forest_data_functions as mcf_data
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as mcf_ps

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def rnd_variable_for_split(x_ind_pos: NDArray[np.int64],
                           x_ai_ind_pos: NDArray[np.int64] | list,
                           cf_cfg: Any,
                           mmm: int,
                           rng: np.random.Generator,
                           all_vars: bool = False,
                           first_attempt: bool = True
                           ) -> list[int]:
    """Find variables used for splitting.

    Parameters
    ----------
    x_ind_pos : NDArray. Indices of all x-variables.
    x_ai_ind_pos : Numpy 1D array or empty list.
         Indices of all x-variables always used for splitting.
    c_dict : Dict. Parameters
    mmm : Number of variables to draw.
    rng : default random number generator.
    all_vars : Boolean, optional.
         Use all variables for splitting. Default is False.
    first_attempt: Boolean, optional.
         Default is True.

    Returns
    -------
    x_i_for_split : List of indices in x of splitting variables.

    """
    if all_vars:
        # return x_ind_pos.tolist()
        return x_ind_pos

    if cf_cfg.m_random_poisson and mmm > cf_cfg.m_random_poisson_min:
        m_l = 1 + int(rng.poisson(mmm - 1))
        if m_l > x_ind_pos.size:
            m_l = int(x_ind_pos.size)
    else:
        m_l = int(mmm)

    if len(x_ai_ind_pos) == 0 or not first_attempt:
        if m_l < len(x_ind_pos):  # Until version 0.7.2 '>' which was wrong
            x_i_for_split = rng.choice(x_ind_pos, m_l, replace=False)
        else:
            x_i_for_split = x_ind_pos.copy()

        return x_i_for_split.tolist()

    if m_l > len(x_ai_ind_pos):
        pool = x_ind_pos[~np.isin(x_ind_pos, x_ai_ind_pos)]
        x_i_for_split = rng.choice(pool,
                                   m_l - len(x_ai_ind_pos),
                                   replace=False
                                   )
        x_i_for_split = np.concatenate((x_i_for_split, x_ai_ind_pos))

        return x_i_for_split.tolist()

    return np.asarray(x_ai_ind_pos, dtype=np.intp).tolist()


def match_cont(d_grid: NDArray[Any],
               y_nn: NDArray[np.floating],
               grid_values: NDArray[Any],
               rng: np.random.Generator
               ) -> NDArray[Any]:
    """
    Select suitable match in case of continuous treatment.

    Parameters
    ----------
    d_grid : Numpy array.
        Discretised treatment.
    y_nn : Numpy array.
        Neighbours.
    grid_values : Numpy array.
        Values (midpoints) used to generate discretised treatment
    rng : Default random number generator object

    Returns
    -------
    y_nn_cont: N x 2 Numpy array.
        Selected neighbours to the left.
    """
    grid = grid_values[1:]    # Controls are not relevant
    min_d_grid, max_d_grid = np.min(d_grid), np.max(d_grid)
    col_no_min = np.argmin(np.abs(grid-min_d_grid))
    col_no_max = np.argmin(np.abs(grid-max_d_grid))
    indices = np.arange(col_no_min, col_no_max + 1)
    nn_indices = rng.choice(indices, size=d_grid.shape[0])
    y_nn_sel = select_one_row_element(y_nn[:, 1:], nn_indices)
    y_nn_red = np.concatenate((y_nn[:, 0].reshape(-1, 1), y_nn_sel), axis=1)

    return y_nn_red


@njit
def select_one_row_element(data: NDArray[np.floating | np.integer],
                           indices: list[int] | NDArray[np.integer]
                           ) -> NDArray[NDArray[np.floating | np.integer]]:
    """Randomly find one element per row."""
    data_selected = np.empty((data.shape[0], 1))
    for idx, val in enumerate(indices):
        data_selected[idx, 0] = data[idx, val]

    return data_selected


def describe_forest(forest: list[list[Any]],
                    m_n_min_ar: NDArray[Any],
                    var_cfg: Any,
                    cf_cfg: Any,
                    gen_cfg: Any,
                    pen_mult: float | int = 0,
                    summary: bool = True
                    ):
    """Describe estimated forest by collecting information in trees."""
    txt = ('\n' + '-' * 100 + '\nParameters of estimation to build random'
           ' forest')
    txt += '\nOutcome variable used to build forest: '
    txt += ' '.join(var_cfg.y_tree_name)
    txt += '\nFeatures used to build forest:          '
    txt += ' '.join(var_cfg.x_name)
    txt += '\nVariables always included in splitting: '
    txt += ' '.join(var_cfg.x_name_always_in)
    txt += (f'\nNumber of replications:     {cf_cfg.boot:<4}')

    match cf_cfg.mtot:
        case 1: splitting_rule = 'MSE+MCE criterion'
        case 3: splitting_rule = 'MSEs of regressions only considered'
        case 2: splitting_rule = '-Var(effect)'
        case 4: splitting_rule = 'Random switching'
        case _: splitting_rule = 'No splitting rule specified.'

    txt += f'\nSplitting rule used:        {splitting_rule:<4}'
    if cf_cfg.p_diff_penalty > 0:
        txt += f'\nPenalty multiplier used in splitting:  {pen_mult}'
        penalty_method = ('MSE of treatment prediction '
                          if cf_cfg.penalty_type == 'mse_d'
                          else 'Difference penalty (Lechner, 2018)')
        txt += f' Penalty method: {penalty_method}'
    txt += '\nShare of data in subsample for forest building:'
    txt += f' {cf_cfg.subsample_share_forest:<4}'
    txt += '\nShare of data in subsample for forest evaluation:'
    txt += f' {cf_cfg.subsample_share_eval:<4}'
    txt += '\nTotal number of variables available for splitting:'
    txt += f' {len(var_cfg.x_name):<4}'
    txt += f'\n# of variables (M) used for split: {m_n_min_ar[0]:<4}'
    if cf_cfg.m_random_poisson:
        txt += '\n           (# of variables drawn from 1+Poisson(M-1))'
        txt += '\nMinimum threshold for using Poisson: '
        txt += f'{cf_cfg.m_random_poisson_min}'
    txt += f'\nMinimum leaf size:                 {m_n_min_ar[1]:<4}'
    txt += f'\nAlpha regularity:                  {m_n_min_ar[2]:5.3f}'
    txt += '\n------------------- Estimated trees ----------------------------'
    leaf_info = get_tree_infos(forest)
    txt += f'\nAverage # of leaves:                      {leaf_info[0]:4.1f}'
    txt += f'\nAverage size of leaves:                   {leaf_info[1]:4.1f}'
    txt += f'\nMedian size of leaves:                    {leaf_info[2]:4.1f}'
    txt += f'\nMin size of leaves:                       {leaf_info[3]:4.0f}'
    txt += f'\nMax size of leaves:                       {leaf_info[4]:4.0f}'
    txt += f'\nAverage # of obs in leaves (single tree): {leaf_info[5]:4.0f}'
    txt += '\n' + '-' * 100
    mcf_ps.print_mcf(gen_cfg, txt, summary=summary)
    report = {}
    report["n_min"] = m_n_min_ar[1]
    report["m_split"] = m_n_min_ar[0]
    report["f_share_build"] = cf_cfg.subsample_share_forest
    report["f_share_fill"] = cf_cfg.subsample_share_eval
    report["alpha"] = m_n_min_ar[2]
    report["y_name_tree"] = var_cfg.y_tree_name[0]
    report["Features"] = ' '.join(var_cfg.x_name)
    report["leaves_mean"] = leaf_info[0]
    report["obs_leaf_mean"] = leaf_info[1]
    report["obs_leaf_med"] = leaf_info[2]
    report["obs_leaf_min"] = leaf_info[3]
    report["obs_leaf_max"] = leaf_info[4]

    return report


def get_tree_infos(forest: list[list[Any]]) -> NDArray[Any]:
    """Obtain some basic information about estimated trees.

    Parameters
    ----------
    forest : List of lists. Collection of node_tables.

    Returns
    -------
    leaf_info : List. Some information about tree.

    """
    leaf_info_tmp = np.zeros([len(forest), 6])
    for boot, tree_dict in enumerate(forest):
        for leaf_info_int in tree_dict['leaf_info_int']:
            if leaf_info_int[7] == 1:   # Terminal leafs only
                leaf_info_tmp[boot, 0] += 1  # Number of leaves
        leaf_info_tree = np.zeros(int(leaf_info_tmp[boot, 0]))
        j = 0
        for leaf_info_int in tree_dict['leaf_info_int']:
            if leaf_info_int[7] == 1:
                leaf_info_tree[j] = leaf_info_int[8]
                j += 1
        leaf_info_tmp[boot, 1] = np.mean(leaf_info_tree)
        leaf_info_tmp[boot, 2] = np.median(leaf_info_tree)
        leaf_info_tmp[boot, 3] = np.min(leaf_info_tree)
        leaf_info_tmp[boot, 4] = np.max(leaf_info_tree)
        leaf_info_tmp[boot, 5] = np.sum(leaf_info_tree)
    leaf_info = np.empty(6)
    list_of_ind = [0, 1, 5]  # Average #, size of leaves, # of obs in leaves
    leaf_info[list_of_ind] = np.mean(leaf_info_tmp[:, list_of_ind], axis=0)
    leaf_info[2] = np.median(leaf_info_tmp[:, 2])   # Med size of leaves
    leaf_info[3] = np.min(leaf_info_tmp[:, 3])      # Min size of leaves
    leaf_info[4] = np.max(leaf_info_tmp[:, 4])      # Max size of leaves

    return leaf_info


def get_terminal_leaf_no(tree_dict: dict,
                         x_dat: NDArray[Any],
                         zero_tol: float = 1e-15,
                         ) -> int:
    """Get the leaf number of the terminal node for single observation.

    Parameters
    ----------
    tree_dict : List of list. Single tree.
    x_dat : Numpy array. Data.

    Returns
    -------
    leaf_no : INT. Number of terminal leaf the observation belongs to.

    Note: This only works if nodes are ordered subsequently. Do not remove
          leafs when pruning. Only changes their activity status.

    """
    leaf_info_int = tree_dict['leaf_info_int']
    cut_off_cont = tree_dict['leaf_info_float'][:, 1]
    cats_prime = tree_dict['cats_prime']
    categorical = any(cat > 0.5 for cat in cats_prime)
    first_leaf_id = 0
    loop = True     # Solve with while loop or directly by recursion; same speed
    if loop:       # recursion via while loop
        if categorical:  # Faster with Numba
            leaf_id = terminal_leaf_loop(leaf_info_int, cut_off_cont,
                                         cats_prime, x_dat, first_leaf_id,
                                         zero_tol=zero_tol,
                                         )
        else:
            leaf_id = terminal_leaf_loop_numba(leaf_info_int, cut_off_cont,
                                               x_dat, first_leaf_id,
                                               zero_tol=zero_tol,
                                               )
    else:  # direct recursion
        if categorical:  # Faster with Numba
            leaf_id = terminal_leaf_recursive(leaf_info_int, cut_off_cont,
                                              cats_prime, x_dat, first_leaf_id,
                                              zero_tol=zero_tol,
                                              )
        else:
            leaf_id = terminal_leaf_recursive_numba(leaf_info_int, cut_off_cont,
                                                    x_dat, first_leaf_id,
                                                    zero_tol=zero_tol,
                                                    )
    return leaf_id


# Optimized version of 4.6.2024
@njit
def terminal_leaf_loop_numba(leaf_info_int: list[Any],
                             cut_off_cont: NDArray[np.floating | np.integer],
                             x_dat: NDArray[np.floating | np.integer],
                             leaf_id: int,
                             zero_tol: float = 1e-15,
                             ) -> int:
    """Get the final leaf number with a loop algorithm."""
    while True:    # Should be equally fast than 'for' loop
        leaf = leaf_info_int[leaf_id]
        if leaf[7] == 1:             # Terminal leaf
            return leaf[0]
        if leaf[7] == 0:
            leaf_id = get_next_leaf_no_numba(leaf,
                                             leaf_id,
                                             x_dat,
                                             cut_off_cont,
                                             zero_tol=zero_tol,
                                             )
        else:
            raise RuntimeError(f'Leaf is still active. {leaf[4]}')

    return leaf_id


def terminal_leaf_loop(leaf_info_int: list[Any],
                       cut_off_cont: NDArray[np.floating | np.integer],
                       cats_prime: list[int],
                       x_dat: NDArray[np.floating | np.integer],
                       leaf_id: int,
                       zero_tol: float = 1e-15,
                       ) -> int:
    """Get the final leaf number with a loop algorithm."""
    while True:    # Should be equally fast than 'for' loop
        leaf = leaf_info_int[leaf_id, :]
        if leaf[7] == 1:             # Terminal leaf
            return leaf[0]
        if leaf[7] == 0:
            leaf_id = get_next_leaf_no(leaf, leaf_id, x_dat, cut_off_cont,
                                       cats_prime, zero_tol=zero_tol,
                                       )
        else:
            raise RuntimeError(f'Leaf is still active. {leaf[4]}')

    return leaf_id


@njit
def terminal_leaf_recursive_numba(leaf_info_int: list[Any],
                                  cut_off_cont: NDArray[np.floating
                                                        | np.integer],
                                  x_dat: NDArray[np.floating | np.integer],
                                  leaf_id: int,
                                  zero_tol: float = 1e-15,
                                  ) -> int:
    """Get the final leaf number with a recursive algorithm."""
    leaf = leaf_info_int[leaf_id]
    # leaf = leaf_info_int[leaf_id, :]   old
    if leaf[7] == 1:             # Terminal leaf, leave recursion
        return leaf[0]

    if leaf[7] == 0:
        # Not final leave, so continue search and update leaf_id
        next_leaf_id = get_next_leaf_no_numba(leaf, leaf_id, x_dat,
                                              cut_off_cont,
                                              zero_tol=zero_tol,
                                              )
    else:
        raise RuntimeError(f'Leaf is still active. {leaf[4]}')

    return terminal_leaf_recursive_numba(leaf_info_int, cut_off_cont,
                                         x_dat, next_leaf_id,
                                         zero_tol=zero_tol,
                                         )


def terminal_leaf_recursive(leaf_info_int: list[Any],
                            cut_off_cont: NDArray[np.floating | np.integer],
                            cats_prime: list[int],
                            x_dat: NDArray,
                            leaf_id: int,
                            zero_tol: float = 1e-15,
                            ) -> int:
    """Get the final leaf number with a recursive algorithm."""
    leaf = leaf_info_int[leaf_id, :]
    if leaf[7] == 1:             # Terminal leaf, leave recursion
        return leaf[0]

    if leaf[7] == 0:
        # Not final leave, so continue search and update leaf_id
        next_leaf_id = get_next_leaf_no(leaf, leaf_id, x_dat, cut_off_cont,
                                        cats_prime, zero_tol=zero_tol,
                                        )
    else:
        raise RuntimeError(f'Leaf is still active. {leaf[4]}')

    return terminal_leaf_recursive(leaf_info_int, cut_off_cont, cats_prime,
                                   x_dat, next_leaf_id, zero_tol=zero_tol,
                                   )


@njit
def get_next_leaf_no_numba(leaf:  list,
                           leaf_id: int,
                           x_dat: NDArray[np.floating | np.integer],
                           cut_off_cont: NDArray[np.floating | np.integer],
                           zero_tol: float = 1e-15,
                           ) -> int:
    """Get next deeper leaf number for a non-active and non-terminal leaf."""
    if leaf[5] == 0:        # Continuous variable
        condition = (x_dat[leaf[4]] - zero_tol) <= cut_off_cont[leaf_id]
    else:                   # Categorical variable
        raise ValueError("""There should be no categorical variables.""")

    return leaf[2] if condition else leaf[3]


def get_next_leaf_no(leaf:  list,
                     leaf_id: int,
                     x_dat: NDArray[np.floating | np.integer],
                     cut_off_cont: NDArray[np.floating | np.integer],
                     cats_prime: list[int],
                     zero_tol: float = 1e-15,
                     ) -> int:
    """Get next deeper leaf number for a non-active and non-terminal leaf."""
    if leaf[5] == 0:        # Continuous variable
        condition = (x_dat[leaf[4]] - zero_tol) <= cut_off_cont[leaf_id]
    else:                   # Categorical variable
        condition = prime_in_leaf(cats_prime[leaf_id],
                                  int(np.round(x_dat[leaf[4]]))
                                  )
    return leaf[2] if condition else leaf[3]


def prime_in_leaf(cats_prime_leaf_id: int, x_dat_leaf4: int) -> bool:
    """Check if primefactor is in primeproduct."""
    prime_factors = mcf_gp.primes_reverse(cats_prime_leaf_id, False)

    return x_dat_leaf4 in prime_factors


def fill_trees_with_y_indices_mp(mcf_: 'ModifiedCausalForest',
                                 data_df: DataFrame,
                                 forest: list[Any],
                                 ) -> tuple[list[Any], list[Any], NDArray[Any]]:
    """Fill trees with indices of outcomes, MP.

    Returns
    -------
    forest_with_y : List of lists. Updated Node_table.
    terminal_nodes: list of np.arrays. No of final node.
    no_of_avg_nodes: INT. Average no of unfilled leafs.

    """
    int_cfg, gen_cfg, cf_cfg = mcf_.int_cfg, mcf_.gen_cfg, mcf_.cf_cfg
    if gen_cfg.with_output and gen_cfg.verbose:
        print("\nFilling trees with indicies of outcomes")
    (x_name, _, _, cf_cfg, _, data_np, _, _, x_i, _, _, d_i, _, _, _
     ) = mcf_data.prepare_data_for_forest(mcf_, data_df, True)
    err_txt = 'Wrong order of variables' + str(x_name) + ': ' + str(
        cf_cfg.x_name_mcf)
    if cf_cfg.x_name_mcf != x_name:
        raise ValueError(err_txt)
    if gen_cfg.d_type == 'continuous':
        d_dat = data_np[:, d_i]
        # substitute those d used for splitting only that have a zero with
        # random element from the positive treatment levels
        d_pos = d_dat[d_dat > int_cfg.zero_tol]
        rng = np.random.default_rng(seed=12366456)
        d_values = rng.choice(d_pos, size=len(d_dat)-len(d_pos), replace=False)
        d_dat_for_x = np.copy(d_dat)
        j = 0
        for i, d_i in enumerate(d_dat):
            if d_i < int_cfg.zero_tol:
                d_dat_for_x[i, 0] = d_values[j]
                j += 1
        x_dat = np.concatenate((data_np[:, x_i], d_dat_for_x), axis=1)
    else:
        x_dat = data_np[:, x_i]
        d_dat = np.int32(np.round(data_np[:, d_i]))
    obs = len(x_dat)
    terminal_nodes = [None for _ in range(cf_cfg.boot)]
    nodes_empty = np.zeros(cf_cfg.boot)
    nodes_merged = np.zeros_like(nodes_empty)
    if gen_cfg.mp_parallel < 1.5:
        maxworkers = 1
    else:
        maxworkers = (mcf_sys.find_no_of_workers(gen_cfg.mp_parallel,
                                                 gen_cfg.sys_share,
                                                 zero_tol=int_cfg.zero_tol,
                                                 )
                      if gen_cfg.mp_automatic else gen_cfg.mp_parallel
                      )
    if gen_cfg.with_output and gen_cfg.verbose:
        print('Number of parallel processes (forest): ', maxworkers)

    # Initialise save seeds for forest computation
    child_seeds = np.random.SeedSequence(1233456).spawn(cf_cfg.boot)

    if maxworkers > 1:
        if not ray.is_initialized():
            ray_err_txt = 'Problems starting ray when filling forests.'
            ray_is_running = mcf_sys.init_ray_with_fallback(
                maxworkers, int_cfg, gen_cfg,
                mem_object_store=int_cfg.mem_object_store_2,
                ray_err_txt=ray_err_txt)
        else:
            ray_is_running = True

        if ray_is_running:
            if (gen_cfg.with_output and gen_cfg.verbose
                    and int_cfg.mem_object_store_2 is not None):
                num = round(int_cfg.mem_object_store_2 / (1024 * 1024))
                txt = f'\nSize of Ray Object Store: {num} MB'
                mcf_ps.print_mcf(gen_cfg, txt, summary=False)
            x_dat_ref = ray.put(x_dat)
            d_dat_ref = ray.put(d_dat)
            still_running = [ray_fill_mp.remote(
                forest[idx], obs, d_dat_ref, x_dat_ref, idx, gen_cfg, cf_cfg,
                seedseq=child_seeds[idx], zero_tol=int_cfg.zero_tol,
                )
                for idx in range(cf_cfg.boot)]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running, num_returns=1)
                finished_res = ray.get(finished)

                for ret_all_i in finished_res:
                    iix = ret_all_i[0]
                    forest[iix] = ret_all_i[1]
                    terminal_nodes[iix] = ret_all_i[2]
                    nodes_empty[iix] = ret_all_i[3]
                    nodes_merged[iix] = ret_all_i[4]
                    if gen_cfg.with_output and gen_cfg.verbose:
                        mcf_gp.share_completed(jdx+1, cf_cfg.boot)
                    jdx += 1
            if 'refs' in int_cfg.mp_ray_del:
                del x_dat_ref, d_dat_ref
            if 'rest' in int_cfg.mp_ray_del:
                del finished_res, finished

        else:  # Ray did not start. No multiprocessing.
            maxworkers = 1

    if maxworkers == 1:
        for idx in range(cf_cfg.boot):
            (_, forest[idx], terminal_nodes[idx], nodes_empty[idx],
             nodes_merged[idx]) = fill_mp(forest[idx], obs, d_dat, x_dat,
                                          idx, gen_cfg, cf_cfg,
                                          seedseq=child_seeds[idx],
                                          zero_tol=int_cfg.zero_tol,
                                          )
            if gen_cfg.with_output and gen_cfg.verbose:
                mcf_gp.share_completed(idx+1, cf_cfg.boot)

    no_of_avg_enodes = np.mean(nodes_empty)
    no_of_avg_mnodes = np.mean(nodes_merged)
    if gen_cfg.with_output and gen_cfg.verbose:
        txt = ('\nShare of merged leaves (w/o all treatments per tree): '
               f'{no_of_avg_mnodes:6.3%}')
        txt += ('\nShare of leaves w/o all treatments per tree (after '
                f'merger): {no_of_avg_enodes:6.3%}')
        if no_of_avg_enodes > 0:
            txt += ('\nIncomplete leafs will not be considered for weight'
                    ' computation.')
        txt += '\n' + '-' * 100
        mem = round(mcf_sys.total_size(forest) / (1024 * 1024), 2)
        txt += f'\nSize of forest: {mem} MB' + '\n' + '-' * 100
        mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    return forest, terminal_nodes, no_of_avg_mnodes


@ray.remote
def ray_fill_mp(tree_dict, obs, d_dat, x_dat, b_idx, gen_cfg, cf_cfg,
                seedseq=None, zero_tol=1e-15,
                ):
    """Make it work under Ray."""
    return fill_mp(tree_dict, obs, d_dat, x_dat, b_idx, gen_cfg, cf_cfg,
                   seedseq=seedseq, zero_tol=zero_tol,
                   )


def fill_mp(tree_dict_g: dict,
            obs: int,
            d_dat: NDArray[np.integer | np.floating],
            x_dat: NDArray[np.floating | np.integer],
            b_idx: int,
            gen_cfg: Any,
            cf_cfg: Any,
            seedseq: np.random.SeedSequence | int | None = None,
            zero_tol: float = 1e-15,
            ) -> tuple[int, dict, NDArray[np.integer], float, float]:
    """Compute new node_table and list of final leaves.

    Parameters
    ----------
    tree_dict : dict.
    obs : Int. Sample size.
    d_dat : Numpy array. Treatment.
    x_dat : Numpy array. Features.
    b_idx : Int. Tree number.
    gen_cfg : GenCfg dataclass, Parameters.
    cf_cfg : CfCfg dataclass. Controls.

    Returns
    -------
    ...

    """
    tree_dict = deepcopy(tree_dict_g)  # Necessary to change leaf_info_int

    subsam = cf_cfg.subsample_share_eval < 1
    indices = np.arange(obs)
    if subsam:
        obs = round(obs * cf_cfg.subsample_share_eval)
        if seedseq is None:
            rng = np.random.default_rng(seed=(10+b_idx)**2+121)
        else:
            rng = np.random.default_rng(seed=seedseq)
        indices = rng.choice(indices, size=obs, replace=False)
    obs_in_leaf = make_zeros(obs)
    for i, idx in enumerate(indices):
        obs_in_leaf[i] = get_terminal_leaf_no(tree_dict, x_dat[idx, :],
                                              zero_tol=zero_tol,
                                              )
    unique_leafs = np.unique(obs_in_leaf)
    if subsam:
        # unique_leafs = unique_leafs[1:]  # remove first index: obs not used
        d_dat = d_dat[indices]
    no_of_treat = (2 if gen_cfg.d_type == 'continuous'
                   else gen_cfg.no_of_treat
                   )
    tree_dict['fill_y_empty_leave'] = np.zeros(
        len(tree_dict['fill_y_indices_list']), dtype=np.int32)
    empty_leaves = merged_leaves = 0
    for leaf_id in unique_leafs:
        if tree_dict['leaf_info_int'][leaf_id, 7] != 1:  # Leaf already merged
            continue
        sel_ind = obs_in_leaf.reshape(-1) == leaf_id
        tree_dict['fill_y_indices_list'][leaf_id] = indices[sel_ind]
        if len(np.unique(d_dat[sel_ind])) < no_of_treat:  # Treatment arm empty
            tree_dict, empty_leaves = merge_leaves(
                tree_dict, leaf_id, obs_in_leaf, indices, d_dat, no_of_treat)
            merged_leaves += 1
    empty_share = empty_leaves / len(unique_leafs)
    merge_share = merged_leaves / len(unique_leafs)

    return b_idx, tree_dict, unique_leafs, empty_share, merge_share


def make_zeros(obs: int) -> NDArray[np.integer]:
    """Make zeros in memory-efficient data format."""
    match obs:
        case x if x < 255:        dt = np.uint8
        case x if x < 65535:      dt = np.uint16
        case x if x < 4294967295: dt = np.uint32
        case _:                   dt = np.uint64

    return np.zeros((obs, 1), dtype=dt)


def merge_leaves(tree_dict: dict,
                 leaf_id: int,
                 obs_in_leaf: NDArray[np.integer],
                 indices: NDArray[np.integer],
                 d_dat: NDArray[np.integer | np.floating],
                 no_of_treat: int
                 ) -> tuple[dict, int]:
    """Merge two leaves."""
    PARENT, LEFTDAUGHTER, RIGHTDAUGHTER = 1, 2, 3
    STATUS6, STATUS7 = 6, 7
    leaf_info = tree_dict['leaf_info_int']
    # Get IDs
    parent_id = leaf_info[leaf_id, PARENT]
    left_id = leaf_info[parent_id, LEFTDAUGHTER]
    right_id = leaf_info[parent_id, RIGHTDAUGHTER]
    left_terminal = leaf_info[left_id, STATUS7] == 1
    right_terminal = leaf_info[right_id, STATUS7] == 1
    if left_terminal and right_terminal:    # Leaves will be merged
        # Remove link of parents to daughters
        leaf_info[parent_id, LEFTDAUGHTER] = -1
        leaf_info[parent_id, RIGHTDAUGHTER] = -1
        # Activity status
        leaf_info[parent_id, STATUS6] = 1
        leaf_info[left_id, STATUS6] = 0
        leaf_info[right_id, STATUS6] = 0
        leaf_info[parent_id, STATUS7] = 1
        leaf_info[left_id, STATUS7] = 0
        leaf_info[right_id, STATUS7] = 0

        tree_dict['leaf_info_int'] = leaf_info  # Redundant but not perf. loss
        # Adjust data
        sel_ind_left = obs_in_leaf.reshape(-1) == left_id
        left_ind = indices[sel_ind_left]
        d_dat_left = d_dat[sel_ind_left]
        sel_ind_right = obs_in_leaf.reshape(-1) == right_id
        right_ind = indices[sel_ind_right]
        d_dat_right = d_dat[sel_ind_right]

        all_indices = np.concatenate((left_ind, right_ind))
        all_d_dat = np.concatenate((d_dat_left, d_dat_right))
        tree_dict['fill_y_indices_list'][parent_id] = all_indices
        tree_dict['fill_y_indices_list'][left_id] = None
        tree_dict['fill_y_indices_list'][right_id] = None
        if len(np.unique(all_d_dat)) < no_of_treat:
            tree_dict['fill_y_empty_leave'][leaf_id] = 1   # Leaf ignored
            still_empty = 1
        else:
            still_empty = 0
    else:  # Leaves will not be merged as they might need too deep aggregation
        still_empty = 1

    return tree_dict, still_empty


def save_forests_in_cf_cfg(forest_dic: dict,
                           forest_list: list,
                           fold: int,
                           no_folds: int,
                           reg_round: int,
                           eff_iate: bool
                           ) -> list:
    """Save forests in dictionary as list of list."""
    # Initialise
    if fold == 0 and reg_round:
        innerlist = [None, None] if eff_iate else [None]
        forest_list = [innerlist for idx in range(no_folds)]
    forest_list[fold][0 if reg_round else 1] = deepcopy(forest_dic)

    return forest_list


def train_save_data(mcf_: 'ModifiedCausalForest', data_df: DataFrame,
                    forest: object,
                    prog_dat: NDArray[Any] | None,
                    prop_dat: NDArray[Any] | None,
                    x_ba_dat: NDArray[Any] | None,
                    ) -> dict:
    """Save data needed in the prediction part of mcf."""
    y_train_df = data_df[mcf_.var_cfg.y_name]
    d_train_df = data_df[mcf_.var_cfg.d_name]
    if mcf_.p_cfg.cluster_std:
        cl_train_df = data_df[mcf_.var_cfg.cluster_name]
    else:
        cl_train_df = None
    if mcf_.gen_cfg.weighted:
        w_train_df = data_df[mcf_.var_cfg.w_name]
    else:
        w_train_df = None
    if mcf_.p_cfg.bt_yes:
        x_bala_train_df = data_df[mcf_.var_cfg.x_name_balance_test]
    else:
        x_bala_train_df = None

    forest_dic = {'forest': forest, 'y_train_df': y_train_df,
                  'd_train_df': d_train_df, 'x_bala_df': x_bala_train_df,
                  'cl_train_df': cl_train_df, 'w_train_df': w_train_df,
                  'prog_dat_np': prog_dat, 'prop_dat_np': prop_dat,
                  'x_ba_dat_np': x_ba_dat,
                  }
    return forest_dic


def not_enough_treated(continuous: bool,
                       n_min_treat: int,
                       d_dat: NDArray[Any],
                       no_of_treat: int
                       ) -> bool:
    """If there are NOT enough treated in new leaf."""
    if continuous or n_min_treat == 1:
        return len(np.unique(d_dat)) < no_of_treat

    ret = np.unique(d_dat, return_counts=True)

    return len(ret[0]) < no_of_treat or np.any(ret[1] < n_min_treat)
