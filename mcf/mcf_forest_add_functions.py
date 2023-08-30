"""
Contains functions for building the forest.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from numba import njit

import numpy as np
import ray
# import pandas as pd

from mcf import mcf_forest_data_functions as mcf_data
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps


def rnd_variable_for_split(x_ind_pos, x_ai_ind_pos, cf_dic, mmm, rng):
    """Generate variables to be used for split.

    Parameters
    ----------
    x_ind_pos : List. Indices of all x-variables.
    x_ai_ind : List. Indices of all x-variables always used for splitting.
    c_dict : Dict. Parameters
    mmm : Number of variables to draw.
    rng : default random number generator.

    Returns
    -------
    x_i_for_split : List of indices in x of splitting variables.

    """
    qqq = len(x_ind_pos)
    if cf_dic['m_random_poisson'] and mmm > cf_dic['m_random_poisson_min']:
        m_l = 1 + rng.poisson(lam=mmm-1, size=1)
        if m_l < 1:
            m_l = 1
        elif m_l > qqq:
            m_l = qqq
    else:
        m_l = mmm
    if x_ai_ind_pos == []:
        x_i_for_split = rng.choice(x_ind_pos, m_l, replace=False)
        x_i_for_split_list = x_i_for_split.tolist()
    else:
        if m_l > len(x_ai_ind_pos):
            x_i_for_split = rng.choice(x_ind_pos, m_l-len(x_ai_ind_pos),
                                       replace=False)
            x_i_for_split = np.concatenate((x_i_for_split, x_ai_ind_pos))
            x_i_for_split = np.unique(x_i_for_split)
            x_i_for_split_list = x_i_for_split.tolist()
        else:
            x_i_for_split_list = x_ai_ind_pos[:]
    return x_i_for_split_list


def init_node_table(n_tr, n_oob, indices_oob):
    """Initialise node table for first leaf.

    Parameters
    ----------
    n_tr : INT. Number of observation in training subsample.
    n_oob : INT. Number of observation in OOB subsample.
    indices_oob: Int.

    Returns
    -------
    node_table : List of lists. First init_node_table

    """
    id_node_0 = 0
    id_parent_1 = id_child_left_2 = id_child_right_3 = None
    active_4 = 2
    leaf_size_tr_5, leaf_size_oob_6 = n_tr, n_oob
    objective_fct_value_oob_7 = next_split_i_8 = cut_off_prime_l_9 = None
    x_type_10 = None
    data_tr_indi_11, data_oob_indi_12 = list(range(n_tr)), list(range(n_oob))
    pot_outcomes_13 = pot_variables_used_indi_14 = leaf_size_pot_15 = None
    indices_oob_16 = indices_oob
    node_table = [
        id_node_0, id_parent_1, id_child_left_2, id_child_right_3, active_4,
        leaf_size_tr_5, leaf_size_oob_6, objective_fct_value_oob_7,
        next_split_i_8, cut_off_prime_l_9, x_type_10, data_tr_indi_11,
        data_oob_indi_12, pot_outcomes_13, pot_variables_used_indi_14,
        leaf_size_pot_15, indices_oob_16]
    return [node_table]


def match_cont(d_grid, y_nn, grid_values, rng):
    """
    Select suitable match in case of continuous treatment.

    Parameters
    ----------
    d_grid : Numpy array.
        Discretised treatment.
    y_nn : Numpy array.
        Neighbours.
    leaf_l : Numpy array.
        Observations going to left leaf. (d < larger splitting value).
    leaf_r : Numpy array.
        Observations going to right leaf (d >= larger splitting value).
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
def select_one_row_element(data, indices):
    """Randomly find one element per row."""
    data_selected = np.empty((data.shape[0], 1))
    for idx, val in enumerate(indices):
        data_selected[idx, 0] = data[idx, val]
    return data_selected


def mcf_mse(y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat, treat_values,
            w_yes=False, splitting=False):
    """Compute average mse for the data passed. Based on different methods.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    w_dat : Numpy Nx1 vector. Weights (or 0)
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : List of INT. Treatment values.
    w_yes: Boolean. Weighted estimation.
    splitting: Boolean. Default is False.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.

    """
    if w_yes or mtot in (2, 3):
        mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_not_numba(
            y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
            treat_values, w_yes, splitting)
    else:
        mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_numba(
            y_dat, y_nn, d_dat, n_obs, mtot, no_of_treat,
            np.array(treat_values, dtype=np.int8))
    return mse_mce, treat_shares, no_of_obs_by_treat


def mcf_mse_not_numba(y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
                      treat_values, w_yes, splitting=False):
    """Compute average mse for the data passed. Based on different methods.

    CURRENTLY ONLY USED FOR WEIGHTED.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_all : Numpy Nx1 vector. Treatment.
    w_dat : Numpy Nx1 vector. Weights (or 0)
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : List of INT. Treatment values.
    w_yes: Boolean. Weighted estimation.
    splitting: Boolean. Default is False.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.

    """
    treat_shares = np.empty(no_of_treat) if mtot in (1, 4) else 0
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    no_of_obs_by_treat = np.zeros(no_of_treat)
    for m_idx in range(no_of_treat):
        d_m = d_dat == treat_values[m_idx]   # d_m is Boolean
        n_m = len(y_dat[d_m])
        no_of_obs_by_treat[m_idx] = n_m
        if w_yes:
            w_m = w_dat[d_m]
            y_m_mean = np.average(y_dat[d_m], weights=w_m, axis=0)
            mse_m = np.average(np.square(y_dat[d_m] - y_m_mean),
                               weights=w_m, axis=0)
        else:
            y_m_mean = np.average(y_dat[d_m], axis=0)
            mse_m = np.dot(y_dat[d_m], y_dat[d_m]) / n_m - (y_m_mean**2)
        if mtot in (1, 4):
            treat_shares[m_idx] = n_m / n_obs
        if mtot in (1, 3, 4):
            mse_mce[m_idx, m_idx] = mse_m
        if mtot != 3:
            mce_ml = 0
            for v_idx in range(m_idx + 1, no_of_treat):
                if mtot == 2:  # Variance of effects mtot = 2
                    d_l = d_dat == treat_values[v_idx]   # d_l is Boolean
                    if w_yes:
                        y_l_mean = np.average(y_dat[d_l], weights=w_dat[d_l],
                                              axis=0)
                    else:
                        y_l_mean = np.average(y_dat[d_l], axis=0)
                    mce_ml = (y_m_mean - y_l_mean)**2
                else:
                    d_ml = (d_dat == treat_values[v_idx]) | (
                        d_dat == treat_values[m_idx])
                    d_ml = d_ml[:, 0]
                    y_nn_m, y_nn_l = y_nn[d_ml, m_idx], y_nn[d_ml, v_idx]
                    if w_yes:
                        w_ml = w_dat[d_ml].reshape(-1)
                        if splitting and (no_of_treat == 2):
                            mce_ml = ((np.average(y_nn_m, weights=w_ml,
                                                  axis=0)) *
                                      (np.average(y_nn_l, weights=w_ml,
                                                  axis=0)) * (-1))
                        else:
                            mce_ml = np.average(
                                (y_nn_m - np.average(y_nn_m, weights=w_ml,
                                                     axis=0)) *
                                (y_nn_l - np.average(y_nn_l, weights=w_ml,
                                                     axis=0)),
                                weights=w_ml, axis=0)
                    else:
                        aaa = np.average(y_nn_m, axis=0) * np.average(y_nn_l,
                                                                      axis=0)
                        bbb = np.dot(y_nn_m, y_nn_l) / len(y_nn_m)
                        mce_ml = bbb - aaa
                mse_mce[m_idx, v_idx] = mce_ml
    return mse_mce, treat_shares, no_of_obs_by_treat


@njit
def mcf_mse_numba(y_dat, y_nn, d_dat, n_obs, mtot, no_of_treat, treat_values):
    """Compute average mse for the data passed. Based on different methods.

       WEIGHTED VERSION DOES NOT YET WORK. TRY with next Numba version.
       Need to change list format soon.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    d_bin_dat : Numpy Nx1 vector. Treatment larger 0.
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : 1D Numpy array of INT. Treatment values.
    cont. Boolean. Continuous treatment.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.
    """
    obs = len(y_dat)
    treat_shares = np.zeros(no_of_treat) if mtot in (1, 3, 4) else np.zeros(1)
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    no_of_obs_by_treat = np.zeros(no_of_treat)
    for m_idx in range(no_of_treat):
        d_m = d_dat == treat_values[m_idx]   # d_m is Boolean
        n_m = np.sum(d_m)
        no_of_obs_by_treat[m_idx] = n_m
        y_m = np.empty(n_m)
        j = 0
        for i in range(obs):
            if d_m[i]:
                y_m[j] = y_dat[i, 0]
                j += 1
        y_m_mean = np.sum(y_m) / n_m
        mse_m = np.dot(y_m, y_m) / n_m - (y_m_mean**2)
        if mtot in (1, 3, 4):
            treat_shares[m_idx] = n_m / n_obs
            mse_mce[m_idx, m_idx] = mse_m
        if mtot != 3:
            mce_ml = 0
            for v_idx in range(m_idx + 1, no_of_treat):
                d_l = d_dat == treat_values[v_idx]   # d_l is Boolean
                n_l = np.sum(d_l)
                if mtot == 2:  # Variance of effects mtot = 2
                    y_l = np.empty(n_l)
                    j = 0
                    for i in range(obs):
                        if d_l[i]:
                            y_l[j] = y_dat[i, 0]
                            j += 1
                    y_l_mean = np.sum(y_l) / n_l
                    mce_ml = (y_m_mean - y_l_mean)**2
                elif mtot in (1, 4):
                    d_ml = (d_dat == treat_values[v_idx]) | (
                        d_dat == treat_values[m_idx])
                    n_ml = np.sum(d_ml)
                    y_nn_l = np.empty(n_ml)
                    y_nn_m = np.empty_like(y_nn_l)
                    j = 0
                    for i in range(obs):
                        if d_ml[i]:
                            y_nn_l[j] = y_nn[i, v_idx]
                            y_nn_m[j] = y_nn[i, m_idx]
                            j += 1
                    aaa = np.sum(y_nn_m) / n_ml * np.sum(y_nn_l) / n_ml
                    bbb = np.dot(y_nn_m, y_nn_l) / n_ml
                    mce_ml = bbb - aaa
                mse_mce[m_idx, v_idx] = mce_ml
    return mse_mce, treat_shares, no_of_obs_by_treat


def get_avg_mse_mce(mse_mce, obs_by_treat, mtot, no_of_treat):
    """Bring MSE_MCE matrix in average form."""
    mse_mce_avg = mse_mce.copy()
    for m_idx in range(no_of_treat):
        mse_mce_avg[m_idx, m_idx] = mse_mce[m_idx, m_idx] / obs_by_treat[m_idx]
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                mse_mce_avg[m_idx, v_idx] = mse_mce[m_idx, v_idx] / (
                    obs_by_treat[m_idx] + obs_by_treat[v_idx])
    return mse_mce_avg


def describe_forest(forest, m_n_min_ar, var_dic, cf_dic, gen_dic, pen_mult=0,
                    summary=True):
    """Describe estimated forest by collecting information in trees.

    Parameters
    ----------
    forest : List of List. Each forest consist of one node_table.
    m_n_min : List of INT. Number of variables and minimum leaf size
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.

    Returns
    -------
    None.

    """
    txt = ('\n' + '-' * 100 + '\nParameters of estimation to build random'
           ' forest')
    txt += '\nOutcome variable used to build forest: '
    txt += ' '.join(var_dic['y_tree_name'])
    txt += '\nFeatures used to build forest:          '
    txt += ' '.join(var_dic['x_name'])
    txt += '\nVariables always included in splitting: '
    txt += ' '.join(var_dic['x_name_always_in'])
    txt += (f'\nNumber of replications:     {cf_dic["boot"]:<4}')
    if cf_dic['mtot'] == 3:
        splitting_rule = 'MSEs of regressions only considered'
    elif cf_dic['mtot'] == 1:
        splitting_rule = 'MSE+MCE criterion'
    elif cf_dic['mtot'] == 2:
        splitting_rule = '-Var(effect)'
    elif cf_dic['mtot'] == 4:
        splitting_rule = 'Random switching'
    txt += f'\nSplitting rule used:        {splitting_rule:<4}'
    if cf_dic['p_diff_penalty'] > 0:
        txt += f'\nPenalty used in splitting:  {pen_mult}'
    txt += '\nShare of data in subsample for forest buildung:'
    txt += f' {cf_dic["subsample_share_forest"]:<4}'
    txt += '\nShare of data in subsample for forest evaluation:'
    txt += f' {cf_dic["subsample_share_eval"]:<4}'
    txt += '\nTotal number of variables available for splitting:'
    txt += f' {len(var_dic["x_name"]):<4}'
    txt += f'\n# of variables (M) used for split: {m_n_min_ar[0]:<4}'
    if cf_dic['m_random_poisson']:
        txt += '\n           (# of variables drawn from 1+Poisson(M-1))'
        txt += '\nMinimum threshold for using Poisson: '
        txt += f'{cf_dic["m_random_poisson_min"]}'
    txt += f'\nMinimum leaf size:                 {m_n_min_ar[1]:<4}'
    txt += f'\nAlpha regularity:                  {m_n_min_ar[2]:5.3f}'
    txt += '\n------------------- Estimated trees ----------------------------'
    leaf_info = get_tree_infos(forest)
    txt += f'\nAverage # of leaves:      {leaf_info[0]:4.1f}'
    txt += f'\nAverage size of leaves:   {leaf_info[1]:4.1f}'
    txt += f'\nMedian size of leaves:    {leaf_info[2]:4.1f}'
    txt += f'\nMin size of leaves:       {leaf_info[3]:4.0f}'
    txt += f'\nMax size of leaves:       {leaf_info[4]:4.0f}'
    txt += f'\nTotal # of obs in leaves: {leaf_info[5]:4.0f}\n' + '-' * 100
    ps.print_mcf(gen_dic, txt, summary=summary)


def get_tree_infos(forest):
    """Obtain some basic information about estimated trees.

    Parameters
    ----------
    forest : List of lists. Collection of node_tables.

    Returns
    -------
    leaf_info : List. Some information about tree.

    """
    leaf_info_tmp = np.zeros([len(forest), 6])
    for boot, tree in enumerate(forest):
        for leaf in tree:
            if leaf[4] == 1:   # Terminal leafs only
                leaf_info_tmp[boot, 0] += 1  # Number of leaves
        leaf_info_tree = np.zeros(int(leaf_info_tmp[boot, 0]))
        j = 0
        for leaf in tree:
            if leaf[4] == 1:
                leaf_info_tree[j] = leaf[5]
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


def get_terminal_leaf_no(node_table, x_dat):
    """Get the leaf number of the terminal node for single observation.

    Parameters
    ----------
    node_table : List of list. Single tree.
    x_dat : Numpy array. Data.

    Returns
    -------
    leaf_no : INT. Number of terminal leaf the observation belongs to.

    Note: This only works if nodes are ordered subsequently. Do not remove
          leafs when pruning. Only changes their activity status.

    """
    not_terminal = True
    leaf_id = 0
    while not_terminal:
        leaf = node_table[leaf_id]
        if leaf[4] not in (0, 1):
            raise RuntimeError(f'Leaf is still active. {leaf[4]}')
        if leaf[4] == 1:             # Terminal leaf
            not_terminal = False
            leaf_no = leaf[0]
        elif leaf[4] == 0:          # Intermediate leaf
            if leaf[10] == 0:        # Continuous variable
                leaf_id = (leaf[2] if (x_dat[leaf[8]] - 1e-15) <= leaf[9]
                           else leaf[3])
            else:                   # Categorical variable
                prime_factors = mcf_gp.primes_reverse(leaf[9], False)
                leaf_id = (leaf[2]
                           if int(np.round(x_dat[leaf[8]])) in prime_factors
                           else leaf[3])
    return leaf_no


def remove_oob_from_leaf0(forest):
    """Save memory by removing OOB indices.

    Parameters
    ----------
    forest : List of list. Node_tables.

    Returns
    -------
    forest_out : List of list. Node_tables.
    """
    for idx, _ in enumerate(forest):
        forest[idx][0][16] = 0
    return forest


def fill_trees_with_y_indices_mp(mcf_, data_df, forest):
    """Fill trees with indices of outcomes, MP.

    Returns
    -------
    forest_with_y : List of lists. Updated Node_table.
    terminal_nodes: Tuple of np.arrays. No of final node.
    no_of_avg_nodes: INT. Average no of unfilled leafs.

    """
    int_dic, gen_dic, cf_dic = mcf_.int_dict, mcf_.gen_dict, mcf_.cf_dict
    if int_dic['with_output'] and int_dic['verbose']:
        print("\nFilling trees with indicies of outcomes")
    (x_name, _, _, cf_dic, _, data_np, _, _, x_i, _, _, d_i, _, _, _
     ) = mcf_data.prepare_data_for_forest(mcf_, data_df, True)
    err_txt = 'Wrong order of variables' + str(x_name) + ': ' + str(
        cf_dic['x_name_mcf'])
    if cf_dic['x_name_mcf'] != x_name:
        raise ValueError(err_txt)
    if gen_dic['d_type'] == 'continuous':
        d_dat = data_np[:, d_i]
        # substitute those d used for splitting only that have a zero with
        # random element from the positive treatment levels
        d_pos = d_dat[d_dat > 1e-15]
        rng = np.random.default_rng(12366456)
        d_values = rng.choice(d_pos, size=len(d_dat)-len(d_pos), replace=False)
        d_dat_for_x = np.copy(d_dat)
        j = 0
        for i, d_i in enumerate(d_dat):
            if d_i < 1e-15:
                d_dat_for_x[i, 0] = d_values[j]
                j += 1
        x_dat = np.concatenate((data_np[:, x_i], d_dat_for_x), axis=1)
    else:
        x_dat = data_np[:, x_i]
        d_dat = np.int16(np.round(data_np[:, d_i]))
    obs = len(x_dat)
    terminal_nodes = [None] * cf_dic['boot']
    nodes_empty = np.zeros(cf_dic['boot'])
    if gen_dic['mp_parallel'] < 1.5:
        maxworkers = 1
    else:
        maxworkers = (mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                 gen_dic['sys_share'])
                      if gen_dic['mp_automatic'] else gen_dic['mp_parallel'])
    if int_dic['with_output'] and int_dic['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1:
        for idx in range(cf_dic['boot']):
            (_, forest[idx], terminal_nodes[idx], nodes_empty[idx]
             ) = fill_mp(forest[idx], obs, d_dat, x_dat, idx, gen_dic, cf_dic)
            if int_dic['with_output'] and int_dic['verbose']:
                mcf_gp.share_completed(idx+1, cf_dic['boot'])
    else:
        if int_dic['ray_or_dask'] == 'ray':
            if int_dic['mem_object_store_2'] is None:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
            else:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False,
                             object_store_memory=int_dic['mem_object_store_2'])
                if int_dic['with_output'] and int_dic['verbose']:
                    num = round(int_dic["mem_object_store_2"] / (1024 * 1024))
                    txt = f'\nSize of Ray Object Store: {num} MB'
                    ps.print_mcf(gen_dic, txt, summary=False)
            x_dat_ref = ray.put(x_dat)
            still_running = [ray_fill_mp.remote(
                forest[idx], obs, d_dat, x_dat_ref, idx, gen_dic, cf_dic)
                for idx in range(cf_dic['boot'])]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for ret_all_i in finished_res:
                    iix = ret_all_i[0]
                    forest[iix] = ret_all_i[1]
                    terminal_nodes[iix] = ret_all_i[2]
                    nodes_empty[iix] = ret_all_i[3]
                    if int_dic['with_output'] and int_dic['verbose']:
                        mcf_gp.share_completed(jdx+1, cf_dic['boot'])
                    jdx += 1
            if 'refs' in int_dic['mp_ray_del']:
                del x_dat_ref
            if 'rest' in int_dic['mp_ray_del']:
                del finished_res, finished
            if int_dic['mp_ray_shutdown']:
                ray.shutdown()
    no_of_avg_enodes = np.mean(nodes_empty)
    if int_dic['with_output'] and int_dic['verbose']:
        txt = ('\nNumber of leaves w/o all treatments per tree: '
               f'{no_of_avg_enodes:6.3%}')
        if no_of_avg_enodes > 0:
            txt += ('\nIncomplete leafs will not be considered for weight'
                    ' computation.')
        txt += '\n' + '-' * 100
        mem = round(mcf_sys.total_size(forest) / (1024 * 1024), 2)
        txt += f'\nSize of forest: {mem} MB' + '\n' + '-' * 100
        ps.print_mcf(gen_dic, txt, summary=True)
    return forest, terminal_nodes, no_of_avg_enodes


@ray.remote
def ray_fill_mp(node_table, obs, d_dat, x_dat, b_idx, gen_dic, cf_dic):
    """Make it work under Ray."""
    return fill_mp(node_table, obs, d_dat, x_dat, b_idx, gen_dic, cf_dic)


def fill_mp(node_table, obs, d_dat, x_dat, b_idx, gen_dic, cf_dic):
    """Compute new node_table and list of final leaves.

    Parameters
    ----------
    node_table : List of lists.
    obs : Int. Sample size.
    d_dat : Numpy array. Treatment.
    x_dat : Numpy array. Features.
    b_idx : Int. Tree number.
    gen_dic, cf_dic : Dict. Controls.

    Returns
    -------
    node_table : List of lists.
    unique_leafs : List.
    b_idx : Int. Tree number.

    """
    subsam = cf_dic['subsample_share_eval'] < 1
    indices = np.arange(obs)
    if subsam:
        obs = round(obs * cf_dic['subsample_share_eval'])
        rng = np.random.default_rng((10+b_idx)**2+121)
        indices = rng.choice(indices, size=obs, replace=False)
    obs_in_leaf = np.zeros((obs, 1), dtype=np.uint32)
    for i, idx in enumerate(indices):
        obs_in_leaf[i] = get_terminal_leaf_no(node_table, x_dat[idx, :])
    unique_leafs = np.unique(obs_in_leaf)
    if subsam:
        unique_leafs = unique_leafs[1:]  # remove first index: obs not used
        d_dat = d_dat[indices]
    nodes_empty = 0
    no_of_treat = (2 if gen_dic['d_type'] == 'continuous'
                   else gen_dic['no_of_treat'])
    for leaf_id in unique_leafs:
        sel_ind = obs_in_leaf.reshape(-1) == leaf_id
        node_table[leaf_id][14] = indices[sel_ind]
        empty_leaf = len(np.unique(d_dat[sel_ind])) < no_of_treat
        if empty_leaf:
            node_table[leaf_id][16] = 1   # Leaf to be ignored
            nodes_empty += 1
    return b_idx, node_table, unique_leafs, nodes_empty/len(unique_leafs)


def save_forests_in_cf_dic(forest_dic, forest_list, fold, no_folds, reg_round,
                           eff_iate):
    """Save forests in dictionary as list of list."""
    # Initialise
    if fold == 0 and reg_round:
        innerlist = [None, None] if eff_iate else [None]
        forest_list = [innerlist for idx in range(no_folds)]
    forest_list[fold][0 if reg_round else 1] = deepcopy(forest_dic)
    return forest_list


def train_save_data(mcf_, data_df, forest):
    """Save data needed in the prediction part of mcf."""
    y_train_df = data_df[mcf_.var_dict['y_name']]
    d_train_df = data_df[mcf_.var_dict['d_name']]
    if mcf_.p_dict['cluster_std']:
        cl_train_df = data_df[mcf_.var_dict['cluster_name']]
    else:
        cl_train_df = None
    if mcf_.gen_dict['weighted']:
        w_train_df = data_df[mcf_.var_dict['w_name']]
    else:
        w_train_df = None
    if mcf_.p_dict['bt_yes']:
        x_bala_train_df = data_df[mcf_.var_dict['x_balance_name']]
    else:
        x_bala_train_df = None
    forest_dic = {'forest': forest, 'y_train_df': y_train_df,
                  'd_train_df': d_train_df, 'x_bala_df': x_bala_train_df,
                  'cl_train_df': cl_train_df, 'w_train_df': w_train_df}
    return forest_dic
