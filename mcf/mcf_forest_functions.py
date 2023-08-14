"""
Contains functions for building the forest.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import copy, deepcopy
from math import inf
from time import time

from numba import njit

import numpy as np
import ray
# import pandas as pd

from mcf import mcf_forest_data_functions as mcf_data
from mcf import mcf_forest_add_functions as mcf_add
from mcf import mcf_general as gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps
from mcf import mcf_variable_importance_functions as vi


def train_forest(mcf_, tree_df, fill_y_df):
    """Train the forest and do variable importance measures."""
    gen_dic, cf_dic, forest_list = mcf_.gen_dict, mcf_.cf_dict, mcf_.forest
    seed, time_vi = 9324561, 0
    ps.print_mcf(gen_dic, '=' * 100 + '\nTraining of Modified Causal Forest')
    if gen_dic['iate_eff']:
        cf_dic['est_rounds'] = ('regular', 'additional')
    else:
        cf_dic['est_rounds'] = ('regular', )
    obs = len(tree_df) + len(fill_y_df)
    if (folds := int(np.ceil(obs / cf_dic['chunks_maxsize']))) > 1:
        index_tr, index_y = tree_df.index, fill_y_df.index
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(index_tr.to_numpy())
        rng.shuffle(index_y.to_numpy())
        index_folds_tr = np.array_split(index_tr, folds)
        index_folds_y = np.array_split(index_y, folds)
    cf_dic['folds'] = folds
    for splits in range(folds):
        if folds > 1:
            tree_fold_df = tree_df.loc[index_folds_tr[splits]]
            fill_y_fold_df = fill_y_df.loc[index_folds_y[splits]]
        else:
            tree_fold_df, fill_y_fold_df = tree_df, fill_y_df
        for round_ in cf_dic['est_rounds']:
            reg_round = round_ == 'regular'
            if not reg_round:
                # Reverse training and fill_with_y_file
                tree_fold_df, fill_y_fold_df = efficient_iate(
                    mcf_, fill_y_fold_df, tree_fold_df, summary=False)
            # Data preparation and stats II (regular, efficient IATE)
            tree_fold_df, mcf_.var_dict = mcf_data.nn_matched_outcomes(
               mcf_, tree_fold_df, print_out=reg_round and splits == 0)
            # Estimate forest structure (regular, efficient IATE)
            if gen_dic['with_output']:
                print(f'\nBuilding {splits+1} / {folds} forests, {round_}')
            forest, x_name_mcf = build_forest(mcf_, tree_fold_df)
            if reg_round and splits == 0:
                cf_dic['x_name_mcf'] = x_name_mcf
            # Variable importance  ONLY REGULAR
            if all((cf_dic['vi_oob_yes'], gen_dic['with_output'], reg_round,
                    splits == 0)):
                time_start = time()
                vi.variable_importance(mcf_, tree_fold_df, forest, x_name_mcf)
                time_vi = time() - time_start
            else:
                time_vi = 0
            forest = mcf_add.remove_oob_from_leaf0(forest)
            # Fill tree with outcomes(regular, , efficient IATE)
            if gen_dic['with_output']:
                print(f'Filling {splits+1} / {folds} forests, {round_}')
            forest, _, _ = mcf_add.fill_trees_with_y_indices_mp(
                mcf_, fill_y_fold_df, forest)
            forest_dic = mcf_add.train_save_data(mcf_, fill_y_fold_df, forest)
            forest_list = mcf_add.save_forests_in_cf_dic(
                forest_dic, forest_list, splits, folds, reg_round,
                gen_dic['iate_eff'])
    return cf_dic, forest_list, time_vi


def add_mse_mce_split(mse_mce_l, mse_mce_r, obs_by_treat_l, obs_by_treat_r,
                      mtot, no_of_treat):
    """Sum up MSE parts of use in splitting rule."""
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat = obs_by_treat_l + obs_by_treat_r
    for m_idx in range(no_of_treat):
        mse_mce[m_idx, m_idx] = (
            mse_mce_l[m_idx, m_idx] * obs_by_treat_l[m_idx]
            + mse_mce_r[m_idx, m_idx] * obs_by_treat_r[m_idx]
            ) / obs_by_treat[m_idx]
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                n_ml_l = obs_by_treat_l[m_idx] + obs_by_treat_l[v_idx]
                n_ml_r = obs_by_treat_r[m_idx] + obs_by_treat_r[v_idx]
                mse_mce[m_idx, v_idx] = (mse_mce_l[m_idx, v_idx] * n_ml_l
                                         + mse_mce_r[m_idx, v_idx] * n_ml_r
                                         ) / (n_ml_l + n_ml_r)
    return mse_mce


def add_rescale_mse_mce(mse_mce, obs_by_treat, mtot, no_of_treat,
                        mse_mce_add_to, obs_by_treat_add_to):
    """Rescale MSE_MCE matrix and update observation count."""
    mse_mce_sc = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat_new = obs_by_treat + obs_by_treat_add_to
    for m_idx in range(no_of_treat):
        mse_mce_sc[m_idx, m_idx] = mse_mce[m_idx, m_idx] * obs_by_treat[m_idx]
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                mse_mce_sc[m_idx, v_idx] = mse_mce[m_idx, v_idx] * (
                    obs_by_treat[m_idx] + obs_by_treat[v_idx])
    mse_mce_new = mse_mce_add_to + mse_mce_sc
    return mse_mce_new, obs_by_treat_new


def best_m_n_min_alpha_reg(forest, gen_dic, cf_dic):
    """Get best forest for the tuning parameters m_try, n_min, alpha_reg.

    Parameters
    ----------
    forest : List of list of lists... Estimated forests.
    c : Dict. Parameters.

    Returns
    -------
    forest_final : List of lists. OOB-optimal forest.
    m_n_final : List. Optimal values of m and n_min.

    """
    grid_for_m = gp.check_if_iterable(cf_dic['m_values'])
    grid_for_n_min = gp.check_if_iterable(cf_dic['n_min_values'])
    grid_for_alpha_reg = gp.check_if_iterable(cf_dic['alpha_reg_values'])
    m_n_min_ar_combi = []
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                m_n_min_ar_combi.append([m_idx, n_min, alpha_reg])
    dim_m_n_min_ar = len(grid_for_m) * len(grid_for_n_min) * len(
        grid_for_alpha_reg)
    if (dim_m_n_min_ar) > 1:       # Find best of trees
        mse_oob = np.zeros(dim_m_n_min_ar)
        trees_without_oob = np.zeros(dim_m_n_min_ar)
        if gen_dic['d_type'] == 'continuous':
            no_of_treat = 2
        else:
            no_of_treat = gen_dic['no_of_treat']
        for trees_m_n_min_ar in forest:                  # different forests
            for j, tree in enumerate(trees_m_n_min_ar):  # trees within forest
                n_lost = n_total = 0
                if no_of_treat is not None:
                    mse_mce_tree = np.zeros((no_of_treat, no_of_treat))
                    obs_t_tree = np.zeros(no_of_treat)
                tree_mse = 0
                for leaf in tree:                        # leaves within tree
                    if leaf[4] == 1:   # Terminal leafs only
                        n_total += np.sum(leaf[6])
                        if leaf[7] is None:
                            if no_of_treat is None:
                                n_lost += leaf[6]
                            else:
                                n_lost += np.sum(leaf[6])  # [6]: Leaf size
                        else:
                            if no_of_treat is None:  # [7]: leaf_mse
                                tree_mse += leaf[6] * leaf[7]
                            else:
                                mse_mce_tree, obs_t_tree = add_rescale_mse_mce(
                                    leaf[7], leaf[6], cf_dic['mtot'],
                                    no_of_treat, mse_mce_tree, obs_t_tree)
                if n_lost > 0:
                    if no_of_treat is None:
                        tree_mse = tree_mse * n_total / (n_total - n_lost)
                    else:
                        if (n_total - n_lost) < 1:
                            trees_without_oob[j] += 1
                if no_of_treat is not None:
                    mse_mce_tree = mcf_add.get_avg_mse_mce(
                        mse_mce_tree, obs_t_tree, cf_dic['mtot'], no_of_treat)
                    tree_mse = compute_mse_mce(mse_mce_tree, cf_dic['mtot'],
                                               no_of_treat)
                mse_oob[j] += tree_mse     # Add MSE to MSE of forest j
        if np.any(trees_without_oob) > 0:
            for j, trees_without_oob_j in enumerate(trees_without_oob):
                if trees_without_oob_j > 0:
                    mse_oob[j] = mse_oob[j] * (
                        cf_dic['boot'] / (cf_dic['boot']
                                          - trees_without_oob_j))
        min_i = np.argmin(mse_oob)
        mse_oob = mse_oob / cf_dic['boot']
        cf_dic['n_min_values'] = gp.check_if_iterable(cf_dic['n_min_values'])
        cf_dic['m_values'] = gp.check_if_iterable(cf_dic['m_values'])
        cf_dic['alpha_reg_values'] = gp.check_if_iterable(
            cf_dic['alpha_reg_values'])
        if gen_dic['with_output']:
            txt = '\n' * 2 + '-' * 100
            txt += ('\nOOB MSE (without penalty) for M_try, minimum leafsize'
                    ' and alpha_reg combinations'
                    '\nNumber of vars / min. leaf size / alpha reg. / '
                    'OOB value. Trees without OOB\n')
            j = 0
            for m_idx in cf_dic['m_values']:
                for n_min in cf_dic['n_min_values']:
                    for alpha_reg in cf_dic['alpha_reg_values']:
                        txt += (f'\n{m_idx:>12} {n_min:>12}'
                                f' {alpha_reg:15.3f}'
                                f' {mse_oob[j]:8.3f}'
                                f' {trees_without_oob[j]:4.0f}')
                        j += 1
            txt += (f'\nMinimum OOB MSE:     {mse_oob[min_i]:7.3f}'
                    f'\nNumber of variables: {m_n_min_ar_combi[min_i][0]}'
                    f'\nMinimum leafsize:    {m_n_min_ar_combi[min_i][1]}'
                    f'\nAlpha regularity:    {m_n_min_ar_combi[min_i][2]}')
            txt += '\n' + '-' * 100
            ps.print_mcf(gen_dic, txt, summary=True)
        forest_final = [trees_m_n_min[min_i] for trees_m_n_min in forest]
        m_n_min_ar_opt = m_n_min_ar_combi[min_i]
    else:       # Find best of trees
        forest_final = [trees_m_n_min_ar[0] for trees_m_n_min_ar in forest]
        m_n_min_ar_opt = m_n_min_ar_combi[0]
    return forest_final, m_n_min_ar_opt


@ray.remote
def ray_build_tree_mcf(data_np, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                       x_type, x_values, x_ind, x_ai_ind, gen_dic, int_dic,
                       cf_dic, ct_dic, fs_dic, boot, pen_mult):
    """Prepare function for Ray."""
    return build_tree_mcf(data_np, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                          x_type, x_values, x_ind, x_ai_ind, gen_dic, int_dic,
                          cf_dic, ct_dic, fs_dic, boot, pen_mult)


@ray.remote
def ray_build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                             x_type, x_values, x_ind, x_ai_ind, gen_dic,
                             int_dic, cf_dic, ct_dic, fs_dic, boot_indices,
                             pen_mult):
    """Prepare function for Ray."""
    return build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i,
                                w_i, x_type, x_values, x_ind, x_ai_ind,
                                gen_dic, int_dic, cf_dic, ct_dic, fs_dic,
                                boot_indices, pen_mult)


def build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                         x_type, x_values, x_ind, x_ai_ind, gen_dic, int_dic,
                         cf_dic, ct_dic, fs_dic, boot_indices, pen_mult):
    """Build larger pieces of the forest (for MP)."""
    little_forest = []
    for boot in boot_indices:
        tree = build_tree_mcf(
            data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i, x_type, x_values,
            x_ind, x_ai_ind, gen_dic, int_dic, cf_dic, ct_dic, fs_dic, boot,
            pen_mult)
        little_forest.append(tree)
    return little_forest


def build_forest(mcf_, tree_df):
    """Build MCF (not yet populated by w and outcomes)."""
    int_dic, gen_dic, var_dic = mcf_.int_dict, mcf_.gen_dict, mcf_.var_dict
    cf_dic, fs_dic, ct_dic = mcf_.cf_dict, mcf_.fs_dict, mcf_.ct_dict
    with_ray = (not int_dic['no_ray_in_forest_building']
                and (int_dic['ray_or_dask'] == 'ray'))
    if not with_ray:
        if int_dic['with_output'] and int_dic['verbose']:
            ps.print_mcf(gen_dic, '\nNo use of ray in forest building.',
                         summary=False)
    (x_name, x_type, x_values, cf_dic, pen_mult, data_np, y_i, y_nn_i, x_i,
     x_ind, x_ai_ind, d_i, w_i, cl_i, d_grid_i
     ) = mcf_data.prepare_data_for_forest(mcf_, tree_df)
    if gen_dic['mp_parallel'] < 1.5:
        maxworkers = 1
    else:
        maxworkers = (mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                 gen_dic['sys_share'])
                      if gen_dic['mp_automatic'] else gen_dic['mp_parallel'])
    if int_dic['with_output'] and int_dic['verbose']:
        ps.print_mcf(gen_dic, f'\nNumber of parallel processes: {maxworkers}',
                     summary=False)
    if maxworkers == 1:
        forest = [None] * cf_dic['boot']
        for idx in range(cf_dic['boot']):
            forest[idx] = build_tree_mcf(
                data_np, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i, x_type,
                x_values, x_ind, x_ai_ind, gen_dic, int_dic, cf_dic, ct_dic,
                fs_dic, idx, pen_mult)
            if int_dic['with_output'] and int_dic['verbose']:
                gp.share_completed(idx+1, cf_dic['boot'])
    else:
        forest = []
        if with_ray:
            if int_dic['mem_object_store_1'] is None:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
            else:
                if not ray.is_initialized():
                    ray.init(
                        num_cpus=maxworkers, include_dashboard=False,
                        object_store_memory=int_dic['mem_object_store_1'])
                if int_dic['with_output'] and int_dic['verbose']:
                    print("Size of Ray Object Store: ", round(
                        int_dic['mem_object_store_1']/(1024*1024)), " MB")
            data_np_ref = ray.put(data_np)
            still_running = [ray_build_tree_mcf.remote(
                data_np_ref, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                x_type, x_values, x_ind, x_ai_ind, gen_dic, int_dic, cf_dic,
                ct_dic, fs_dic, boot, pen_mult)
                for boot in range(cf_dic['boot'])]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for ret_all_i in finished_res:
                    forest.append(ret_all_i)
                    if int_dic['with_output'] and int_dic['verbose']:
                        gp.share_completed(jdx+1, cf_dic['boot'])
                    jdx += 1
                if jdx % 50 == 0:   # every 50'th tree
                    mcf_sys.auto_garbage_collect(50)  # do if half mem full
            if 'refs' in int_dic['mp_ray_del']:
                del data_np_ref
            if 'rest' in int_dic['mp_ray_del']:
                del finished_res, finished
            if int_dic['mp_ray_shutdown']:
                ray.shutdown()
        else:
            raise RuntimeError('USE RAY')
        if len(forest) != cf_dic['boot']:
            raise RuntimeError(f'Forest has wrong size: {len(forest)}'
                               'Bug in Multiprocessing.')
    # find best forest given the saved oob values
    forest_final, m_n_final = best_m_n_min_alpha_reg(forest, gen_dic, cf_dic)
    del forest    # Free memory
    # Describe final tree
    if int_dic['with_output']:
        mcf_add.describe_forest(forest_final, m_n_final, var_dic, cf_dic,
                                gen_dic, pen_mult)
    # x_name: List. Order of x_name as used by tree building
    return forest_final, x_name


def build_tree_mcf(data_np, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i, x_type,
                   x_values, x_ind, x_ai_ind, gen_dic, int_dic, cf_dic, ct_dic,
                   fs_dic, boot, pen_mult):
    """Build single trees for all values of tuning parameters.

    Parameters
    ----------
    data_np : Numpy 2D array. Data.
    y_i : Position of Outcome in DATA.
    y_nn_i: Position of Matched outcomes.
    x_i : Position of Covariates.
    d_i : Position of Treatment.
    d_grid_i: Position of discretized treatments (continuous case)
    cl_i : Position of Cluster variable.
    x_type : List of INT. Type of variable: 0,1,2
    x_values: List of lists. Values of variable (if few or categorical)
    x_ind : List of INT. Identifier of variables
    x_ai_ind : List of INT. 1 if variable is included in every split
    c_dict : Dict. Control parameters
    boot : INT. Counter for bootstrap replication (currently not used)

    Returns
    -------
    tree_all : LIST (m_grid x N_min_grid x alpha_grid) with trees for all
               values of tuning parameters
    """
    # split data into OOB and tree data
    n_obs = data_np.shape[0]
    # Random number initialisation. This seeds rnd generator within process
    rng = np.random.default_rng((10+boot)**2+121)
    if gen_dic['panel_in_rf']:
        cl_unique = np.unique(data_np[:, cl_i])
        n_cl = cl_unique.shape[0]
        n_train = round(n_cl * int_dic['share_forest_sample'])
        indices_cl = list(rng.choice(n_cl, size=n_train, replace=False))
        indices = [i for i in range(n_obs) if data_np[i, cl_i] in indices_cl]
    else:
        n_train = round(n_obs * int_dic['share_forest_sample'])
        indices = list(rng.choice(n_obs, size=n_train, replace=False))
    data_tr, data_oob = data_np[indices], np.delete(data_np, indices, axis=0)
    n_tr, n_oob = data_tr.shape[0], data_oob.shape[0]
    node_t_init = mcf_add.init_node_table(
        n_tr, n_oob, np.delete(range(n_obs), indices, axis=0))
    # build trees for all m,n combinations
    grid_for_m = gp.check_if_iterable(cf_dic['m_values'])
    grid_for_n_min = gp.check_if_iterable(cf_dic['n_min_values'])
    grid_for_alpha_reg = gp.check_if_iterable(cf_dic['alpha_reg_values'])
    tree_all = [None] * len(grid_for_m) * len(grid_for_n_min) * len(
        grid_for_alpha_reg)
    j = 0
    ct_grid_nn_val, fs_yes = ct_dic['grid_nn_val'], fs_dic['yes']
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                node_table_0 = deepcopy(node_t_init)  # empty table
                tree_all[j] = build_single_tree(
                    data_tr, data_oob, y_i, y_nn_i, d_i, d_grid_i, x_i, w_i,
                    x_type, x_values, x_ind, x_ai_ind, cf_dic, gen_dic,
                    ct_grid_nn_val, fs_yes, m_idx, n_min,
                    alpha_reg, node_table_0, pen_mult, rng)
                j += 1
    return tree_all


def build_single_tree(data, data_oob, y_i, y_nn_i, d_i, d_grid_i, x_i, w_i,
                      x_type, x_values, x_ind, x_ai_ind, cf_dic, gen_dic,
                      ct_grid_nn_val, fs_yes, mmm, n_min, alpha_reg,
                      node_table, pen_mult, rng):
    """Build single tree given random sample split.

    Parameters
    ----------
    data : Nympy array. Training data
    data_oob : Numpy array. OOB data
    y_i : List. Position of y in numpy array.
    y_nn_i : List. Position of y_nn in numpy array.
    d_i : Int. Position of d in numpy array.
    d_i_grid: Int. Position of d_grid in numpy array.
    x_i : List. Position of x in numpy array.
    x_type : List of INT. Type of covariate (0,1,2).
    x_values: List of lists. Values of covariate (if not too many)
    x_ind : List. Postion of covariate in x for easy reference.
    x_ai_ind : List. Postion of covariate always-in in x for easy reference.
    cf_dic, gen_dic : Dict. Parameters.
    m : INT. Number of covariates to be included.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. alpha regularity.
    node_table : List of list of lists. Initial tree (basic leaf)
    pen_mult: Float. Multiplier of penalty.
    rng : Default random number generator object.

    Returns
    -------
    node_table : List of list of lists. Final tree.
    """
    continue_to_split = True
    while continue_to_split:
        len_table = len(node_table)
        active_knots = 0
        for node_i in range(len_table):
            if node_table[node_i][4] == 2:
                current = deepcopy(node_table[node_i])
                left, right, current, terminal = next_split(
                    current, data, data_oob, y_i, y_nn_i, d_i, d_grid_i, x_i,
                    w_i, x_type, x_values, x_ind, x_ai_ind, cf_dic, gen_dic,
                    ct_grid_nn_val, fs_yes, mmm, n_min, alpha_reg, pen_mult,
                    len(node_table), rng)
                node_table[node_i] = deepcopy(current)
                if not terminal:
                    active_knots += 1
                    node_table.append(deepcopy(left))
                    node_table.append(deepcopy(right))
        if active_knots == 0:
            continue_to_split = False  # Tree completed
    return node_table


def compute_mse_mce(mse_mce, mtot, no_of_treat):
    """Sum up MSE parts for use in splitting rule and else."""
    if no_of_treat > 4:
        if mtot in (1, 4):
            mse = no_of_treat * np.trace(mse_mce) - mse_mce.sum()
        elif mtot == 2:
            mse = 2 * np.trace(mse_mce) - mse_mce.sum()
        elif mtot == 3:
            mse = np.trace(mse_mce)
    else:
        mse = mce = 0
        for m_idx in range(no_of_treat):
            mse_a = ((no_of_treat - 1) * mse_mce[m_idx, m_idx]
                     if mtot in (1, 4) else mse_mce[m_idx, m_idx])
            mse += mse_a
            if mtot != 3:
                for v_idx in range(m_idx+1, no_of_treat):
                    mce += mse_mce[m_idx, v_idx]
        mse -= 2 * mce
    return mse


def efficient_iate(mcf_, fill_y_df, tree_df, summary=False):
    """Get more efficient iates."""
    if mcf_.int_dict['with_output'] and mcf_.int_dict['verbose']:
        ps.print_mcf(mcf_.gen_dict,
                     '\nSecond round of estimation to get better IATEs',
                     summary=summary)
    fill_y_df, tree_df = tree_df, fill_y_df
    return tree_df, fill_y_df


def next_split(current_node, data_tr, data_oob, y_i, y_nn_i, d_i, d_grid_i,
               x_i, w_i, x_type, x_values, x_ind, x_ai_ind, cf_dic, gen_dic,
               ct_grid_nn_val, fs_yes, mmm, n_min, alpha_reg, pen_mult, trl,
               rng):
    """Find best next split of leaf (or terminate splitting for this leaf).

    Parameters
    ----------
    current_node : List of list: Information about leaf to split.
    data_tr: Numpy array. All training data.
    data_oob : Numpy array: All OOB data.
    y_i : INT. Location of Y in data matrix.
    y_nn_i :  List of INT. Location of Y_NN in data matrix.
    d_i : INT. Location of D in data matrix.
    d_grid_i : INT. Location of D_grid in data matrix.
    x_i : List of INT. Location of X in data matrix.
    x_type : List of INT (0,1,2). Type of X.
    x_ind : List INT. Location of X in X matrix.
    x_ai_ind : List of INT. Location of X_always in X matrix.
    ... : DICT. Parameters.
    mmm : INT. Number of X-variables to choose for splitting.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. Alpha regularity.
    pen_mult: Float. Penalty multiplier.
    trl: Int. Tree length.
    rng: Numpy default random number generator object.

    Returns
    -------
    left : List of lists. Information about left leaf.
    right : List of lists. Information about right leaf.
    current : List of lists. Updated information about this leaf.
    terminal : INT. 1: No splits for this leaf. 0: Leaf splitted
    """
    data_tr_ns = data_tr[current_node[11], :]   # Train. data of node
    data_oob_ns = data_oob[current_node[12], :]   # OOB data of  node
    terminal = split_done = False
    if current_node[5] < (2 * n_min):
        terminal = True
    elif np.all(data_tr_ns[:, d_i] == data_tr_ns[0, d_i]):
        terminal = True
    else:
        if current_node[5] < 200:  # Otherwise, too slow:
            if gen_dic['d_type'] == 'continuous':
                terminal = not (2 <= np.sum(data_tr_ns[:, d_i] == 0)
                                <= current_node[5] - 2)
            else:
                ret = np.unique(data_tr_ns[:, d_i], return_counts=True)
                terminal = (len(ret[0]) < gen_dic['no_of_treat']
                            or np.any(ret[1] < 2 * cf_dic['n_min_treat']))
    mtot, w_yes = cf_dic['mtot'], gen_dic['weighted']
    if gen_dic['d_type'] == 'continuous':
        no_of_treat, d_values, continuous = 2, [0, 1], True
        d_split_in_x_ind = np.max(x_ind) + 1
    else:
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
        d_bin_dat, continuous = None, False
    if not terminal:
        obs_min = max([round(current_node[5] * alpha_reg), n_min])
        best_mse = inf   # Initialisation: Infinity as default values
        for _ in range(3):
            x_ind_split = mcf_add.rnd_variable_for_split(
                x_ind, x_ai_ind, cf_dic, mmm, rng)
            x_type_split = x_type[x_ind_split].copy()
            x_values_split = [x_values[v_idx].copy() for v_idx in x_ind_split]
            # Check if split is possible ... sequential order to minimize costs
            # Check if enough variation in the data to do splitting (costly)
            with_d_oob = continuous
            (y_dat, _, d_dat, d_oob, d_grid_dat, _, x_dat, x_oob,
             terminal, terminal_x, x_no_varia) = term_or_data(
                 data_tr_ns, data_oob_ns, y_i, d_i, d_grid_i,
                 x_i[x_ind_split], no_of_treat, with_d_oob=with_d_oob)
            if not terminal_x:
                break
        if terminal_x:
            terminal = True  # No variation in drawn X. Splitting stops.
    if not terminal:         # ML 25.5.2022
        if mtot in (1, 4):
            y_nn = data_tr_ns[:, y_nn_i]
        else:
            y_nn = y_nn_l = y_nn_r = 0
        w_dat = data_tr_ns[:, [w_i]] if w_yes else [1]
        if continuous:
            d_bin_dat = d_dat > 1e-15   # Binary treatment indicator
            x_no_varia.append(np.all(d_bin_dat == d_bin_dat[0]))
            x_ind_split.append(d_split_in_x_ind)
            x_type_split = np.append(x_type_split, 0)
        p_x = len(x_ind_split)  # indices refer to order of x in data_*
        d_cont_split = False
        for j in range(p_x):  # Loops over the variables
            if not x_no_varia[j]:  # No variation of this x -> no split
                d_cont_split = continuous and (j == p_x - 1)
                if d_cont_split:
                    x_j, x_oob_j = np.copy(d_dat), np.copy(d_oob)
                    x_j_pos = x_j[x_j > 1e-15]  # Positive treatment values
                    nr_pos, nr_all = len(x_j_pos),  len(x_j)
                    nr_0 = nr_all - nr_pos
                    nr_all_oob = len(x_oob_j)
                    if nr_0 < 2 or nr_pos < 2:  # Too few controls
                        continue
                    split_values = np.unique(x_j_pos).tolist()
                    if len(split_values) > 1:
                        split_values = split_values[:-1]  # 0 not included
                else:
                    x_j, x_oob_j = np.copy(x_dat[:, j]), np.copy(x_oob[:, j])
                    if x_type_split[j] > 0:
                        x_j = x_j.astype(np.int32)
                        x_oob_j = x_oob_j.astype(np.int32)
                    split_values = get_split_values(
                        y_dat, w_dat, x_j, x_type_split[j], x_values_split[j],
                        current_node[5], cf_dic['random_thresholds'],
                        gen_dic['weighted'], rng=rng)
                    split_values_unord_j = []
                if d_cont_split:
                    # Randomly allocate half the controls to left leaf
                    rnd_in = rng.choice([True, False], size=(nr_all, 1))
                    # Somewhat inefficient as it is also applied to treated
                    treat_0 = (x_j - 1e-15) <= 0
                    zeros_l = treat_0 & rnd_in
                    rnd_in_oob = rng.choice([True, False],
                                            size=(nr_all_oob, 1))
                    # Somewhat inefficient as it is also applied to treated
                    treat_0_oob = (x_oob_j - 1e-15) <= 0
                    zeros_l_oob = treat_0_oob & rnd_in_oob
                for val in split_values:  # Loops over values of variables
                    if x_type_split[j] == 0:
                        if d_cont_split:   # Treated and selected non-treated
                            treated_l = np.invert(treat_0) & (x_j <= val)
                            leaf_l = (treated_l | zeros_l).flatten()
                        else:
                            leaf_l = (x_j - 1e-15) <= val  # because of float
                    else:                          # ordered with few vals.
                        # Categorial variable: Either in group or not
                        split_values_unord_j.append(val)
                        leaf_l = np.isin(x_j, split_values_unord_j)
                    n_l = np.count_nonzero(leaf_l)
                    n_r = current_node[5] - n_l
                    # Check if enough observations available
                    if (n_l < obs_min) or (n_r < obs_min):
                        continue
                    if x_type_split[j] == 0:
                        if d_cont_split:   # Treated and selected non-treated
                            treated_l_oob = np.invert(treat_0_oob
                                                      ) & (x_oob_j <= val)
                            leaf_oob_l = (treated_l_oob | zeros_l_oob
                                          ).flatten()
                        else:
                            leaf_oob_l = (x_oob_j - 1e-15) <= val
                    else:
                        leaf_oob_l = np.isin(x_oob_j, split_values_unord_j)
                    n_oob_l = np.count_nonzero(leaf_oob_l)
                    n_oob_r = current_node[6] - n_oob_l
                    # Next we check if any obs in each treatment
                    d_dat_l = (d_bin_dat[leaf_l]
                               if continuous else d_dat[leaf_l])
                    if continuous or cf_dic['n_min_treat'] == 1:
                        if len(np.unique(d_dat_l)) < no_of_treat:
                            continue
                    else:
                        ret = np.unique(d_dat_l, return_counts=True)
                        if len(ret[0]) < no_of_treat:
                            continue
                        if np.any(ret[1] < cf_dic['n_min_treat']):
                            continue
                    leaf_r = np.invert(leaf_l)  # Reverses True to False
                    d_dat_r = (d_bin_dat[leaf_r]
                               if continuous else d_dat[leaf_r])
                    if continuous or cf_dic['n_min_treat'] == 1:
                        if len(np.unique(d_dat_r)) < no_of_treat:
                            continue   # Splits possible?
                    else:
                        ret = np.unique(d_dat_r, return_counts=True)
                        if len(ret[0]) < no_of_treat:
                            continue
                        if np.any(ret[1] < cf_dic['n_min_treat']):
                            continue
                    leaf_oob_r = np.invert(leaf_oob_l)
                    if mtot in (1, 4):
                        if continuous:
                            y_nn_l = mcf_add.match_cont(
                                d_grid_dat[leaf_l], y_nn[leaf_l, :],
                                ct_grid_nn_val, rng)
                            y_nn_r = mcf_add.match_cont(
                                d_grid_dat[leaf_r], y_nn[leaf_r, :],
                                ct_grid_nn_val, rng)
                        else:
                            y_nn_l, y_nn_r = y_nn[leaf_l, :], y_nn[leaf_r, :]
                    else:
                        y_nn_l = y_nn_r = 0
                    if w_yes:
                        w_l, w_r = w_dat[leaf_l], w_dat[leaf_r]
                    else:
                        w_l = w_r = 0
                    # compute objective functions given particular method
                    mse_mce_l, shares_l, obs_by_treat_l = mcf_add.mcf_mse(
                        y_dat[leaf_l], y_nn_l, d_dat_l, w_l, n_l, mtot,
                        no_of_treat, d_values, w_yes)
                    mse_mce_r, shares_r, obs_by_treat_r = mcf_add.mcf_mse(
                        y_dat[leaf_r], y_nn_r, d_dat_r, w_r, n_r, mtot,
                        no_of_treat, d_values, w_yes)
                    mse_mce = add_mse_mce_split(
                        mse_mce_l, mse_mce_r, obs_by_treat_l,
                        obs_by_treat_r, mtot, no_of_treat)
                    mse_split = compute_mse_mce(mse_mce, mtot, no_of_treat)
                    # add penalty for this split
                    if ((cf_dic['mtot'] == 1) or ((cf_dic['mtot'] == 4)
                                                  and (rng.random() > 0.5))):
                        penalty = mcf_penalty(shares_l, shares_r)
                        mse_split = mse_split + pen_mult * penalty
                    if mse_split < best_mse:
                        split_done = True
                        best_mse = mse_split
                        best_var_i = copy(x_ind_split[j])
                        best_type = copy(x_type_split[j])
                        best_n_l, best_n_r = n_l, n_r
                        best_leaf_l = np.copy(leaf_l)
                        best_leaf_r = np.copy(leaf_r)
                        best_leaf_oob_l = np.copy(leaf_oob_l)
                        best_leaf_oob_r = np.copy(leaf_oob_r)
                        best_n_oob_l, best_n_oob_r = n_oob_l, n_oob_r
                        best_value = (copy(val) if best_type == 0 else
                                      split_values_unord_j[:])  # left
    if not split_done:
        terminal = True
    if terminal:
        current_node[4] = 1  # terminal
        w_oob = data_oob_ns[:, [w_i]] if w_yes else 0
        n_oob = np.copy(current_node[6])
        if continuous:
            d_oob = data_oob_ns[:, d_i] > 1e-15
        else:
            d_oob = data_oob_ns[:, d_i]
        if len(np.unique(d_oob)) < no_of_treat:
            current_node[7] = None      # MSE cannot be computed
        else:
            if continuous:
                y_nn = mcf_add.match_cont(data_oob_ns[:, d_grid_i],
                                          data_oob_ns[:, y_nn_i],
                                          ct_grid_nn_val, rng)
            else:
                y_nn = data_oob_ns[:, y_nn_i]
            current_node[7], shares_r, current_node[6] = mcf_add.mcf_mse(
                data_oob_ns[:, y_i], y_nn, d_oob, w_oob, n_oob, mtot,
                no_of_treat, d_values, w_yes)
        current_node[11] = current_node[12] = 0  # Data no longer needed
        newleaf_l, newleaf_r = [], []
    else:
        newleaf_l = deepcopy(current_node)
        newleaf_r = deepcopy(current_node)
        newleaf_l[0], newleaf_r[0] = trl, trl + 1  # Tree length, starts with 0
        newleaf_l[1] = deepcopy(current_node[0])  # Parent nodes
        newleaf_r[1] = deepcopy(current_node[0])
        newleaf_l[2] = newleaf_r[2] = None             # Following splits l
        newleaf_l[3] = newleaf_r[3] = None             # Following splits r
        newleaf_l[4] = newleaf_r[4] = 2                # Node is active
        newleaf_l[5], newleaf_r[5] = best_n_l, best_n_r   # Leaf size training
        newleaf_l[6], newleaf_r[6] = best_n_oob_l, best_n_oob_r  # Leafsize OOB
        newleaf_l[7] = newleaf_r[7] = None         # OOB MSE without penalty
        newleaf_l[8] = newleaf_r[8] = None         # Variable for next split
        newleaf_l[9] = newleaf_r[9] = newleaf_l[10] = newleaf_r[10] = None
        train_list = np.array(current_node[11], copy=True)
        oob_list = np.array(current_node[12], copy=True)
        newleaf_l[11] = train_list[best_leaf_l].tolist()
        newleaf_r[11] = train_list[best_leaf_r].tolist()
        newleaf_l[12] = oob_list[best_leaf_oob_l].tolist()
        newleaf_r[12] = oob_list[best_leaf_oob_r].tolist()
        newleaf_l[13] = newleaf_r[13] = newleaf_l[14] = newleaf_r[14] = None
        newleaf_l[15] = newleaf_r[15] = None
        current_node[2] = copy(newleaf_l[0])  # ID of daughter leaf
        current_node[3] = copy(newleaf_r[0])
        current_node[4] = 0     # not active, not terminal - intermediate
        current_node[8] = copy(best_var_i)
        if best_type > 0:  # Save as product of primes
            best_value = gp.list_product(best_value)   # int
        current_node[9] = copy(best_value)    # <= -> left
        current_node[10] = copy(best_type)
        current_node[11] = current_node[12] = 0   # Data, no longer needed
        if current_node[0] != 0:
            current_node[16] = 0
        else:    # Need to keep OOB data in first leaf for VIB, Feature select
            if (not cf_dic['vi_oob_yes']) and (not fs_yes):
                current_node[16] = 0    # Data, no longer needed, saves memory
    return newleaf_l, newleaf_r, current_node, terminal


def get_split_values(y_dat, w_dat, x_dat, x_type, x_values, leaf_size,
                     random_thresholds, w_yes, rng=None):
    """Determine the values used for splitting.

    Parameters
    ----------
    y_dat : Numpy array. Outcome.
    x_dat : 1-d Numpy array. Splitting variable.
    w_dat : 1-d Numpy array. Weights.
    x_type : Int. Type of variables used for splitting.
    x_values: List.
    leaf_size. INT. Size of leaf.
    c_dict: Dict. Parameters
    rng : Random Number Generator object

    Returns
    -------
    splits : List. Splitting values to use.
    """
    if rng is None:
        rng = np.random.default_rng()
        raise Warning('unseeded random number generator used')
    if x_type == 0:
        if bool(x_values):  # Limited number of values in x_value
            min_x, max_x = np.amin(x_dat), np.amax(x_dat)
            del_values = [j for j, val in enumerate(x_values)
                          if (val < (min_x - 1e-15))
                          or (val > (max_x + 1e-15))]
            if del_values:  # List is not empty
                splits_x = [x for x in x_values if x not in del_values]
            else:
                splits_x = x_values[:]
            if len(splits_x) > 1:
                splits_x = splits_x[:-1]
                if 0 < random_thresholds < len(splits_x):
                    splits = np.unique(
                        rng.choice(splits_x, size=random_thresholds,
                                   replace=False, shuffle=False))
                else:
                    splits = splits_x
        else:  # Continoues variable with very many values; x_values empty
            if 0 < random_thresholds < (leaf_size - 1):
                x_vals_np = rng.choice(
                    x_dat, random_thresholds, replace=False,
                    shuffle=False)
                x_vals_np = np.unique(x_vals_np)
                splits = x_vals_np.tolist()
            else:
                x_vals_np = np.unique(x_dat)
                splits = x_vals_np.tolist()
                if len(splits) > 1:
                    splits = splits[:-1]
    else:
        y_mean_by_cat = np.empty(len(x_values))  # x_vals comes as list
        x_vals_np = np.array(x_values, dtype=np.int32, copy=True)
        used_values = []
        for v_idx, val in enumerate(x_vals_np):
            value_equal = np.isclose(x_dat, val)
            if np.any(value_equal):  # Position of empty cells do not matter
                if w_yes:
                    y_mean_by_cat[v_idx] = np.average(
                        y_dat[value_equal], weights=w_dat[value_equal], axis=0)
                else:
                    y_mean_by_cat[v_idx] = np.average(
                        y_dat[value_equal], axis=0)
                used_values.append(v_idx)
        x_vals_np = x_vals_np[used_values]
        sort_ind = np.argsort(y_mean_by_cat[used_values])
        x_vals_np = x_vals_np[sort_ind]
        splits = x_vals_np.tolist()
        splits = splits[:-1]  # Last category not needed
    return splits


@njit
def mcf_penalty(shares_l, shares_r):
    """Generate the (unscaled) penalty.

    Parameters
    ----------
    shares_l : Numpy array. Treatment shares left.
    shares_r : Numpy array. Treatment shares right.

    Returns
    -------
    penalty : Numpy INT. Penalty of split.

    """
    diff = (shares_l - shares_r) ** 2
    penalty = 1 - (np.sum(diff) / len(shares_l))
    return penalty


def term_or_data(data_tr_ns, data_oob_ns, y_i, d_i, d_grid_i, x_i_ind_split,
                 no_of_treat, with_d_oob=True):
    """Check if terminal leaf. If not, provide data.

    Parameters
    ----------
    data_tr_ns : Numpy array. Data used for splitting.
    data_oob_ns : Numpy array. OOB Data.
    y_i : List of INT. Indices of y in data.
    d_i : List of INT. Indices of d in data.
    d_grid_i : List of INT. Indices of d_grid in data.
    x_i_ind_split : List of INT. Ind. of x used for splitting. Pos. in data.
    no_of_treat: INT.

    Returns
    -------
    y_dat : Numpy array. Data.
    y_oob : Numpy array. OOB Data.
    d_dat : Numpy array. Data.
    d_oob : Numpy array. OOB Data.
    x_dat : Numpy array. Data.
    x_oob : Numpy array. OOB Data.
    terminal : Boolean. True if no further split possible. End splitting.
    terminal2 : Boolean. Try new variables.

    """
    terminal = terminal_x = False
    y_oob = d_dat = d_oob = d_grid_dat = d_grid_oob = x_dat = x_oob = None
    x_no_variation = []
    y_dat = data_tr_ns[:, y_i]
    if np.all(np.isclose(y_dat, y_dat[0])):    # all elements are equal
        terminal = True
    else:
        y_oob = data_oob_ns[:, y_i]
        d_dat = data_tr_ns[:, d_i]
        if d_grid_i is not None:
            d_grid_dat = data_tr_ns[:, d_grid_i]
        terminal = len(np.unique(d_dat)) < no_of_treat
        if not terminal:
            if with_d_oob:
                d_oob = data_oob_ns[:, d_i]
                if d_grid_i is not None:
                    d_grid_oob = data_oob_ns[:, d_grid_i]
            x_dat = data_tr_ns[:, x_i_ind_split]
            x_no_variation = [np.all(np.isclose(x_dat[:, cols], x_dat[0, cols])
                                     ) for cols, _ in enumerate(x_i_ind_split)]
            if np.all(x_no_variation):
                terminal_x = True
            else:
                x_oob = data_oob_ns[:, x_i_ind_split]
    return (y_dat, y_oob, d_dat, d_oob, d_grid_dat, d_grid_oob, x_dat, x_oob,
            terminal, terminal_x, x_no_variation)


def oob_in_tree(obs_in_leaf, y_dat, y_nn, d_dat, w_dat, mtot, no_of_treat,
                treat_values, w_yes, cont=False):
    """Compute OOB values for a tree.

    Parameters
    ----------
    obs_in_leaf : List of int. Terminal leaf no of observation
    y : Numpy array.
    y_nn : Numpy array.
    d : Numpy array.
    w : Numpy array.
    mtot : INT. Method used.
    no_of_treat : INT.
    treat_values : INT.
    w_yes : INT.
    cont : Boolean. Default is False.

    Returns
    -------
    oob_tree : INT. OOB value of the MSE of the tree

    """
    leaf_no = np.unique(obs_in_leaf[:, 1])
    oob_tree = n_lost = n_total = 0
    mse_mce_tree = np.zeros((no_of_treat, no_of_treat))
    obs_t_tree = np.zeros(no_of_treat)
    for leaf in leaf_no:
        in_leaf = obs_in_leaf[:, 1] == leaf
        w_l = w_dat[in_leaf] if w_yes else 0
        n_l = np.count_nonzero(in_leaf)
        d_dat_in_leaf = d_dat[in_leaf]  # makes a copy
        if n_l < no_of_treat:
            enough_data_in_leaf = False
        else:
            enough_data_in_leaf = True
            if n_l < 40:          # this is done for efficiency reasons
                if set(d_dat_in_leaf.reshape(-1)) < set(treat_values):
                    enough_data_in_leaf = False
            else:
                if len(np.unique(d_dat_in_leaf)) < no_of_treat:  # No MSE
                    enough_data_in_leaf = False
        if enough_data_in_leaf:
            mse_mce_leaf, _, obs_by_treat_leaf = mcf_add.mcf_mse(
                y_dat[in_leaf], y_nn[in_leaf], d_dat_in_leaf, w_l, n_l,
                mtot, no_of_treat, treat_values, w_yes, cont)
            mse_mce_tree, obs_t_tree = add_rescale_mse_mce(
                mse_mce_leaf, obs_by_treat_leaf, mtot, no_of_treat,
                mse_mce_tree, obs_t_tree)
        else:
            n_lost += n_l
        n_total += n_l
    mse_mce_tree = mcf_add.get_avg_mse_mce(mse_mce_tree, obs_t_tree, mtot,
                                           no_of_treat)
    oob_tree = compute_mse_mce(mse_mce_tree, mtot, no_of_treat)
    return oob_tree
