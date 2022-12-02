"""Created on Fri Apr  3 11:05:15 2020.

Contains the functions needed for the tree and forest computations of MCF
@author: MLechner
-*- coding: utf-8 -*-
"""
from concurrent import futures
import math
import copy
from dask.distributed import Client, as_completed
from numba import njit

import numpy as np
import ray

from mcf import general_purpose as gp
from mcf import general_purpose_system_files as gp_sys
from mcf import mcf_general_purpose as mcf_gp
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_forest_add_functions as mcf_forest_add


def build_forest(indatei, v_dict, v_x_type, v_x_values, c_dict, regrf=False):
    """Build MCF (not yet populated by w and outcomes).

    Parameters
    ----------
    datendatei : string. Data  contained in csv file
    v : Dictionary. Variable names.
    v_x_type : Dictionary. Key: Variable name. Values: 0,1,2
    v_x_values: Dictionary. Key: Variable name. Values: List with INT/Float
    c : Dictionary. Control parameters
    regrf: Boolean. True if regression random forest. Default=False.

    Returns
    -------
    forest_final : Dictionary. All info needed for the forest estimated
    x_name: List. Order of x_name as used by tree building

    """
    old_ray_or_dask = c_dict['_ray_or_dask']
    if c_dict['no_ray_in_forest_building'] and c_dict['_ray_or_dask'] == 'ray':
        if c_dict['with_output'] and c_dict['verbose']:
            print('No use of ray in forest building.')
        c_dict['_ray_or_dask'] = 'dask'
    (x_name, x_type, x_values, c_dict, pen_mult, data_np, y_i, y_nn_i, x_i,
     x_ind, x_ai_ind, d_i, w_i, cl_i, d_grid_i
     ) = mcf_data.prepare_data_for_forest(indatei, v_dict, v_x_type,
                                          v_x_values, c_dict, regrf=regrf)
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        maxworkers = (mcf_gp.find_no_of_workers(c_dict['no_parallel'],
                                                c_dict['sys_share'])
                      if c_dict['mp_automatic'] else c_dict['no_parallel'])
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1:
        forest = [None] * c_dict['boot']
        for idx in range(c_dict['boot']):
            forest[idx] = build_tree_mcf(
                data_np, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i, x_type,
                x_values, x_ind, x_ai_ind, c_dict, idx, pen_mult, regrf)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, c_dict['boot'])
    else:
        if (c_dict['mem_object_store_1'] is not None
                and c_dict['_ray_or_dask'] == 'ray'):
            boot_by_boot = 1
        else:
            boot_by_boot = c_dict['boot_by_boot']
        forest = []
        if boot_by_boot == 1:
            if c_dict['_ray_or_dask'] == 'ray':
                if c_dict['mem_object_store_1'] is None:
                    if not ray.is_initialized():
                        ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    if not ray.is_initialized():
                        ray.init(
                            num_cpus=maxworkers, include_dashboard=False,
                            object_store_memory=c_dict['mem_object_store_1'])
                    if c_dict['with_output'] and c_dict['verbose']:
                        print("Size of Ray Object Store: ", round(
                            c_dict['mem_object_store_1']/(1024*1024)), " MB")
                data_np_ref = ray.put(data_np)
                still_running = [ray_build_tree_mcf.remote(
                    data_np_ref, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                    x_type, x_values, x_ind, x_ai_ind, c_dict, boot, pen_mult,
                    regrf) for boot in range(c_dict['boot'])]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for ret_all_i in finished_res:
                        forest.append(ret_all_i)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, c_dict['boot'])
                        jdx += 1
                    if jdx % 50 == 0:   # every 50'th tree
                        gp_sys.auto_garbage_collect(50)  # do if half mem full
                if 'refs' in c_dict['_mp_ray_del']:
                    del data_np_ref
                if 'rest' in c_dict['_mp_ray_del']:
                    del finished_res, finished
                if c_dict['_mp_ray_shutdown']:
                    ray.shutdown()
            elif c_dict['_ray_or_dask'] == 'dask':
                with Client(n_workers=maxworkers) as clt:
                    data_np_ref = clt.scatter(data_np)
                    ret_fut = [clt.submit(
                        build_tree_mcf, data_np_ref, y_i, y_nn_i, x_i, d_i,
                        d_grid_i, cl_i, w_i, x_type, x_values, x_ind, x_ai_ind,
                        c_dict, boot, pen_mult, regrf)
                            for boot in range(c_dict['boot'])]
                    jdx = 0
                    for _, res in as_completed(ret_fut, with_results=True):
                        jdx += 1
                        forest.append(res)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, c_dict['boot'])
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        build_tree_mcf, data_np, y_i, y_nn_i, x_i, d_i,
                        d_grid_i, cl_i, w_i, x_type, x_values, x_ind, x_ai_ind,
                        c_dict, boot, pen_mult, regrf):
                            boot for boot in range(c_dict['boot'])}
                    for jdx, frx in enumerate(futures.as_completed(ret_fut)):
                        forest.append(frx.result())
                        del ret_fut[frx]
                        del frx
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, c_dict['boot'])
        else:
            if (c_dict['boot'] / maxworkers) > c_dict['boot_by_boot']:
                no_of_split = round(c_dict['boot'] / c_dict['boot_by_boot'])
            else:
                no_of_split = maxworkers
            boot_indx_list = np.array_split(range(c_dict['boot']), no_of_split)
            if c_dict['with_output'] and c_dict['verbose']:
                print('Avg. number of bootstraps per process:',
                      f' {round(c_dict["boot"] / no_of_split, 2)}')
            if c_dict['_ray_or_dask'] == 'ray':
                if c_dict['mem_object_store_1'] is None:
                    if not ray.is_initialized():
                        ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    if not ray.is_initialized():
                        ray.init(
                            num_cpus=maxworkers, include_dashboard=False,
                            object_store_memory=c_dict['mem_object_store_1'])
                data_np_ref = ray.put(data_np)
                still_running = [ray_build_many_trees_mcf.remote(
                    data_np_ref, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                    x_type, x_values, x_ind, x_ai_ind, c_dict, boot, pen_mult,
                    regrf) for boot in boot_indx_list]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for ret_all_i in finished_res:
                        forest.extend(ret_all_i)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, len(boot_indx_list))
                        jdx += 1
                    if jdx % 50 == 0:   # every 50'th tree
                        gp_sys.auto_garbage_collect(50)  # do if 0.5 mem. full
                if 'refs' in c_dict['_mp_ray_del']:
                    del data_np_ref
                if 'rest' in c_dict['_mp_ray_del']:
                    del finished_res, finished
                if c_dict['_mp_ray_shutdown']:
                    ray.shutdown()
            elif c_dict['_ray_or_dask'] == 'dask':
                with Client(n_workers=maxworkers) as clt:
                    data_np_ref = clt.scatter(data_np)
                    ret_fut = [clt.submit(
                        build_many_trees_mcf, data_np_ref, y_i, y_nn_i, x_i,
                        d_i, d_grid_i, cl_i, w_i, x_type, x_values, x_ind,
                        x_ai_ind, c_dict, boot, pen_mult, regrf)
                        for boot in boot_indx_list]
                    jdx = 0
                    for _, res in as_completed(ret_fut, with_results=True):
                        jdx += 1
                        forest.extend(res)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, len(boot_indx_list))
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        build_many_trees_mcf, data_np, y_i, y_nn_i, x_i, d_i,
                        d_grid_i, cl_i, w_i, x_type, x_values, x_ind, x_ai_ind,
                        c_dict, boot, pen_mult, regrf):
                            boot for boot in boot_indx_list}
                    for jdx, frx in enumerate(futures.as_completed(ret_fut)):
                        forest.extend(frx.result())
                        del ret_fut[frx]
                        del frx
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, len(boot_indx_list))
        if len(forest) != c_dict['boot']:
            raise Exception('Forest has wrong size: ', len(forest),
                            'Bug in Multiprocessing.')
    # find best forest given the saved oob values
    forest_final, m_n_final = best_m_n_min_alpha_reg(forest, c_dict)
    del forest    # Free memory
    # Describe final tree
    if c_dict['with_output']:
        mcf_forest_add.describe_forest(forest_final, m_n_final, v_dict, c_dict,
                                       pen_mult, regrf)
    if c_dict['no_ray_in_forest_building'] and old_ray_or_dask == 'ray':
        c_dict['_ray_or_dask'] = old_ray_or_dask
    return forest_final, x_name


def oob_in_tree(obs_in_leaf, y_dat, y_nn, d_dat, w_dat, mtot, no_of_treat,
                treat_values, w_yes, regrf=False, cont=False):
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
    regrf : Boolean. Default is False.
    cont : Boolean. Default is False.

    Returns
    -------
    oob_tree : INT. OOB value of the MSE of the tree

    """
    leaf_no = np.unique(obs_in_leaf[:, 1])
    oob_tree = n_lost = n_total = 0
    if not regrf:
        mse_mce_tree = np.zeros((no_of_treat, no_of_treat))
        obs_t_tree = np.zeros(no_of_treat)
    for leaf in leaf_no:
        in_leaf = obs_in_leaf[:, 1] == leaf
        w_l = w_dat[in_leaf] if w_yes else 0
        n_l = np.count_nonzero(in_leaf)
        if regrf:
            if n_l > 1:
                mse_oob = regrf_mse(y_dat[in_leaf],  w_l, n_l, w_yes)
                oob_tree += mse_oob * n_l
        else:
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
                mse_mce_leaf, _, obs_by_treat_leaf = mcf_mse(
                    y_dat[in_leaf], y_nn[in_leaf], d_dat_in_leaf, w_l, n_l,
                    mtot, no_of_treat, treat_values, w_yes, cont)
                mse_mce_tree, obs_t_tree = add_rescale_mse_mce(
                    mse_mce_leaf, obs_by_treat_leaf, mtot, no_of_treat,
                    mse_mce_tree, obs_t_tree)
            else:
                n_lost += n_l
            n_total += n_l
    if not regrf:
        mse_mce_tree = get_avg_mse_mce(mse_mce_tree, obs_t_tree, mtot,
                                       no_of_treat)
        oob_tree = compute_mse_mce(mse_mce_tree, mtot, no_of_treat)
    return oob_tree


def best_m_n_min_alpha_reg(forest, c_dict):
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
    m_n_min_ar_combi = []
    grid_for_m = ([c_dict['grid_m']] if np.size(c_dict['grid_m']) == 1
                  else c_dict['grid_m'])
    grid_for_n_min = ([c_dict['grid_n_min']]
                      if np.size(c_dict['grid_n_min']) == 1
                      else c_dict['grid_n_min'])
    grid_for_alpha_reg = ([c_dict['grid_alpha_reg']]
                          if np.size(c_dict['grid_alpha_reg']) == 1
                          else c_dict['grid_alpha_reg'])
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                m_n_min_ar_combi.append([m_idx, n_min, alpha_reg])
    dim_m_n_min_ar = np.size(c_dict['grid_m']) * np.size(
        c_dict['grid_n_min']) * np.size(c_dict['grid_alpha_reg'])
    if (dim_m_n_min_ar) > 1:       # Find best of trees
        mse_oob = np.zeros(dim_m_n_min_ar)
        trees_without_oob = np.zeros(dim_m_n_min_ar)
        if c_dict['d_type'] == 'continuous':
            no_of_treat = 2
        else:
            no_of_treat = c_dict['no_of_treat']
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
                                    leaf[7], leaf[6], c_dict['mtot'],
                                    no_of_treat, mse_mce_tree, obs_t_tree)
                if n_lost > 0:
                    if no_of_treat is None:
                        tree_mse = tree_mse * n_total / (n_total - n_lost)
                    else:
                        if (n_total - n_lost) < 1:
                            trees_without_oob[j] += 1
                if no_of_treat is not None:
                    mse_mce_tree = get_avg_mse_mce(
                        mse_mce_tree, obs_t_tree, c_dict['mtot'], no_of_treat)
                    tree_mse = compute_mse_mce(mse_mce_tree, c_dict['mtot'],
                                               no_of_treat)
                mse_oob[j] += tree_mse     # Add MSE to MSE of forest j
        if np.any(trees_without_oob) > 0:
            for j, trees_without_oob_j in enumerate(trees_without_oob):
                if trees_without_oob_j > 0:
                    mse_oob[j] = mse_oob[j] * (
                        c_dict['boot'] / (c_dict['boot']
                                          - trees_without_oob_j))
        min_i = np.argmin(mse_oob)
        mse_oob = mse_oob / c_dict['boot']
        if not isinstance(c_dict['grid_n_min'], (list, tuple, np.ndarray)):
            c_dict['grid_n_min'] = [c_dict['grid_n_min']]
        if not isinstance(c_dict['grid_m'], (list, tuple, np.ndarray)):
            c_dict['grid_m'] = [c_dict['grid_m']]
        if not isinstance(c_dict['grid_alpha_reg'], (list, tuple, np.ndarray)):
            c_dict['grid_alpha_reg'] = [c_dict['grid_alpha_reg']]
        if c_dict['with_output']:
            print('\n')
            print('-' * 80,
                  '\nOOB MSE (without penalty) for M_try, minimum leafsize',
                  ' and alpha_reg combinations', '\n')
            j = 0
            print('\nNumber of vars / min. leaf size / alpha reg. / OOB value',
                  'Trees without OOB')
            for m_idx in c_dict['grid_m']:
                for n_min in c_dict['grid_n_min']:
                    for alpha_reg in c_dict['grid_alpha_reg']:
                        print(f'{m_idx:>12}', f'{n_min:>12} {alpha_reg:15.3f}',
                              f' {mse_oob[j]:8.3f}',
                              f' {trees_without_oob[j]:4.0f}')
                        j += 1
            print(f'Minimum OOB MSE:      {mse_oob[min_i]:8.3f}')
            print('Number of variables: ', m_n_min_ar_combi[min_i][0])
            print('Minimum leafsize:    ', m_n_min_ar_combi[min_i][1])
            print('Alpha regularity:    ', m_n_min_ar_combi[min_i][2])
            print('-' * 80)
        # forest_final = []
        # for trees_m_n_min in forest:
        #     forest_final.append(trees_m_n_min[min_i])
        forest_final = [trees_m_n_min[min_i] for trees_m_n_min in forest]
        m_n_min_ar_opt = m_n_min_ar_combi[min_i]
    else:       # Find best of trees
        # forest_final = []
        # for trees_m_n_min_ar in forest:
        #     forest_final.append(trees_m_n_min_ar[0])
        forest_final = [trees_m_n_min_ar[0] for trees_m_n_min_ar in forest]
        m_n_min_ar_opt = m_n_min_ar_combi[0]
    return forest_final, m_n_min_ar_opt


def regrf_mse(y_dat, w_dat, obs, w_yes):
    """Compute average mse for the data passed. Regression Forest.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    w_dat : Numpy Nx1 vector. Weights.
    obs : INT. Leaf size for this split.
    w_yes: Boolean. Weighted estimation.

    Returns
    -------
    mse : Mean squared error.

    """
    if w_yes:
        y_mean = np.average(y_dat, weights=w_dat, axis=0)
        mse = np.average(np.square(y_dat - y_mean), weights=w_dat, axis=0)
    else:
        y_dat = y_dat.reshape(-1)
        mse = np.inner(y_dat, y_dat) / obs - np.mean(y_dat)**2
    return mse


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
    if w_yes:
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
            # mse_mce[m_idx, m_idx] = mse_m
        if mtot in (1, 3, 4):
            # elif mtot == 3:
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
    treat_shares = np.zeros(no_of_treat) if mtot in (1, 4) else np.zeros(1)
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
                else:
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


def add_mse_mce_split(mse_mce_l, mse_mce_r, obs_by_treat_l, obs_by_treat_r,
                      mtot, no_of_treat):
    """Sum up MSE parts of use in splitting rule."""
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat = np.empty(no_of_treat)
    for m_idx in range(no_of_treat):
        obs_by_treat[m_idx] = obs_by_treat_l[m_idx] + obs_by_treat_r[m_idx]
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


def term_or_data(data_tr_ns, data_oob_ns, y_i, d_i, d_grid_i, x_i_ind_split,
                 no_of_treat, regrf=False, with_d_oob=True):
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
        if not regrf:
            d_dat = data_tr_ns[:, d_i]
            if d_grid_i is not None:
                d_grid_dat = data_tr_ns[:, d_grid_i]
            terminal = len(np.unique(d_dat)) < no_of_treat
        if not terminal:
            if not regrf:
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


def next_split(current_node, data_tr, data_oob, y_i, y_nn_i, d_i, d_grid_i,
               x_i, w_i, x_type, x_values, x_ind, x_ai_ind, c_dict, mmm, n_min,
               alpha_reg, pen_mult, trl, rng, regrf=False):
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
    c_dict : DICT. Parameters.
    mmm : INT. Number of X-variables to choose for splitting.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. Alpha regularity.
    pen_mult: Float. Penalty multiplier.
    trl: Int. Tree length.
    rng: Numpy default random number generator object.
    regrf: Boolean. Regression Random Forest. Default is False.

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
    elif not regrf and np.all(data_tr_ns[:, d_i] == data_tr_ns[0, d_i]):
        terminal = True
    else:
        if not regrf and current_node[5] < 200:  # Otherwise, too slow:
            if c_dict['d_type'] == 'continuous':
                terminal = not (2 <= np.sum(data_tr_ns[:, d_i] == 0)
                                <= current_node[5] - 2)
            else:
                ret = np.unique(data_tr_ns[:, d_i], return_counts=True)
                terminal = (len(ret[0]) < c_dict['no_of_treat']
                            or np.any(ret[1] < 2 * c_dict['n_min_treat']))
    mtot, w_yes = c_dict['mtot'], c_dict['w_yes']
    if c_dict['d_type'] == 'continuous':
        no_of_treat, d_values, continuous = 2, [0, 1], True
        d_split_in_x_ind = np.max(x_ind) + 1
    else:
        no_of_treat, d_values = c_dict['no_of_treat'], c_dict['d_values']
        d_bin_dat, continuous = None, False
    if not terminal:
        obs_min = max([round(current_node[5] * alpha_reg), n_min])
        best_mse = math.inf   # Initialisation: Infinity as default values
        x_ind_split = rnd_variable_for_split(x_ind, x_ai_ind, c_dict, mmm, rng)
        x_type_split = x_type[x_ind_split].copy()
        x_values_split = [x_values[v_idx].copy() for v_idx in x_ind_split]
        # Check if split is possible ... sequential order to minimize costs
        # Check if enough variation in the data to do splitting (costly)
        if regrf:
            (y_dat, _, _, _, _, _, x_dat, x_oob, terminal, terminal_x,
             x_no_varia) = term_or_data(
                 data_tr_ns, data_oob_ns, y_i, d_i, d_grid_i, x_i[x_ind_split],
                 None, regrf=True, with_d_oob=False)
        else:
            with_d_oob = continuous
            (y_dat, _, d_dat, d_oob, d_grid_dat, _, x_dat, x_oob,
             terminal, terminal_x, x_no_varia) = term_or_data(
                 data_tr_ns, data_oob_ns, y_i, d_i, d_grid_i, x_i[x_ind_split],
                 no_of_treat, regrf=False, with_d_oob=with_d_oob)
        if terminal_x:
            terminal = True  # No variation in drawn X. Splitting stops.
    if not terminal:         # ML 25.5.2022
        if not regrf:
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
                        current_node[5], c_dict, rng=rng)
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
                    if not regrf:
                        d_dat_l = (d_bin_dat[leaf_l]
                                   if continuous else d_dat[leaf_l])
                        if continuous or c_dict['n_min_treat'] == 1:
                            if len(np.unique(d_dat_l)) < no_of_treat:
                                continue
                        else:
                            ret = np.unique(d_dat_l, return_counts=True)
                            if len(ret[0]) < no_of_treat:
                                continue
                            if np.any(ret[1] < c_dict['n_min_treat']):
                                continue
                    leaf_r = np.invert(leaf_l)  # Reverses True to False
                    if not regrf:
                        d_dat_r = (d_bin_dat[leaf_r]
                                   if continuous else d_dat[leaf_r])
                        if continuous or c_dict['n_min_treat'] == 1:
                            if len(np.unique(d_dat_r)) < no_of_treat:
                                continue   # Splits possible?
                        else:
                            ret = np.unique(d_dat_r, return_counts=True)
                            if len(ret[0]) < no_of_treat:
                                continue
                            if np.any(ret[1] < c_dict['n_min_treat']):
                                continue
                    leaf_oob_r = np.invert(leaf_oob_l)
                    if not regrf:
                        if mtot in (1, 4):
                            if continuous:
                                y_nn_l = mcf_forest_add.match_cont(
                                    d_grid_dat[leaf_l], y_nn[leaf_l, :],
                                    c_dict['ct_grid_nn_val'], rng)
                                y_nn_r = mcf_forest_add.match_cont(
                                    d_grid_dat[leaf_r], y_nn[leaf_r, :],
                                    c_dict['ct_grid_nn_val'], rng)
                            else:
                                y_nn_l = y_nn[leaf_l, :]
                                y_nn_r = y_nn[leaf_r, :]
                        else:
                            y_nn_l = y_nn_r = 0
                    if w_yes:
                        w_l, w_r = w_dat[leaf_l], w_dat[leaf_r]
                    else:
                        w_l = w_r = 0
                    # compute objective functions given particular method
                    if regrf:
                        mse_l = regrf_mse(y_dat[leaf_l],  w_l, n_l, w_yes)
                        mse_r = regrf_mse(y_dat[leaf_r],  w_r, n_r, w_yes)
                        mse_split = (mse_l * n_l + mse_r * n_r) / (n_l + n_r)
                    else:
                        mse_mce_l, shares_l, obs_by_treat_l = mcf_mse(
                            y_dat[leaf_l], y_nn_l, d_dat_l, w_l, n_l, mtot,
                            no_of_treat, d_values, w_yes)
                        mse_mce_r, shares_r, obs_by_treat_r = mcf_mse(
                            y_dat[leaf_r], y_nn_r, d_dat_r, w_r, n_r, mtot,
                            no_of_treat, d_values, w_yes)
                        mse_mce = add_mse_mce_split(
                            mse_mce_l, mse_mce_r, obs_by_treat_l,
                            obs_by_treat_r, mtot, no_of_treat)
                        mse_split = compute_mse_mce(mse_mce, mtot, no_of_treat)
                    # add penalty for this split
                    if not regrf:
                        if (c_dict['mtot'] == 1) or ((c_dict['mtot'] == 4) and
                                                     (rng.random() > 0.5)):
                            penalty = mcf_penalty(shares_l, shares_r)
                            mse_split = mse_split + pen_mult * penalty
                    if mse_split < best_mse:
                        split_done = True
                        best_mse = mse_split
                        best_var_i = copy.copy(x_ind_split[j])
                        best_type = copy.copy(x_type_split[j])
                        best_n_l, best_n_r = n_l, n_r
                        best_leaf_l = np.copy(leaf_l)
                        best_leaf_r = np.copy(leaf_r)
                        best_leaf_oob_l = np.copy(leaf_oob_l)
                        best_leaf_oob_r = np.copy(leaf_oob_r)
                        best_n_oob_l, best_n_oob_r = n_oob_l, n_oob_r
                        best_value = (copy.copy(val) if best_type == 0 else
                                      split_values_unord_j[:])  # left
    if not split_done:
        terminal = True
    if terminal:
        current_node[4] = 1  # terminal
        w_oob = data_oob_ns[:, [w_i]] if w_yes else 0
        n_oob = np.copy(current_node[6])
        if regrf:
            if n_oob > 1:
                current_node[7] = regrf_mse(data_oob_ns[:, y_i],  w_oob, n_oob,
                                            w_yes)
            elif n_oob == 1:
                current_node[7] = 0
            else:
                current_node[7] = None      # MSE cannot be computed
        else:
            if continuous:
                d_oob = data_oob_ns[:, d_i] > 1e-15
            else:
                d_oob = data_oob_ns[:, d_i]
            if len(np.unique(d_oob)) < no_of_treat:
                current_node[7] = None      # MSE cannot be computed
            else:
                if continuous:
                    y_nn = mcf_forest_add.match_cont(
                        data_oob_ns[:, d_grid_i], data_oob_ns[:, y_nn_i],
                        c_dict['ct_grid_nn_val'], rng)
                else:
                    y_nn = data_oob_ns[:, y_nn_i]
                current_node[7], shares_r, current_node[6] = mcf_mse(
                    data_oob_ns[:, y_i], y_nn, d_oob, w_oob, n_oob, mtot,
                    no_of_treat, d_values, w_yes)
        current_node[11] = current_node[12] = 0  # Data no longer needed
        newleaf_l, newleaf_r = [], []
    else:
        newleaf_l = copy.deepcopy(current_node)
        newleaf_r = copy.deepcopy(current_node)
        newleaf_l[0], newleaf_r[0] = trl, trl + 1  # Tree length, starts with 0
        newleaf_l[1] = copy.deepcopy(current_node[0])  # Parent nodes
        newleaf_r[1] = copy.deepcopy(current_node[0])
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
        current_node[2] = copy.copy(newleaf_l[0])  # ID of daughter leaf
        current_node[3] = copy.copy(newleaf_r[0])
        current_node[4] = 0     # not active, not terminal - intermediate
        current_node[8] = copy.copy(best_var_i)
        if best_type > 0:  # Save as product of primes
            best_value = gp.list_product(best_value)   # int
        current_node[9] = copy.copy(best_value)    # <= -> left
        current_node[10] = copy.copy(best_type)
        current_node[11] = current_node[12] = 0   # Data, no longer needed
        if current_node[0] != 0:
            current_node[16] = 0
        else:    # Need to keep OOB data in first leaf for VIB, Feature select
            if (not c_dict['var_import_oob']) and (not c_dict['fs_yes']):
                current_node[16] = 0    # Data, no longer needed, saves memory
    return newleaf_l, newleaf_r, current_node, terminal


def rnd_variable_for_split(x_ind_pos, x_ai_ind_pos, c_dict, mmm, rng):
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
    if c_dict['m_random_poisson']:
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


def build_single_tree(data, data_oob, y_i, y_nn_i, d_i, d_grid_i, x_i, w_i,
                      x_type, x_values, x_ind, x_ai_ind, c_dict, mmm, n_min,
                      alpha_reg, node_table, pen_mult, rng, regrf=False):
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
    c_dict : Dict. Parameters.
    m : INT. Number of covariates to be included.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. alpha regularity.
    node_table : List of list of lists. Initial tree (basic leaf)
    pen_mult: Float. Multiplier of penalty.
    rng : Default random number generator object.
    regrf: Boolean. Regression Random Forest. Default is False.

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
                current = copy.deepcopy(node_table[node_i])
                left, right, current, terminal = next_split(
                    current, data, data_oob, y_i, y_nn_i, d_i, d_grid_i, x_i,
                    w_i, x_type, x_values, x_ind, x_ai_ind, c_dict, mmm, n_min,
                    alpha_reg, pen_mult, len(node_table), rng, regrf)
                node_table[node_i] = copy.deepcopy(current)
                if not terminal:
                    active_knots += 1
                    node_table.append(copy.deepcopy(left))
                    node_table.append(copy.deepcopy(right))
        if active_knots == 0:
            continue_to_split = False  # Tree completed
    return node_table


def init_node_table(n_tr, n_oob, indices_oob):
    """Initialise Node table for first leaf.

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


@ray.remote
def ray_build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                             x_type, x_values, x_ind, x_ai_ind, c_dict,
                             boot_indices, pen_mult, regrf=False):
    """Prepare function for Ray."""
    return build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i,
                                w_i, x_type, x_values, x_ind, x_ai_ind, c_dict,
                                boot_indices, pen_mult, regrf)


def build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                         x_type, x_values, x_ind, x_ai_ind, c_dict,
                         boot_indices, pen_mult, regrf=False):
    """Build larger pieces of the forest (for MP)."""
    little_forest = []
    for boot in boot_indices:
        tree = build_tree_mcf(
            data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i, x_type, x_values,
            x_ind, x_ai_ind, c_dict, boot, pen_mult, regrf)
        little_forest.append(tree)
    return little_forest


@ray.remote
def ray_build_tree_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                       x_type, x_values, x_ind, x_ai_ind, c_dict, boot,
                       pen_mult, regrf=False):
    """Prepare function for Ray."""
    return build_tree_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                          x_type, x_values, x_ind, x_ai_ind, c_dict, boot,
                          pen_mult, regrf)


def build_tree_mcf(data, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i, x_type,
                   x_values, x_ind, x_ai_ind, c_dict, boot, pen_mult,
                   regrf=False):
    """Build single trees for all values of tuning parameters.

    Parameters
    ----------
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
    regrf: Boolean. Regression Random Forest. Default=False.

    Returns
    -------
    tree_all : LIST (m_grid x N_min_grid x alpha_grid) with trees for all
               values of tuning parameters

    """
    # split data into OOB and tree data
    n_obs = data.shape[0]
    # Random number initialisation. This seeds rnd generator within process
    rng = np.random.default_rng((10+boot)**2+121)
    if c_dict['panel_in_rf']:
        cl_unique = np.unique(data[:, cl_i])
        n_cl = cl_unique.shape[0]
        n_train = round(n_cl * c_dict['subsam_share_forest'])
        indices_cl = list(rng.choice(n_cl, size=n_train, replace=False))
        indices = [i for i in range(n_obs) if data[i, cl_i] in indices_cl]
    else:
        n_train = round(n_obs * c_dict['subsam_share_forest'])
        indices = list(rng.choice(n_obs, size=n_train, replace=False))
    data_tr, data_oob = data[indices], np.delete(data, indices, axis=0)
    n_tr, n_oob = data_tr.shape[0], data_oob.shape[0]
    node_t_init = init_node_table(n_tr, n_oob, np.delete(range(n_obs), indices,
                                                         axis=0))
    # build trees for all m,n combinations
    grid_for_m = ([c_dict['grid_m']] if np.size(c_dict['grid_m']) == 1
                  else c_dict['grid_m'])
    grid_for_n_min = ([c_dict['grid_n_min']]
                      if np.size(c_dict['grid_n_min']) == 1
                      else c_dict['grid_n_min'])
    grid_for_alpha_reg = ([c_dict['grid_alpha_reg']]
                          if np.size(c_dict['grid_alpha_reg']) == 1
                          else c_dict['grid_alpha_reg'])
    tree_all = [None] * len(grid_for_m) * len(grid_for_n_min) * len(
        grid_for_alpha_reg)
    j = 0
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                node_table_0 = copy.deepcopy(node_t_init)  # emty table
                tree_all[j] = build_single_tree(
                    data_tr, data_oob, y_i, y_nn_i, d_i, d_grid_i, x_i, w_i,
                    x_type, x_values, x_ind, x_ai_ind, c_dict, m_idx, n_min,
                    alpha_reg, node_table_0, pen_mult, rng, regrf)
                j += 1
    return tree_all


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


def get_split_values(y_dat, w_dat, x_dat, x_type, x_values, leaf_size, c_dict,
                     rng=None):
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
    if x_type == 0:
        if bool(x_values):  # Limited number of values in x_value
            min_x, max_x = np.amin(x_dat), np.amax(x_dat)
            del_values = [j for j, val in enumerate(x_values)
                          if (val < (min_x - 1e-15))
                          or (val > (max_x + 1e-15))]
            # del_values = []
            # for j, val in enumerate(x_values):
            #     if (val < (min_x - 1e-15)) or (val > (max_x + 1e-15)):
            #         del_values.append(j)
            if del_values:  # List is not empty
                splits_x = [x for x in x_values if x not in del_values]
            else:
                splits_x = x_values[:]
            if len(splits_x) > 1:
                splits_x = splits_x[:-1]
                if 0 < c_dict['random_thresholds'] < len(splits_x):
                    splits = np.unique(
                        rng.choice(splits_x, size=c_dict['random_thresholds'],
                                   replace=False, shuffle=False))
                else:
                    splits = splits_x
        else:  # Continoues variable with very many values; x_values empty
            if 0 < c_dict['random_thresholds'] < (leaf_size - 1):
                x_vals_np = rng.choice(
                    x_dat, c_dict['random_thresholds'], replace=False,
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
                if c_dict['w_yes']:
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
