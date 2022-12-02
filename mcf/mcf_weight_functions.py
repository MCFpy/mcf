"""Created on Fri Apr  3 11:05:15 2020.

Contains the functions needed for the weight computation of MCF
@author: MLechner
-*- coding: utf-8 -*-
"""
from concurrent import futures
from dask.distributed import Client, as_completed

import numpy as np
import pandas as pd
from scipy import sparse
import ray

from mcf import general_purpose as gp
from mcf import general_purpose_system_files as gp_sys
from mcf import mcf_general_purpose as mcf_gp
from mcf import mcf_forest_add_functions as mcf_forest_add


def get_weights_mp(forest, x_file, y_file, v_dict, c_dict, x_name,
                   regrf=False):
    """Get weights for obs in pred_data & outcome and cluster from y_data.

    Parameters
    ----------
    forest : Tuple of lists. Node_table.
    x_file : String. csv-file with data to make predictions for.
    y_file : String. csv-file with outcome data.
    v_dict :  Dict. Variables.
    c_dict :  Dict. Parameters.
    x_name : List of str.
    regrf : Bool. Honest regression Random Forest. Default is False.

    Returns
    -------
    weights : Tuple of lists (N_pred x 1 (x no_of_treat + 1).
    y_data : N_y x number of outcomes-Numpy array. Outcome variables.
    x_bala : N_y x number of balancing-vars-Numpy array. Balancing test vars.
    cl_dat : N_y x 1 Numpy array. Cluster number.
    w_dat: N_y x 1 Numpy array. Sampling weights (if used).
    """
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nObtaining weights from estimated forest')
    data_x = pd.read_csv(x_file)
    x_dat = data_x[x_name].to_numpy()
    del data_x
    data_y = pd.read_csv(y_file)
    y_dat = data_y[v_dict['y_name']].to_numpy()
    d_dat = None if regrf else data_y[v_dict['d_name']].to_numpy()
    if not regrf:
        d_dat = np.int16(np.round(d_dat))
    n_x, n_y = len(x_dat), len(y_dat)
    cl_dat = (data_y[v_dict['cluster_name']].to_numpy()
              if c_dict['cluster_std'] else 0)
    w_dat = data_y[v_dict['w_name']].to_numpy() if c_dict['w_yes'] else 0
    x_bala = 0
    if not regrf:
        if c_dict['balancing_test_w']:
            x_bala = data_y[v_dict['x_balance_name']].to_numpy()
    del data_y
    empty_leaf_counter = merge_leaf_counter = 0
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        if c_dict['mp_automatic']:
            maxworkers = mcf_gp.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share'])
        else:
            maxworkers = c_dict['no_parallel']
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1 or c_dict['_ray_or_dask'] == 'ray':
        mp_over_boots = False
    else:
        mp_over_boots = (False if regrf
                         else bool(c_dict['mp_type_weights'] == 2))
    no_of_treat = (len(c_dict['ct_grid_w_val'])
                   if c_dict['d_type'] == 'continuous'
                   else c_dict['no_of_treat'])
    if maxworkers == 1 or mp_over_boots:
        weights = initialise_weights(c_dict, n_x, n_y, regrf)
        split_forest = False
        for idx in range(n_x):
            results_fut_idx = weights_obs_i(
                idx, n_y, forest, x_dat, d_dat, c_dict, regrf,
                mp_over_boots, maxworkers, False)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, n_x)
            if c_dict['weight_as_sparse']:
                for d_idx in range(no_of_treat):
                    indices = results_fut_idx[1][d_idx][0]
                    weights_obs = results_fut_idx[1][d_idx][1]
                    weights[d_idx][idx, indices] = weights_obs
                    if regrf:
                        break
            else:
                weights[idx] = results_fut_idx[1]
            empty_leaf_counter += results_fut_idx[2]
            merge_leaf_counter += results_fut_idx[3]
        if c_dict['weight_as_sparse']:
            weights = weights_to_csr(weights, no_of_treat, regrf)
    else:
        no_of_splits_i, max_size_i = maxworkers, 1000
        if n_x / no_of_splits_i > max_size_i:
            while True:
                no_of_splits_i += maxworkers
                if n_x / no_of_splits_i <= max_size_i:
                    break
        if c_dict['with_output'] and c_dict['verbose']:
            print('\nOperational characteristics of weight estimation I:')
            print(f'Number of workers {maxworkers:2}')
            print(f'Number of observation chunks: {no_of_splits_i:5}')
            print('Average # of observations per chunck:',
                  f' {n_x / no_of_splits_i:5.2f}')
        all_idx_split = np.array_split(range(n_x), no_of_splits_i)
        split_forest = False
        if c_dict['_ray_or_dask'] != 'ray':
            if c_dict['mp_weights_tree_batch'] > 1:  # User def. # of batches
                no_of_boot_splits = c_dict['mp_weights_tree_batch']
                split_forest = True
                if c_dict['with_output'] and c_dict['verbose']:
                    print('\nUser determined number of tree batches')
            elif c_dict['mp_weights_tree_batch'] == 0:  # Automatic # of batch
                size_of_forest_mb = gp_sys.total_size(forest) / (1024 * 1024)
                no_of_boot_splits = mcf_gp.no_of_boot_splits_fct(
                    size_of_forest_mb, maxworkers, c_dict['with_output'])
                if no_of_boot_splits > 1:
                    split_forest = True
                else:
                    if c_dict['with_output']:
                        print('No tree batching')
        if split_forest:
            boot_indx_list = np.array_split(range(c_dict['boot']),
                                            no_of_boot_splits)
            total_bootstraps = c_dict['boot']
            if c_dict['with_output'] and c_dict['verbose']:
                print(f'Number of bootstrap chunks: {no_of_boot_splits:5}')
                print('Average # of bootstraps per chunck:',
                      f' {c_dict["boot"]/no_of_boot_splits:5.2f}')
        else:
            boot_indx_list = range(1)
        if (not c_dict['weight_as_sparse']) and split_forest:
            if c_dict['with_output'] and c_dict['verbose']:
                print('XXXXXXXXDANGERXXXX' * 3)
                print('Bootstrap splitting requires using sparse',
                      ' matrices.')
                print('Programme continues without bootstrap splitting',
                      ' but may crash due to insufficient memory.')
                print('XXXXXXXXDANGERXXXX' * 3)
        for b_i, boots_ind in enumerate(boot_indx_list):
            weights = initialise_weights(c_dict, n_x, n_y, regrf)
            if split_forest:
                # get subforest
                # forest_temp = [forest[boot] for boot in boots_ind]
                forest_temp = forest[boots_ind[0]:boots_ind[-1]+1]
                if c_dict['with_output'] and c_dict['verbose'] and b_i == 0:
                    print('Size of each submitted forest ',
                          f'{gp_sys.total_size(forest_temp)/(1024*1024):6.2f}',
                          ' MB')
                c_dict['boot'] = len(boots_ind)
                # weights über trees addieren
                if c_dict['with_output'] and c_dict['verbose']:
                    print()
                    print(f'Boot Chunk {b_i+1:2} of {no_of_boot_splits:2}')
                    gp_sys.memory_statistics()
            else:
                if c_dict['_ray_or_dask'] != 'ray':
                    forest_temp = forest
            if c_dict['_ray_or_dask'] == 'ray':
                if c_dict['mem_object_store_3'] is None:
                    if not ray.is_initialized():
                        ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    if not ray.is_initialized():
                        ray.init(
                            num_cpus=maxworkers, include_dashboard=False,
                            object_store_memory=c_dict['mem_object_store_3'])
                    if c_dict['with_output'] and c_dict['verbose']:
                        print("Size of Ray Object Store: ",
                              round(c_dict['mem_object_store_3']/(1024*1024)),
                              " MB")
                x_dat_ref = ray.put(x_dat)
                forest_ref = ray.put(forest)
                still_running = [ray_weights_many_obs_i.remote(
                    idx_list, n_y, forest_ref, x_dat_ref, d_dat, c_dict, regrf,
                    split_forest) for idx_list in all_idx_split]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for results_fut_idx in finished_res:
                        if c_dict['with_output'] and c_dict['verbose']:
                            if jdx == 0:
                                print()
                                print(f'   Obs chunk {jdx+1:2}',
                                      f' ({no_of_splits_i:2})', end='')
                            else:
                                print(f' {jdx+1:2} ({no_of_splits_i:2})',
                                      end='')
                            jdx += 1
                        for idx, val_list in enumerate(results_fut_idx[0]):
                            if c_dict['weight_as_sparse']:
                                for d_idx in range(no_of_treat):
                                    indices = results_fut_idx[1][idx][d_idx][0]
                                    weights_obs = results_fut_idx[1][idx][d_idx
                                                                          ][1]
                                    idx_resized = np.resize(indices,
                                                            (len(indices), 1))
                                    weights[d_idx][val_list, idx_resized
                                                   ] = weights_obs
                                    if regrf:
                                        break
                            else:
                                weights[val_list] = results_fut_idx[1][idx]
                        empty_leaf_counter += results_fut_idx[2]
                        merge_leaf_counter += results_fut_idx[3]
                if 'refs' in c_dict['_mp_ray_del']:
                    del x_dat_ref, forest_ref
                if 'rest' in c_dict['_mp_ray_del']:
                    del finished_res, finished
                if c_dict['_mp_ray_shutdown']:
                    ray.shutdown()
            elif c_dict['_ray_or_dask'] == 'dask':
                with Client(n_workers=maxworkers) as clt:
                    x_dat_ref = clt.scatter(x_dat)
                    d_dat_ref = clt.scatter(d_dat)
                    forest_temp_ref = clt.scatter(forest_temp)
                    ret_fut = [clt.submit(
                        weights_many_obs_i, idx_list, n_y, forest_temp_ref,
                        x_dat_ref, d_dat_ref, c_dict, regrf, split_forest)
                        for idx_list in all_idx_split]
                    jdx = 0
                    for _, res in as_completed(ret_fut, with_results=True):
                        if c_dict['with_output'] and c_dict['verbose']:
                            if jdx == 0:
                                print()
                                print(f'   Obs chunk {jdx+1:2}',
                                      f' ({no_of_splits_i:2})', end='')
                            else:
                                print(f' {jdx+1:2} ({no_of_splits_i:2})',
                                      end='')
                            jdx += 1
                        idx_list = res[0]
                        for idx, val_list in enumerate(idx_list):
                            if c_dict['weight_as_sparse']:
                                for d_idx in range(no_of_treat):
                                    indices = res[1][idx][d_idx][0]
                                    weights_obs = res[1][idx][d_idx][1]
                                    weights[d_idx][val_list, indices
                                                   ] = weights_obs
                                    if regrf:
                                        break
                            else:
                                weights[val_list] = res[1][idx]
                        empty_leaf_counter += res[2]
                        merge_leaf_counter += res[3]
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        weights_many_obs_i, idx_list, n_y, forest_temp, x_dat,
                        d_dat, c_dict, regrf, split_forest):
                        idx_list for idx_list in all_idx_split}
                    for jdx, fur in enumerate(futures.as_completed(ret_fut)):
                        results_fut_idx = fur.result()
                        del ret_fut[fur]
                        del fur
                        if c_dict['with_output'] and c_dict['verbose']:
                            if jdx == 0:
                                print()
                                print(f'   Obs chunk {jdx+1:2}',
                                      f' ({no_of_splits_i:2})', end='')
                            else:
                                print(f' {jdx+1:2} ({no_of_splits_i:2})',
                                      end='')
                        idx_list = results_fut_idx[0]
                        for idx, val_list in enumerate(idx_list):
                            if c_dict['weight_as_sparse']:
                                for d_idx in range(no_of_treat):
                                    indices = results_fut_idx[1][idx][d_idx][0]
                                    weights_obs = results_fut_idx[1][idx][d_idx
                                                                          ][1]
                                    weights[d_idx][val_list, indices
                                                   ] = weights_obs
                                    if regrf:
                                        break
                            else:
                                weights[val_list] = results_fut_idx[1][idx]
                        empty_leaf_counter += results_fut_idx[2]
                        merge_leaf_counter += results_fut_idx[3]
            if c_dict['weight_as_sparse']:
                weights = weights_to_csr(weights, no_of_treat, regrf)
            if split_forest:
                if b_i == 0:  # only in case of splitted forests
                    weights_all = weights
                else:
                    for d_idx in range(no_of_treat):
                        weights_all[d_idx] += weights[d_idx]
                        if regrf:
                            break
                if c_dict['with_output'] and c_dict['verbose']:
                    print()
                    mcf_gp.print_size_weight_matrix(
                        weights_all, c_dict['weight_as_sparse'], no_of_treat)
                    gp_sys.memory_statistics()
    if split_forest:
        c_dict['boot'] = total_bootstraps
        weights = normalize_weights(weights_all, no_of_treat, regrf,
                                    c_dict['weight_as_sparse'], n_x)
    weights = tuple(weights)
    if (((empty_leaf_counter > 0) or (merge_leaf_counter > 0))
            and c_dict['with_output']) and c_dict['verbose']:
        print('\n')
        print(f'{merge_leaf_counter:5} observations attributed in merged',
              ' leaves')
        print(f'{empty_leaf_counter:5} observations attributed to leaf w/o',
              ' observations')
    return weights, y_dat, x_bala, cl_dat, w_dat


def normalize_weights(weights, no_of_treat, regrf, sparse_m, n_x):
    """Normalise weight matrix (needed when forest is split)."""
    for d_idx in range(no_of_treat):
        if sparse_m:
            row_sum = 1 / weights[d_idx].sum(axis=1)
            for i in range(n_x):
                weights[d_idx][i, :] = weights[d_idx][i, :].multiply(
                    row_sum[i])
            weights[d_idx] = weights[d_idx].astype(np.float32,
                                                   casting='same_kind')
        else:
            for i in range(n_x):
                weights[i][d_idx][1] = weights[i][d_idx][1] / np.sum(
                    weights[i][d_idx][1])
                weights[i][d_idx][1] = weights[i][d_idx][1].astype(np.float32)
        if regrf:
            break
    return weights


def weights_to_csr(weights, no_of_treat, regrf):
    """Convert list of lil sparse matrices to csr format."""
    for d_idx in range(no_of_treat):
        weights[d_idx] = weights[d_idx].tocsr()
        if regrf:
            break
    return weights


def initialise_weights(c_dict, n_x, n_y, regrf):
    """Initialise the weights matrix."""
    no_of_treat = (len(c_dict['ct_grid_w_val'])
                   if c_dict['d_type'] == 'continuous'
                   else c_dict['no_of_treat'])
    if c_dict['weight_as_sparse']:
        weights = []
        for _ in range(no_of_treat):
            weights.append(sparse.lil_matrix((n_x, n_y), dtype=np.float32))
            if regrf:
                break
    else:
        weights = [None] * n_x
    return weights


@ray.remote
def ray_weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, c_dict,
                           regrf=False, split_forest=False):
    """Make function compatible with Ray."""
    return weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, c_dict,
                              regrf, split_forest)


def weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, c_dict,
                       regrf=False, split_forest=False):
    """
    Create chunks of task to be efficiently executed by MP.

    Parameters
    ----------
    idx_list : List of Int. Counter.
    n_y: Int. Length of training data.
    forest : List of Lists.
    x_dat : Numpy array. Prediction sample.
    d_dat: Numpy array. Training sample.
    c_dict : Dict. Parameters
    regrf: Bool. Regression random forest. Default is False.
    split_forest : Boolean. True if chunks of bootstraps. Default is False.

    Returns
    -------
    idx : List of Int. Int. Counter.
    weights_i : List of lists.
    empty_leaf_counter : Int.
    merge_leaf_counter : Int.

    """
    weights = [None] * len(idx_list)
    empty_leaf_counter = merge_leaf_counter = 0
    for idx, val in enumerate(idx_list):
        results_fut_idx = weights_obs_i(
            val, n_y, forest, x_dat, d_dat, c_dict, regrf, mp_over_boots=False,
            maxworkers=1, split_forest=split_forest)
        weights[idx] = results_fut_idx[1]
        empty_leaf_counter += results_fut_idx[2]
        merge_leaf_counter += results_fut_idx[3]
    return idx_list, weights, empty_leaf_counter, merge_leaf_counter


def weights_obs_i(idx, n_y, forest, x_dat, d_dat, c_dict, regrf=False,
                  mp_over_boots=False, maxworkers=1, split_forest=False):
    """
    Compute weight for single observation to predict.

    Parameters
    ----------
    idx : Int. Counter.
    n_y: Int. Length of training data.
    forest : List of Lists.
    x_dat : Numpy array. Prediction sample.
    d_dat: Numpy array. Training sample.
    c_dict : Dict. Parameters
    regrf: Bool. Regression random forest. Default is False.
    mp_over_boots : Bool. Multiprocessing at level of bootstraps.
                          Default is False.
    maxworkers: Int. Number of workers if MP.
    split_forest : Boolean. True if chunks of bootstraps. Default is False.

    Returns
    -------
    idx : Int. Counter.
    weights_i : List of lists.
    empty_leaf_counter : Int.
    merge_leaf_counter : Int.

    """
    empty_leaf_counter = merge_leaf_counter = 0
    if c_dict['d_type'] == 'continuous':
        no_of_treat = len(c_dict['ct_grid_w_val'])
        d_values = c_dict['ct_grid_w_val']
        continuous, regrf = True, False
    else:
        no_of_treat, d_values = c_dict['no_of_treat'], c_dict['d_values']
        continuous = False
    if regrf:
        weights_i_np = np.zeros((n_y, 2))
        weights_i = [None]
        continuous = False
    else:
        weights_i_np = np.zeros((n_y, no_of_treat + 1))
        weights_i = [None] * no_of_treat  # 1 list for each treatment
    weights_i_np[:, 0] = np.arange(n_y)  # weight for index of outcomes
    x_dat_i = x_dat[idx, :]
    if mp_over_boots:
        with futures.ProcessPoolExecutor(max_workers=maxworkers) as fpp:
            ret_fut = {fpp.submit(
                weights_obs_i_inside_boot, forest[boot], x_dat_i, regrf, n_y,
                no_of_treat, d_values, d_dat, continuous):
                    boot for boot in range(c_dict['boot'])}
            for fur in futures.as_completed(ret_fut):
                result = fur.result()
                del ret_fut[fur]
                del fur
                if result[1]:
                    empty_leaf_counter += 1
                else:
                    weights_i_np[:, 1:] += result[0]
    else:
        for boot in range(c_dict['boot']):
            weights_ij_np, empty_leaf = weights_obs_i_inside_boot(
                forest[boot], x_dat_i, regrf, n_y, no_of_treat, d_values,
                d_dat, continuous)
            if empty_leaf:
                empty_leaf_counter += 1
            else:
                weights_i_np[:, 1:] += weights_ij_np
    obs_without_leaf = 1 if empty_leaf_counter == c_dict['boot'] else 0
    normalize = not split_forest
    if regrf:
        weights_i = final_trans(weights_i_np, None, regrf, normalize)
    else:
        weights_i = final_trans(weights_i_np, no_of_treat, regrf, normalize)
    return idx, weights_i, obs_without_leaf, merge_leaf_counter


def weights_obs_i_inside_boot(forest_b, x_dat_i, regrf, n_y, no_of_treat,
                              d_values, d_dat, continuous):
    """Allow for MP at bootstrap level (intermediate procedure)."""
    # here is the information for the different treatments needed
    # Problem ist dass man hier für die x alle potential outcomes berechnet
    # Bei continuous treatments gehören aber verschiedene leafs zu den gleichen
    # X. Ausserdem werden d=0 und die anderen Werte von d verschieden behandelt
    # x_dat_i ist eine design matrix, die dann auch die verschiedenen werte von
    # d enthalten sollte
    # d = 0 zu setzen scheint aber überflüssig bei der designmatrix bzgl. d
    if continuous:
        leaf_id_list = []
        # We get a list of leafs that contains relevant elements, some of them
        # may be identical, but this does not matter
        for treat in d_values[1:]:
            x_dat_i_t = np.append(x_dat_i, treat)
            leaf_id = (mcf_forest_add.get_terminal_leaf_no(forest_b,
                                                           x_dat_i_t))
            leaf_id_list.append(leaf_id)
            if forest_b[leaf_id][14] is None:  # Leave will be ignored
                empty_leaf, weights_ij_np = True, 0
                # If any of the subleaves is None, then stop ...
                break
            empty_leaf = False

    else:
        leaf_id = mcf_forest_add.get_terminal_leaf_no(forest_b, x_dat_i)
        if forest_b[leaf_id][14] is None:  # Leave will be ignored
            empty_leaf, weights_ij_np = True, 0
        else:
            empty_leaf = False
    if not empty_leaf:
        if continuous:
            fb_lid_14_list = [forest_b[leaf_id][14]
                              for leaf_id in leaf_id_list]
        else:
            fb_lid_14 = forest_b[leaf_id][14]
        if regrf:  # continuous is set to False in previous function
            weights_ij_np = np.zeros((n_y, 1))
            n_x_i = len(fb_lid_14)
            weights_ij_np[fb_lid_14, 0] += 1 / n_x_i
        else:
            weights_ij_np = np.zeros((n_y, no_of_treat))
            if continuous:
                # We need to collect information over various leafs for the 0
                # For the other leaves, the leaves are treatment specific
                leaf_0_complete, leaf_pos_complete = False, True
                # Zuerst 0 einsammeln
                for jdx, fb_lid_14 in enumerate(fb_lid_14_list):
                    # fb_lid_14 = fb_lid_14_list[jdx]
                    d_ib = d_dat[fb_lid_14].reshape(-1)  # view
                    indices_ibj_0 = d_ib < 1e-15
                    indices_ibj_pos = d_ib >= 1e-15
                    if np.any(indices_ibj_0):  # any valid observations?
                        fb_lid_14_indi_0 = fb_lid_14[indices_ibj_0]
                        n_x_i_0 = len(fb_lid_14_indi_0)
                        weights_ij_np[fb_lid_14_indi_0, 0] += 1 / n_x_i_0
                        leaf_0_complete = True
                    if np.any(indices_ibj_pos):  # any valid observations?
                        fb_lid_14_indi_pos = fb_lid_14[indices_ibj_pos]
                        n_x_i_pos = len(fb_lid_14_indi_pos)
                        weights_ij_np[fb_lid_14_indi_pos, jdx+1] += (
                            1 / n_x_i_pos)
                    else:
                        leaf_pos_complete = False
                        break
                leaf_complete = leaf_0_complete and leaf_pos_complete
            else:
                d_ib = d_dat[fb_lid_14].reshape(-1)  # view
                leaf_complete = True
                for jdx, treat in enumerate(d_values):
                    indices_ibj = d_ib == treat
                    if np.any(indices_ibj):  # any valid observations?
                        fb_lid_14_indi = fb_lid_14[indices_ibj]
                        n_x_i = len(fb_lid_14_indi)
                        weights_ij_np[fb_lid_14_indi, jdx] += 1 / n_x_i
                    else:
                        leaf_complete = False
                        break
            if not leaf_complete:
                empty_leaf, weights_ij_np = True, 0
    return weights_ij_np, empty_leaf


def final_trans(weights_i_np, no_of_treat, regrf, normalize=True):
    """
    Compute last transformations of (positive only) weights.

    Parameters
    ----------
    weights_i_np : 2D Numpy array. Weights including zeros.
    no_of_treat : Int. Number of treatments.
    regrf : Bool.
    normalize : Bool. Normalize weights to row sum of 1. Default is True.

    Returns
    -------
    weights_i: List of lists.

    """
    if regrf:
        iterator, weights_i = 1, [None]
    else:
        iterator, weights_i = no_of_treat, [None] * no_of_treat
    for jdx in range(iterator):
        weights_t = weights_i_np[weights_i_np[:, jdx+1] > 1e-14]
        weights_ti = np.int32(weights_t[:, 0])  # Indices
        weights_tw = (weights_t[:, jdx+1] / np.sum(weights_t[:, jdx+1])
                      if normalize else weights_t[:, jdx+1].copy())
        weights_tw = weights_tw.astype(np.float32)
        weights_i[jdx] = [weights_ti, weights_tw]
        if regrf:
            break
    return weights_i
